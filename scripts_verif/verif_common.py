from __future__ import annotations

import argparse
import copy
import csv
import re
from pathlib import Path
import shutil

import numpy as np
import riley
from scipy.spatial.transform import Rotation

CASES = {
    0: (0.0, 0.0, 0.0, 0.0, 0.0),
    1: (-0.15, 0.5, 0.0, 0.0, 0.0),
    2: (0.15, -0.3, 0.0, 0.0, 0.0),
    3: (-0.50, 5.0, -10.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0, 0.002, -0.0015),
    5: (-0.25, 1.2, -0.5, 0.001, 0.0008),
    6: (-1.8, 12.0, -25.0, 0.005, -0.004),
}
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "out"
PIXELS = (1280, 960)
PIXEL_SIZE = (4.0e-6, 4.0e-6)
FOCAL = 50.0e-3
MATCHED_ROI = (0.0125, 0.0175, 0.0005)
MATCHED_CAM0_POS = (0.0125, 0.0175, 0.160864856482)
MATCHED_CAM1_POS = (0.067348011198, 0.0175, 0.151193672270)
CAMERA_PAIR_SCALE = 0.6
POSES = 128
FOV_FRACTION = 1.0
# These ranges deliberately constrain dot centres, not the complete plate, to the
# two camera sensors.  The plate edge is allowed to leave the image.
MOTION_LIMITS = riley.CalTargetMotionLimits(
    translation=((-0.001, 0.001), (-0.001, 0.001), (-0.002, 0.002)),
    rotation_deg=((-7.0, 7.0), (-7.0, 7.0), (-7.0, 7.0)),
)


def distortion(case: int) -> dict[str, float]:
    k1, k2, k3, p1, p2 = CASES[case]
    return {"distortion_model": 1, "distortion_k1": k1, "distortion_k2": k2,
            "distortion_k3": k3, "distortion_p1": p1, "distortion_p2": p2}


def target_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = riley.data.stereocal_case_path()
    coords = riley.load_csv(data / "coords.csv")
    connect = riley.load_csv(data / "connect.csv", dtype=np.int64)
    uvs = riley.load_csv(data / "uvs.csv")
    center = riley.roi_cent_from_coords(coords)
    shift = np.asarray(MATCHED_ROI) - center
    coords = coords + shift
    return coords, connect, uvs, np.asarray(MATCHED_ROI)


def target_texture() -> np.ndarray:
    return riley.load_texture_mono_u8(riley.data.cal_target_texture_path())


def cameras(case: int, coords: np.ndarray, center: np.ndarray) -> tuple[riley.Camera, riley.Camera]:
    cam0_rot = (0.0, 0.0, 0.0)
    cam0_pos = riley.pos_frame_coords(
        coords, PIXELS, PIXEL_SIZE, FOCAL, cam0_rot,
        fov_scale=1.10, fit_mode=riley.FrameFitMode.contain,
    )
    
    cam1_rot = (0.0, float(np.deg2rad(20.0)), 0.0)
    cam1_pos = riley.pos_frame_coords(
        coords, PIXELS, PIXEL_SIZE, FOCAL, cam1_rot,
        fov_scale=1.10, fit_mode=riley.FrameFitMode.contain,
    )
    
    cam0 = riley.Camera(
        pixels_num=PIXELS, pixels_size=PIXEL_SIZE, pos_world=cam0_pos,
        rot_world=cam0_rot, roi_cent_world=tuple(center), focal_length=FOCAL,
        sub_sample=8, **distortion(case),
    )

    cam1 = copy.deepcopy(cam0)
    cam1.pos_world = cam1_pos
    cam1.rot_world = (0.0, float(np.deg2rad(20.0)), 0.0)

    return cam0, cam1


def stereo_ground_truth(case: int) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV camera-1-from-camera-0 translation (mm) and XYZ Euler angles."""
    coords, _, _, center = target_geometry()
    cam0, cam1 = cameras(case, coords, center)
    rot0 = Rotation.from_euler("ZYX", np.asarray(cam0.rot_world)[::-1])
    rot1 = Rotation.from_euler("ZYX", np.asarray(cam1.rot_world)[::-1])
    opengl_to_opencv = np.diag((1.0, -1.0, -1.0))
    rotation = opengl_to_opencv @ (rot1.inv() * rot0).as_matrix() @ opengl_to_opencv
    translation = -rotation @ (opengl_to_opencv @ (
        np.asarray(cam1.pos_world) - np.asarray(cam0.pos_world))) * 1.0e3
    return translation, Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)


def motion(coords: np.ndarray, cam0: riley.Camera, cam1: riley.Camera) -> tuple[np.ndarray, ...]:
    disp = riley.caltarget_motion_from_limits(
        coords, POSES, limits=MOTION_LIMITS,
        sampling=riley.ECalTargetMotionSampling.HYPERCUBE,
        include_reference_frame=True, seed=2026,
    )
    disp = tuple(np.ascontiguousarray(item.copy()) for item in disp)
    if any(item.shape[1] != POSES for item in disp):
        raise RuntimeError(f"Riley did not generate exactly {POSES} poses")
    return disp


def output_dir(case: int) -> Path:
    return OUT_ROOT / f"verif{case}_brownconrady"


def render_case(case: int) -> Path:
    out = output_dir(case)
    if out.exists():
        shutil.rmtree(out)
    (out / "riley").mkdir(parents=True)
    coords, connect, uvs, center = target_geometry()
    cam0, cam1 = cameras(case, coords, center)
    disp = motion(coords, cam0, cam1)
    riley.save_stereo_pair(str(out), "stereo_data_opengl.csv", cam0, cam1)
    cam0_cv = copy.copy(cam0)
    cam1_cv = copy.copy(cam1)
    cam0_cv.coord_sys = riley.CameraCoordSys.opencv
    cam1_cv.coord_sys = riley.CameraCoordSys.opencv
    riley.save_stereo_pair(str(out), "stereo_data_opencv.csv", cam0_cv, cam1_cv)
    mesh = riley.create_mesh(
        convention=riley.ConnectConvention(riley.EElemType.TRI3, riley.EConnectAxis.ROW, 0,
                                             riley.ENodeOrder.RILEY),
        mesh_type=riley.MeshType.tri3, coords=coords, connect=connect, disp=disp,
        shader=riley.TextureShader(uvs=uvs, texture=target_texture()))
    config = riley.create_raster_config(num_frames=POSES, total_threads=8,
                                        save_strategy=riley.SaveStrategy.disk)
    config.background_value = 128.0
    config.save_scaling = riley.ScaleStrategy.none
    riley.raster([mesh], [cam0, cam1], config, out_dir=str(out / "riley"))
    return out


def images(out: Path) -> tuple[list[Path], list[Path]]:
    files = sorted((out / "riley").glob("*.bmp"))
    cam0 = [p for p in files if re.search(r"cam0", p.stem, re.I)]
    cam1 = [p for p in files if re.search(r"cam1", p.stem, re.I)]
    if len(cam0) != POSES or len(cam1) != POSES:
        raise RuntimeError(f"Expected {POSES} images per camera, found {len(cam0)} and {len(cam1)}")
    return cam0, cam1


def args(cases: list[int] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=sorted(CASES))
    parser.add_argument("--cases", type=int, nargs="+", choices=sorted(CASES))
    return parser.parse_args()


def selected(ns: argparse.Namespace) -> list[int]:
    return ns.cases if ns.cases else ([ns.case] if ns.case is not None else sorted(CASES))
