from __future__ import annotations

import csv

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from verif0_pyvale_identify_calibration import detect_case
from verif_common import FOCAL, PIXEL_SIZE, distortion, output_dir, stereo_ground_truth


def identify_case(case: int) -> None:
    out = output_dir(case)
    dots0, dots1, grid, _, _ = detect_case(case)
    object_points = [np.asarray(item, dtype=np.float32) for item in grid]
    image_points0 = [np.asarray(item, dtype=np.float32) for item in dots0]
    image_points1 = [np.asarray(item, dtype=np.float32) for item in dots1]
    image_size = (1280, 960)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-12)
    calibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if case == 0:
        calibration_flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    riley_intrinsics = np.array(
        [[FOCAL / PIXEL_SIZE[0], 0.0, 640.0],
         [0.0, FOCAL / PIXEL_SIZE[1], 480.0],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    rms0, k0, d0, rvecs0, tvecs0 = cv2.calibrateCamera(
        object_points, image_points0, image_size, riley_intrinsics.copy(), np.zeros(5),
        flags=calibration_flags, criteria=criteria)
    rms1, k1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(
        object_points, image_points1, image_size, riley_intrinsics.copy(), np.zeros(5),
        flags=calibration_flags, criteria=criteria)
    d0 = np.asarray(d0).reshape(-1)
    d1 = np.asarray(d1).reshape(-1)
    # Retain the independent single-camera estimate for comparison.  Case 0
    # subsequently fixes the known ideal intrinsics for stereo calibration.
    single_k0, single_d0 = k0.copy(), d0.copy()
    single_k1, single_d1 = k1.copy(), d1.copy()
    if case == 0:
        k0 = riley_intrinsics.copy()
        k1 = riley_intrinsics.copy()
        d0 = np.zeros(5, dtype=np.float64)
        d1 = np.zeros(5, dtype=np.float64)
    r = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    e = np.empty(3, dtype=np.float64)
    f = np.empty((3, 3), dtype=np.float64)
    stereo_rms, k0, d0, k1, d1, r, t, e, f = cv2.stereoCalibrate(
        object_points, image_points0, image_points1, k0, d0, k1, d1, image_size,
        r, t, e, f, cv2.CALIB_FIX_INTRINSIC, criteria)
    storage = cv2.FileStorage(str(out / "opencv_stereo_calibration.yaml"), cv2.FILE_STORAGE_WRITE)
    storage.write("image_size", np.asarray(image_size))
    storage.write("K0", k0)
    storage.write("D0", d0)
    storage.write("K1", k1)
    storage.write("D1", d1)
    storage.write("R", r)
    storage.write("T", t)
    storage.write("E", e)
    storage.write("F", f)
    storage.release()
    values = distortion(case)
    riley_distortion = (values["distortion_k1"], values["distortion_k2"],
                        values["distortion_p1"], values["distortion_p2"],
                        values["distortion_k3"])
    identified = {
        "cam0_fx_px": single_k0[0, 0], "cam0_fy_px": single_k0[1, 1], "cam0_fs_px": single_k0[0, 1],
        "cam0_cx_px": single_k0[0, 2], "cam0_cy_px": single_k0[1, 2],
        "cam0_k1": single_d0[0], "cam0_k2": single_d0[1], "cam0_p1": single_d0[2],
        "cam0_p2": single_d0[3], "cam0_k3": single_d0[4],
        "cam1_fx_px": single_k1[0, 0], "cam1_fy_px": single_k1[1, 1], "cam1_fs_px": single_k1[0, 1],
        "cam1_cx_px": single_k1[0, 2], "cam1_cy_px": single_k1[1, 2],
        "cam1_k1": single_d1[0], "cam1_k2": single_d1[1], "cam1_p1": single_d1[2],
        "cam1_p2": single_d1[3], "cam1_k3": single_d1[4],
    }
    rows = [("parameter", "riley_opencv", "pyvale_identified", "opencv_identified")]
    riley_cam = (FOCAL / PIXEL_SIZE[0], FOCAL / PIXEL_SIZE[1], 0.0, 640.0, 480.0)
    for camera, (matrix, coefficients) in enumerate(((k0, d0), (k1, d1))):
        for name, value in zip(("fx_px", "fy_px", "fs_px", "cx_px", "cy_px"), riley_cam):
            rows.append((f"cam{camera}_{name}", value, "", identified[f"cam{camera}_{name}"]))
        for name, value in zip(("k1", "k2", "p1", "p2", "k3"), riley_distortion):
            rows.append((f"cam{camera}_{name}", value, "", coefficients[("k1", "k2", "p1", "p2", "k3").index(name)]))
    translation, rotation = stereo_ground_truth(case)
    opencv_rotation = Rotation.from_matrix(r).as_euler("xyz", degrees=True)
    for name, riley_value, opencv_value in zip(
        ("tx_mm", "ty_mm", "tz_mm", "theta_deg", "phi_deg", "psi_deg"),
        (*translation, *rotation), (*t.reshape(-1), *opencv_rotation),
    ):
        rows.append((name, riley_value, "", opencv_value))
    comparison_path = out / "comparison.csv"
    existing = {}
    if comparison_path.exists():
        with comparison_path.open(newline="") as stream:
            existing = {row["parameter"]: row.get("pyvale_identified", "") for row in csv.DictReader(stream)}
    with comparison_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*rows[0], "pyvale_minus_riley", "opencv_minus_riley"))
        for row in rows[1:]:
            pyvale_value = existing.get(row[0], "")
            opencv_value = row[3]
            pyvale_delta = "" if pyvale_value == "" else float(pyvale_value) - row[1]
            opencv_delta = "" if opencv_value == "" else opencv_value - row[1]
            writer.writerow((row[0], row[1], pyvale_value, opencv_value,
                             pyvale_delta, opencv_delta))
    print(f"case {case}: OpenCV RMS cam0={rms0:.6f} cam1={rms1:.6f} stereo={stereo_rms:.6f}")


def main() -> None:
    identify_case(0)


if __name__ == "__main__":
    main()
