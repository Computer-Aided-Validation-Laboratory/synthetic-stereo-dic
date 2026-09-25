from __future__ import annotations

import csv

import cv2
import numpy as np
import pyvale.calib as calib

from verif_common import FOCAL, PIXEL_SIZE, distortion, images, output_dir, stereo_ground_truth


def detect_case(case: int):
    out = output_dir(case)
    cam0, cam1 = images(out)
    dots0 = []
    dots1 = []
    grid = []
    files0 = []
    files1 = []
    for image0, image1 in zip(cam0, cam1):
        try:
            detected0, detected1, detected_grid, detected_files0, detected_files1 = calib.detect_dots(
                cam0=[image0], cam1=[image1], grid_height=9, grid_width=12,
                grid_spacing=1.25, hollow_dots=[(9, 6), (2, 2), (2, 6)],
                min_dot_fraction=0.5,
            )
        except (RuntimeError, ValueError, IndexError):
            continue
        if detected0 and detected1:
            values = distortion(case)
            coefficients = np.array((values["distortion_k1"], values["distortion_k2"],
                                     values["distortion_p1"], values["distortion_p2"],
                                     values["distortion_k3"]), dtype=np.float64)
            camera_matrix = np.array(((FOCAL / PIXEL_SIZE[0], 0.0, 640.0),
                                      (0.0, FOCAL / PIXEL_SIZE[1], 480.0),
                                      (0.0, 0.0, 1.0)), dtype=np.float64)
            object_points = np.asarray(detected_grid[0], dtype=np.float32)
            valid = True
            for image_points in (detected0[0], detected1[0]):
                solved, rotation, translation = cv2.solvePnP(
                    object_points, np.asarray(image_points, dtype=np.float32),
                    camera_matrix, coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
                if not solved:
                    valid = False
                    break
                projected, _ = cv2.projectPoints(object_points, rotation, translation,
                                                 camera_matrix, coefficients)
                error = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
                if error > 1.0:
                    valid = False
                    break
            if not valid:
                continue
        dots0.extend(detected0)
        dots1.extend(detected1)
        grid.extend(detected_grid)
        files0.extend(detected_files0)
        files1.extend(detected_files1)
    if not grid or len(dots0) < 50 or len(dots1) < 50:
        raise RuntimeError(f"Pyvale accepted {len(dots0)}/{len(dots1)} pairs, expected at least 50")
    return dots0, dots1, grid, files0, files1


def identify_case(case: int) -> None:
    out = output_dir(case)
    dots0, dots1, grid, files0, files1 = detect_case(case)
    print(f"case {case}: frame 0 accepted dots={len(dots0[0])}, total pairs={len(dots0)}")
    identified, err0, err1 = calib.calibrate_stereo(
        dots0, dots1, grid, [1280, 960], optimize_distortion=case != 0,
        num_threads=8, error_formulation="RMSE",
    )
    calib.savetxt(identified, out / "pyvale_identified.txt")
    np.savetxt(out / "pyvale_reprojection_cam0.txt", err0)
    np.savetxt(out / "pyvale_reprojection_cam1.txt", err1)
    rows = [("parameter", "riley_opencv", "pyvale_identified", "opencv_identified")]
    riley_cam = (FOCAL / PIXEL_SIZE[0], FOCAL / PIXEL_SIZE[1], 0.0,
                 1280.0 / 2.0, 960.0 / 2.0)
    for cam_index, py_cam in enumerate((identified.cam0, identified.cam1)):
        for name, riley_value, py_value in zip(
            ("fx_px", "fy_px", "fs_px", "cx_px", "cy_px"), riley_cam,
            (py_cam.fx, py_cam.fy, py_cam.fs, py_cam.cx, py_cam.cy)):
            rows.append((f"cam{cam_index}_{name}", riley_value, py_value, ""))
    translation, rotation = stereo_ground_truth(case)
    for name, riley_value, pyvale_value in zip(
        ("tx_mm", "ty_mm", "tz_mm", "theta_deg", "phi_deg", "psi_deg"),
        (*translation, *rotation), (*identified.translation, *identified.rotation),
    ):
        rows.append((name, riley_value, pyvale_value, ""))
        values = distortion(case)
        riley_distortion = (values["distortion_k1"], values["distortion_k2"],
                            values["distortion_p1"], values["distortion_p2"],
                            values["distortion_k3"])
        for name, riley_value, py_value in zip(
            ("k1", "k2", "p1", "p2", "k3"), riley_distortion,
            identified.cam0.distortion if cam_index == 0 else identified.cam1.distortion):
            rows.append((f"cam{cam_index}_{name}", riley_value, py_value, ""))
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*rows[0], "pyvale_minus_riley", "opencv_minus_riley"))
        for parameter, riley_value, pyvale_value, opencv_value in rows[1:]:
            pyvale_delta = "" if pyvale_value == "" else pyvale_value - riley_value
            opencv_delta = "" if opencv_value == "" else opencv_value - riley_value
            writer.writerow((parameter, riley_value, pyvale_value, opencv_value,
                             pyvale_delta, opencv_delta))
    print(f"case {case}: Pyvale calibration written to {out}")


def main() -> None:
    identify_case(0)


if __name__ == "__main__":
    main()
