"""Calibrate every existing Verif0 render without invoking Riley."""

from __future__ import annotations

from verifall_opencv_calibration import main as opencv_main
from verifall_pyvale_identify_calibration import main as pyvale_main


def main() -> None:
    print("identifying all existing cases with Pyvale")
    pyvale_main()
    print("identifying all existing cases with OpenCV and updating comparisons")
    opencv_main()


if __name__ == "__main__":
    main()
