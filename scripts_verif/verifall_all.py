"""Render and calibrate every Verif0 Brown--Conrady case."""

from __future__ import annotations

from verifall_opencv_calibration import main as opencv_main
from verifall_pyvale_identify_calibration import main as pyvale_main
from verifall_riley_render_caltarget import main as riley_main


def main() -> None:
    print("rendering Riley cases 0--6")
    riley_main()
    print("identifying all cases with Pyvale")
    pyvale_main()
    print("identifying all cases with OpenCV and updating comparisons")
    opencv_main()


if __name__ == "__main__":
    main()
