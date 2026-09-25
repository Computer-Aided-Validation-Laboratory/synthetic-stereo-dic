from __future__ import annotations

from verif0_opencv_calibration import main as opencv_main
from verif0_pyvale_identify_calibration import main as pyvale_main
from verif0_riley_render_caltarget import main as riley_main


def main() -> None:
    print("rendering Riley Case 0")
    riley_main()
    print("identifying calibration with Pyvale")
    pyvale_main()
    print("identifying calibration with OpenCV and updating comparison")
    opencv_main()


if __name__ == "__main__":
    main()
