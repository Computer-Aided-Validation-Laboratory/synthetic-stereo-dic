from __future__ import annotations

from verif0_opencv_calibration import identify_case
from verif_common import args, selected


def main() -> None:
    for case in selected(args()):
        identify_case(case)


if __name__ == "__main__":
    main()
