from __future__ import annotations

from verif_common import args, render_case, selected


def main() -> None:
    for case in selected(args()):
        print(f"rendering verification case {case}")
        print(render_case(case))


if __name__ == "__main__":
    main()
