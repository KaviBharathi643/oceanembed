#!/usr/bin/env python3
"""
OceanEmbed GLORYS12V1 acquisition (thetao).

Uses copernicusmarine.subset() with native 0.083° resolution and native
vertical levels from the surface through 1000 m. No regrid and no
interpolation to the 15 target depths.

Existing successful --test files are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import add_common_args, run_copernicus_source  # noqa: E402


def parse_arguments():
    parser = add_common_args(
        argparse.ArgumentParser(
            description="Copernicus GLORYS12V1 (thetao) acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    code = run_copernicus_source(
        source_key="glorys",
        title="OceanEmbed - Copernicus GLORYS Data Acquisition Module",
        config_path=args.config,
        test=args.test,
        force=args.force,
        output_dir_override=args.output_dir,
        include_depth=True,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
