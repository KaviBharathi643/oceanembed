#!/usr/bin/env python3
"""
OceanEmbed DUACS SLA acquisition.

Copernicus dataset cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D.
Preserves native 0.125° resolution. No regrid during download.
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
            description="Copernicus DUACS SLA acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    code = run_copernicus_source(
        source_key="sla",
        title="OceanEmbed - Copernicus DUACS SLA Acquisition Module",
        config_path=args.config,
        test=args.test,
        force=args.force,
        output_dir_override=args.output_dir,
        include_depth=False,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
