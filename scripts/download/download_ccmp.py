#!/usr/bin/env python3
"""
OceanEmbed CCMP v3.1 10 m winds acquisition via earthaccess.

CMR: CCMP_WINDS_10M6HR_L4_V3.1
Native temporal resolution is 6-hourly. This script does not aggregate to daily.
Native variables uwnd, vwnd are stored as wind_u, wind_v.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import add_common_args, run_earthaccess_source  # noqa: E402


def parse_arguments():
    parser = add_common_args(
        argparse.ArgumentParser(
            description="CCMP v3.1 winds acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    code = run_earthaccess_source(
        source_key="ccmp",
        title="OceanEmbed - CCMP v3.1 Winds Acquisition Module",
        config_path=args.config,
        test=args.test,
        force=args.force,
        output_dir_override=args.output_dir,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
