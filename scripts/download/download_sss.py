#!/usr/bin/env python3
"""
OceanEmbed SMAP RSS L3 SSS acquisition via earthaccess.

CMR: SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6
Native product is an 8-day running mean generated daily.
This script does not treat those files as independent daily observations.
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
            description="SMAP RSS L3 SSS acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    code = run_earthaccess_source(
        source_key="sss",
        title="OceanEmbed - SMAP RSS L3 SSS Acquisition Module",
        config_path=args.config,
        test=args.test,
        force=args.force,
        output_dir_override=args.output_dir,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
