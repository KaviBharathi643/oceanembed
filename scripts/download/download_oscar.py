#!/usr/bin/env python3
"""
OceanEmbed OSCAR surface currents acquisition via earthaccess.

CMR: OSCAR_L4_OC_FINAL_V2.0
Native variables u, v are stored as current_u, current_v.
No regrid.
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
            description="OSCAR surface currents acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    code = run_earthaccess_source(
        source_key="oscar",
        title="OceanEmbed - OSCAR Surface Currents Acquisition Module",
        config_path=args.config,
        test=args.test,
        force=args.force,
        output_dir_override=args.output_dir,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
