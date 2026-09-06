#!/usr/bin/env python3
"""
OceanEmbed multi-source acquisition runner.

Default behavior is --test for all enabled sources.
Full-year download requires an explicit --full flag and is not started
by this framework task.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import credentials_configured, get_project_root, load_configuration  # noqa: E402

DEFAULT_ORDER = ["glorys", "sst", "sla", "sss", "oscar", "ccmp", "argo"]
SCRIPTS = {
    "glorys": "download_glorys.py",
    "sst": "download_sst.py",
    "sla": "download_sla.py",
    "sss": "download_sss.py",
    "oscar": "download_oscar.py",
    "ccmp": "download_ccmp.py",
    "argo": "download_argo.py",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OceanEmbed downloaders in a controlled order",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=get_project_root() / "config" / "dataset_config.yaml",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run each selected source in --test mode (3-day window from config).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the configured full period. Refused unless explicitly passed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forward --force to each downloader.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated source keys to run (default: all enabled sources).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining sources if one fails.",
    )
    return parser.parse_args()


def selected_sources(config: dict, only: str) -> list:
    requested = [item.strip() for item in only.split(",") if item.strip()] if only else list(DEFAULT_ORDER)
    unknown = [key for key in requested if key not in SCRIPTS]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Valid: {list(SCRIPTS)}")
    enabled = []
    for key in requested:
        source = config.get("sources", {}).get(key, {})
        if source.get("enabled", True):
            enabled.append(key)
        else:
            print(f"[!] Skipping disabled source: {key}")
    return enabled


def main() -> None:
    args = parse_arguments()
    if args.full and args.test:
        raise SystemExit("Use either --test or --full, not both.")
    if not args.full and not args.test:
        print(
            "Refusing to start a full-year download by default.\n"
            "Pass --test for 3-day validation, or --full after explicit approval.",
            file=sys.stderr,
        )
        sys.exit(2)

    config = load_configuration(args.config)
    flags = credentials_configured()
    print(f"[*] Copernicus credentials configured: {flags['copernicus']}")
    print(f"[*] Earthdata credentials configured: {flags['earthdata']}")

    sources = selected_sources(config, args.only)
    if not sources:
        raise SystemExit("No sources selected.")

    python = sys.executable
    failures = []
    for key in sources:
        script = _SCRIPT_DIR / SCRIPTS[key]
        cmd = [python, str(script), "--config", str(args.config)]
        if args.test:
            cmd.append("--test")
        if args.force:
            cmd.append("--force")
        print("\n" + "=" * 70)
        print(f"RUNNING {key}: {' '.join(cmd)}")
        print("=" * 70)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures.append(key)
            if not args.continue_on_error:
                print(f"[ERROR] {key} failed with exit code {completed.returncode}", file=sys.stderr)
                sys.exit(completed.returncode)
            print(f"[ERROR] {key} failed; continuing because --continue-on-error is set.")

    if failures:
        print(f"[ERROR] Failed sources: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("[DONE] Selected downloaders completed.")


if __name__ == "__main__":
    main()
