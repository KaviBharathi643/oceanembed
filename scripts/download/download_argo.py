#!/usr/bin/env python3
"""
OceanEmbed Argo profile acquisition for independent validation.

Primary: Ifremer/Coriolis ERDDAP tabledap (ArgoFloats).
Fallback: argopy DataFetcher.
Not a model input. No regrid, no collocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    add_common_args,
    append_manifest,
    build_filename,
    credentials_configured,
    file_size_mb,
    get_poc_bbox,
    get_source,
    get_time_window,
    load_configuration,
    manifest_path,
    network_settings,
    planned_chunks,
    print_banner,
    print_specs,
    print_test_banner,
    raw_output_dir,
    retry_call,
    skip_existing,
    validate_netcdf,
    write_netcdf_atomic,
)


def parse_arguments():
    parser = add_common_args(
        argparse.ArgumentParser(
            description="Argo independent-validation acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def erddap_url(source: dict, bbox: dict, start, end) -> str:
    base = str(source["erddap_base"]).rstrip("/")
    variables = ",".join(source["variables"])
    return (
        f"{base}.nc?{variables}"
        f"&time>={start.isoformat()}T00:00:00Z"
        f"&time<={end.isoformat()}T23:59:59Z"
        f"&latitude>={bbox['lat_min']}"
        f"&latitude<={bbox['lat_max']}"
        f"&longitude>={bbox['lon_min']}"
        f"&longitude<={bbox['lon_max']}"
    )


def download_erddap(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    with requests.get(url, stream=True, timeout=180) as response:
        if response.status_code != 200:
            snippet = response.text[:300].replace("\n", " ")
            raise RuntimeError(f"ERDDAP HTTP {response.status_code}: {snippet}")
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise RuntimeError("ERDDAP returned HTML instead of NetCDF.")
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    if tmp.stat().st_size < 512:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ERDDAP response was empty or too small.")
    tmp.replace(destination)


def download_argopy(source: dict, bbox: dict, start, end, destination: Path) -> None:
    try:
        from argopy import DataFetcher
    except ImportError as exc:
        raise RuntimeError("argopy is not installed. Run: pip install -r requirements.txt") from exc

    box = [
        bbox["lon_min"],
        bbox["lon_max"],
        bbox["lat_min"],
        bbox["lat_max"],
        0,
        2000,
        start.isoformat(),
        end.isoformat(),
    ]
    print("[*] ERDDAP failed or incomplete; falling back to argopy ...")
    ds = DataFetcher(src="erddap").region(box).to_xarray()
    keep = [name for name in source["variables"] if name in ds.data_vars or name in ds.coords]
    if "TEMP" not in ds and "TEMP" not in ds.data_vars:
        ds.close()
        raise RuntimeError("argopy result does not contain TEMP.")
    subset = ds[keep] if keep else ds
    subset.attrs["oceanembed_role"] = "independent_validation"
    write_netcdf_atomic(subset, destination)
    subset.close()
    ds.close()


def run(args) -> int:
    print_banner("OceanEmbed - Argo Independent Validation Acquisition")
    print(f"[*] Loading configuration from: {args.config.resolve()}")
    config = load_configuration(args.config)
    source = get_source(config, "argo")
    bbox = get_poc_bbox(config)
    start, end = get_time_window(config, test=args.test)
    if args.test:
        print_test_banner(start, end)

    flags = credentials_configured()
    print(f"[*] Copernicus credentials configured: {flags['copernicus']}")
    print(f"[*] Earthdata credentials configured: {flags['earthdata']}")
    print("[*] Argo ERDDAP requires no login")

    output_dir = raw_output_dir(config, source, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = manifest_path(output_dir, config)
    variables = list(source["variables"])
    variable_tag = "TEMP_PRES"
    chunks = planned_chunks(config, start, end, test=args.test)
    required = ["TEMP", "PRES"]

    print_specs(
        [
            ("Dataset", source.get("dataset_id")),
            ("Role", source.get("role")),
            ("Variables", ", ".join(variables)),
            ("Region", bbox["name"]),
            ("Latitude", f"[{bbox['lat_min']}N, {bbox['lat_max']}N]"),
            ("Longitude", f"[{bbox['lon_min']}E, {bbox['lon_max']}E]"),
            ("Time range", f"{start} to {end}"),
            ("Notes", source.get("notes", "")),
            ("Output dir", output_dir.resolve()),
        ]
    )

    created: List[Path] = []
    skipped: List[Path] = []
    try:
        for chunk_start, chunk_end in chunks:
            filename = build_filename("argo", variable_tag, bbox, chunk_start, chunk_end, args.test)
            output_file = output_dir / filename
            if skip_existing(output_file, force=args.force, required_vars=required):
                skipped.append(output_file)
                append_manifest(
                    log_path,
                    {
                        "source": "argo",
                        "status": "skipped",
                        "product": source.get("dataset_id"),
                        "variables": variables,
                        "start": str(chunk_start),
                        "end": str(chunk_end),
                        "bounds": bbox,
                        "file": str(output_file),
                        "size_mb": round(file_size_mb(output_file), 3),
                    },
                )
                continue

            url = erddap_url(source, bbox, chunk_start, chunk_end)
            print(f"[*] Querying Ifremer ERDDAP {chunk_start} to {chunk_end} ...")
            settings = network_settings(config)
            try:
                retry_call(
                    lambda: download_erddap(url, output_file),
                    retries=settings["retries"],
                    backoff=settings["backoff_seconds"],
                    what=f"ERDDAP query {chunk_start}",
                )
            except Exception as erddap_exc:
                print(f"[!] ERDDAP download failed: {erddap_exc}")
                download_argopy(source, bbox, chunk_start, chunk_end, output_file)

            ok, reason = validate_netcdf(output_file, required_vars=required)
            if not ok:
                raise RuntimeError(f"Argo file failed validation: {reason}")
            created.append(output_file)
            append_manifest(
                log_path,
                {
                    "source": "argo",
                    "status": "success",
                    "product": source.get("dataset_id"),
                    "variables": variables,
                    "role": "independent_validation",
                    "start": str(chunk_start),
                    "end": str(chunk_end),
                    "bounds": bbox,
                    "file": str(output_file),
                    "size_mb": round(file_size_mb(output_file), 3),
                },
            )
            print(f"[SUCCESS] {output_file.name} ({file_size_mb(output_file):.2f} MB)")
    except Exception as exc:
        append_manifest(
            log_path,
            {
                "source": "argo",
                "status": "failure",
                "product": source.get("dataset_id"),
                "variables": variables,
                "start": str(start),
                "end": str(end),
                "bounds": bbox,
                "error": str(exc),
            },
        )
        print(f"[ERROR] Argo download failed: {exc}", file=sys.stderr)
        return 1

    print(f"[DONE] argo: created={len(created)} skipped={len(skipped)}")
    return 0


def main() -> None:
    sys.exit(run(parse_arguments()))


if __name__ == "__main__":
    main()
