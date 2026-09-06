#!/usr/bin/env python3
"""
OceanEmbed NOAA OISST v2.1 SST acquisition.

Preferred route: NOAA PSL THREDDS / OPeNDAP spatial-temporal subset.
Fallback: official PSL HTTPS annual file, then local subset.
No regrid. Native 0.25° daily SST only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List
from urllib.request import urlretrieve

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    add_common_args,
    append_manifest,
    build_filename,
    credentials_configured,
    expected_output_vars,
    file_size_mb,
    get_poc_bbox,
    get_source,
    get_time_window,
    load_configuration,
    manifest_path,
    planned_chunks,
    print_banner,
    print_specs,
    print_test_banner,
    raw_output_dir,
    select_variables,
    skip_existing,
    spatial_subset,
    time_subset,
    validate_netcdf,
    write_netcdf_atomic,
)


def parse_arguments():
    parser = add_common_args(
        argparse.ArgumentParser(
            description="NOAA OISST v2.1 SST acquisition for OceanEmbed",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def _open_remote(url: str) -> Any:
    import xarray as xr

    return xr.open_dataset(url)


def _download_https(url: str, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    destination = staging / Path(url).name
    if destination.exists() and destination.stat().st_size > 1024:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".partial")
    print(f"[*] HTTPS fallback download: {url}")
    urlretrieve(url, tmp)
    tmp.replace(destination)
    return destination


def _open_year_dataset(source: dict, year: int, staging: Path) -> Any:
    opendap = str(source["opendap_url"]).format(year=year)
    https = str(source["https_url"]).format(year=year)
    try:
        print(f"[*] Opening OPeNDAP: {opendap}")
        return _open_remote(opendap)
    except Exception as exc:
        print(f"[!] OPeNDAP failed ({exc.__class__.__name__}). Using HTTPS annual file.")
        local = _download_https(https, staging)
        import xarray as xr

        return xr.open_dataset(local)


def run(args) -> int:
    print_banner("OceanEmbed - NOAA OISST v2.1 SST Acquisition Module")
    print(f"[*] Loading configuration from: {args.config.resolve()}")
    config = load_configuration(args.config)
    source = get_source(config, "sst")
    bbox = get_poc_bbox(config)
    start, end = get_time_window(config, test=args.test)
    if args.test:
        print_test_banner(start, end)

    flags = credentials_configured()
    print(f"[*] Copernicus credentials configured: {flags['copernicus']}")
    print(f"[*] Earthdata credentials configured: {flags['earthdata']}")
    print("[*] NOAA OISST requires no login")

    output_dir = raw_output_dir(config, source, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".staging"
    log_path = manifest_path(output_dir, config)
    variables = list(source["variables"])
    variable_tag = "_".join(expected_output_vars(source))
    chunks = planned_chunks(config, start, end, test=args.test)

    print_specs(
        [
            ("Dataset ID", source.get("dataset_id")),
            ("Variables", ", ".join(variables)),
            ("Region", bbox["name"]),
            ("Latitude", f"[{bbox['lat_min']}N, {bbox['lat_max']}N]"),
            ("Longitude", f"[{bbox['lon_min']}E, {bbox['lon_max']}E]"),
            ("Time range", f"{start} to {end}"),
            ("Native res", f"{source.get('native_resolution_deg')} deg, {source.get('native_temporal')}"),
            ("Output dir", output_dir.resolve()),
        ]
    )

    created: List[Path] = []
    skipped: List[Path] = []
    open_cache = {}
    try:
        for chunk_start, chunk_end in chunks:
            filename = build_filename("sst", variable_tag, bbox, chunk_start, chunk_end, args.test)
            output_file = output_dir / filename
            if skip_existing(output_file, force=args.force, required_vars=variables):
                skipped.append(output_file)
                append_manifest(
                    log_path,
                    {
                        "source": "sst",
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

            years = range(chunk_start.year, chunk_end.year + 1)
            pieces = []
            for year in years:
                if year not in open_cache:
                    open_cache[year] = _open_year_dataset(source, year, staging)
                ds = open_cache[year]
                selected = select_variables(ds, variables)
                selected = time_subset(selected, chunk_start, chunk_end)
                selected = spatial_subset(selected, bbox)
                pieces.append(selected.load())

            if len(pieces) == 1:
                out = pieces[0]
            else:
                import xarray as xr

                out = xr.concat(pieces, dim="time")

            if out["sst"].size == 0:
                raise RuntimeError("OISST subset is empty. Check bbox/time against the PSL file.")
            write_netcdf_atomic(out, output_file)
            ok, reason = validate_netcdf(output_file, required_vars=variables)
            if not ok:
                raise RuntimeError(f"OISST file failed validation: {reason}")
            created.append(output_file)
            append_manifest(
                log_path,
                {
                    "source": "sst",
                    "status": "success",
                    "product": source.get("dataset_id"),
                    "variables": variables,
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
                "source": "sst",
                "status": "failure",
                "product": source.get("dataset_id"),
                "variables": variables,
                "start": str(start),
                "end": str(end),
                "bounds": bbox,
                "error": str(exc),
            },
        )
        print(f"[ERROR] SST download failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for ds in open_cache.values():
            try:
                ds.close()
            except Exception:
                pass

    print(f"[DONE] sst: created={len(created)} skipped={len(skipped)}")
    return 0


def main() -> None:
    sys.exit(run(parse_arguments()))


if __name__ == "__main__":
    main()
