#!/usr/bin/env python3
"""Shared acquisition helpers for OceanEmbed download scripts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

COPERNICUS_USER_VAR = "COPERNICUSMARINE_SERVICE_USERNAME"
COPERNICUS_PASS_VAR = "COPERNICUSMARINE_SERVICE_PASSWORD"
EARTHDATA_USER_VAR = "EARTHDATA_USERNAME"
EARTHDATA_PASS_VAR = "EARTHDATA_PASSWORD"

LAT_NAMES = ("latitude", "lat", "nav_lat", "y")
LON_NAMES = ("longitude", "lon", "nav_lon", "x")
TIME_NAMES = ("time", "time_counter", "valid_time", "t")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_env() -> None:
    dotenv_path = get_project_root() / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        load_dotenv(override=False)


def _env_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def credentials_configured() -> Dict[str, str]:
    """Return YES/NO flags only. Never include secret values."""
    load_env()
    return {
        "copernicus": "YES" if (_env_set(COPERNICUS_USER_VAR) and _env_set(COPERNICUS_PASS_VAR)) else "NO",
        "earthdata": "YES" if (_env_set(EARTHDATA_USER_VAR) and _env_set(EARTHDATA_PASS_VAR)) else "NO",
    }


def require_copernicus() -> Tuple[str, str]:
    load_env()
    username = os.getenv(COPERNICUS_USER_VAR, "").strip()
    password = os.getenv(COPERNICUS_PASS_VAR, "").strip()
    missing = [n for n, v in ((COPERNICUS_USER_VAR, username), (COPERNICUS_PASS_VAR, password)) if not v]
    if missing:
        raise RuntimeError(
            "Copernicus credentials not found in environment (.env).\n"
            f"Missing: {', '.join(missing)}\n"
            "Fill COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD."
        )
    os.environ[COPERNICUS_USER_VAR] = username
    os.environ[COPERNICUS_PASS_VAR] = password
    return username, password


def require_earthdata() -> Tuple[str, str]:
    load_env()
    username = os.getenv(EARTHDATA_USER_VAR, "").strip()
    password = os.getenv(EARTHDATA_PASS_VAR, "").strip()
    missing = [n for n, v in ((EARTHDATA_USER_VAR, username), (EARTHDATA_PASS_VAR, password)) if not v]
    if missing:
        raise RuntimeError(
            "Earthdata credentials not found in environment (.env).\n"
            f"Missing: {', '.join(missing)}\n"
            "Fill EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )
    os.environ[EARTHDATA_USER_VAR] = username
    os.environ[EARTHDATA_PASS_VAR] = password
    return username, password


def redact(text: str, *secrets: str) -> str:
    cleaned = str(text)
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "******")
    return cleaned


def load_configuration(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not config:
        raise ValueError(f"Configuration file is empty: {config_path}")
    return config


def parse_iso_date(value: Any) -> dt.date:
    return dt.datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()


def get_poc_bbox(config: Dict[str, Any]) -> Dict[str, Any]:
    region_key = config.get("acquisition", {}).get("region", "poc")
    poc = config.get("domain", {}).get(region_key, {})
    required = ("lat_min", "lat_max", "lon_min", "lon_max")
    missing = [k for k in required if k not in poc]
    if missing:
        raise KeyError(f"Missing domain.{region_key} keys: {missing}")
    lat_min, lat_max = float(poc["lat_min"]), float(poc["lat_max"])
    lon_min, lon_max = float(poc["lon_min"]), float(poc["lon_max"])
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError(f"Invalid bounds lat [{lat_min}, {lat_max}] lon [{lon_min}, {lon_max}]")
    return {
        "name": str(poc.get("name", region_key)),
        "slug": str(poc.get("name", region_key)).replace(" ", "_").lower(),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def get_time_window(config: Dict[str, Any], test: bool) -> Tuple[dt.date, dt.date]:
    time_cfg = config.get("time", {})
    start = parse_iso_date(time_cfg["start"])
    end = parse_iso_date(time_cfg["end"])
    if start > end:
        raise ValueError(f"Start date {start} is after end date {end}")
    if not test:
        return start, end
    test_start = parse_iso_date(time_cfg.get("test_start", start))
    test_end = parse_iso_date(time_cfg.get("test_end", start + dt.timedelta(days=2)))
    if test_start < start:
        test_start = start
    if test_end > end:
        test_end = end
    if test_start > test_end:
        raise ValueError(f"Invalid test window {test_start} to {test_end}")
    return test_start, test_end


def get_source(config: Dict[str, Any], source_key: str) -> Dict[str, Any]:
    sources = config.get("sources", {})
    if source_key not in sources:
        raise KeyError(f"Source '{source_key}' not found in config.sources")
    return dict(sources[source_key])


def raw_output_dir(config: Dict[str, Any], source: Dict[str, Any], override: Optional[Path] = None) -> Path:
    if override is not None:
        return Path(override)
    raw_root = config.get("acquisition", {}).get("raw_root", "data/raw")
    subdir = source.get("raw_subdir")
    if not subdir:
        raise KeyError("Source is missing raw_subdir")
    return get_project_root() / raw_root / subdir


def date_tag(value: dt.date) -> str:
    return value.strftime("%Y%m%d")


def month_chunks(start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    chunks: List[Tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            month_end = dt.date(cursor.year, 12, 31)
        else:
            month_end = dt.date(cursor.year, cursor.month + 1, 1) - dt.timedelta(days=1)
        chunk_end = min(end, month_end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def planned_chunks(config: Dict[str, Any], start: dt.date, end: dt.date, test: bool) -> List[Tuple[dt.date, dt.date]]:
    if test or (end - start).days < 28:
        return [(start, end)]
    mode = str(config.get("acquisition", {}).get("chunking", "monthly")).lower()
    if mode == "monthly":
        return month_chunks(start, end)
    return [(start, end)]


def add_common_args(parser: argparse.ArgumentParser, default_output: Optional[Path] = None) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=Path,
        default=get_project_root() / "config" / "dataset_config.yaml",
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Lightweight acquisition using time.test_start/test_end (does not modify config).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing valid output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Override destination directory.",
    )
    return parser


def print_banner(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_test_banner(start: dt.date, end: dt.date) -> None:
    print("\n" + "!" * 70)
    print(">>> TEST MODE ACTIVATED <<<")
    print(f"Downloading small subset only: {start} to {end}")
    print("Project configuration on disk is not modified.")
    print("!" * 70 + "\n")


def print_specs(rows: Sequence[Tuple[str, Any]]) -> None:
    print("-" * 70)
    print("DOWNLOAD SPECIFICATIONS:")
    for label, value in rows:
        print(f"  {label:<18}: {value}")
    print("-" * 70 + "\n")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def manifest_path(output_dir: Path, config: Dict[str, Any]) -> Path:
    name = config.get("acquisition", {}).get("manifest_name", "acquisition_manifest.jsonl")
    return output_dir / name


def append_manifest(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def find_coord(dataset: Any, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in dataset.coords or name in getattr(dataset, "variables", {}):
            return name
    dims = getattr(dataset, "dims", {})
    for name in names:
        if name in dims:
            return name
    return None


def spatial_subset(dataset: Any, bbox: Dict[str, Any]) -> Any:
    if "latitude" in dataset.dims and "lat" in dataset.coords and "lat" not in dataset.dims:
        dataset = dataset.swap_dims({"latitude": "lat"})
    if "longitude" in dataset.dims and "lon" in dataset.coords and "lon" not in dataset.dims:
        dataset = dataset.swap_dims({"longitude": "lon"})

    lat_name = find_coord(dataset, LAT_NAMES)
    lon_name = find_coord(dataset, LON_NAMES)
    if lat_name is None or lon_name is None:
        raise RuntimeError(f"Could not find lat/lon coordinates. Coords={list(dataset.coords)}")
    lat = dataset[lat_name]
    lon = dataset[lon_name]
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    if lat.size > 1 and float(lat[0]) > float(lat[-1]):
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    lon_values = lon.values if hasattr(lon, "values") else lon
    lon_min_data = float(min(lon_values))
    lon_max_data = float(max(lon_values))
    if lon_min_data >= 0 and lon_max_data > 180 and lon_min < 0:
        raise RuntimeError("Requested negative longitudes against a 0-360 dataset; convert first.")
    subset = dataset.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})
    if subset[lat_name].size == 0 or subset[lon_name].size == 0:
        raise RuntimeError("Spatial subset produced empty coordinates. Check longitude convention.")
    return subset


def time_subset(dataset: Any, start: dt.date, end: dt.date) -> Any:
    time_name = find_coord(dataset, TIME_NAMES)
    if time_name is None:
        return dataset
    start_ts = dt.datetime.combine(start, dt.time.min)
    end_ts = dt.datetime.combine(end, dt.time.max)
    return dataset.sel({time_name: slice(str(start_ts), str(end_ts))})


def format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def network_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    net = (config or {}).get("acquisition", {}).get("network", {})
    return {
        "timeout_seconds": int(net.get("timeout_seconds", 120)),
        "retries": int(net.get("retries", 5)),
        "backoff_seconds": float(net.get("backoff_seconds", 4.0)),
    }


def apply_requests_timeout(timeout_seconds: int) -> None:
    """Use a longer default timeout for Earthdata/CMR HTTPS calls."""
    import requests

    existing = getattr(requests.sessions.Session.request, "_oceanembed_timeout", None)
    if existing == timeout_seconds:
        return
    original = getattr(requests.sessions.Session.request, "_oceanembed_original", requests.sessions.Session.request)

    def patched(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("timeout", timeout_seconds)
        return original(self, method, url, **kwargs)

    patched._oceanembed_timeout = timeout_seconds  # type: ignore[attr-defined]
    patched._oceanembed_original = original  # type: ignore[attr-defined]
    requests.sessions.Session.request = patched  # type: ignore[method-assign]


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    retryable_words = ["timeout", "connection", "remoteendclosed", "temporarily", "reset", "503", "502", "504", "500", "429"]
    return any(w in name or w in text for w in retryable_words)


def retry_call(func: Callable[[], Any], *, retries: int, backoff: float, what: str) -> Any:
    last: Optional[BaseException] = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last = exc
            if (not _is_retryable(exc)) or attempt >= attempts:
                raise RuntimeError(
                    f"{what} failed after {attempt} attempt(s): {type(exc).__name__}: {exc}"
                ) from None
            delay = backoff * (2 ** (attempt - 1))
            print(f"[!] {what} failed ({type(exc).__name__}); retry {attempt}/{attempts} in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"{what} failed: {last}") from None


def sanitize_for_netcdf(dataset: Any) -> Any:
    """Copy arrays into a clean Dataset so OPeNDAP encodings cannot collide on write."""
    import xarray as xr

    coords = {}
    for name, coord in dataset.coords.items():
        coords[name] = xr.DataArray(np.asarray(coord.values), dims=tuple(coord.dims), attrs=dict(coord.attrs))
    data_vars = {}
    for name, array in dataset.data_vars.items():
        if name in coords:
            continue
        data_vars[name] = xr.DataArray(np.asarray(array.values), dims=tuple(array.dims), attrs=dict(array.attrs))
    clean_attrs = {
        k: v for k, v in dataset.attrs.items()
        if not str(k).startswith("_NC") and not str(k).startswith("DODS")
    }
    clean = xr.Dataset(data_vars=data_vars, coords=coords, attrs=clean_attrs)
    if hasattr(clean, "drop_encoding"):
        clean = clean.drop_encoding()
    elif hasattr(clean, "reset_encoding"):
        clean = clean.reset_encoding()
    for variable in clean.variables.values():
        variable.encoding = {}
    return clean


def select_variables(dataset: Any, variables: Sequence[str]) -> Any:
    missing = [name for name in variables if name not in dataset.data_vars]
    if missing:
        available = list(dataset.data_vars)
        raise RuntimeError(f"Missing variables {missing}. Available: {available}")
    keep = list(variables)
    return dataset[keep] if len(keep) > 1 else dataset[[keep[0]]]


def rename_variables(dataset: Any, mapping: Optional[Dict[str, str]]) -> Any:
    if not mapping:
        return dataset
    present = {src: dst for src, dst in mapping.items() if src in dataset.data_vars}
    renamed = dataset.rename(present)
    for src, dst in present.items():
        if dst in renamed.data_vars:
            renamed[dst].attrs["native_name"] = src
    return renamed


def write_netcdf_atomic(dataset: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    clean = sanitize_for_netcdf(dataset)
    encoding = {}
    for name, array in clean.data_vars.items():
        encoding[name] = {"zlib": True, "complevel": 4}
        if np.asarray(array.values).dtype.kind == "f":
            encoding[name]["dtype"] = "float32"
    try:
        clean.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
        tmp.replace(destination)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        try:
            clean.close()
        except Exception:
            pass


def validate_netcdf(path: Path, required_vars: Optional[Sequence[str]] = None, min_bytes: int = 512) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    if path.stat().st_size < min_bytes:
        return False, f"file too small ({path.stat().st_size} bytes)"
    try:
        import xarray as xr
    except ImportError:
        return True, "xarray not installed; size check only"
    try:
        ds = xr.open_dataset(path)
        try:
            if required_vars:
                available = set(ds.data_vars) | set(ds.coords) | set(ds.variables)
                available_lower = {name.lower(): name for name in available}
                missing = []
                for name in required_vars:
                    if name not in available and name.lower() not in available_lower:
                        missing.append(name)
                if missing:
                    return False, f"missing variables {missing}; have {sorted(available)}"
            sizes = getattr(ds, "sizes", ds.dims)
            if all(size == 0 for size in sizes.values()):
                return False, "all dimensions have size 0"
        finally:
            ds.close()
        return True, "ok"
    except Exception as exc:
        return False, f"failed to open NetCDF: {exc}"


def skip_existing(path: Path, force: bool, required_vars: Optional[Sequence[str]] = None) -> bool:
    if force or not path.exists():
        return False
    ok, reason = validate_netcdf(path, required_vars=required_vars)
    if ok:
        print(f"[!] Skipping existing valid file ({file_size_mb(path):.2f} MB): {path}")
        print("[!] Use --force to overwrite.")
        return True
    print(f"[!] Existing file failed validation ({reason}). Re-downloading: {path}")
    return False


def expected_output_vars(source: Dict[str, Any]) -> List[str]:
    native = list(source.get("variables", []))
    mapping = source.get("rename") or {}
    return [mapping.get(name, name) for name in native]


def login_earthaccess(config: Optional[Dict[str, Any]] = None) -> None:
    require_earthdata()
    settings = network_settings(config)
    apply_requests_timeout(settings["timeout_seconds"])
    try:
        import earthaccess
    except ImportError:
        raise RuntimeError(
            "Package 'earthaccess' is not installed. Run: pip install -r requirements.txt"
        ) from None

    def _login() -> Any:
        auth = earthaccess.login(strategy="environment")
        if not auth:
            raise RuntimeError("Earthdata login was rejected. Credentials are configured=YES.")
        return auth

    retry_call(
        _login,
        retries=settings["retries"],
        backoff=settings["backoff_seconds"],
        what="Earthdata login",
    )


def search_earthaccess_granules(
    short_name: str,
    start: dt.date,
    end: dt.date,
    bbox: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    import earthaccess

    settings = network_settings(config)
    apply_requests_timeout(settings["timeout_seconds"])
    temporal = (f"{start.isoformat()}T00:00:00Z", f"{end.isoformat()}T23:59:59Z")
    bounding_box = (bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"])

    def _search() -> List[Any]:
        granules = earthaccess.search_data(
            short_name=short_name,
            temporal=temporal,
            bounding_box=bounding_box,
        )
        if not granules:
            raise RuntimeError(
                f"No granules found for {short_name} between {start} and {end} in the requested bbox."
            )
        return list(granules)

    return retry_call(
        _search,
        retries=settings["retries"],
        backoff=settings["backoff_seconds"],
        what=f"CMR search ({short_name})",
    )


def download_granules(
    granules: Sequence[Any],
    staging_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    import earthaccess

    settings = network_settings(config)
    apply_requests_timeout(settings["timeout_seconds"])
    staging_dir.mkdir(parents=True, exist_ok=True)

    def _download() -> List[Path]:
        downloaded = earthaccess.download(granules, local_path=str(staging_dir))
        paths: List[Path] = []
        for item in downloaded:
            path = Path(item)
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded granule is missing or empty: {path}")
            paths.append(path)
        if not paths:
            raise RuntimeError("earthaccess.download returned no files.")
        return paths

    return retry_call(
        _download,
        retries=settings["retries"],
        backoff=settings["backoff_seconds"],
        what="Earthdata granule download",
    )


def open_local_dataset(path: Path) -> Any:
    import xarray as xr

    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, decode_times=False)


def subset_local_files(
    files: Sequence[Path],
    bbox: Dict[str, Any],
    start: dt.date,
    end: dt.date,
    source: Dict[str, Any],
) -> Any:
    import xarray as xr

    datasets = []
    native_vars = list(source.get("variables", []))
    for path in files:
        ds = open_local_dataset(path)
        try:
            selected = select_variables(ds, native_vars)
            try:
                selected = time_subset(selected, start, end)
            except Exception:
                pass
            selected = spatial_subset(selected, bbox)
            datasets.append(selected.load())
        finally:
            ds.close()
    if not datasets:
        raise RuntimeError("No datasets remained after subsetting granules.")
    time_name = find_coord(datasets[0], TIME_NAMES)
    if len(datasets) == 1:
        combined = datasets[0]
    elif time_name:
        combined = xr.concat(datasets, dim=time_name)
        combined = combined.sortby(time_name)
        _, unique_idx = np.unique(combined[time_name].values, return_index=True)
        combined = combined.isel({time_name: sorted(unique_idx.tolist())})
    else:
        combined = xr.concat(datasets, dim="time")
    combined = rename_variables(combined, source.get("rename"))
    combined.attrs["oceanembed_native_temporal"] = str(source.get("native_temporal", ""))
    if source.get("notes"):
        combined.attrs["oceanembed_notes"] = str(source["notes"])
    return combined.load()


def copernicus_subset(
    *,
    dataset_id: str,
    variables: Sequence[str],
    bbox: Dict[str, Any],
    start: dt.date,
    end: dt.date,
    output_file: Path,
    username: str,
    password: str,
    force: bool,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
) -> None:
    try:
        import copernicusmarine
    except ImportError as exc:
        raise RuntimeError(
            "Package 'copernicusmarine' is not installed. Run: pip install -r requirements.txt"
        ) from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "variables": list(variables),
        "minimum_longitude": bbox["lon_min"],
        "maximum_longitude": bbox["lon_max"],
        "minimum_latitude": bbox["lat_min"],
        "maximum_latitude": bbox["lat_max"],
        "start_datetime": f"{start.isoformat()}T00:00:00",
        "end_datetime": f"{end.isoformat()}T23:59:59",
        "output_filename": output_file.name,
        "output_directory": str(output_file.parent),
        "username": username,
        "password": password,
    }
    if depth_min is not None:
        kwargs["minimum_depth"] = depth_min
    if depth_max is not None:
        kwargs["maximum_depth"] = depth_max
    if force:
        kwargs["overwrite"] = True
    try:
        copernicusmarine.subset(**kwargs)
    except Exception as exc:
        raise RuntimeError(redact(str(exc), password, username)) from None


def build_filename(source_key: str, variable_tag: str, bbox: Dict[str, Any], start: dt.date, end: dt.date, test: bool) -> str:
    suffix = "_test" if test else ""
    return f"{source_key}_{variable_tag}_{bbox['slug']}_{date_tag(start)}_{date_tag(end)}{suffix}.nc"


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def run_copernicus_source(
    *,
    source_key: str,
    title: str,
    config_path: Path,
    test: bool,
    force: bool,
    output_dir_override: Optional[Path],
    include_depth: bool = False,
) -> int:
    print_banner(title)
    print(f"[*] Loading configuration from: {config_path.resolve()}")
    config = load_configuration(config_path)
    source = get_source(config, source_key)
    bbox = get_poc_bbox(config)
    start, end = get_time_window(config, test=test)
    if test:
        print_test_banner(start, end)

    flags = credentials_configured()
    print(f"[*] Copernicus credentials configured: {flags['copernicus']}")
    username, password = require_copernicus()

    output_dir = raw_output_dir(config, source, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = manifest_path(output_dir, config)
    variables = list(source["variables"])
    variable_tag = "_".join(expected_output_vars(source))
    chunks = planned_chunks(config, start, end, test=test)
    depth_min = float(source["depth_min"]) if include_depth else None
    depth_max = float(source["depth_max"]) if include_depth else None

    print_specs(
        [
            ("Dataset ID", source.get("dataset_id")),
            ("Variables", ", ".join(variables)),
            ("Region", bbox["name"]),
            ("Latitude", f"[{bbox['lat_min']}N, {bbox['lat_max']}N]"),
            ("Longitude", f"[{bbox['lon_min']}E, {bbox['lon_max']}E]"),
            ("Time range", f"{start} to {end}"),
            ("Chunks", len(chunks)),
            ("Native res", f"{source.get('native_resolution_deg')} deg, {source.get('native_temporal')}"),
            ("Output dir", output_dir.resolve()),
        ]
    )

    created: List[Path] = []
    skipped: List[Path] = []
    try:
        for chunk_start, chunk_end in chunks:
            filename = build_filename(source_key, variable_tag, bbox, chunk_start, chunk_end, test)
            output_file = output_dir / filename
            if skip_existing(output_file, force=force, required_vars=expected_output_vars(source)):
                skipped.append(output_file)
                append_manifest(
                    log_path,
                    {
                        "source": source_key,
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
            print(f"[*] Requesting Copernicus subset {chunk_start} to {chunk_end} ...")
            copernicus_subset(
                dataset_id=str(source["dataset_id"]),
                variables=variables,
                bbox=bbox,
                start=chunk_start,
                end=chunk_end,
                output_file=output_file,
                username=username,
                password=password,
                force=force,
                depth_min=depth_min,
                depth_max=depth_max,
            )
            ok, reason = validate_netcdf(output_file, required_vars=variables)
            if not ok:
                raise RuntimeError(f"Downloaded file failed validation: {reason}")
            created.append(output_file)
            append_manifest(
                log_path,
                {
                    "source": source_key,
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
        err = redact(str(exc), password, username)
        append_manifest(
            log_path,
            {
                "source": source_key,
                "status": "failure",
                "product": source.get("dataset_id"),
                "variables": variables,
                "start": str(start),
                "end": str(end),
                "bounds": bbox,
                "error": err,
            },
        )
        print(f"[ERROR] {source_key} download failed: {err}", file=sys.stderr)
        return 1

    print(f"[DONE] {source_key}: created={len(created)} skipped={len(skipped)}")
    return 0


def run_earthaccess_source(
    *,
    source_key: str,
    title: str,
    config_path: Path,
    test: bool,
    force: bool,
    output_dir_override: Optional[Path],
) -> int:
    print_banner(title)
    print(f"[*] Loading configuration from: {config_path.resolve()}")
    config = load_configuration(config_path)
    source = get_source(config, source_key)
    bbox = get_poc_bbox(config)
    start, end = get_time_window(config, test=test)
    if test:
        print_test_banner(start, end)

    flags = credentials_configured()
    print(f"[*] Earthdata credentials configured: {flags['earthdata']}")
    _username, password = require_earthdata()

    output_dir = raw_output_dir(config, source, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".staging"
    log_path = manifest_path(output_dir, config)
    native_vars = list(source["variables"])
    output_vars = expected_output_vars(source)
    variable_tag = "_".join(output_vars)
    chunks = planned_chunks(config, start, end, test=test)
    short_name = str(source["cmr_short_name"])

    print_specs(
        [
            ("CMR short name", short_name),
            ("Native variables", ", ".join(native_vars)),
            ("Saved as", ", ".join(output_vars)),
            ("Region", bbox["name"]),
            ("Latitude", f"[{bbox['lat_min']}N, {bbox['lat_max']}N]"),
            ("Longitude", f"[{bbox['lon_min']}E, {bbox['lon_max']}E]"),
            ("Time range", f"{start} to {end}"),
            ("Native temporal", source.get("native_temporal")),
            ("Notes", source.get("notes", "")),
            ("Output dir", output_dir.resolve()),
        ]
    )

    created: List[Path] = []
    skipped: List[Path] = []
    try:
        login_earthaccess(config)
        print("[+] earthaccess login succeeded")
        for chunk_start, chunk_end in chunks:
            filename = build_filename(source_key, variable_tag, bbox, chunk_start, chunk_end, test)
            output_file = output_dir / filename
            if skip_existing(output_file, force=force, required_vars=output_vars):
                skipped.append(output_file)
                append_manifest(
                    log_path,
                    {
                        "source": source_key,
                        "status": "skipped",
                        "product": short_name,
                        "variables": output_vars,
                        "start": str(chunk_start),
                        "end": str(chunk_end),
                        "bounds": bbox,
                        "file": str(output_file),
                        "size_mb": round(file_size_mb(output_file), 3),
                    },
                )
                continue
            print(f"[*] Searching CMR granules {chunk_start} to {chunk_end} ...")
            granules = search_earthaccess_granules(short_name, chunk_start, chunk_end, bbox, config=config)
            print(f"[+] Found {len(granules)} granule(s)")
            print("[*] Downloading native granules (no regrid, no daily aggregation) ...")
            local_files = download_granules(granules, staging_dir, config=config)
            subset = subset_local_files(local_files, bbox, chunk_start, chunk_end, source)
            try:
                write_netcdf_atomic(subset, output_file)
            finally:
                try:
                    subset.close()
                except Exception:
                    pass
            cleanup_paths(local_files)
            ok, reason = validate_netcdf(output_file, required_vars=output_vars)
            if not ok:
                if output_file.exists():
                    output_file.unlink()
                raise RuntimeError(f"Subset file failed validation: {reason}")
            created.append(output_file)
            append_manifest(
                log_path,
                {
                    "source": source_key,
                    "status": "success",
                    "product": short_name,
                    "variables": output_vars,
                    "granules": len(granules),
                    "start": str(chunk_start),
                    "end": str(chunk_end),
                    "bounds": bbox,
                    "file": str(output_file),
                    "size_mb": round(file_size_mb(output_file), 3),
                    "native_temporal": source.get("native_temporal"),
                },
            )
            print(f"[SUCCESS] {output_file.name} ({file_size_mb(output_file):.2f} MB)")
    except Exception as exc:
        err = redact(str(exc), password)
        append_manifest(
            log_path,
            {
                "source": source_key,
                "status": "failure",
                "product": short_name,
                "variables": output_vars,
                "start": str(start),
                "end": str(end),
                "bounds": bbox,
                "error": err,
            },
        )
        print(f"[ERROR] {source_key} download failed: {err}", file=sys.stderr)
        return 1

    print(f"[DONE] {source_key}: created={len(created)} skipped={len(skipped)}")
    return 0
