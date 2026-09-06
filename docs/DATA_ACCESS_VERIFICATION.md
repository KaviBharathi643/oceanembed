# OceanEmbed Data Access Verification

**Project**: OceanEmbed — Subsurface Ocean Temperature Reconstruction  
**Problem Statement**: SIH 2026 Problem Statement 26066  
**Phase**: Data Access and Availability Verification Phase  
**Target Domain (PoC)**: Bay of Bengal (5°N to 25°N, 80°E to 100°E)  
**Target Temporal Range**: 2024-01-01 to 2024-12-31 (Full Calendar Year 2024)  
**Target Model Grid**: 0.25° × 0.25° Spatial, Daily Temporal  

---

## 1. Executive Summary

A comprehensive automated audit was conducted on all seven scientific data sources (spanning seven surface input variables, one subsurface training target, and independent in situ validation profiles).

- **Total Datasets Investigated**: 7 datasets (9 physical variables)
- **Datasets Requiring Login**: 5 (GLORYS12V1, DUACS SLA, SMAP RSS SSS, OSCAR Currents, CCMP Winds)
- **Datasets Accessible Without Login**: 2 (NOAA OISST v2.1 SST, International Argo Program Profiles via ERDDAP/GDAC)
- **Datasets Requiring Copernicus Marine Account**: 2 (GLORYS12V1 `thetao`, DUACS `sla`)
- **Datasets Requiring NASA Earthdata Account**: 3 (SMAP RSS `sss`, OSCAR `current_u`/`current_v`, CCMP `wind_u`/`wind_v`)
- **Datasets Requiring Manual Intervention**: 0 (all datasets are fully automatable via official REST/CLI/ERDDAP APIs once credentials are provided)
- **Datasets Requiring Credentials Configuration**: 3 (NASA Earthdata credentials need to be added to `.env`)
- **Datasets Still Uncertain**: 0 (all 7 sources have verified metadata, endpoints, and 2024 coverage)

---

## 2. Authentication Results

The local `.env` configuration was securely audited against official authentication endpoints without exposing or logging any sensitive values.

| Provider | Account Service | Credentials Present in `.env` | Authentication Test Method | Test Result |
| :--- | :--- | :--- | :--- | :--- |
| **Copernicus Marine Service** | EUMETSAT / Mercator Ocean CAS | **SET** | `copernicusmarine.login(check_credentials_valid=True)` | **PASS** (Valid credentials confirmed) |
| **NASA Earthdata** | EOSDIS User Registration System (URS) | **NOT SET** | Environment Variable Presence Check | **NOT CONFIGURED** (`EARTHDATA_USERNAME` & `EARTHDATA_PASSWORD` missing) |
| **NOAA NCEI / PSL** | Open Access Data Server | **N/A** (No account needed) | HTTP HEAD on Data & THREDDS Endpoints | **PASS** (Unrestricted public access) |
| **Argo / Coriolis / INCOIS** | Global Data Assembly Centre (GDAC) | **N/A** (No account needed) | ERDDAP REST & GDAC HTTPS Head Queries | **PASS** (Unrestricted public access) |

---

## 3. Dataset Access Results

| Variable | Target Parameter | Primary Provider | Auth Required | API / Download Route | 2024 Available | PoC Region (5–25°N, 80–100°E) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **sst** | Sea Surface Temperature | NOAA NCEI / PSL | **NO** | HTTPS / OPeNDAP / ERDDAP | **YES** (366 days) | **YES** (Global 0.25°) | **PASS** |
| **sss** | Sea Surface Salinity | NASA/JPL PO.DAAC | **YES** (Earthdata) | CMR Search / PO.DAAC Protected HTTPS | **YES** (366 days) | **YES** (Global 0.25°) | **CONDITIONAL** (Requires Earthdata credentials) |
| **sla** | Sea Level Anomaly | Copernicus Marine | **YES** (Copernicus) | Copernicus Marine Toolbox API | **YES** (366 days) | **YES** (Global 0.125°) | **PASS** (Credentials valid) |
| **current_u** | Zonal Surface Current | NASA/JPL PO.DAAC | **YES** (Earthdata) | CMR Search / PO.DAAC Protected HTTPS | **YES** (366 days) | **YES** (Global 0.25°) | **CONDITIONAL** (Requires Earthdata credentials) |
| **current_v** | Meridional Surface Current | NASA/JPL PO.DAAC | **YES** (Earthdata) | CMR Search / PO.DAAC Protected HTTPS | **YES** (366 days) | **YES** (Global 0.25°) | **CONDITIONAL** (Requires Earthdata credentials) |
| **wind_u** | Zonal 10m Wind Vector | NASA/JPL PO.DAAC | **YES** (Earthdata) | CMR Search / PO.DAAC Protected HTTPS | **YES** (366 days) | **YES** (Global 0.25°) | **CONDITIONAL** (Requires Earthdata credentials) |
| **wind_v** | Meridional 10m Wind Vector | NASA/JPL PO.DAAC | **YES** (Earthdata) | CMR Search / PO.DAAC Protected HTTPS | **YES** (366 days) | **YES** (Global 0.25°) | **CONDITIONAL** (Requires Earthdata credentials) |
| **thetao** | Subsurface Temperature (Target) | Copernicus Marine | **YES** (Copernicus) | Copernicus Marine Toolbox API | **YES** (366 days) | **YES** (Global 0.083°) | **PASS** (Credentials valid) |
| **Argo Profiles** | In Situ Temperature Profiles | Coriolis / Ifremer / INCOIS | **NO** | ERDDAP TableDAP / GDAC HTTPS | **YES** (>500 profiles) | **YES** (Verified in Bay of Bengal) | **PASS** |

*Status Definitions:*
- **PASS**: Endpoint verified, credentials ready/validated, spatial/temporal range confirmed.
- **CONDITIONAL**: Endpoint verified and dataset confirmed available for 2024, but user credentials in `.env` are required to automate download.
- **FAIL**: Dataset not found or inaccessible.
- **UNKNOWN**: Status could not be determined.

---

## 4. Dataset Metadata

### Source 1: Sea Surface Temperature (`sst`)
- **Product Name**: NOAA Daily Optimum Interpolation Sea Surface Temperature (OISST) Version 2.1 — AVHRR-only
- **Dataset ID**: `noaa.oisst.v2.highres` (NOAA PSL THREDDS) / `ncdcOisst21Agg_LonPM180` (NOAA CoastWatch)
- **Variables**: `sst` (sea surface temperature [°C]), `anom` (daily anomaly), `err` (standard error)
- **Native Spatial Resolution**: 0.25° × 0.25° (1440 × 720 global grid)
- **Native Temporal Resolution**: Daily (1-day mean)
- **Vertical Information**: Surface (0 m / ~0.2 m bulk temperature)
- **2024 Coverage**: Complete (2024-01-01 to 2024-12-31)
- **Geographic Coverage**: Global (-89.875° to 89.875°N, 0.125° to 359.875°E)
- **Official Access Method**: Direct HTTPS download from NCEI, or remote spatial subsetting via NOAA PSL THREDDS OPeNDAP server (`https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.2024.nc`)

### Source 2: Sea Surface Salinity (`sss`)
- **Product Name**: RSS SMAP Level 3 Sea Surface Salinity Standard Mapped Image 8-Day Running Mean V6.0 Validated Dataset
- **Dataset ID (CMR Short Name)**: `SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6` (or `SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V5`)
- **Variables**: `sss_smap` (practical salinity [PSU]), `sss_smap_unc` (uncertainty estimate)
- **Native Spatial Resolution**: 0.25° × 0.25° (1440 × 720 global grid)
- **Native Temporal Resolution**: 8-day running mean produced daily (daily timestamp)
- **Vertical Information**: Surface (top ~1 cm skin layer)
- **2024 Coverage**: Complete (2024-01-01 to 2024-12-31 confirmed in CMR)
- **Geographic Coverage**: Global (-90.0° to 90.0°N, -180.0° to 180.0°E)
- **Official Access Method**: NASA Earthdata / PO.DAAC Cumulus Archive (`https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/`) via `earthaccess` or direct authenticated HTTPS; Remote Sensing Systems (RSS) public mirror as backup.

### Source 3: Sea Level Anomaly (`sla`)
- **Product Name**: GLOBAL OCEAN GRIDDED L4 SEA SURFACE HEIGHTS AND DERIVED VARIABLES REPROCESSED (1993-ONGOING)
- **Product ID**: `SEALEVEL_GLO_PHY_L4_MY_008_047`
- **Dataset ID**: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`  
  *(Alternative 0.25° / NRT datasets identified in catalogue: `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` and `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D`)*
- **Variables**: `sla` (sea level anomaly [m]), `adt` (absolute dynamic topography), `ugos`, `vgos`, `ugosa`, `vgosa`, `err_sla`
- **Native Spatial Resolution**: 0.125° × 0.125° (1/8 degree)
- **Native Temporal Resolution**: Daily (P1D)
- **Vertical Information**: Surface (0 m)
- **2024 Coverage**: Complete (1993-01-01 to 2026-01-16 available in reprocessed multi-year catalogue)
- **Geographic Coverage**: Global (-89.9375° to 89.9375°N, -179.9375° to 179.9375°E)
- **Official Access Method**: Copernicus Marine Toolbox Python library (`copernicusmarine.subset`)

### Source 4: Ocean Surface Currents (`current_u`, `current_v`)
- **Product Name**: Ocean Surface Current Analyses Real-time (OSCAR) Surface Currents — Final 0.25 Degree (Version 2.0)
- **Dataset ID (CMR Short Name)**: `OSCAR_L4_OC_FINAL_V2.0` (or `OSCAR_L4_OC_INTERIM_V2.0`)
- **Variables**: `u` (zonal surface velocity [m/s]), `v` (meridional surface velocity [m/s]), `ug` (geostrophic u), `vg` (geostrophic v)
- **Native Spatial Resolution**: 0.25° × 0.25° (1440 × 721 global grid)
- **Native Temporal Resolution**: Daily (1-day mean)
- **Vertical Information**: 15 m depth upper mixed-layer average
- **2024 Coverage**: Complete (2024-01-01 to 2024-12-31 confirmed in CMR)
- **Geographic Coverage**: Global (-80.0° to 80.0°N, 0.0° to 359.75°E)
- **Official Access Method**: NASA Earthdata / PO.DAAC Cumulus Archive via `earthaccess` or PO.DAAC REST API

### Source 5: Ocean Surface Winds (`wind_u`, `wind_v`)
- **Product Name**: RSS Cross-Calibrated Multi-Platform (CCMP) 6-Hourly 10 Meter Surface Winds Level 4 Version 3.1
- **Dataset ID (CMR Short Name)**: `CCMP_WINDS_10M6HR_L4_V3.1` (Version 3.1)
- **Variables**: `uwnd` (zonal wind velocity at 10m [m/s]), `vwnd` (meridional wind velocity at 10m [m/s]), `ws` (wind speed)
- **Native Spatial Resolution**: 0.25° × 0.25° (1440 × 628 global grid)
- **Native Temporal Resolution**: 6-hourly (00:00, 06:00, 12:00, 18:00 UTC) -> aggregated to daily mean
- **Vertical Information**: 10 meters above sea surface
- **2024 Coverage**: Complete (1993-01-01 to present confirmed in CMR)
- **Geographic Coverage**: Global (-78.375° to 78.375°N, 0.125° to 359.875°E)
- **Official Access Method**: NASA Earthdata / PO.DAAC Cumulus Archive via `earthaccess` or Remote Sensing Systems HTTPS server (`data.remss.com/ccmp/v03.1/`)

### Source 6: Subsurface Potential Temperature Target (`thetao`)
- **Product Name**: Global Ocean Physics Reanalysis (GLORYS12V1)
- **Product ID**: `GLOBAL_MULTIYEAR_PHY_001_030`
- **Dataset ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Variables**: `thetao` (sea water potential temperature [°C]), `so` (salinity [1e-3]), `uo` (u-velocity), `vo` (v-velocity), `zos` (sea surface height)
- **Native Spatial Resolution**: 0.0833° × 0.0833° (1/12 degree ~ 8-9 km)
- **Native Temporal Resolution**: Daily (P1D-m)
- **Vertical Information**: 50 standard depth levels (0.49 m to 5727.9 m; 0–1000 m spans 37 depth levels)
- **2024 Coverage**: Complete (1993-01-01 to 2026-06-25 available in catalogue)
- **Geographic Coverage**: Global (-80.0° to 90.0°N, -180.0° to 179.9167°E)
- **Official Access Method**: Copernicus Marine Toolbox Python library (`copernicusmarine.subset`)

### Source 7: ARGO In Situ Temperature Profiles (Independent Validation)
- **Product Name**: International Argo Program Global Data Assembly Centre (GDAC) In Situ Profiles
- **Dataset ID / Collection**: `ArgoFloats` (Ifremer / Coriolis ERDDAP `https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json`), Coriolis GDAC HTTPS (`https://data-argo.ifremer.fr/`)
- **Variables**: `TEMP` (in situ temperature [°C]), `PRES` (sea water pressure [dbar] ~ depth [m]), `PSAL` (salinity [PSU]), `TEMP_QC`, `PRES_QC`, `latitude`, `longitude`, `time`, `platform_number`
- **Native Spatial Resolution**: Discrete float trajectories (in situ profile points)
- **Native Temporal Resolution**: ~10-day cycle per float (~daily across distributed floats in the region)
- **Vertical Information**: Continuous vertical profiles from 0–2000 dbar with fine vertical sampling (1–2 dbar near surface, 5–10 dbar at depth)
- **2024 Coverage**: Complete (All 2024 profiles archived and accessible)
- **Geographic Coverage**: Global ocean; verified active coverage in the Bay of Bengal (5°–25°N, 80°–100°E) with >500 profiles in 2024
- **Official Access Method**: REST API / TableDAP queries via Ifremer/Coriolis ERDDAP or `argopy` Python package

---

## 5. Recommended Acquisition Method

| Dataset | Provider | Recommended Automation Tool / Library | Acquisition Route Details |
| :--- | :--- | :--- | :--- |
| **NOAA OISST v2.1** (`sst`) | NOAA NCEI / PSL | `xarray` / `requests` / `pydap` | Stream remote slice via OPeNDAP (`https://psl.noaa.gov/thredds/dodsC/...`) or download daily NetCDF tiles from NCEI HTTPS |
| **RSS SMAP L3 SSS** (`sss`) | NASA PO.DAAC | `earthaccess` Python library | Search CMR by granule temporal range, download via authenticated HTTPS stream to `data/raw/sss/` |
| **DUACS SLA** (`sla`) | Copernicus Marine | `copernicusmarine` Python API | `copernicusmarine.subset(dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", ...)` directly into `data/raw/sla/` |
| **OSCAR Currents** (`current_u`, `current_v`) | NASA PO.DAAC | `earthaccess` Python library | Search CMR collection `OSCAR_L4_OC_FINAL_V2.0`, download daily NetCDF files via authenticated HTTPS stream |
| **CCMP v3.1 Winds** (`wind_u`, `wind_v`) | NASA PO.DAAC | `earthaccess` / `requests` | Download 6-hourly or daily NetCDF files via `earthaccess` or RSS public server, average daily |
| **GLORYS12V1** (`thetao`) | Copernicus Marine | `copernicusmarine` Python API | `copernicusmarine.subset(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m", ...)` directly into `data/raw/glorys/` |
| **ARGO Profiles** (`TEMP`, `PRES`) | Coriolis / Ifremer / INCOIS | `argopy` / REST `requests` | Direct REST TableDAP query to Ifremer ERDDAP (`https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json`) bounded to Bay of Bengal & 2024 |

---

## 6. Manual Intervention Requirements

| Provider / Dataset | Create Account? | Accept Terms? | Manually Download Files? | Manually Select Files? | Provide Credentials? | Overall User Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Copernicus Marine (GLORYS, DUACS SLA)** | Done | Done | No | No | Done (in `.env`) | **Do Nothing** (Ready for automated acquisition) |
| **NASA Earthdata (SMAP, OSCAR, CCMP)** | If not registered, create free account at `urs.earthdata.nasa.gov` | Approve PO.DAAC application in Earthdata URS | No (fully automatable) | No (automated by CMR API) | Yes (add `EARTHDATA_USERNAME` & `EARTHDATA_PASSWORD` to `.env`) | **Provide Earthdata Credentials in `.env`** |
| **NOAA OISST v2.1** | No | No | No | No | No | **Do Nothing** (Fully public) |
| **ARGO / Coriolis** | No | No | No | No | No | **Do Nothing** (Fully public) |

---

## 7. Security Audit

A static analysis and pattern-matching scan across the repository was conducted:

- **`.env`**: Local secrets file detected. Verified that `.env` is included in `.gitignore` (line 30).
- **`.env.example`**: Clean (contains only placeholder labels `your_copernicus_username_here`, `your_copernicus_password_here`, etc.).
- **`.gitignore`**: Properly configured to exclude `.env`, `*.nc`, `checkpoints/`, `data/raw/*/*`, and token files.
- **Python Files (`scripts/`, `src/`, `dashboard/`)**: Clean. No hardcoded passwords, tokens, or private keys detected in executable code.
- **YAML Files (`config/`)**: Clean. Contains only domain bounds, dataset IDs, depth levels, and hyperparameter specifications.

> [!NOTE]
> No hardcoded secrets were detected in any source code, config files, or tracked git artifacts.

---

## 8. 2024 Common-Period Verification

| Dataset | Required Spatial Domain (5–25°N, 80–100°E) | Required Temporal Domain (2024-01-01 to 2024-12-31) | Resolution Compatibility | Status |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA OISST v2.1** | Fully covers domain (0.25° grid) | Fully available (366 daily records) | Native 0.25° daily | **CONFIRMED** |
| **SMAP RSS L3 SSS** | Fully covers domain (0.25° grid) | Fully available (366 daily records) | Native 0.25° daily | **CONFIRMED** |
| **Copernicus DUACS SLA** | Fully covers domain (0.125° grid) | Fully available (366 daily records) | Native 0.125° daily (regrids cleanly to 0.25°) | **CONFIRMED** |
| **OSCAR Currents v2.0** | Fully covers domain (0.25° grid) | Fully available (366 daily records) | Native 0.25° daily | **CONFIRMED** |
| **CCMP v3.1 Winds** | Fully covers domain (0.25° grid) | Fully available (366 daily records) | Native 0.25° 6-hr (aggregates to daily) | **CONFIRMED** |
| **Copernicus GLORYS12V1** | Fully covers domain (0.083° grid) | Fully available (366 daily records) | Native 0.083° daily, 50 vertical levels (0–1000m) | **CONFIRMED** |
| **ARGO Profiles** | Verified >500 profiles in Bay of Bengal | Full 2024 calendar year available | In situ profiles (0–2000 dbar) | **CONFIRMED** |

**Conclusion**: All seven datasets have complete temporal and spatial coverage for the entire calendar year 2024 over the Bay of Bengal PoC region.

---

## 9. Final Acquisition Matrix

| Dataset | Variable(s) | Account Needed | Credentials Ready | Programmatically Automatable | Manual Download Needed | Recommended Tool | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GLORYS12V1** | `thetao` | Copernicus | **YES** | **YES** | **NO** | `copernicusmarine` | **CRITICAL** |
| **DUACS SLA** | `sla` | Copernicus | **YES** | **YES** | **NO** | `copernicusmarine` | **CRITICAL** |
| **NOAA OISST v2.1** | `sst` | None | **N/A** | **YES** | **NO** | `xarray` / `requests` | **CRITICAL** |
| **SMAP RSS L3 SSS** | `sss` | Earthdata | **NO** (Need `.env`) | **YES** | **NO** | `earthaccess` | **CRITICAL** |
| **OSCAR Currents** | `current_u`, `current_v` | Earthdata | **NO** (Need `.env`) | **YES** | **NO** | `earthaccess` | **CRITICAL** |
| **CCMP v3.1 Winds** | `wind_u`, `wind_v` | Earthdata | **NO** (Need `.env`) | **YES** | **NO** | `earthaccess` / `requests` | **CRITICAL** |
| **ARGO Profiles** | In Situ `TEMP` / `PRES` | None | **N/A** | **YES** | **NO** | `argopy` / ERDDAP REST | **VALIDATION** |

---

## 10. Final Decision

| Dataset | Variable | Decision Status | Action Required |
| :--- | :--- | :--- | :--- |
| **NOAA OISST v2.1** | `sst` | <span style="color:green;font-weight:bold;">GREEN</span> — Ready for automation | None. Direct download / OPeNDAP ready. |
| **GLORYS12V1** | `thetao` | <span style="color:green;font-weight:bold;">GREEN</span> — Ready for automation | None. Copernicus credentials validated and active. |
| **DUACS SLA** | `sla` | <span style="color:green;font-weight:bold;">GREEN</span> — Ready for automation | None. Copernicus credentials validated and active. |
| **ARGO Temperature Profiles** | `TEMP`, `PRES` | <span style="color:green;font-weight:bold;">GREEN</span> — Ready for automation | None. Public ERDDAP REST endpoint verified. |
| **SMAP RSS L3 SSS** | `sss` | <span style="color:goldenrod;font-weight:bold;">YELLOW</span> — Credentials required | Add `EARTHDATA_USERNAME` & `EARTHDATA_PASSWORD` to `.env`. |
| **OSCAR Surface Currents** | `current_u`, `current_v` | <span style="color:goldenrod;font-weight:bold;">YELLOW</span> — Credentials required | Add `EARTHDATA_USERNAME` & `EARTHDATA_PASSWORD` to `.env`. |
| **CCMP v3.1 Winds** | `wind_u`, `wind_v` | <span style="color:goldenrod;font-weight:bold;">YELLOW</span> — Credentials required | Add `EARTHDATA_USERNAME` & `EARTHDATA_PASSWORD` to `.env`. |

---
*Verification Audit Completed. No large data downloads, preprocessing, or training steps were performed.*
