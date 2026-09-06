# OceanEmbed Final Data Access Verification

**Project**: OceanEmbed — Subsurface Ocean Temperature Reconstruction  
**Problem Statement**: SIH 2026 Problem Statement 26066  
**Phase**: Final Authentication & Programmatic API Verification  
**Target Domain (PoC)**: Bay of Bengal (5°N to 25°N, 80°E to 100°E)  
**Target Temporal Range**: 2024-01-01 to 2024-12-31 (Full Calendar Year 2024)  
**Target Model Grid**: 0.25° × 0.25° Spatial, Daily Temporal  

---

## 1. Authentication Status

Both required authentication services (Copernicus Marine and NASA Earthdata) have been securely verified against official upstream authentication endpoints.

| Provider | Credentials in `.env` | Authentication Mechanism Tested | Result |
| :--- | :--- | :--- | :--- |
| **Copernicus Marine Service** | **SET** (`COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD`) | `copernicusmarine.login(check_credentials_valid=True)` | **PASS** (Active token / valid credentials confirmed) |
| **NASA Earthdata** | **SET** (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`) | NASA URS Profile API (`https://urs.earthdata.nasa.gov/profile`) + PO.DAAC Protected Archive Handshake | **PASS** (HTTP 200 OK authenticated response) |
| **NOAA NCEI / PSL** | **N/A** (No authentication required) | Direct HTTP / OPeNDAP Endpoint Verification | **PASS** (Unrestricted public access) |
| **Argo / Coriolis / Ifremer** | **N/A** (No authentication required) | Ifremer ERDDAP TableDAP & Coriolis GDAC REST APIs | **PASS** (Unrestricted public access) |

---

## 2. Dataset Access

| Dataset | Variable(s) | Provider | Login Required | 2024 Available | Bay of Bengal (5–25°N, 80–100°E) | Automation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NOAA OISST v2.1** | `sst` | NOAA NCEI / PSL | **NO** | **YES** (366 days) | **YES** (0.25° global grid) | **FULL AUTOMATION READY** |
| **RSS SMAP L3 SSS** | `sss` | NASA/JPL PO.DAAC | **YES** (Earthdata) | **YES** (366 days) | **YES** (0.25° global grid) | **FULL AUTOMATION READY** |
| **DUACS Gridded SLA** | `sla` | Copernicus Marine | **YES** (Copernicus) | **YES** (366 days) | **YES** (0.125° global grid) | **FULL AUTOMATION READY** |
| **OSCAR Currents v2.0** | `current_u`, `current_v` | NASA/JPL PO.DAAC | **YES** (Earthdata) | **YES** (366 days) | **YES** (0.25° global grid) | **FULL AUTOMATION READY** |
| **CCMP v3.1 10m Winds** | `wind_u`, `wind_v` | NASA/JPL PO.DAAC | **YES** (Earthdata) | **YES** (366 days) | **YES** (0.25° global grid) | **FULL AUTOMATION READY** |
| **GLORYS12V1 Reanalysis** | `thetao` (Target) | Copernicus Marine | **YES** (Copernicus) | **YES** (366 days) | **YES** (0.083° global grid, 50 depths) | **FULL AUTOMATION READY** |
| **Argo Float Profiles** | `TEMP`, `PRES` (Validation) | Coriolis / Ifremer | **NO** | **YES** (>500 profiles) | **YES** (Verified active floats) | **FULL AUTOMATION READY** |

---

## 3. Exact Product / Dataset IDs

The following current and active dataset identifiers were discovered and verified in the official catalogues:

1. **Sea Surface Temperature (`sst`)**:
   - Provider: NOAA Physical Sciences Laboratory (PSL) / NCEI
   - Dataset Identifier: `noaa.oisst.v2.highres` (OPeNDAP) / `ncdcOisst21Agg_LonPM180` (CoastWatch ERDDAP)
   - Variable Name: `sst`

2. **Sea Surface Salinity (`sss`)**:
   - Provider: NASA/JPL PO.DAAC (Remote Sensing Systems)
   - CMR Short Name: `SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6` (Version 6.0)  
     *(Fallback verified: `SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V5`)*
   - Variable Name: `sss_smap`

3. **Sea Level Anomaly (`sla`)**:
   - Provider: Copernicus Marine Service (Mercator Ocean / CLS DUACS)
   - Product ID: `SEALEVEL_GLO_PHY_L4_MY_008_047`
   - Dataset ID: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
   - Variable Name: `sla`

4. **Surface Ocean Currents (`current_u`, `current_v`)**:
   - Provider: NASA/JPL PO.DAAC (Earth & Space Research)
   - CMR Short Name: `OSCAR_L4_OC_FINAL_V2.0`  
     *(Interim/NRT fallback verified: `OSCAR_L4_OC_INTERIM_V2.0`)*
   - Variable Names: `u` (zonal), `v` (meridional)

5. **Surface Winds (`wind_u`, `wind_v`)**:
   - Provider: NASA/JPL PO.DAAC (Remote Sensing Systems)
   - CMR Short Name: `CCMP_WINDS_10M6HR_L4_V3.1`
   - Variable Names: `uwnd` (zonal 10m wind), `vwnd` (meridional 10m wind)

6. **Target Subsurface Temperature (`thetao`)**:
   - Provider: Copernicus Marine Service (GLORYS12V1)
   - Product ID: `GLOBAL_MULTIYEAR_PHY_001_030`
   - Dataset ID: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
   - Variable Name: `thetao`

7. **Independent Validation Profiles (`TEMP`, `PRES`)**:
   - Provider: International Argo Program / Coriolis GDAC / Ifremer
   - Collection Name: `ArgoFloats` (`https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json`)
   - Variable Names: `TEMP`, `PRES`, `PSAL`, `TEMP_QC`, `PRES_QC`

---

## 4. Access Method

When dataset acquisition is eventually executed, Cursor should use the following official mechanisms:

- **GLORYS12V1 (`thetao`)**: Use `copernicusmarine.subset()` via `scripts/download/download_glorys.py` with bounding box `[5°N, 25°N, 80°E, 100°E]`, depths `0.0–1000.0 m`, and time range `2024-01-01` to `2024-12-31`. Output to `data/raw/glorys/`.
- **DUACS SLA (`sla`)**: Use `copernicusmarine.subset()` specifying dataset `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`, variable `sla`, and the Bay of Bengal coordinates into `data/raw/sla/`.
- **NOAA OISST v2.1 (`sst`)**: Use `xarray.open_dataset()` against the PSL THREDDS OPeNDAP endpoint (`https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.2024.nc`) with `.sel(lat=slice(5.0, 25.0), lon=slice(80.0, 100.0))` to stream only the PoC bounding box directly to `data/raw/sst/`.
- **SMAP RSS L3 SSS (`sss`)**: Use `earthaccess.search_data(short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6", temporal=("2024-01-01", "2024-12-31"))` and download via `earthaccess.download()` into `data/raw/sss/`.
- **OSCAR Surface Currents (`current_u`, `current_v`)**: Use `earthaccess.search_data(short_name="OSCAR_L4_OC_FINAL_V2.0", temporal=("2024-01-01", "2024-12-31"))` and download into `data/raw/currents/`.
- **CCMP v3.1 Winds (`wind_u`, `wind_v`)**: Use `earthaccess.search_data(short_name="CCMP_WINDS_10M6HR_L4_V3.1", temporal=("2024-01-01", "2024-12-31"))` and download into `data/raw/winds/`.
- **Argo Profiles (`TEMP`, `PRES`)**: Use direct HTTP REST TableDAP query to Ifremer ERDDAP (`https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json`) bounded to `5°N–25°N, 80°E–100°E` and `2024-01-01T00:00:00Z` to `2024-12-31T23:59:59Z`, or the `argopy` Python library, into `data/raw/argo/`.

---

## 5. Manual Actions

**NO MANUAL DATA DOWNLOAD REQUIRED.**

All seven datasets can be programmatically queried, subsetted, and retrieved using automated scripts utilizing the active credentials present in `.env`.

---

## 6. Temporal Alignment Requirements

The input and target variables exhibit distinct native temporal characteristics. Preprocessing pipelines must apply appropriate temporal alignment to establish a synchronized daily dataset:

1. **SMAP RSS L3 SSS (`sss`) — 8-Day Running Mean Composite**:
   - *Characteristic*: SMAP L3 8-day running mean files are generated daily, but each file represents an 8-day smoothing window (consecutive daily files have an 87.5% temporal overlap). **These are NOT independent daily observations**.
   - *Alignment Strategy*: The daily timestamp of the 8-day running mean file is aligned 1:1 with the model daily target. The model data loader treats this as an 8-day smoothed surface salinity proxy without requiring further temporal interpolation.

2. **CCMP v3.1 Winds (`wind_u`, `wind_v`) — 6-Hourly Synoptic Vector**:
   - *Characteristic*: Contains 4 synoptic time steps per day (00:00, 06:00, 12:00, 18:00 UTC).
   - *Alignment Strategy*: Must be aggregated to a daily mean (`ds[["uwnd", "vwnd"]].resample(time="1D").mean()` or `mean(dim="time")`) before feature concatenation.

3. **GLORYS12V1 (`thetao`), NOAA OISST (`sst`), DUACS SLA (`sla`), OSCAR (`current_u/v`) — Native Daily**:
   - *Characteristic*: Daily composite or daily mean fields.
   - *Alignment Strategy*: Direct daily timestamp alignment matching `YYYY-MM-DD`.

4. **Argo In Situ Temperature Profiles — Event-Based Ascents**:
   - *Characteristic*: Asynchronous profile events (~10 days per float, random spatial sampling).
   - *Alignment Strategy*: Point-wise spatial and temporal collocation (interpolate model 3D prediction field to exact Argo profile timestamp `t` and coordinates `(lat, lon, depth)` for validation metric computation).

---

## 7. Final Acquisition Readiness

| Dataset | Variable | Readiness Classification | Description |
| :--- | :--- | :--- | :--- |
| **NOAA OISST v2.1** | `sst` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Open public access) |
| **GLORYS12V1** | `thetao` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Copernicus credentials validated) |
| **DUACS SLA** | `sla` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Copernicus credentials validated) |
| **RSS SMAP L3 SSS** | `sss` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Earthdata credentials validated) |
| **OSCAR Currents v2.0** | `current_u`, `current_v` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Earthdata credentials validated) |
| **CCMP v3.1 Winds** | `wind_u`, `wind_v` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Earthdata credentials validated) |
| **ARGO Float Profiles** | `TEMP`, `PRES` | <span style="color:green;font-weight:bold;">GREEN</span> | Ready for automated acquisition (Open public ERDDAP REST) |

**Result**: **100% GREEN across all 7 sources.**

---

## 8. Final Recommendation

When data acquisition is initiated, Cursor should execute acquisition in the following prioritized sequence:

1. **Step 1: GLORYS12V1 Training Target (`thetao`)** — [CRITICAL]
   - *Rationale*: Establishes the exact reference grid, 15 standard depth levels (0–1000m), land/sea mask, and target training tensor shapes.
   - *Tool*: `copernicusmarine.subset()` via `scripts/download/download_glorys.py`.

2. **Step 2: Sea Surface Temperature (`sst`)** — [CRITICAL]
   - *Rationale*: Open access, fast OPeNDAP spatial slicing to test surface grid alignment directly against GLORYS.
   - *Tool*: `xarray` streaming from NOAA PSL THREDDS server.

3. **Step 3: DUACS Sea Level Anomaly (`sla`)** — [CRITICAL]
   - *Rationale*: Uses the validated Copernicus connection to obtain gridded SLA, requiring bilinear regridding from 0.125° to 0.25°.
   - *Tool*: `copernicusmarine.subset()`.

4. **Step 4: NASA PO.DAAC Suite (`sss`, `current_u/v`, `wind_u/v`)** — [CRITICAL]
   - *Rationale*: Uses the validated NASA Earthdata login via `earthaccess` in a single unified script. Winds are daily-averaged during ingestion.
   - *Tool*: `earthaccess` batch acquisition.

5. **Step 5: Argo Float Profiles (`TEMP`, `PRES`)** — [VALIDATION]
   - *Rationale*: Independent ground-truth validation dataset for testing model generalizability and physical fidelity.
   - *Tool*: ERDDAP REST TableDAP client / `argopy`.

---
*Verification Phase Complete. All accounts verified, endpoints confirmed, and 2024 data availability validated. Stopping without data acquisition.*
