# OceanEmbed: Oceanographic Foundation Model & Embeddings

OceanEmbed is an oceanographic representation learning framework designed for spatio-temporal modeling and embedding of ocean dynamics using multi-source satellite and in-situ observations.

## Directory Structure

```text
.
├── config/
│   └── dataset_config.yaml     # Single source of truth for acquisition & preprocessing
│
├── data/
│   ├── raw/
│   │   ├── glorys/             # GLORYS12V1 ocean reanalysis target (thetao)
│   │   ├── sst/                # Sea Surface Temperature
│   │   ├── sss/                # Sea Surface Salinity
│   │   ├── sla/                # Sea Level Anomaly
│   │   ├── currents/           # Current velocities (current_u, current_v)
│   │   ├── winds/              # Wind velocities (wind_u, wind_v)
│   │   └── argo/               # In-situ vertical profiles from Argo floats
│   │
│   └── processed/
│       ├── harmonized/         # Regridded, quality-controlled spatio-temporal grids
│       └── training/           # Normalized tensors ready for model training
│
├── scripts/
│   ├── download/               # Acquisition scripts per exact data product
│   ├── preprocess/             # Regridding and spatial harmonization pipelines
│   └── inspect/                # Quality inspection and statistics scripts
│
├── src/
│   ├── data/                   # PyTorch datasets, DataLoaders, and collators
│   ├── models/                 # Neural architectures and embedding encoders
│   └── evaluation/             # Metrics, probe tasks, and vertical profile validation
│
├── checkpoints/                # Model checkpoints and weights
│
├── dashboard/
│   └── app.py                  # Interactive Streamlit dashboard
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules for ocean data, models, and cache
└── README.md
```

## Configuration

Dataset parameters, bounding boxes, target grid resolution, and variables are governed strictly by [`config/dataset_config.yaml`](file:///e:/dp/projects/SIH26066/config/dataset_config.yaml).

- **Domain (Full)**: Lat [5.0, 30.0], Lon [45.0, 105.0]
- **POC Domain (Bay of Bengal)**: Lat [5.0, 25.0], Lon [80.0, 100.0]
- **Time Window**: 2024-01-01 to 2024-03-31
- **Grid**: 0.25° resolution, daily
- **Input Variables**: `sst`, `sss`, `sla`, `current_u`, `current_v`, `wind_u`, `wind_v`
- **Training Target**: `thetao` (GLORYS)

## Getting Started

### 1. Environment Setup

It is recommended to use Python 3.10+ in a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Dashboard

```bash
streamlit run dashboard/app.py
```
