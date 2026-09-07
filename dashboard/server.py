"""
OceanEmbed (SIH 2026 Problem Statement 26066)
FastAPI Backend Server

Serves the judge-facing working prototype:
- High-performance REST API for live inference and metadata
- Static asset delivery for the single-page application
- Absolute protection of frozen checkpoints and datasets
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.engine import OceanEmbedEngine

app = FastAPI(
    title="OceanEmbed Prototype API",
    description="On-demand model inference and oceanographic dashboard for SIH 2026 Problem Statement 26066",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log structured development diagnostics on invalid payloads without hiding the 422 status."""
    errors = exc.errors()
    failing_fields = [
        ".".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        for err in errors
    ]
    try:
        raw_body = await request.body()
        body = raw_body.decode("utf-8", errors="replace")
    except Exception:
        body = "<unreadable body>"

    error_summary = "; ".join(f"{err.get('loc', [])}: {err.get('msg', '')}" for err in errors)
    print(f"[API Validation 422] Failed field(s): {failing_fields} | Error(s): {error_summary} | Payload: {body}")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": f"Invalid request parameter(s): {error_summary}",
            "detail": errors,
            "failing_fields": failing_fields,
        },
    )


# Static directory setup
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request Model
class PredictRequest(BaseModel):
    lat: float = Field(..., ge=5.0, le=25.0, description="Latitude (5°N to 25°N)")
    lon: float = Field(..., ge=80.0, le=100.0, description="Longitude (80°E to 100°E)")
    date: str = Field("2024-04-06", description="Calendar date (YYYY-MM-DD) in 2024")


@app.get("/")
def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "OceanEmbed Prototype API is running. Frontend index.html being assembled."}


@app.get("/api/health")
def get_health():
    engine = OceanEmbedEngine.get_instance()
    return {
        "status": "healthy",
        "service": "OceanEmbed Working Prototype",
        "problem_statement": "SIH 2026 PS 26066",
        "model_loaded": True,
        "checkpoint": os.path.basename(engine.checkpoint_path),
        "total_parameters": 120655,
        "indexed_samples": len(engine.lookup),
        "available_dates_count": len(engine.available_dates),
        "date_range": [engine.available_dates[0], engine.available_dates[-1]],
    }


@app.get("/api/demo-locations")
def get_demo_locations():
    engine = OceanEmbedEngine.get_instance()
    return {"demo_locations": engine.get_demo_locations()}


@app.post("/api/predict")
def predict_profile(req: PredictRequest):
    engine = OceanEmbedEngine.get_instance()
    result = engine.predict(lat=req.lat, lon=req.lon, date=req.date)
    return result


@app.get("/api/predict-get")
def predict_profile_get(
    lat: float = Query(14.00, ge=5.0, le=25.0),
    lon: float = Query(87.00, ge=80.0, le=100.0),
    date: str = Query("2024-04-06"),
):
    engine = OceanEmbedEngine.get_instance()
    return engine.predict(lat=lat, lon=lon, date=date)


@app.get("/api/data-catalog")
def get_data_catalog():
    engine = OceanEmbedEngine.get_instance()
    return {"catalog": engine.get_data_catalog()}


@app.get("/api/model-specs")
def get_model_specs():
    engine = OceanEmbedEngine.get_instance()
    return engine.get_model_specs()


@app.get("/api/validation-argo")
def get_argo_validation():
    argo_path = Path("checkpoints/argo_validation.json")
    if argo_path.exists():
        with open(argo_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="ARGO validation report not found.")


@app.get("/api/test-evaluation")
def get_test_evaluation():
    test_path = Path("checkpoints/test_evaluation.json")
    if test_path.exists():
        with open(test_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Test evaluation report not found.")


@app.get("/api/validation-summary")
def get_val_summary():
    val_path = Path("checkpoints/validation_analysis.json")
    if val_path.exists():
        with open(val_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Validation analysis report not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=False)
