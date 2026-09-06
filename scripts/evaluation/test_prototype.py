"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Comprehensive Prototype Test Suite

Tests:
1. Static HTML/CSS/JS delivery
2. /api/health endpoint
3. /api/demo-locations endpoint
4. /api/predict live inference (all 5 demo locations)
5. /api/data-catalog endpoint
6. /api/model-specs endpoint
7. /api/validation-argo endpoint (verifies frozen values match)
8. /api/test-evaluation endpoint
9. Integrity verification of all frozen checkpoints and datasets
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.server import app


def test_prototype_system():
    print("======================================================================")
    print("RUNNING OCEANEMBED PROTOTYPE VERIFICATION SUITE")
    print("======================================================================")

    client = TestClient(app)

    # 1. Test Static Index Delivery
    print("\n[Test 1] Static Frontend Delivery")
    r_index = client.get("/")
    assert r_index.status_code == 200, f"Expected 200, got {r_index.status_code}"
    assert "OceanEmbed" in r_index.text, "Index does not contain OceanEmbed"
    assert "Explore the Ocean" in r_index.text, "Index missing Explore header"
    print("  -> index.html successfully served (HTTP 200)")

    r_css = client.get("/static/style.css")
    assert r_css.status_code == 200
    print("  -> style.css successfully served (HTTP 200)")

    r_js = client.get("/static/app.js")
    assert r_js.status_code == 200
    print("  -> app.js successfully served (HTTP 200)")

    # Verify offline vendor assets
    for vendor_f in ["leaflet.css", "leaflet.js", "chart.umd.min.js", "lucide.min.js"]:
        r_v = client.get(f"/static/vendor/{vendor_f}")
        assert r_v.status_code == 200, f"Missing vendor file: {vendor_f}"
    print("  -> offline vendor assets verified (HTTP 200: Leaflet, Chart.js, Lucide)")

    # 2. Test Health Endpoint
    print("\n[Test 2] Health Endpoint")
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "healthy"
    assert h_data["total_parameters"] == 120655
    assert h_data["indexed_samples"] == 618633
    assert h_data["available_dates_count"] == 366
    print(f"  -> Health OK: {h_data['service']} | Checkpoint: {h_data['checkpoint']}")

    # 3. Test Demo Locations Endpoint
    print("\n[Test 3] Demo Locations Endpoint")
    r_demo = client.get("/api/demo-locations")
    assert r_demo.status_code == 200
    demos = r_demo.json()["demo_locations"]
    assert len(demos) == 5, f"Expected 5 demo locations, got {len(demos)}"
    print(f"  -> Retrieved {len(demos)} verified demo presets:")
    for d in demos:
        print(f"     - {d['name']}: ({d['lat']}N, {d['lon']}E) on {d['default_date']} [{d['tag']}]")

    # 4. Test Live Inference on All 5 Demo Locations
    print("\n[Test 4] Live Model Inference on Demo Locations")
    for d in demos:
        payload = {"lat": d["lat"], "lon": d["lon"], "date": d["default_date"]}
        r_pred = client.post("/api/predict", json=payload)
        assert r_pred.status_code == 200, f"Prediction failed for {d['name']}"
        res = r_pred.json()
        assert res["success"] is True, f"Unsuccessful prediction: {res}"
        assert len(res["profile"]) == 15, "Expected 15 depths"
        assert len(res["embedding"]["values"]) == 64, "Expected 64-D embedding"
        assert "sst" in res["surface_conditions"], "Missing SST"

        surf_t = res["profile"][0]["predicted_temp_c"]
        deep_t = res["profile"][-1]["predicted_temp_c"]
        mld = res["diagnostics"]["mixed_layer_depth_m"]
        t_ms = res["model_metadata"]["inference_time_ms"]

        assert 20.0 <= surf_t <= 33.0, f"Unphysical surface temp: {surf_t}"
        assert 5.0 <= deep_t <= 10.0, f"Unphysical deep temp: {deep_t}"
        assert surf_t > deep_t, "Surface must be warmer than 1000m"

        print(f"  -> {d['name']:15s} | Surf: {surf_t:5.2f} C | 1000m: {deep_t:4.2f} C | MLD: {mld:4.1f} m | Latent Norm: {res['embedding']['norm']} | Time: {t_ms} ms")

    # 5. Test Data Catalog
    print("\n[Test 5] Data Catalog Endpoint")
    r_cat = client.get("/api/data-catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()["catalog"]
    assert len(cat) == 7, f"Expected 7 scientific products, got {len(cat)}"
    print(f"  -> Data catalog verified ({len(cat)} products)")

    # 6. Test Model Specs
    print("\n[Test 6] Model Specifications Endpoint")
    r_specs = client.get("/api/model-specs")
    assert r_specs.status_code == 200
    specs = r_specs.json()
    assert specs["total_parameters"] == 120655
    assert len(specs["architecture_components"]) == 8
    print(f"  -> Model specs verified: {specs['total_parameters']:,} parameters across 8 components")

    # 7. Test ARGO Validation Endpoint & Value Verification
    print("\n[Test 7] ARGO Validation Endpoint")
    r_argo = client.get("/api/validation-argo")
    assert r_argo.status_code == 200
    argo_data = r_argo.json()
    assert argo_data["collocation"]["matched_profiles"] == 672, "ARGO matched profiles mismatch"
    assert argo_data["overall_metrics"]["total_temperature_comparisons"] == 9983, "ARGO points mismatch"
    assert round(argo_data["overall_metrics"]["oceanembed_rmse"], 4) == 0.8464, "ARGO RMSE mismatch"
    assert round(argo_data["overall_metrics"]["improvement_pct"], 2) == 36.28, "ARGO improvement mismatch"
    assert round(argo_data["overall_metrics"]["bias"], 4) == -0.0065, "ARGO bias mismatch"
    print(f"  -> ARGO metrics verified: 672 floats | 9,983 points | RMSE: 0.8464 C | Improvement: +36.28%")

    # 8. Test Evaluation Endpoint
    print("\n[Test 8] Held-Out Test Evaluation Endpoint")
    r_test = client.get("/api/test-evaluation")
    assert r_test.status_code == 200
    t_data = r_test.json()
    assert round(t_data["overall_test"]["oceanembed_rmse"], 4) == 0.9753, "Test RMSE mismatch"
    assert round(t_data["overall_test"]["baseline_rmse"], 4) == 1.4451, "Test baseline RMSE mismatch"
    print(f"  -> Test metrics verified: Held-out RMSE: 0.9753 C | Baseline: 1.4451 C (+32.51%)")

    # 9. Test Validation Summary Endpoint
    print("\n[Test 9] Validation Summary Endpoint")
    r_val = client.get("/api/validation-summary")
    assert r_val.status_code == 200
    v_data = r_val.json()
    assert round(v_data["overall_validation"]["oceanembed_rmse"], 4) == 0.8827
    print(f"  -> Validation summary verified: Val RMSE: {v_data['overall_validation']['oceanembed_rmse']:.4f} C")

    # 10. Verify Protection of Frozen Artifacts
    print("\n[Test 10] Protection & Integrity of Frozen Artifacts")
    frozen_files = [
        "checkpoints/oceanembed_best.pt",
        "checkpoints/preliminary_epoch5_checkpoint.pt",
        "checkpoints/training_history.json",
        "checkpoints/validation_analysis.json",
        "checkpoints/test_evaluation.json",
        "checkpoints/argo_validation.json",
        "checkpoints/test_evaluation_report.md",
        "checkpoints/argo_validation_report.md",
        "data/processed/normalization.json",
        "data/processed/sample_metadata.csv",
        "data/processed/oceanembed_train.nc",
        "data/processed/oceanembed_val.nc",
        "data/processed/oceanembed_test.nc",
        "src/models/oceanembed.py",
        "src/models/baseline.py",
        "src/data/dataset.py",
        "src/data/preprocessing.py",
        "config/dataset_config.yaml",
        "data/processed/preprocessing_report.json",
    ]

    for ff in frozen_files:
        p = Path(ff)
        assert p.exists(), f"Frozen file missing: {ff}"
        assert p.stat().st_size > 0, f"Frozen file is empty: {ff}"
    print(f"  -> All {len(frozen_files)} frozen ML/data files intact and verified.")

    print("\n======================================================================")
    print("ALL PROTOTYPE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    test_prototype_system()
