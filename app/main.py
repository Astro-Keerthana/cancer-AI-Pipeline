"""
app/main.py — Cancer Classification API
FastAPI + XGBoost | TCGA-BRCA Gene Expression Classifier
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
import uvicorn

# ─── Load Model ───────────────────────────────────────────────
MODEL_PATH = "models/xgb_cancer_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Expected features : {model.n_features_in_}")
except FileNotFoundError:
    raise RuntimeError(
        f"❌ Model not found at '{MODEL_PATH}'. "
        "Run 03_train_model.py first."
    )

# ─── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title       = "🧬 Cancer Classification API",
    description = (
        "Real-time **Tumor vs. Normal** tissue classification "
        "using TCGA-BRCA gene expression data.\n\n"
        "- `/predict` — single sample inference\n"
        "- `/predict/batch` — multi-sample inference\n"
        "- `/health` — server health check"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ─── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    allow_credentials = False
)

# ─── Schemas ──────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    gene_expression: List[float] = Field(
        ...,
        description = (
            "Normalized gene expression values. "
            f"Must match model's expected feature count "
            f"({model.n_features_in_} features)."
        )
    )
    sample_id: str = Field(
        default     = "unknown",
        description = "Optional sample identifier for tracking"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sample_id": "sample-001",
                    "gene_expression": [0.12, -0.34, 1.22, 0.88, -1.05]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    sample_id:          str
    prediction:         str
    prediction_code:    int
    probability_tumor:  float
    probability_normal: float
    confidence:         str


class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total:       int
    successful:  int
    failed:      int


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    expected_features: int


# ─── Endpoints ────────────────────────────────────────────────

@app.get(
    "/",
    tags    = ["Health"],
    summary = "Root — API status"
)
def root():
    """Quick check that the API is alive."""
    return {
        "status":           "🟢 API is running",
        "model":            "XGBoost TCGA-BRCA Classifier",
        "version":          "1.0.0",
        "docs":             "http://127.0.0.1:8000/docs",
        "expected_features": model.n_features_in_
    }


@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["Health"],
    summary        = "Health check"
)
def health_check():
    """Returns server health and model status."""
    return HealthResponse(
        status            = "healthy",
        model_loaded      = model is not None,
        expected_features = model.n_features_in_
    )


@app.post(
    "/predict",
    response_model = PredictionResponse,
    tags           = ["Inference"],
    summary        = "Single sample prediction"
)
def predict(request: PredictionRequest):
    """
    Classify a single sample as **Tumor** or **Normal**.

    - **gene_expression**: list of normalized float values
    - **sample_id**: optional label for tracking

    Returns prediction label, class probabilities, and confidence tier.
    """
    try:
        X = np.array(request.gene_expression).reshape(1, -1)

        # ── Feature count validation ──────────────────────────
        expected = model.n_features_in_
        if X.shape[1] != expected:
            raise HTTPException(
                status_code = 422,
                detail      = (
                    f"Feature count mismatch: "
                    f"expected {expected}, got {X.shape[1]}. "
                    "Ensure input is preprocessed with the same "
                    "pipeline used during training."
                )
            )

        # ── Inference ─────────────────────────────────────────
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        label      = "Tumor" if pred == 1 else "Normal"
        max_prob   = float(max(proba))
        confidence = (
            "High"   if max_prob > 0.85 else
            "Medium" if max_prob > 0.65 else
            "Low"
        )

        return PredictionResponse(
            sample_id          = request.sample_id,
            prediction         = label,
            prediction_code    = pred,
            probability_tumor  = round(float(proba[1]), 4),
            probability_normal = round(float(proba[0]), 4),
            confidence         = confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model = BatchPredictionResponse,
    tags           = ["Inference"],
    summary        = "Batch prediction for multiple samples"
)
def predict_batch(requests: List[PredictionRequest]):
    """
    Classify **multiple samples** in a single request.

    Each item follows the same schema as `/predict`.
    Failed samples are included in the response with an `error` field
    instead of being silently dropped.
    """
    results    = []
    successful = 0
    failed     = 0

    for req in requests:
        try:
            result = predict(req)
            results.append(result.model_dump())
            successful += 1
        except HTTPException as e:
            results.append({
                "sample_id": req.sample_id,
                "error":     e.detail
            })
            failed += 1
        except Exception as e:
            results.append({
                "sample_id": req.sample_id,
                "error":     str(e)
            })
            failed += 1

    return BatchPredictionResponse(
        predictions = results,
        total       = len(results),
        successful  = successful,
        failed      = failed
    )


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "127.0.0.1",
        port    = 8000,
        reload  = True
    )
