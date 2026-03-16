# 🧬 Cancer Classification API

Tumor vs. Normal tissue classifier using TCGA-BRCA gene expression data.
Built with XGBoost + FastAPI.

## How to Run

1. Install dependencies
pip install fastapi uvicorn xgboost scikit-learn joblib numpy pydantic

2. Start the API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

3. Test the API
python scripts/test_api.py

## API Endpoints

| Endpoint         | Method | Description              |
|------------------|--------|--------------------------|
| /                | GET    | API status               |
| /health          | GET    | Health check             |
| /predict         | POST   | Single sample prediction |
| /predict/batch   | POST   | Batch prediction         |

## Results
- Model: XGBoost
- Features: 41,410 gene expression values
- Classes: Tumor / Normal