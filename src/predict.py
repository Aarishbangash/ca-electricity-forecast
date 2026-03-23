# src/predict.py
# Model loading and inference

import os
import json
import pickle
import numpy as np

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models")

HORIZON = 24
_CACHE  = {}


def load_models():
    global _CACHE
    if _CACHE:
        return _CACHE

    print("Loading models...")

    fc_path = os.path.join(MODELS_DIR, "feature_cols.json")
    with open(fc_path) as f:
        _CACHE["feature_cols"] = json.load(f)

    xgb_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            _CACHE["xgboost"] = pickle.load(f)
        print("  XGBoost loaded")
    else:
        raise FileNotFoundError(
            f"model_xgboost.pkl not found in {MODELS_DIR}")

   
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _CACHE["metrics"] = json.load(f)

    print("Models ready")
    return _CACHE


def predict_24h(X_features):
    models = load_models()
    model  = models["xgboost"]
    Y_pred = model.predict(X_features)
    if Y_pred.ndim == 2:
        return Y_pred[0].ravel()[:HORIZON]
    return Y_pred.ravel()[:HORIZON]


# src/predict.py
def get_metrics(reload=True):
    """
    Return latest metrics. If reload=True, read fresh from disk.
    """
    global _CACHE
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    
    if reload or "metrics" not in _CACHE:
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                _CACHE["metrics"] = json.load(f)
        else:
            _CACHE["metrics"] = {
                "last_retrain": "Not available",
                "xgboost_mape": "N/A",
                "train_rows": "N/A",
                "data_from": "N/A",
                "data_to": "N/A",
            }
    
    return _CACHE["metrics"]


def reload_models():
    global _CACHE
    _CACHE = {}
    return load_models()
