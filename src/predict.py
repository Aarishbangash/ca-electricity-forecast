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
    """Load models once and cache in memory."""
    global _CACHE
    if _CACHE:
        return _CACHE

    print("Loading models...")

    # Feature columns
    fc_path = os.path.join(MODELS_DIR, "feature_cols.json")
    with open(fc_path) as f:
        _CACHE["feature_cols"] = json.load(f)

    # XGBoost model
    xgb_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            _CACHE["xgboost"] = pickle.load(f)
        print("  XGBoost loaded")
    else:
        raise FileNotFoundError(
            f"model_xgboost.pkl not found in {MODELS_DIR}")

    # Metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _CACHE["metrics"] = json.load(f)

    print("Models ready")
    return _CACHE


def predict_24h(X_features):
    """
    Run XGBoost 24h forecast.
    Clips predictions to realistic CA demand range.
    CA demand: 15,000 - 52,000 MWh typically.
    """
    models = load_models()
    model  = models["xgboost"]

    Y_pred = model.predict(X_features)

    if Y_pred.ndim == 2:
        result = Y_pred[0].ravel()[:HORIZON]
    else:
        result = Y_pred.ravel()[:HORIZON]

    # Clip to realistic range — demand cannot be negative
    result = np.clip(result, 10000, 60000)

    return result


def get_metrics():
    """Return latest training metrics."""
    models = load_models()
    return models.get("metrics", {
        "last_retrain" : "Not available",
        "xgboost_mape" : "N/A",
        "train_rows"   : "N/A",
        "data_from"    : "N/A",
        "data_to"      : "N/A",
    })


def reload_models():
    """Force reload from disk after retrain."""
    global _CACHE
    _CACHE = {}
    return load_models()
