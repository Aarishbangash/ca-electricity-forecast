# src/predict.py - UPDATED VERSION

import os
import json
import pickle
import numpy as np

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models")

HORIZON = 24
_CACHE  = {}

# Historical demand range for California (based on your data)
# Adjust these numbers based on actual historical min/max from your data
# CAISO demand typically: min ~18,000 MW (night), max ~45,000 MW (summer peak)
MIN_HISTORICAL_DEMAND = 18000  # Typical minimum demand in CA (MW)
MAX_HISTORICAL_DEMAND = 45000  # Typical maximum demand in CA (MW)


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

    # Load or compute demand stats for clipping
    stats_path = os.path.join(MODELS_DIR, "demand_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            _CACHE["demand_stats"] = json.load(f)
    else:
        # Fallback stats - adjust these based on your actual data
        _CACHE["demand_stats"] = {
            "min": MIN_HISTORICAL_DEMAND,
            "max": MAX_HISTORICAL_DEMAND,
            "mean": 28000  # Typical CA demand mean
        }
   
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _CACHE["metrics"] = json.load(f)

    print("Models ready")
    return _CACHE


def predict_24h(X_features):
    """
    Predict 24-hour demand with post-processing to ensure non-negative,
    physically plausible values.
    """
    models = load_models()
    model  = models["xgboost"]
    stats  = models.get("demand_stats", {
        "min": MIN_HISTORICAL_DEMAND,
        "max": MAX_HISTORICAL_DEMAND,
        "mean": 28000
    })
    
    Y_pred = model.predict(X_features)
    
    # Handle different output shapes
    if Y_pred.ndim == 2:
        preds = Y_pred[0].ravel()[:HORIZON]
    else:
        preds = Y_pred.ravel()[:HORIZON]
    
    # FIX 1: Clip negative values to a reasonable minimum
    # Use historical minimum for realistic predictions
    min_demand = stats.get("min", MIN_HISTORICAL_DEMAND)
    # Don't clip below historical minimum (or slightly below for night hours)
    preds = np.maximum(preds, min_demand * 0.9)  # At least 90% of historical min
    
    # FIX 2: Cap extreme high values
    max_demand = stats.get("max", MAX_HISTORICAL_DEMAND)
    preds = np.minimum(preds, max_demand * 1.1)  # Don't exceed 110% of historical max
    
    # FIX 3: Smooth out extreme outliers (optional, but helps with stability)
    mean_demand = stats.get("mean", 28000)
    
    # If any predictions are still unrealistic (very low or very high)
    # replace with reasonable values based on time of day pattern
    for i in range(len(preds)):
        if preds[i] < min_demand * 0.5:  # Less than 50% of historical min
            # Too low - replace with typical night demand
            preds[i] = min_demand * 0.95
        elif preds[i] > max_demand * 1.5:  # More than 150% of historical max
            # Too high - replace with typical peak demand
            preds[i] = max_demand
    
    # FIX 4: Ensure daily pattern makes sense (optional smoothing)
    # Apply a simple moving average to smooth out unrealistic spikes
    if len(preds) > 2:
        smoothed = np.copy(preds)
        for i in range(1, len(preds)-1):
            if preds[i] > preds[i-1] * 1.5 and preds[i] > preds[i+1] * 1.5:
                # Unrealistic spike - smooth it
                smoothed[i] = (preds[i-1] + preds[i+1]) / 2
        preds = smoothed
    
    return preds


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