# src/retrain.py
# Daily retraining pipeline
# Runs on GitHub servers automatically every day at 2AM UTC
# Your laptop/internet does NOT need to be on

import os
import sys
import json
import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingestion import fetch_eia_day, fetch_weather_archive
from src.features  import build_features

TIMEZONE  = "America/Los_Angeles"
DATA_PATH = os.path.join(ROOT, "data", "ca_electricity_raw.csv")
MODEL_DIR = os.path.join(ROOT, "models")
HORIZON   = 24


def update_dataset(api_key: str) -> bool:
    yesterday = (datetime.now() - timedelta(days=1)
                 ).strftime("%Y-%m-%d")

    print(f"\n── STEP 1: Fetch {yesterday} ────────────────────")

    # Retry EIA up to 3 times with 30s delay
    eia_df = None
    for attempt in range(1, 4):
        print(f"  EIA attempt {attempt}/3...")
        eia_df = fetch_eia_day(api_key, yesterday)
        if eia_df is not None:
            break
        if attempt < 3:
            print(f"  EIA failed — waiting 30s before retry...")
            time.sleep(30)

    # Fallback to day before yesterday
    if eia_df is None:
        print(f"  EIA failed for {yesterday} — trying day before...")
        day_before = (datetime.now() - timedelta(days=2)
                      ).strftime("%Y-%m-%d")
        for attempt in range(1, 3):
            print(f"  EIA fallback attempt {attempt}/2...")
            eia_df = fetch_eia_day(api_key, day_before)
            if eia_df is not None:
                yesterday = day_before
                break
            time.sleep(30)

    if eia_df is None:
        print("  ERROR: EIA completely unavailable")
        return False

    # Fetch weather
    weather_df = fetch_weather_archive(yesterday)
    if weather_df is None:
        print("  ERROR: Weather fetch failed")
        return False

    # Normalize timestamps
    eia_df["timestamp"] = pd.to_datetime(
        eia_df["timestamp"]).dt.tz_convert(TIMEZONE)
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"]).dt.tz_convert(TIMEZONE)

    # Merge demand + weather
    new_day = pd.merge(
        eia_df, weather_df, on="timestamp", how="inner")
    print(f"  New rows: {len(new_day)}")

    # Load existing dataset
    existing = pd.read_csv(DATA_PATH)
    existing["timestamp"] = pd.to_datetime(
        existing["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)

    # Append + deduplicate
    combined = (
        pd.concat([existing, new_day], ignore_index=True)
          .drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    # Rolling 5-year window
    cutoff   = (pd.Timestamp.now(tz=TIMEZONE)
                - pd.DateOffset(years=5))
    combined = combined[
        combined["timestamp"] >= cutoff
    ].reset_index(drop=True)

    # Save
    combined.to_csv(DATA_PATH, index=False)

    print(f"  Dataset: {len(combined):,} rows")
    print(f"  Range  : {combined.timestamp.min().date()}"
          f" → {combined.timestamp.max().date()}")
    return True


def prepare_data():
    print(f"\n── STEP 2: Feature engineering ─────────────────")

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["timestamp"] = pd.to_datetime(
        df_raw["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)

    df = build_features(df_raw)
    df = df.dropna().reset_index(drop=True)

    with open(os.path.join(MODEL_DIR,
                           "feature_cols.json")) as f:
        FEATURE_COLS = json.load(f)

    n        = len(df)
    n_train  = int(n * 0.85)
    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:].copy()

    print(f"  Total  : {n:,} rows")
    print(f"  Train  : {len(df_train):,} rows")
    print(f"  Val    : {len(df_val):,} rows")

    return df_train, df_val, FEATURE_COLS


def retrain_xgboost(df_train, df_val, FEATURE_COLS):
    print(f"\n── STEP 3: Retrain XGBoost ──────────────────────")

    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_percentage_error

    def build_targets(df, horizon=HORIZON):
        targets = np.stack([
            df["demand_mwh"].shift(-h).values
            for h in range(1, horizon + 1)
        ], axis=1)
        valid = ~np.isnan(targets).any(axis=1)
        return df[valid].copy(), targets[valid]

    train_df, Y_train = build_targets(df_train)
    val_df,   Y_val   = build_targets(df_val)

    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values

    best_params = {
        "n_estimators"    : 1000,
        "max_depth"       : 8,
        "learning_rate"   : 0.05,
        "min_child_weight": 1,
        "subsample"       : 0.7,
        "colsample_bytree": 0.7,
        "gamma"           : 0.0,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 1.0,
        "random_state"    : 42,
        "n_jobs"          : -1,
        "tree_method"     : "hist",
        "verbosity"       : 0,
    }

    print(f"  Training XGBoost...")
    model = MultiOutputRegressor(
        XGBRegressor(**best_params), n_jobs=-1)
    model.fit(X_train, Y_train)

    Y_val_pred = model.predict(X_val)
    val_mape   = float(
        mean_absolute_percentage_error(
            Y_val.ravel(), Y_val_pred.ravel()) * 100)

    print(f"  Val MAPE: {val_mape:.2f}%")
    return model, val_mape


def save_results(model, val_mape, df_train, df_val):
    print(f"\n── STEP 4: Save ─────────────────────────────────")

    model_path = os.path.join(MODEL_DIR, "model_xgboost.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model → {model_path}")

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["timestamp"] = pd.to_datetime(
        df_raw["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)

    metrics = {
        "last_retrain" : datetime.now().isoformat(),
        "train_rows"   : len(df_train),
        "val_rows"     : len(df_val),
        "xgboost_mape" : round(val_mape, 4),
        "data_from"    : str(df_raw["timestamp"].min().date()),
        "data_to"      : str(df_raw["timestamp"].max().date()),
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {metrics_path}")
    print(f"  Metrics: {metrics}")


def run_retrain():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "EIA_API_KEY not set in GitHub Secrets")

    print(f"\n{'='*55}")
    print(f"  DAILY RETRAIN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}")

    if not update_dataset(api_key):
        print("Aborted — data fetch failed")
        sys.exit(1)

    df_train, df_val, FEATURE_COLS = prepare_data()
    model, val_mape = retrain_xgboost(
        df_train, df_val, FEATURE_COLS)
    save_results(model, val_mape, df_train, df_val)

    print(f"\n{'='*55}")
    print(f"  RETRAIN COMPLETE")
    print(f"  XGBoost val MAPE : {val_mape:.2f}%")
    print(f"  Time             : "
          f"{datetime.now().strftime('%H:%M UTC')}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_retrain()