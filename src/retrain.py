# src/retrain.py - COMPLETE WORKING VERSION

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import fetch_eia_day, fetch_weather_archive

TIMEZONE = "America/Los_Angeles"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ca_electricity_raw.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def save_demand_stats(df):
    """
    Save demand statistics for post-processing clipping.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    demand_stats = {
        "min": float(df["demand_mwh"].min()),
        "max": float(df["demand_mwh"].max()),
        "mean": float(df["demand_mwh"].mean()),
        "median": float(df["demand_mwh"].median()),
        "std": float(df["demand_mwh"].std()),
        "p1": float(df["demand_mwh"].quantile(0.01)),
        "p5": float(df["demand_mwh"].quantile(0.05)),
        "p95": float(df["demand_mwh"].quantile(0.95)),
        "p99": float(df["demand_mwh"].quantile(0.99)),
    }
    
    stats_path = os.path.join(MODELS_DIR, "demand_stats.json")
    with open(stats_path, "w") as f:
        json.dump(demand_stats, f, indent=2)
    
    print(f"  Demand stats saved: min={demand_stats['min']:.0f} MW, max={demand_stats['max']:.0f} MW")
    print(f"  Stats saved to: {stats_path}")
    
    return demand_stats


def update_dataset(api_key: str) -> bool:
    """
    Update dataset with latest 5 days of data.
    """
    # --- Define last 5 days ---
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=4)
    date_range = pd.date_range(start=start_date, end=end_date)

    print(f"\n── STEP 1: Fetch last 5 days ────────────────────")
    print(f"  Range: {start_date.date()} → {end_date.date()}")

    all_new_data = []

    # --- Loop through each day ---
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n  Fetching {date_str}...")

        # =========================
        # EIA FETCH (with retries)
        # =========================
        eia_df = None
        for attempt in range(1, 4):
            eia_df = fetch_eia_day(api_key, date_str)
            if eia_df is not None:
                break
            print(f"    EIA retry {attempt}/3...")
            time.sleep(10)

        if eia_df is None:
            print(f"    Skipping {date_str} (EIA failed)")
            continue

        # =========================
        # WEATHER FETCH
        # =========================
        weather_df = fetch_weather_archive(date_str)
        if weather_df is None:
            print(f"    Skipping {date_str} (Weather failed)")
            continue

        # =========================
        # NORMALIZE TIMESTAMPS
        # =========================
        eia_df["timestamp"] = pd.to_datetime(
            eia_df["timestamp"]
        ).dt.tz_convert(TIMEZONE)

        weather_df["timestamp"] = pd.to_datetime(
            weather_df["timestamp"]
        ).dt.tz_convert(TIMEZONE)

        # =========================
        # MERGE
        # =========================
        merged = pd.merge(
            eia_df, weather_df, on="timestamp", how="inner"
        )

        print(f"    Rows: {len(merged)}")

        all_new_data.append(merged)

    # =========================
    # CHECK IF ANY DATA FETCHED
    # =========================
    if not all_new_data:
        print("  ERROR: No data fetched for any day")
        return False

    new_data = pd.concat(all_new_data, ignore_index=True)
    print(f"\n  Total new rows: {len(new_data)}")

    # =========================
    # LOAD EXISTING DATA
    # =========================
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH)
        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"], utc=True
        ).dt.tz_convert(TIMEZONE)
    else:
        print("  No existing data found, creating new dataset")
        existing = pd.DataFrame()

    # =========================
    # APPEND + REMOVE DUPLICATES
    # =========================
    if len(existing) > 0:
        combined = (
            pd.concat([existing, new_data], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        combined = new_data.sort_values("timestamp").reset_index(drop=True)

    # =========================
    # ROLLING 5-YEAR WINDOW
    # =========================
    cutoff = (
        pd.Timestamp.now(tz=TIMEZONE)
        - pd.DateOffset(years=5)
    )

    combined = combined[
        combined["timestamp"] >= cutoff
    ].reset_index(drop=True)

    # =========================
    # SAVE DATASET
    # =========================
    combined.to_csv(DATA_PATH, index=False)

    print(f"\n  Dataset saved: {len(combined):,} rows")
    print(
        f"  Range  : {combined['timestamp'].min().date()} "
        f"→ {combined['timestamp'].max().date()}"
    )

    # =========================
    # SAVE DEMAND STATS - IMPORTANT!
    # =========================
    save_demand_stats(combined)

    return True


def main():
    """
    Main retrain function to be called by GitHub Actions.
    """
    print("=" * 60)
    print("Starting retrain pipeline")
    print("=" * 60)
    
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("ERROR: EIA_API_KEY not found in environment")
        return False
    
    success = update_dataset(api_key)
    
    if success:
        print("\n✅ Dataset update successful")
        print("✅ demand_stats.json created/updated")
    else:
        print("\n❌ Dataset update failed")
    
    print("=" * 60)
    return success


if __name__ == "__main__":
    main()