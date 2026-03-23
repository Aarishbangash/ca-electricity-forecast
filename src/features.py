# src/features.py
# Feature engineering — identical to Colab pipeline

import pandas as pd
import numpy as np
import holidays

BASE_TEMP = 13.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features from raw demand + weather data."""
    df = df.copy().sort_values(
        "timestamp").reset_index(drop=True)

    if "demand_mwh" in df.columns:
        df["demand_mwh"] = df["demand_mwh"].interpolate(
            method="linear")

    # Calendar
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["quarter"]      = df["timestamp"].dt.quarter
    df["day_of_year"]  = df["timestamp"].dt.dayofyear
    df["week_of_year"] = (df["timestamp"]
                           .dt.isocalendar()
                           .week.astype(int))
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    year_min = int(df["timestamp"].dt.year.min())
    year_max = int(df["timestamp"].dt.year.max()) + 2
    ca_hols  = holidays.US(
        state="CA", years=range(year_min, year_max))
    df["is_holiday"]  = df["timestamp"].dt.date.apply(
        lambda d: 1 if d in ca_hols else 0)
    df["is_off_peak"] = (
        (df["is_weekend"] == 1) | (df["is_holiday"] == 1)
    ).astype(int)

    # Fourier
    t = np.arange(len(df))
    for k in [1, 2]:
        df[f"sin_daily_{k}"]  = np.sin(2 * np.pi * k * t / 24)
        df[f"cos_daily_{k}"]  = np.cos(2 * np.pi * k * t / 24)
        df[f"sin_weekly_{k}"] = np.sin(2 * np.pi * k * t / 168)
        df[f"cos_weekly_{k}"] = np.cos(2 * np.pi * k * t / 168)

    # Weather derived
    df["CDD"] = np.maximum(df["temperature_c"] - BASE_TEMP, 0)
    df["HDD"] = np.maximum(BASE_TEMP - df["temperature_c"], 0)
    df["heat_index"] = (
        df["temperature_c"]
        + 0.33 * (df["humidity_pct"] / 100 * 6.105
                  * np.exp(17.27 * df["temperature_c"]
                  / (237.7 + df["temperature_c"])))
        - 4.0
    )
    df["is_daytime"]    = (
        (df["hour"] >= 7) & (df["hour"] <= 19)).astype(int)
    df["solar_x_CDD"]   = df["solar_radiation_wm2"] * df["CDD"]
    df["solar_daytime"] = df["solar_radiation_wm2"] * df["is_daytime"]
    df["temp_bin"]      = pd.cut(
        df["temperature_c"],
        bins   = [-np.inf, 5, 10, 15, 20, 25, 30, np.inf],
        labels = [0, 1, 2, 3, 4, 5, 6]
    ).astype(float)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"demand_lag_{lag}h"] = df["demand_mwh"].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24, 48, 168]:
        df[f"demand_roll_mean_{window}h"] = (
            df["demand_mwh"].shift(1)
              .rolling(window, min_periods=1).mean())
    for window in [24, 168]:
        df[f"demand_roll_std_{window}h"] = (
            df["demand_mwh"].shift(1)
              .rolling(window, min_periods=1).std())
    for window in [3, 24]:
        df[f"temp_roll_mean_{window}h"] = (
            df["temperature_c"].shift(1)
              .rolling(window, min_periods=1).mean())
        df[f"solar_roll_mean_{window}h"] = (
            df["solar_radiation_wm2"].shift(1)
              .rolling(window, min_periods=1).mean())

    return df


def build_inference_row(historical_df, forecast_weather,
                         target_date, feature_cols):
    """
    Build feature rows for inference on target_date.
    Uses historical data for lag features.
    Uses forecast weather for temperature etc.
    """
    TIMEZONE = "America/Los_Angeles"

    forecast_weather = forecast_weather.copy()
    forecast_weather["timestamp"] = pd.to_datetime(
        forecast_weather["timestamp"]
    ).dt.tz_convert(TIMEZONE)

    historical_df = historical_df.copy()
    historical_df["timestamp"] = pd.to_datetime(
        historical_df["timestamp"]
    ).dt.tz_convert(TIMEZONE)

    # Sort and fill null demand
    historical_df = historical_df.sort_values(
        "timestamp").reset_index(drop=True)
    historical_df["demand_mwh"] = (
        historical_df["demand_mwh"]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Get target date rows from forecast weather
    target_dt   = pd.to_datetime(target_date).date()
    available   = sorted(
        forecast_weather["timestamp"].dt.date.unique())

    print(f"  Target date     : {target_dt}")
    print(f"  Available dates : {available}")

    # Use closest available date if exact not found
    if target_dt not in available:
        target_dt = available[-1]
        print(f"  Using closest  : {target_dt}")

    target_rows = forecast_weather[
        forecast_weather["timestamp"].dt.date == target_dt
    ].copy()

    # Use last known demand as placeholder for target rows
    last_demand = historical_df["demand_mwh"].iloc[-1]
    target_rows["demand_mwh"] = last_demand

    # Keep only needed columns
    keep_cols = ["timestamp", "demand_mwh",
                 "temperature_c", "humidity_pct",
                 "wind_speed_kmh", "solar_radiation_wm2",
                 "precipitation_mm"]
    hist_cols = [c for c in keep_cols
                 if c in historical_df.columns]

    # Combine history + target
    combined = pd.concat(
        [historical_df[hist_cols],
         target_rows[keep_cols]],
        ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

    # Build features
    featured = build_features(combined)

    # Return only target date rows
    mask   = (featured["timestamp"].dt.date == target_dt)
    result = featured[mask].copy()

    # Ensure all feature cols exist
    for c in feature_cols:
        if c not in result.columns:
            result[c] = 0.0

    # Fill any remaining nulls
    result[feature_cols] = result[feature_cols].fillna(0)

    print(f"  Result rows  : {len(result)}")
    print(f"  Result nulls : {result[feature_cols].isnull().sum().sum()}")

    return result[feature_cols].reset_index(drop=True)
