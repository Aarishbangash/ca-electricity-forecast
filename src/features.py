# src/features.py
# Feature engineering — identical to Colab pipeline

import pandas as pd
import numpy as np
import holidays

BASE_TEMP = 13.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(
        "timestamp").reset_index(drop=True)

    if "demand_mwh" in df.columns:
        df["demand_mwh"] = df["demand_mwh"].interpolate(
            method="linear")

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

    # ── Fourier features based on ACTUAL time, not row index ──
    # This ensures inference rows get the same encoding as training
    # hour_of_week = 0..167, hour_of_year = 0..8759
    hour_of_day  = df["timestamp"].dt.hour.astype(float)
    day_of_year  = df["timestamp"].dt.dayofyear.astype(float)
    day_of_week  = df["timestamp"].dt.dayofweek.astype(float)
    hour_of_week = day_of_week * 24 + hour_of_day

    for k in [1, 2]:
        df[f"sin_daily_{k}"]  = np.sin(
            2 * np.pi * k * hour_of_day / 24)
        df[f"cos_daily_{k}"]  = np.cos(
            2 * np.pi * k * hour_of_day / 24)
        df[f"sin_weekly_{k}"] = np.sin(
            2 * np.pi * k * hour_of_week / 168)
        df[f"cos_weekly_{k}"] = np.cos(
            2 * np.pi * k * hour_of_week / 168)

    df["CDD"] = np.maximum(
        df["temperature_c"] - BASE_TEMP, 0)
    df["HDD"] = np.maximum(
        BASE_TEMP - df["temperature_c"], 0)
    df["heat_index"] = (
        df["temperature_c"]
        + 0.33 * (df["humidity_pct"] / 100 * 6.105
                  * np.exp(17.27 * df["temperature_c"]
                  / (237.7 + df["temperature_c"])))
        - 4.0
    )
    df["is_daytime"]    = (
        (df["hour"] >= 7) & (df["hour"] <= 19)
    ).astype(int)
    df["solar_x_CDD"]   = (
        df["solar_radiation_wm2"] * df["CDD"])
    df["solar_daytime"] = (
        df["solar_radiation_wm2"] * df["is_daytime"])

    df["temp_bin"] = pd.cut(
        df["temperature_c"],
        bins   = [-np.inf, 5, 10, 15, 20, 25, 30, np.inf],
        labels = [0, 1, 2, 3, 4, 5, 6]
    ).astype(float)

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"demand_lag_{lag}h"] = df["demand_mwh"].shift(lag)

    for window in [3, 6, 12, 24, 48, 168]:
        df[f"demand_roll_mean_{window}h"] = (
            df["demand_mwh"].shift(1)
              .rolling(window, min_periods=1).mean()
        )
    for window in [24, 168]:
        df[f"demand_roll_std_{window}h"] = (
            df["demand_mwh"].shift(1)
              .rolling(window, min_periods=1).std()
        )
    for window in [3, 24]:
        df[f"temp_roll_mean_{window}h"] = (
            df["temperature_c"].shift(1)
              .rolling(window, min_periods=1).mean()
        )
        df[f"solar_roll_mean_{window}h"] = (
            df["solar_radiation_wm2"].shift(1)
              .rolling(window, min_periods=1).mean()
        )

    return df


def build_inference_row(historical_df, forecast_weather,
                         target_date, feature_cols):
    TIMEZONE = "America/Los_Angeles"

    forecast_weather = forecast_weather.copy()
    forecast_weather["timestamp"] = pd.to_datetime(
        forecast_weather["timestamp"]
    ).dt.tz_convert(TIMEZONE)

    historical_df = historical_df.copy()
    historical_df["timestamp"] = pd.to_datetime(
        historical_df["timestamp"]
    ).dt.tz_convert(TIMEZONE)

    # Need at least 168 rows for the largest lag (168h = 1 week)
    historical_df = historical_df.sort_values(
        "timestamp").tail(200).reset_index(drop=True)

    print(f"  History rows : {len(historical_df)}")
    print(f"  History end  : {historical_df['timestamp'].max()}")
    print(f"  Null demand  : {historical_df['demand_mwh'].isna().sum()}")

    # Fix deprecated fillna(method=...) 
    historical_df["demand_mwh"] = (
        historical_df["demand_mwh"]
        .ffill()
        .bfill()
    )

    target_dt = pd.to_datetime(target_date).date()

    available_dates = sorted(
        forecast_weather["timestamp"].dt.date.unique())

    print(f"  Available forecast dates: {available_dates}")
    print(f"  Requested target date   : {target_dt}")

    # Pick the closest available forecast date to target_dt
    # Prefer exact match, otherwise use nearest future date,
    # fallback to nearest past date
    if target_dt in available_dates:
        use_dt = target_dt
    else:
        future_dates = [d for d in available_dates if d >= target_dt]
        if future_dates:
            use_dt = future_dates[0]
        else:
            use_dt = available_dates[-1]
        print(f"  Target date not in forecast — using: {use_dt}")

    target_rows = forecast_weather[
        forecast_weather["timestamp"].dt.date == use_dt
    ].copy()

    # Restamp target_rows to the actual target_date
    # so that lag features and Fourier features are correct
    if use_dt != target_dt:
        delta = pd.Timestamp(target_dt) - pd.Timestamp(use_dt)
        target_rows["timestamp"] = target_rows["timestamp"] + delta

    # Add placeholder demand (last known value)
    last_known_demand = historical_df["demand_mwh"].iloc[-1]
    target_rows["demand_mwh"] = last_known_demand

    keep_cols = ["timestamp", "demand_mwh",
                 "temperature_c", "humidity_pct",
                 "wind_speed_kmh", "solar_radiation_wm2",
                 "precipitation_mm"]

    hist_cols = [c for c in keep_cols
                 if c in historical_df.columns]

    combined = pd.concat(
        [historical_df[hist_cols],
         target_rows[keep_cols]],
        ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"  Combined rows: {len(combined)}")

    featured = build_features(combined)

    # Extract target date rows using the actual target_dt
    mask   = (featured["timestamp"].dt.date == target_dt)
    result = featured[mask].copy()

    print(f"  Result rows  : {len(result)}")
    print(f"  Result nulls : {result[feature_cols].isnull().sum().sum()}")

    for c in feature_cols:
        if c not in result.columns:
            result[c] = 0.0

    result[feature_cols] = result[feature_cols].fillna(0)

    return result[feature_cols].reset_index(drop=True)