# app.py — Flask application

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

from flask import (Flask, jsonify, render_template,
                   request, send_file, Response)
import plotly.graph_objects as go
import plotly.utils

from src.ingestion import (fetch_weather_forecast,
                            fetch_eia_day,
                            fetch_weather_archive)
from src.features  import build_inference_row
from src.predict   import load_models, predict_24h, get_metrics

app      = Flask(__name__)
TIMEZONE = "America/Los_Angeles"
DATA_PATH= "data/ca_electricity_raw.csv"

print("Starting — loading models...")
load_models()
print("Ready")


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def _load_and_fill_history():
    """
    Load historical data from CSV.
    Fill any gap between last saved date and yesterday.
    Uses real EIA data if available, else repeats last day pattern.
    """
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    last_date = df["timestamp"].max().date()
    yesterday = (pd.Timestamp.now(tz=TIMEZONE)
                 - timedelta(days=1)).date()
    api_key   = os.environ.get("EIA_API_KEY", "")

    print(f"  Data ends : {last_date}")
    print(f"  Yesterday : {yesterday}")

    # Fill each missing day
    cur_date = last_date + timedelta(days=1)
    while cur_date <= yesterday:
        date_str = cur_date.strftime("%Y-%m-%d")
        filled   = False

        # Try real EIA + weather data
        if api_key:
            try:
                eia_df     = fetch_eia_day(api_key, date_str)
                weather_df = fetch_weather_archive(date_str)

                if eia_df is not None and weather_df is not None:
                    eia_df["timestamp"] = pd.to_datetime(
                        eia_df["timestamp"]
                    ).dt.tz_convert(TIMEZONE)
                    weather_df["timestamp"] = pd.to_datetime(
                        weather_df["timestamp"]
                    ).dt.tz_convert(TIMEZONE)

                    new_day = pd.merge(
                        eia_df, weather_df,
                        on="timestamp", how="inner")

                    df = pd.concat(
                        [df, new_day], ignore_index=True
                    ).drop_duplicates("timestamp"
                    ).sort_values("timestamp"
                    ).reset_index(drop=True)

                    print(f"  EIA added : {date_str} ({len(new_day)} rows)")
                    filled = True
            except Exception as e:
                print(f"  EIA error : {e}")

        # Fallback — repeat last known day pattern
        if not filled:
            last_available = df["timestamp"].max().date()
            last_day_rows  = df[
                df["timestamp"].dt.date == last_available
            ].copy()

            if len(last_day_rows) > 0:
                new_rows = last_day_rows.copy()
                days_diff = (cur_date - last_available).days
                new_rows["timestamp"] = (
                    new_rows["timestamp"]
                    + pd.Timedelta(days=days_diff))
                df = pd.concat(
                    [df, new_rows], ignore_index=True
                ).drop_duplicates("timestamp"
                ).sort_values("timestamp"
                ).reset_index(drop=True)
                print(f"  Pattern   : {date_str} filled")

        cur_date += timedelta(days=1)

    return df.tail(300).copy()


def _get_target_date(df):
    """
    Prediction target = day after last data.
    Data till March 21 → predict March 22.
    Data till March 23 → predict March 24.
    """
    last_date   = df["timestamp"].max().date()
    target_date = last_date + timedelta(days=1)
    return target_date.strftime("%Y-%m-%d")


def _run_forecast():
    """
    Full forecast pipeline.
    Returns (predictions, timestamps, target_date).
    """
    models       = load_models()
    FEATURE_COLS = models["feature_cols"]

    # Load + fill history
    df_hist     = _load_and_fill_history()
    target_date = _get_target_date(df_hist)

    print(f"  Predicting: {target_date}")

    # Get weather forecast
    forecast_weather = fetch_weather_forecast()
    if forecast_weather is None:
        raise ValueError("Weather API unavailable")

    # Build features
    X_feat = build_inference_row(
        df_hist, forecast_weather,
        target_date, FEATURE_COLS)

    if len(X_feat) == 0:
        raise ValueError(
            f"Could not build features for {target_date}")

    # Predict
    predictions = predict_24h(X_feat.values)

    # Build timestamps (hour 0 to 23 of target day)
    base = pd.Timestamp(target_date, tz=TIMEZONE)
    timestamps = [
        (base + pd.Timedelta(hours=h)).isoformat()
        for h in range(24)
    ]

    return predictions, timestamps, target_date


# ════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status"   : "ok",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/metrics")
def metrics():
    return jsonify(get_metrics())


@app.route("/predict")
def predict():
    try:
        predictions, timestamps, target_date = _run_forecast()
        return jsonify({
            "date"        : target_date,
            "generated_at": datetime.now().isoformat(),
            "unit"        : "MWh",
            "timestamps"  : timestamps,
            "predictions" : [round(float(v), 1)
                             for v in predictions],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chart")
def chart():
    try:
        predictions, timestamps, target_date = _run_forecast()
        preds = [round(float(v), 1) for v in predictions]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x         = timestamps,
            y         = preds,
            name      = "XGBoost Forecast",
            line      = dict(color="#1a237e", width=2.5),
            mode      = "lines+markers",
            marker    = dict(size=5),
            fill      = "tozeroy",
            fillcolor = "rgba(26,35,126,0.07)",
        ))

        peak_idx = int(np.argmax(preds))
        fig.add_annotation(
            x           = timestamps[peak_idx],
            y           = preds[peak_idx],
            text        = f"Peak: {preds[peak_idx]:,.0f} MWh",
            showarrow   = True,
            arrowhead   = 2,
            arrowcolor  = "#c62828",
            font        = dict(color="#c62828", size=12),
            bgcolor     = "white",
            bordercolor = "#c62828",
            borderwidth = 1,
        )

        fig.update_layout(
            title     = (f"24h Electricity Demand Forecast"
                         f" — California ({target_date})"),
            xaxis     = dict(
                title      = "Time",
                tickformat = "%H:%M",
                showgrid   = True,
                gridcolor  = "#f0f0f0"),
            yaxis     = dict(
                title    = "Demand (MWh)",
                showgrid = True,
                gridcolor= "#f0f0f0"),
            template  = "plotly_white",
            hovermode = "x unified",
            height    = 420,
            margin    = dict(l=60, r=20, t=60, b=60),
        )

        return Response(
            plotly.utils.PlotlyJSONEncoder().encode(fig),
            mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download")
def download():
    try:
        predictions, timestamps, target_date = _run_forecast()

        rows = [{
            "hour"        : f"h+{i+1}",
            "timestamp"   : ts,
            "forecast_mwh": round(float(predictions[i]), 1),
            "model"       : "XGBoost",
            "generated_at": datetime.now().isoformat(),
        } for i, ts in enumerate(timestamps)]

        df  = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            mimetype       = "text/csv",
            as_attachment  = True,
            download_name  = f"ca_forecast_{target_date}.csv",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    """Debug endpoint — shows data status."""
    try:
        df_hist     = _load_and_fill_history()
        target_date = _get_target_date(df_hist)
        fw          = fetch_weather_forecast()

        return jsonify({
            "target_date"  : target_date,
            "hist_rows"    : len(df_hist),
            "hist_date_min": str(df_hist["timestamp"].min().date()),
            "hist_date_max": str(df_hist["timestamp"].max().date()),
            "weather_rows" : len(fw) if fw is not None else 0,
            "weather_dates": [str(d) for d in sorted(
                fw["timestamp"].dt.date.unique()
            )] if fw is not None else [],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        debug = False,
        host  = "0.0.0.0",
        port  = int(os.environ.get("PORT", 7860)),
    )
