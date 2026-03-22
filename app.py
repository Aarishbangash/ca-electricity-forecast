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

from src.ingestion import fetch_weather_forecast
from src.features  import build_inference_row
from src.predict   import load_models, predict_24h, get_metrics

app      = Flask(__name__)
TIMEZONE = "America/Los_Angeles"
DATA_PATH= "data/ca_electricity_raw.csv"

print("Starting — loading models...")
load_models()
print("Ready")


def _get_historical():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)
    return df.tail(300).copy()


def _default_date():
    return (datetime.now() + timedelta(days=1)
            ).strftime("%Y-%m-%d")


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
        target_date  = request.args.get("date", _default_date())
        models       = load_models()
        FEATURE_COLS = models["feature_cols"]

        df_hist          = _get_historical()
        forecast_weather = fetch_weather_forecast()

        if forecast_weather is None:
            return jsonify({"error": "Weather API failed"}), 503

        X_feat = build_inference_row(
            df_hist, forecast_weather,
            target_date, FEATURE_COLS)

        if len(X_feat) == 0:
            return jsonify({
                "error": f"No data for {target_date}"
            }), 400

        predictions = predict_24h(X_feat.values)

        base = pd.Timestamp(target_date, tz=TIMEZONE)
        timestamps = [
            (base + pd.Timedelta(hours=h)).isoformat()
            for h in range(24)
        ]

        return jsonify({
            "date"        : target_date,
            "generated_at": datetime.now().isoformat(),
            "unit"        : "MWh",
            "timestamps"  : timestamps,
            "predictions" : [
                round(float(v), 1) for v in predictions
            ],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chart")
def chart():
    try:
        target_date = request.args.get("date", _default_date())

        models       = load_models()
        FEATURE_COLS = models["feature_cols"]

        df_hist          = _get_historical()
        forecast_weather = fetch_weather_forecast()

        if forecast_weather is None:
            return jsonify({"error": "Weather API failed"}), 503

        X_feat = build_inference_row(
            df_hist, forecast_weather,
            target_date, FEATURE_COLS)

        if len(X_feat) == 0:
            return jsonify({
                "error": f"No data for {target_date}"
            }), 400

        predictions = predict_24h(X_feat.values)

        base = pd.Timestamp(target_date, tz=TIMEZONE)
        timestamps = [
            (base + pd.Timedelta(hours=h)).isoformat()
            for h in range(24)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x         = timestamps,
            y         = [round(float(v), 1) for v in predictions],
            name      = "XGBoost Forecast",
            line      = dict(color="#1a237e", width=2.5),
            mode      = "lines+markers",
            marker    = dict(size=5),
            fill      = "tozeroy",
            fillcolor = "rgba(26,35,126,0.07)",
        ))

        peak_idx = int(np.argmax(predictions))
        fig.add_annotation(
            x           = timestamps[peak_idx],
            y           = float(predictions[peak_idx]),
            text        = f"Peak: {predictions[peak_idx]:,.0f} MWh",
            showarrow   = True,
            arrowhead   = 2,
            arrowcolor  = "#c62828",
            font        = dict(color="#c62828", size=12),
            bgcolor     = "white",
            bordercolor = "#c62828",
            borderwidth = 1,
        )

        fig.update_layout(
            title     = f"24h Electricity Demand Forecast — {target_date}",
            xaxis     = dict(title="Time", tickformat="%H:%M",
                             showgrid=True, gridcolor="#f0f0f0"),
            yaxis     = dict(title="Demand (MWh)",
                             showgrid=True, gridcolor="#f0f0f0"),
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
        target_date = request.args.get("date", _default_date())

        models       = load_models()
        FEATURE_COLS = models["feature_cols"]

        df_hist          = _get_historical()
        forecast_weather = fetch_weather_forecast()

        if forecast_weather is None:
            return jsonify({"error": "Weather API failed"}), 503

        X_feat = build_inference_row(
            df_hist, forecast_weather,
            target_date, FEATURE_COLS)

        predictions = predict_24h(X_feat.values)

        base = pd.Timestamp(target_date, tz=TIMEZONE)
        timestamps = [
            (base + pd.Timedelta(hours=h)).isoformat()
            for h in range(24)
        ]

        rows = []
        for i, ts in enumerate(timestamps):
            rows.append({
                "hour"        : f"h+{i+1}",
                "timestamp"   : ts,
                "forecast_mwh": round(float(predictions[i]), 1),
                "model"       : "XGBoost",
                "generated_at": datetime.now().isoformat(),
            })

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
    try:
        target_date  = request.args.get("date", _default_date())
        models       = load_models()
        FEATURE_COLS = models["feature_cols"]
        df_hist      = _get_historical()
        forecast_weather = fetch_weather_forecast()

        X_feat = build_inference_row(
            df_hist, forecast_weather,
            target_date, FEATURE_COLS)

        return jsonify({
            "target_date"     : target_date,
            "hist_rows"       : len(df_hist),
            "hist_date_min"   : str(df_hist["timestamp"].min()),
            "hist_date_max"   : str(df_hist["timestamp"].max()),
            "X_feat_shape"    : list(X_feat.shape),
            "X_feat_nulls"    : int(X_feat.isnull().sum().sum()),
            "X_feat_sample"   : X_feat.head(2).to_dict(),
            "forecast_rows"   : len(forecast_weather),
            "forecast_min"    : str(forecast_weather["timestamp"].min()),
            "forecast_max"    : str(forecast_weather["timestamp"].max()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        debug = False,
        host  = "0.0.0.0",
        port  = int(os.environ.get("PORT", 7860)),
    )
