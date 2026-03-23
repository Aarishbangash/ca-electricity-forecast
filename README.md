---
title: Ca Electricity Forecast
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# California Electricity Demand Forecast

24-hour ahead demand forecasting for California (CISO region).

- Model: XGBoost (3.38% test MAPE)
- Data: EIA API + Open-Meteo
- Retrain: Every day 2AM UTC via GitHub Actions
- Deploy: HuggingFace Spaces (Flask + Docker)

## API Endpoints

- GET /           Dashboard
- GET /predict    JSON forecast
- GET /chart      Plotly chart
- GET /download   CSV download
- GET /metrics    Model metrics
- GET /health     Health check
