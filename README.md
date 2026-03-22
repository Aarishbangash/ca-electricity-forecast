<<<<<<< HEAD
# California Electricity Demand Forecast

24-hour ahead demand forecasting for California (CISO region).

- Model: XGBoost (3.38% test MAPE)
- Data: EIA API + Open-Meteo
- Retrain: Every day 2AM UTC via GitHub Actions (automatic)
- Serve: HuggingFace Spaces (Flask + Docker)

## Files you need to add

After extracting this project, copy your Colab downloads:

```
models/
  model_xgboost.pkl      <- from Colab
  feature_cols.json      <- from Colab

data/
  ca_electricity_raw.csv <- from Colab
```

## Deployment Steps

1. Add your files (see above)
2. Push to GitHub
3. Add EIA_API_KEY to GitHub Secrets
4. Create HuggingFace Space (Docker SDK)
5. Link Space to your GitHub repo

## API Endpoints

- GET /              Dashboard
- GET /predict?date  JSON forecast
- GET /chart?date    Plotly chart
- GET /download?date CSV download
- GET /metrics       Model metrics
- GET /health        Health check
=======
# ca-electricity-forecast
>>>>>>> 90dc84f6739d6cff804c2dd4cc3a3ea0d73479d6
