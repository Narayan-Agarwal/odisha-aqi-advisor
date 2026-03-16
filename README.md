# Odisha Industrial AQI Analytics and Next-Day Health Advisor

Air quality analytics and next-day AQI prediction for 10 Odisha cities across the industrial corridor. Built with XGBoost + Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)

**Live App**: YOUR_APP_URL

## City Coverage

| City | Tier | Primary Industry |
|------|------|-----------------|
| Jharsuguda | Heavy Industrial | Vedanta aluminium smelter, JSPL thermal power |
| Angul | Heavy Industrial | NALCO aluminium, NTPC Kaniha |
| Talcher | Heavy Industrial | Talcher coalfields, NTPC Talcher |
| Rourkela | Heavy Industrial | SAIL steel plant, fertiliser and chemical industry |
| Sambalpur | Heavy Industrial | Cement plants, coal belt proximity |
| Bhubaneswar | Urban | Traffic, construction activity, urban growth |
| Cuttack | Urban | Dense traffic, old city road network, biomass burning |
| Bhadrak | Urban | Port-adjacent industry, light manufacturing |
| Ganjam | Clean Baseline | Coastal city, sea breeze dispersion |
| Koraput | Clean Baseline | Hilly terrain, bauxite mining |

## Setup

```bash
git clone https://github.com/Narayan-Agarwal/odisha-aqi-advisor.git
cd odisha-aqi-advisor
pip install -r requirements.txt
# Run notebooks 01-04 in order
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelling.ipynb
streamlit run app.py
```

## Project Structure

```
odisha-aqi-advisor/
├── data/processed/     # featured.csv, model_results.csv (committed)
├── models/             # 20 .joblib files + feature_columns.json
├── charts/             # 16 PNG charts
├── notebooks/          # 01-04 pipeline notebooks
├── src/                # Python library modules
├── app.py              # Streamlit entry point
└── requirements.txt
```

## Models

- Linear Regression (baseline)
- XGBoost Regressor (primary): n_estimators=200, learning_rate=0.05, max_depth=5

Features: aqi_yesterday, aqi_7day_avg, pm25_lag1, pm10_lag1, so2_lag1, no2_lag1, month, day_of_week, is_winter, is_monsoon, is_diwali_week, is_industrial_peak, tier_encoded

## Key Findings

- Industrial corridor cities (Jharsuguda, Angul, Talcher) consistently show highest AQI values
- Monsoon season (Jul-Sep) brings significant AQI reduction across all cities
- Diwali causes sharp PM2.5 spikes visible across all city tiers
- XGBoost outperforms Linear Regression for industrial cities with spike patterns

## Known Limitations

- No weather data (wind speed, temperature, humidity) included
- Odisha station data sparsity for some cities in early years
- Training window limited to 2019-2023

## Future Scope

- Sem 7: Weather API integration, LSTM model, national city expansion

## License

MIT License — see [LICENSE](LICENSE) file.

## Authors

Narayan Agarwal (Regd. 2302041076), VSSUT Burla CSE
