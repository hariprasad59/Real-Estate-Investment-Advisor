# Real Estate Investment Advisor

Predict 5-year property prices and classify investment suitability for Indian residential real estate using scikit-learn, XGBoost, and MLflow — served through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

## What It Does

| Task | Target | Type | Best Model | Key Metric |
|------|--------|------|------------|------------|
| Price Prediction | `Future_Price_5Y` | Regression | XGBRegressor | RMSE, R² |
| Investment Classification | `Good_Investment` | Binary Classification | XGBClassifier | AUC, Precision, Recall |

The Streamlit app provides four views:

- **Home** — KPI dashboard with quick search and top-cities chart
- **EDA** — 20 interactive analyses (distributions, correlations, outliers, comparisons)
- **Filters & Search** — Slice the dataset by city, property type, BHK, and price with CSV export
- **Predict** — Enter property details, get a 5-year price estimate and investment verdict with confidence scores

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data & EDA | pandas, numpy, matplotlib, seaborn, Plotly |
| Preprocessing | scikit-learn (`ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`) |
| Models | Linear/Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Experiment Tracking | MLflow (tracking, model registry, Production stage management) |
| UI | Streamlit (multi-page app) |

## Architecture

```
Raw CSV (250K rows) → EDA (20 analyses)
                    → Feature Engineering (Amenity_Count, growth rates)
                    → Target Engineering (Future_Price_5Y, Good_Investment)
                    → Train/Test Split (80/20, seed=42)
                    → ColumnTransformer (StandardScaler + OneHotEncoder)
                    → Regression Models (LR, RF, GB, XGB)  → MLflow Tracking
                    → Classification Models (LogReg, RF, XGB) → MLflow Registry
                    → Best models promoted to Production
                    → Streamlit App loads Production models for inference
```

## How to Run

### 1. Clone & install

```bash
git clone https://github.com/HariPrasad599/RealEstateInvestmentAdvisor.git
cd RealEstateInvestmentAdvisor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost mlflow streamlit plotly matplotlib seaborn
```

### 2. Train models (notebook)

Open `RealEstateInvestment.ipynb` and run all cells. This will:
- Load `india_housing_prices.csv`, perform EDA, engineer features and targets
- Train regression and classification models with MLflow tracking
- Register the best models and promote them to Production stage

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Models are loaded from the local `mlruns/` directory.

## Project Structure

```
RealEstateInvestmentAdvisor/
├── app.py                          ← Streamlit multi-page app
├── RealEstateInvestment.ipynb      ← Full ML pipeline (EDA → training → registry)
├── PROJECT_FLOW.md                 ← Detailed architecture & design documentation
├── india_housing_prices.csv        ← Raw dataset (250K rows, 23 columns)
├── target_indian_house_price.csv   ← Engineered dataset with targets
├── regression_results.csv          ← Model comparison results
├── classification_results.csv      ← Model comparison results
└── mlruns/                         ← MLflow tracking store & registered models
```

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| **Pipeline wrapping** | Each model is wrapped in a `sklearn.pipeline.Pipeline` with the preprocessor, so transforms travel with the artifact |
| **MLflow registry** | Best model is auto-selected (lowest RMSE / highest AUC), registered, and promoted — reproducible model management |
| **Simulated targets** | `Future_Price_5Y` uses data-driven growth rates (city + locality + age + infrastructure + noise) since real historical data isn't available; documented transparently |
| **Leakage-free labels** | `Good_Investment` is derived from `growth_rate_effective > median`, not from `Price_per_SqFt` which is a training feature |
| **Permutation importance** | The Predict page offers local explainability via `sklearn.inspection.permutation_importance` |

## Limitations

- **Simulated targets** — `Future_Price_5Y` is a methodology demo; real deployment would need actual historical price data
- **Single-user** — Local MLflow file store, no auth or multi-tenancy
- **No CI/CD** — Manual training and deployment; see `PROJECT_FLOW.md` §10 for a retraining pipeline design

## License

MIT
