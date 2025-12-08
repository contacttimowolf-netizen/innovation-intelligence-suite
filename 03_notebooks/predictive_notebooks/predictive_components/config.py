from pathlib import Path

# Repo root = innovation-intelligence-suite
BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "01_data" / "predictive_model"

PATH_CORPUS     = DATA_DIR / "df_auto_corpus_area_tech.parquet"
PATH_TS         = DATA_DIR / "area_tech_timeseries.parquet"
PATH_GROWTH     = DATA_DIR / "forecast_area_tech_growth.parquet"
PATH_TRANSITION = DATA_DIR / "forecast_area_tech_transition.parquet"
