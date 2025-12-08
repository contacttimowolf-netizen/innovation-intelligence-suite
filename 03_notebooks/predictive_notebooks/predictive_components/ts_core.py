import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

from .config import PATH_TS


def load_ts_from_parquet() -> pd.DataFrame:
    df_ts = pd.read_parquet(PATH_TS)
    df_ts["date"] = pd.to_datetime(df_ts["date"])
    return df_ts


def prepare_time_series(group_df: pd.DataFrame, value_col: str):
    g = group_df.sort_values("date")
    g = g[["date", value_col]].dropna()

    if len(g) < 4:
        return None, None

    t = np.arange(len(g)).reshape(-1, 1)
    y = g[value_col].values.astype(float)
    return t, y


def linear_forecast(group_df: pd.DataFrame, value_col: str, horizon: int = 12):
    g = group_df.sort_values("date")
    t, y = prepare_time_series(g, value_col)
    if t is None:
        return g, None

    model = LinearRegression()
    model.fit(t, y)

    last_t = t[-1, 0]
    future_t = np.arange(last_t + 1, last_t + 1 + horizon).reshape(-1, 1)
    y_pred = model.predict(future_t)

    last_date = g["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        f"forecast_{value_col}": y_pred,
    })

    return g, forecast_df


def _logistic_fn(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


def logistic_forecast(group_df: pd.DataFrame, value_col: str, horizon: int = 12):
    g = group_df.sort_values("date")
    t, y = prepare_time_series(g, value_col)
    if t is None:
        return g, None

    t = t.astype(float).ravel()
    y_min, y_max = float(y.min()), float(y.max())

    if y_max == y_min:
        return linear_forecast(g, value_col, horizon=horizon)

    y_norm = (y - y_min) / (y_max - y_min)

    p0 = [1.0, 0.3, np.median(t)]

    try:
        params, _ = curve_fit(_logistic_fn, t, y_norm, p0=p0, maxfev=10000)
        K, r, t0 = params

        last_t = t[-1]
        future_t = np.arange(last_t + 1, last_t + 1 + horizon)

        t_all = np.concatenate([t, future_t])
        y_pred_norm_all = _logistic_fn(t_all, K, r, t0)
        y_pred_norm_all = np.clip(y_pred_norm_all, 0.0, 1.0)

        y_pred_all = y_min + y_pred_norm_all * (y_max - y_min)
        y_future = y_pred_all[len(t):]

        last_date = g["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        forecast_df = pd.DataFrame({
            "date": future_dates,
            f"logistic_{value_col}": y_future,
        })

        return g, forecast_df

    except Exception:
        return linear_forecast(g, value_col, horizon=horizon)


def estimate_growth(df_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (area, tech), g in df_ts.groupby(
        ["auto_focus_area", "auto_tech_cluster"]
    ):
        t, y = prepare_time_series(g, "n_total")
        if t is None:
            continue

        model = LinearRegression()
        model.fit(t, y)
        slope = float(model.coef_[0])

        rows.append({
            "auto_focus_area": area,
            "auto_tech_cluster": tech,
            "growth_slope_n_total": slope,
            "n_total_last": y[-1],
        })

    return pd.DataFrame(rows)


def forecast_paper_to_patent_shift(
    df_ts: pd.DataFrame,
    horizon: int = 12,
    min_app: float = 0.7,
    max_research: float = 0.3,
    growth_thr: float = 0.03,
    decline_thr: float = -0.03,
) -> pd.DataFrame:

    def classify_stage(last_share_patent: float, future_mean: float) -> str:
        delta = future_mean - last_share_patent

        if last_share_patent >= min_app and delta <= decline_thr:
            return "Over-Mature"

        if last_share_patent >= min_app:
            return "Application Now"

        if last_share_patent <= max_research and delta <= growth_thr:
            return "Still Research"

        return "Transitioning"

    rows = []

    for (area, tech), g in df_ts.groupby(
        ["auto_focus_area", "auto_tech_cluster"]
    ):
        actual, forecast = linear_forecast(g, "share_patent", horizon=horizon)
        if forecast is None:
            continue

        last_val = actual["share_patent"].iloc[-1]
        future_mean = float(forecast["forecast_share_patent"].mean())
        stage = classify_stage(last_val, future_mean)

        rows.append({
            "auto_focus_area": area,
            "auto_tech_cluster": tech,
            "last_share_patent": last_val,
            "forecast_share_patent_mean": future_mean,
            "delta_share_patent": future_mean - last_val,
            "tech_stage": stage,
        })

    return pd.DataFrame(rows)
