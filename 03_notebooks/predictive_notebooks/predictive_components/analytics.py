import pandas as pd
import matplotlib.pyplot as plt
from .ts_core import (
    load_ts_from_parquet,
    estimate_growth,
    forecast_paper_to_patent_shift,
)


def load_area_tech_ts() -> pd.DataFrame:
    return load_ts_from_parquet()


def get_fastest_growing_topics(df_ts: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    df_growth = estimate_growth(df_ts)
    return df_growth.sort_values(
        "growth_slope_n_total", ascending=False
    ).head(top_n)


def get_transitioning_technologies(
    df_ts: pd.DataFrame,
    horizon: int = 12,
) -> pd.DataFrame:
    return forecast_paper_to_patent_shift(df_ts, horizon=horizon)


def get_likely_to_mature_next_year(
    df_ts: pd.DataFrame,
    horizon: int = 12,
    quantile_delta: float = 0.5,
    max_last_share: float = 0.9,
) -> pd.DataFrame:
    """
    'Likely to mature next year':

    - Şu an patent payı max_last_share'in altında
    - Patent payı artışı (delta) üst quantile’da (örn 0.75)
    - Tahmini patent payı, tüm dağılımın medyanından yüksek
    """

    df_trans = forecast_paper_to_patent_shift(df_ts, horizon=horizon)
    if df_trans.empty:
        return df_trans

    delta_thr = df_trans["delta_share_patent"].quantile(quantile_delta)
    forecast_med = df_trans["forecast_share_patent_mean"].median()

    mask = (
        (df_trans["last_share_patent"] < max_last_share)
        & (df_trans["delta_share_patent"] >= delta_thr)
        & (df_trans["forecast_share_patent_mean"] >= forecast_med)
    )

    return df_trans.loc[mask].sort_values(
        "forecast_share_patent_mean", ascending=False
    ).copy()

def plot_simple_timeseries(
    df_ts: pd.DataFrame,
    area: str,
    tech: str,
    value_col: str = "n_total"
):
    """
    Streamlit içinde gösterilebilecek basit zaman serisi grafiği üretir.

    Beklenen kolonlar:
    - auto_focus_area
    - auto_tech_cluster
    - date
    - value_col (default: n_total)
    """

    g = df_ts[
        (df_ts["auto_focus_area"] == area) &
        (df_ts["auto_tech_cluster"] == tech)
    ].sort_values("date")

    if g.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data found for this selection")
        return fig

    fig, ax = plt.subplots()
    ax.plot(g["date"], g[value_col])
    ax.set_title(f"{tech} – {value_col}")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)

    return fig