import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .ts_core import (
    load_ts_from_parquet,
    estimate_growth,
    forecast_paper_to_patent_shift,
)


def load_area_tech_ts() -> pd.DataFrame:
    return load_ts_from_parquet()

def _to_quarterly(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly time series -> quarterly toplulaştırma.

    - n_paper / n_patent yoksa, n_total + share_patent veya paper/patent'ten türetir.
    - Sadece TAMAMLANMIŞ quarter'ları (3 farklı ay içeren) bırakır.
    """
    df = df_ts.copy()
    df = df.sort_values("date")

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    # n_paper / n_patent garanti
    if ("n_paper" not in df.columns) or ("n_patent" not in df.columns):
        if {"n_total", "share_patent"}.issubset(df.columns):
            df["n_patent"] = (df["n_total"] * df["share_patent"]).fillna(0).round().astype(int)
            df["n_paper"] = (df["n_total"] - df["n_patent"]).astype(int)
        elif {"paper", "patent"}.issubset(df.columns):
            df["n_paper"] = df["paper"].fillna(0).astype(int)
            df["n_patent"] = df["patent"].fillna(0).astype(int)
        else:
            raise ValueError(
                "Time series data must contain either "
                "'n_paper'/'n_patent' or 'n_total'+'share_patent' "
                "or 'paper'/'patent' columns."
            )

    df["quarter"] = df["date"].dt.to_period("Q")

    # Hangi quarter'ların 3 farklı ayı var? (tamamlanmış quarter filtresi)
    month_counts = (
        df.groupby("quarter")["date"]
        .agg(lambda x: x.dt.month.nunique())
    )
    full_quarters = month_counts[month_counts == 3].index

    df = df[df["quarter"].isin(full_quarters)]

    agg = (
        df.groupby(["auto_focus_area", "auto_tech_cluster", "quarter"])
        .agg(
            n_paper_q=("n_paper", "sum"),
            n_patent_q=("n_patent", "sum"),
        )
        .reset_index()
    )

    agg["quarter_end"] = agg["quarter"].dt.to_timestamp(how="end")
    return agg




def get_fastest_growing_topics(df_ts: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Fastest Growing Tech (GLOBAL):

    - Her teknoloji için TÜM alanlardaki (auto_focus_area) paper sayıları
      birleştirilir -> global teknoloji seviyesi.
    - Ölçüt: Son quarter'daki paper sayısının, bir önceki quarter'a göre
      yüzdesel artışı ( (last_q - prev_q) / prev_q ).
    - Quarterly bazda yalnızca PAPER kullanır (n_paper_q).
    - Dönen df SÜREKLİ 'Global' area içerir:
        auto_focus_area     -> "Global"
        auto_tech_cluster
        growth_slope_n_total  -> growth rate (son quarter vs önceki quarter)
        n_total_last          -> son quarter toplam paper sayısı (GLOBAL)
    """

    # Aylıktan çeyrek veriye
    qdf = _to_quarterly(df_ts)

    # --- KRİTİK NOKTA ---
    # Tüm auto_focus_area'ları toplayıp teknoloji bazına indiriyoruz
    qdf_global = (
        qdf.groupby(["auto_tech_cluster", "quarter"])
        .agg(
            n_paper_q=("n_paper_q", "sum"),
            quarter_end=("quarter_end", "max"),
        )
        .reset_index()
    )

    rows = []
    for tech, g in qdf_global.groupby("auto_tech_cluster"):
        g = g.sort_values("quarter_end")

        # En az 3 quarter verisi olsun
        if len(g) < 3:
            continue

        last_q = float(g["n_paper_q"].iloc[-1])
        prev_q = float(g["n_paper_q"].iloc[-2])

        if prev_q <= 0:
            # Çok eski/seyrek veriler; growth hesaplamak anlamsız
            continue

        growth_rate = (last_q - prev_q) / prev_q  # örn. 0.4 = %40

        rows.append(
            {
                "auto_focus_area": "Global",      # TEK AREA: GLOBAL
                "auto_tech_cluster": tech,
                "growth_slope_n_total": growth_rate,
                "n_total_last": int(last_q),      # 50 + 70 = 120 gibi
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "auto_focus_area",
                "auto_tech_cluster",
                "growth_slope_n_total",
                "n_total_last",
            ]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("growth_slope_n_total", ascending=False)

    # En hızlı büyüyen ilk N global teknoloji
    return out.head(top_n)





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
    top_n: int = 5,   # <-- EKLEDİK
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

    out = df_trans.loc[mask].sort_values(
        "forecast_share_patent_mean", ascending=False
    ).copy()

    # SADECE EN İYİ top_n TEKNOLOJİ
    return out.head(top_n)


from matplotlib.ticker import FuncFormatter
def plot_simple_timeseries(
    df_ts: pd.DataFrame,
    area: str,      # çağrı imzası bozulmasın diye duruyor, içeride kullanmıyoruz
    tech: str,
    max_quarters: int = 5,
):
    """
    Commercial interest momentum (relative publication index):

    - Teknoloji tüm alanlardaki kümülatif paper sayısına göre çizilir
      (A=50, B=70 ise grafikte 120 olarak temsil edilir).
    - Bar: seçilen teknolojinin index’i
    - Kesikli çizgi: aynı quarter'lardaki tüm teknolojilerin ortalama index’i
    """

    qdf = _to_quarterly(df_ts)

    # --- 1) Seçilen tech: TÜM alanlardan toplanmış quarterly paper sayısı ---
    tech_df = (
        qdf[qdf["auto_tech_cluster"] == tech]
        .groupby("quarter", as_index=False)
        .agg(n_paper_q=("n_paper_q", "sum"))
    )

    if tech_df.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#DDDDDD")
        ax.set_axis_off()
        fig.patch.set_alpha(0.0)
        return fig

    tech_df = tech_df.sort_values("quarter")

    # --- 2) Global baseline: tüm teknolojiler (tüm area'lar) toplam paper ---
    overall_q = (
        qdf.groupby("quarter", as_index=False)["n_paper_q"]
        .sum()
        .rename(columns={"n_paper_q": "overall_paper_q"})
    )

    g = tech_df.merge(overall_q, on="quarter", how="left")

    # Son max_quarters quarter
    g = g.tail(max_quarters).reset_index(drop=True)

    # --- 3) Index hesapları (yüzde şeklinde gösterilecek) ---
    tech_mean = g["n_paper_q"].mean()
    if pd.isna(tech_mean) or tech_mean == 0:
        tech_mean = 1.0
    g["tech_index"] = (g["n_paper_q"] / tech_mean) * 100.0

    overall_mean = g["overall_paper_q"].mean()
    if pd.isna(overall_mean) or overall_mean == 0:
        overall_mean = 1.0
    g["overall_index"] = (g["overall_paper_q"] / overall_mean) * 100.0

    # --- 4) Plot ---
    x = np.arange(len(g))
    labels = g["quarter"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(6, 3))

    # Arka plan şeffaf
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Barlar: seçilen teknoloji
    tech_label = tech.replace("_", " ")
    ax.bar(
        x,
        g["tech_index"].values,
        width=0.6,
        label=tech_label,
    )

    # Kesikli çizgi: tüm teknolojilerin baseline index'i (quarter-quarter değişen)
    ax.plot(
        x,
        g["overall_index"].values,
        linestyle="--",
        linewidth=1.0,
        label="All technologies baseline",
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Y ekseni: % formatı
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}%"))
    ax.set_ylabel("Relative publication momentum", color="#DDDDDD")

    ax.tick_params(axis="x", colors="#DDDDDD")
    ax.tick_params(axis="y", colors="#DDDDDD")
    for spine in ax.spines.values():
        spine.set_color("#666666")

    ax.legend(loc="upper left", frameon=False, labelcolor="#DDDDDD")

    fig.tight_layout()
    return fig








def plot_maturity_derivatives(
    df_ts: pd.DataFrame,
    area: str,
    tech: str,
    max_quarters: int = 8,
):
    """
    Tech Maturity paneli:
    - Sol: Patent momentum (1. türev)
    - Sağ: Momentum ivmesi (2. türev)
    """

    qdf = _to_quarterly(df_ts)

    g = qdf[(qdf["auto_focus_area"] == area) & (qdf["auto_tech_cluster"] == tech)].copy()
    if g.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#DDDDDD")
        ax.set_axis_off()
        return fig

    g = g.sort_values("quarter_end")
    g["cum_patent"] = g["n_patent_q"].cumsum()

    # Türevler
        # Türevler
    g["d1"] = g["cum_patent"].diff()
    g["d2"] = g["d1"].diff()

    # Baştaki tamamen düz dönemleri biraz kes (grafik çok düz olmasın)
    nonflat = g[(g["d1"] != 0) | (g["d2"] != 0)]
    if not nonflat.empty:
        first_idx = nonflat.index[0]
        g = g.loc[first_idx:].copy()

    # Son max_quarters
    g = g.tail(max_quarters).copy()

    # --- Son türev değerlerine göre kısa yorum başlıkları ---
    last_d1 = g["d1"].iloc[-1]
    last_d2 = g["d2"].iloc[-1]

    # 1. türev yorumu
    if last_d1 > 0:
        momentum_title = "Momentum: positive – activity still expanding"
    elif last_d1 < 0:
        momentum_title = "Momentum: negative – activity in contraction"
    else:
        momentum_title = "Momentum: flat – no net change in activity"

    # 2. türev + outlook yorumu
    if (last_d2 > 0) and (last_d1 > 0):
        outlook_title = "Outlook: growth accelerating in the near term"
    elif (last_d2 < 0) and (last_d1 > 0):
        outlook_title = "Outlook: growth slowing – risk of plateau"
    elif (last_d2 > 0) and (last_d1 <= 0):
        outlook_title = "Outlook: early recovery signals – momentum turning up"
    elif (last_d2 < 0) and (last_d1 < 0):
        outlook_title = "Outlook: decline deepening – downside pressure"
    else:
        outlook_title = "Outlook: broadly stable – no strong inflection"

    x = np.arange(len(g))
    labels = [str(q) for q in g["quarter"].astype(str)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)


    # Arka plan şeffaf
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")

    line_color = "#4DB6AC"

    # 1. türev – “Patent momentum”
    axes[0].plot(x, g["d1"], marker="o", color=line_color)
    axes[0].axhline(0, linestyle="--", linewidth=0.8, color="#888888")
    axes[0].set_title(
        momentum_title,
        color="#DDDDDD",
        fontsize=9,
    )
    axes[0].set_ylabel("Δ patents", color="#DDDDDD")

    # 2. türev – “Momentum acceleration”
    axes[1].plot(x, g["d2"], marker="o", color=line_color)
    axes[1].axhline(0, linestyle="--", linewidth=0.8, color="#888888")
    axes[1].set_title(
        outlook_title,
        color="#DDDDDD",
        fontsize=9,
    )
    axes[1].set_ylabel("Δ² patents", color="#DDDDDD")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.tick_params(axis="x", colors="#DDDDDD")
        ax.tick_params(axis="y", colors="#DDDDDD")
        for spine in ax.spines.values():
            spine.set_color("#666666")

    # figure-level başlık YOK – Streamlit üst başlığı veriyor
    fig.tight_layout()
    return fig



