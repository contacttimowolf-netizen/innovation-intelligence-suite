import os
import re
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime as dt
from typing import Any, List
import ast

# =========================
# BigQuery client
# =========================
SERVICE_ACCOUNT_PATH = r"serviceaccount"
PROJECT_ID = "projectid"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_PATH
)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

def _is_na_scalar(x: Any) -> bool:
    """
    Liste/dict değilse ve pandas açısından NaN ise True.
    (Liste/dict için kesinlikle True dönmez → ValueError engellenir)
    """
    if isinstance(x, (list, dict)):
        return False
    try:
        val = pd.isna(x)
    except Exception:
        return False
    if isinstance(val, bool):
        return val
    return False

def extract_text_only(cell: Any) -> str:
    """
    Localized format:
      [{'text': '...', 'language': 'en', 'truncated': False}]
    veya bu yapının string hali
    -> sadece 'text' döner.
    """
    if cell is None:
        return ""

    # 1) Bazı durumlarda string repr gelebilir → parse etmeyi dene
    if isinstance(cell, str) and cell.startswith("[") and "text" in cell:
        try:
            cell = ast.literal_eval(cell)
        except Exception:
            # parse edemezsek olduğu gibi bırak
            return cell

    # 2) Liste ise
    if isinstance(cell, list) and len(cell) > 0:
        item = cell[0]
        if isinstance(item, dict) and "text" in item:
            return str(item["text"])

    # 3) Tek dict ise
    if isinstance(cell, dict) and "text" in cell:
        return str(cell["text"])

    # 4) Son çare: stringe çevir
    return str(cell)


def extract_cpc_codes(cell: Any) -> List[str]:
    """
    CPC / IPC formatı:
      [{'code': 'A61P31/14', 'inventive': True, ...}]
    -> sadece ['A61P31/14', ...] döner.
    """
    codes: List[str] = []

    if isinstance(cell, list):
        for item in cell:
            if isinstance(item, dict) and "code" in item:
                codes.append(str(item["code"]))
    elif isinstance(cell, dict) and "code" in cell:
        codes.append(str(cell["code"]))
    elif isinstance(cell, str):
        # nadiren tek string kod olarak gelebilir
        codes.append(cell)

    return codes


def _names_to_str(cell: Any) -> str:
    """
    assignee_harmonized / inventor_harmonized
    -> 'NAME1;NAME2;…'
    """
    if cell is None or _is_na_scalar(cell):
        return ""

    names: List[str] = []

    if isinstance(cell, list):
        for item in cell:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(str(name))
            else:
                names.append(str(item))
    elif isinstance(cell, dict):
        name = cell.get("name")
        if name:
            names.append(str(name))
    else:
        names.append(str(cell))

    return ";".join(names)


def _count_list(cell: Any) -> int:
    """
    citation gibi liste kolonları için eleman sayısı.
    """
    if isinstance(cell, list):
        return len(cell)
    return 0

def clean_patent_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # title / abstract
    df["title"] = df.get("title_localized", "").apply(extract_text_only)
    df["abstract"] = df.get("abstract_localized", "").apply(extract_text_only)

    # citation sayısı
    df["citation_count"] = df.get("citation", []).apply(_count_list)

    if "assignee_harmonized" in df.columns:
        df["applicant"] = df["assignee_harmonized"].apply(_names_to_str)
    else:
        df["applicant"] = ""

# description
    if "description_localized" in df.columns:
        df["description"] = df["description_localized"].apply(extract_text_only)
    else:
        df["description"] = ""

# claims
    if "claims_localized" in df.columns:
        df["claims"] = df["claims_localized"].apply(extract_text_only)
    else:
        df["claims"] = ""
    # gereksiz nested kolonları at
    drop_cols = [
        "ipc",
        "title_localized",
        "abstract_localized",        
        "assignee_harmonized",
        "inventor_harmonized",
        "citation",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
# =========================
# AI / IoT keyword regex (Python tarafında)
# =========================
PATENT_AI_REGEX = re.compile(
    r"(artificial intelligence|machine learning|deep learning|neural network|"
    r"robotics|big data|internet of things|internet-of-things|"
    r"intelligent systems|data[- ]driven|large[- ]scale dataset|smart objects|"
    r"\bai\b|\bai[-/]\b|\biot\b|\biot[-/]\b)",
    flags=re.IGNORECASE,
)


def fetch_month_patents_ai(year: int, month: int, per_month: int = 100) -> pd.DataFrame:
    """
    1 ay için patent-public-data'dan AI/IoT alakalı patentleri çeker,
    temizler, df_master formatına çevirir ve en fazla `per_month` satır döner.
    """

    y = str(year)
    m = f"{month:02d}"

    # ---- tarih aralığı ----
    start_dt = dt.date(year, month, 1)
    first_next = start_dt + dt.timedelta(days=32)
    last_dt = first_next.replace(day=1) - dt.timedelta(days=1)

    # INT64 format (YYYYMMDD) → publication_date ile aynı format
    start_key = int(start_dt.strftime("%Y%m%d"))
    last_key = int(last_dt.strftime("%Y%m%d"))

    # ---- BigQuery: o ayki patentler (biraz fazla çek → Python'da filtrele) ----
    raw_limit = per_month * 10  # buffer

    query = f"""
    SELECT
        publication_number,
        -- INT64 → DATE kolonu üretelim
        PARSE_DATE('%Y%m%d', CAST(publication_date AS STRING)) AS pub_date,
        title_localized,
        abstract_localized,
        cpc,
        ipc,
        assignee_harmonized,
        citation
    FROM `patents-public-data.patents.publications`
    WHERE
        publication_date BETWEEN {start_key} AND {last_key}
    ORDER BY ARRAY_LENGTH(citation) DESC, publication_date DESC
    LIMIT {raw_limit}
    """

    df_raw = client.query(query).to_dataframe()

    # ---- senin cleaner ile sadeleştir ----
    df_clean = clean_patent_dataframe(df_raw)

    # ---- keyword filtresi (title + abstract) ----
    full_text = (
        df_clean["title"].fillna("").astype(str)
        + " "
        + df_clean["abstract"].fillna("").astype(str)
    )

    mask = full_text.apply(lambda t: bool(PATENT_AI_REGEX.search(t)))
    df_filt = df_clean[mask].copy()

    if df_filt.empty:
        print(f"{y}-{m} → 0 patent (filtre sonrası)")
        return df_filt

    # ---- df_master formatı ----
    df_master = pd.DataFrame()
    df_master["doc_id"] = df_filt["publication_number"]
    df_master["title"] = df_filt["title"]
    df_master["text"] = df_filt["abstract"]           # abstract = text
    df_master["date"] = pd.to_datetime(df_filt["pub_date"])
    df_master["year"] = df_master["date"].dt.year
    df_master["month"] = df_master["date"].dt.month
    df_master["source_type"] = "patent"
    df_master["tech_field"] = df_filt["cpc"].astype(str)
    df_master["trl_true"] = pd.NA                      # henüz label yok → NaN

    # en ilgili ilk N: citation_count (yüksek) + yeni tarih
    if "citation_count" in df_filt.columns:
        df_master = df_master.join(df_filt["citation_count"])
        df_master = df_master.sort_values(
            ["citation_count", "date"], ascending=[False, False]
        )
    else:
        df_master = df_master.sort_values("date", ascending=False)

    df_master = df_master.head(per_month).reset_index(drop=True)

    # ---- csv kaydet ----
    out_dir = "patent_ai_monthly"
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"patent_ai_{y}-{m}.csv")
    df_master.to_csv(outfile, index=False)

    print(f"{y}-{m} → {len(df_master)} rows | saved → {outfile}")
    return df_master

if __name__ == "__main__":
    print("\n=== PATENT AYLIK SCRAPE BAŞLADI ===\n")

    for year in range(2019, 2026):
        for month in range(1, 13):

            # SON DURMA NOKTASI: 2025 Kasım
            if year == 2025 and month == 12:
                print("\n=== 2025 Kasım → SON AY TAMAMLANDI ===")
                break

            fetch_month_patents_ai(year, month)

    print("\n=== TÜM PATENT VERİLERİ TAMAMLANDI ===\n")
