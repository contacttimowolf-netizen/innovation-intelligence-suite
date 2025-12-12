import os
import datetime as dt
from google.cloud import bigquery
from google.oauth2 import service_account

# ===========================
# Google Cloud Ayarları
# ===========================

SERVICE_ACCOUNT_PATH = r"serviceaccount"
PROJECT_ID = "projectid"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_PATH
)

client = bigquery.Client(
    credentials=credentials,
    project=PROJECT_ID,
)

# ===========================
# Çıktı klasörü
# ===========================

OUT_DIR = "gdelt_ai_monthly"
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# Aylık veri çekme fonksiyonu
# ===========================


def run_month(year: int, month: int, per_month: int = 100) -> None:
    """
    GDELT GKG'den belirli yıl/ay için en fazla 'per_month' adet
    AI / data / IoT temalı haber çeker ve CSV olarak kaydeder.
    """

    # ---- tarih aralığı ----
    y = str(year)
    m = f"{month:02d}"

    start_date = f"{y}-{m}-01"
    first_next = dt.date(year, month, 1) + dt.timedelta(days=32)
    last_day = (first_next.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    # ---- BigQuery sorgusu ----
    query = f"""
    WITH base AS (
      SELECT
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS ts,
        DocumentIdentifier AS url,
        V2Themes AS themes,
        V2Tone AS tone
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE
        -- 1) Partition filtresi (kritik, maliyeti düşürür)
        DATE(_PARTITIONTIME) BETWEEN '{start_date}' AND '{last_day}'

        -- 2) AI / data / IoT temaları
        AND (
          LOWER(V2Themes) LIKE '%artificial_intelligence%' OR
          LOWER(V2Themes) LIKE '%machine_learning%' OR
          LOWER(V2Themes) LIKE '%deep_learning%' OR
          LOWER(V2Themes) LIKE '%neural_network%' OR
          LOWER(V2Themes) LIKE '%autonomous%' OR
          LOWER(V2Themes) LIKE '%robotics%' OR
          LOWER(V2Themes) LIKE '%big_data%' OR
          LOWER(V2Themes) LIKE '%internet_of_things%' OR
          LOWER(V2Themes) LIKE '%intelligent_systems%' OR
          LOWER(V2Themes) LIKE '%data-driven%' OR
          LOWER(V2Themes) LIKE '%large-scale-dataset%' OR
          LOWER(V2Themes) LIKE '%smart_objects%' OR

          -- AI ve IoT kısaltmaları (kelime bazlı, 'brain' gibi kelimeleri alma)
          REGEXP_CONTAINS(LOWER(V2Themes), r'\\bai\\b') OR
          REGEXP_CONTAINS(LOWER(V2Themes), r'\bai[- ]') OR
          REGEXP_CONTAINS(LOWER(V2Themes), r'\\biot\\b')
        )

        -- 3) Ağırlıklı İngilizce ~ .com / .co.uk / .org domainleri
        AND REGEXP_CONTAINS(DocumentIdentifier, r'https?://[^/]+\\.(com|co\\.uk|org)/')
    ),

    monthly AS (
      SELECT
        ts AS date,
        EXTRACT(YEAR FROM ts) AS year,
        EXTRACT(MONTH FROM ts) AS month,
        url,
        themes,
        tone
      FROM base
    )

    SELECT *
    FROM (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY year, month
          ORDER BY date DESC
        ) AS rn
      FROM monthly
    )
    WHERE rn <= {per_month}
    ORDER BY year, month, date DESC
    """

    df = client.query(query).to_dataframe()
    outfile = f"{OUT_DIR}/gdelt_ai_{y}-{m}.csv"
    df.to_csv(outfile, index=False)

    print(f"{y}-{m} → {len(df)} rows  | saved → {outfile}")


# ===========================
# Full dönem çek (2015-01 → 2025-11)
# ===========================

print("\n=== GDELT AYLIK NEWS SCRAPE BAŞLADI ===\n")

for year in range(2015, 2026):  # 2015 dahil, 2025 dahil
    for month in range(1, 13):
        # 2025 Kasım'da dur
        if year == 2025 and month == 12:
            print("\n=== 2025 Kasım → SON AY TAMAMLANDI ===")
            break

        run_month(year, month)

print("\n=== TÜM AYLIK VERİLER TAMAMLANDI ===\n")
