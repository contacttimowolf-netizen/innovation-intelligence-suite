# functions/paper_data.py

import os
import re
from typing import Any, List, Tuple

import pandas as pd
from tqdm.auto import tqdm  # <-- progress bar

from openalex_data import (
    fetch_openalex_works,
    abstract_from_inverted_index,
    DEFAULT_MAILTO,
)

# ============================
# 1) AI / IoT regex (GDELT ile aynı)
# ============================

PAPER_AI_REGEX = re.compile(
    r"(artificial intelligence|machine learning|deep learning|neural network|"
    r"autonomous|robotics|big data|internet of things|internet-of-things|"
    r"intelligent systems|data[- ]driven|large[- ]scale dataset|smart objects|"
    r"\bai\b|\bai[- ]|\biot\b)",
    flags=re.IGNORECASE,
)

# ============================
# 2) Yardımcılar
# ============================

def _reconstruct_abstract(row: pd.Series) -> str:
    """OpenAlex abstract_inverted_index → normal text."""
    inv = row.get("abstract_inverted_index")
    if isinstance(inv, dict):
        txt = abstract_from_inverted_index(inv)
        return txt or ""
    # bazen direkt string olarak gelebilir
    txt = row.get("abstract")
    return txt or ""


def _main_tech_field(concepts: Any) -> str:
    """concepts listesinden en güçlü kavram adını seç."""
    if not isinstance(concepts, list) or not concepts:
        return ""
    concepts_sorted = sorted(
        concepts,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    for c in concepts_sorted:
        name = c.get("display_name")
        if name:
            return str(name)
    return ""

# ============================
# 3) Ham OpenAlex DF → (df_master, df_full)
# ============================

def clean_openalex_to_master(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_raw  : fetch_openalex_works çıktısı (ham OpenAlex verisi)

    df_master : Proje2 ana korpus satırları (sadece gerekli kolonlar)
        - doc_id, title, text(=abstract), date, year, month,
          source_type='paper', tech_field, trl_true

    df_full : filtrelenmiş ama tüm ek kolonları taşıyan tablo
              (embedding / ek analiz için kullanacağız)
    """
    df = df_raw.copy()

    # Başlık, tarih, abstract
    df["title"] = df.get("display_name", "")
    df["abstract_text"] = df.apply(_reconstruct_abstract, axis=1)

    # Tarih & yıl/ay
    df["date"] = pd.to_datetime(df.get("publication_date"), errors="coerce")
    df["year"] = df.get("publication_year")
    df["month"] = df["date"].dt.month

    # Tech field (en baskın concept)
    df["tech_field"] = df.get("concepts", []).apply(_main_tech_field)

    # AI / IoT filtresi: title + abstract üzerinden
    full_text = (
        df["title"].fillna("").astype(str) + " " +
        df["abstract_text"].fillna("").astype(str)
    )
    mask = full_text.apply(lambda t: bool(PAPER_AI_REGEX.search(t)))
    df_ai = df[mask].copy()

    if df_ai.empty:
        return pd.DataFrame(), pd.DataFrame()

    # df_master: Proje2 korpus formatı
    df_master = pd.DataFrame({
        "doc_id": df_ai["id"],
        "title": df_ai["title"],
        "text": df_ai["abstract_text"],   # SADECE abstract
        "date": df_ai["date"],
        "year": df_ai["year"],
        "month": df_ai["month"],
        "source_type": "paper",
        "tech_field": df_ai["tech_field"],
        "trl_true": pd.NA,                # henüz label yok
    })

    # df_full: tüm ek kolonlar korunmuş (embedding / network için)
    df_full = df_ai.reset_index(drop=True)

    return df_master.reset_index(drop=True), df_full

# ============================
# 4) 2015–2025 korpusu kur ve kaydet
# ============================

def build_paper_ai_corpus(
    start_year: int = 2015,
    end_year: int = 2025,
    per_year: int = 2000,
    out_master_path: str = "paper_ai_master_2015_2025.csv",
    out_full_path: str = "paper_ai_full_2015_2025.parquet",
    mailto: str = DEFAULT_MAILTO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - start_year–end_year arası, her yıl için:
        language:en & has_abstract:true filtresiyle OpenAlex'ten raw veri çeker
    - AI/IoT regex'i ile filtreler
    - df_master ve df_full üretir, disk'e kaydeder
    """
    all_raw = []

    years = range(start_year, end_year + 1)
    for year in tqdm(years, desc="OpenAlex years"):
        filter_str = f"publication_year:{year},language:en,has_abstract:true"
        df_y = fetch_openalex_works(
            filter_str=filter_str,
            max_records=per_year,
            mailto=mailto,
            extra_params={"sort": "publication_date:desc"},
        )
        print(f"{year} → {len(df_y)} raw works")
        if not df_y.empty:
            all_raw.append(df_y)

    if not all_raw:
        print("Hiç veri gelmedi.")
        return pd.DataFrame(), pd.DataFrame()

    df_raw = pd.concat(all_raw, ignore_index=True)
    print(f"Toplam raw works: {len(df_raw)}")

    df_master, df_full = clean_openalex_to_master(df_raw)
    print(f"AI/IoT filtre sonrası: {len(df_master)} paper")

    # klasörleri oluştur
    if out_master_path:
        os.makedirs(os.path.dirname(out_master_path) or ".", exist_ok=True)
        df_master.to_csv(out_master_path, index=False)
        print(f"df_master ({len(df_master)} satır) → {out_master_path}")

    if out_full_path:
        os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)
        df_full.to_parquet(out_full_path, index=False)
        print(f"df_full   ({len(df_full)} satır) → {out_full_path}")

    return df_master, df_full

def quick_openalex_test_with_files():
    """
    Küçük bir demo: 2023 yılından yalnızca 5 kayıt çeker,
    temizler, mini CSV ve mini Parquet olarak kaydeder.
    """
    test_year = 2023
    test_limit = 5

    print(f"[TEST] {test_year} yılı için {test_limit} kayıt çekiliyor...")

    filter_str = f"publication_year:{test_year},language:en,has_abstract:true"

    df_raw = fetch_openalex_works(
        filter_str=filter_str,
        max_records=test_limit,
        mailto=DEFAULT_MAILTO,
        extra_params={"sort": "publication_date:desc"},
    )

    print(f"[TEST] Ham kayıt sayısı: {len(df_raw)}")

    df_master, df_full = clean_openalex_to_master(df_raw)

    print(f"[TEST] Filtrelenmiş kayıt: {len(df_master)}")

    # Kaydet
    os.makedirs("paper_test_outputs", exist_ok=True)

    mini_master_path = "paper_test_outputs/mini_paper_master.csv"
    mini_full_path = "paper_test_outputs/mini_paper_full.parquet"

    df_master.to_csv(mini_master_path, index=False)
    df_full.to_parquet(mini_full_path, index=False)

    print(f"[TEST] mini_master CSV oluşturuldu → {mini_master_path}")
    print(f"[TEST] mini_full Parquet oluşturuldu → {mini_full_path}")
    print("[TEST] İlk 3 başlık:")
    print(df_master['title'].head(3).to_string(index=False))

# ============================
# 5) Script olarak çalıştırma
# ============================

if __name__ == "__main__":
    
    #quick_openalex_test_with_files()
    build_paper_ai_corpus()
