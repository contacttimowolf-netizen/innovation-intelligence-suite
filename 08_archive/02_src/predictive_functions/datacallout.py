
import os, re, time, json, requests, pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_table(df: pd.DataFrame, outdir: str, name: str):
    ensure_dir(outdir)
    base = os.path.join(outdir, name)
    df.to_csv(base + ".csv", index=False, encoding="utf-8")
    try:
        df.to_parquet(base + ".parquet", index=False)
    except Exception:
        pass

def _make_session():
    s = requests.Session()
    # HTTPAdapter + urllib3 Retry (429/5xx için)
    retry_cfg = Retry(
        total=3, connect=3, read=3, backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry_cfg, pool_connections=4, pool_maxsize=4))
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Encoding": "identity"  # gzip kapat: bazı ağlarda chunk takılıyor
    })
    s.trust_env = False  # sistem proxy değişkenlerini yok say
    return s

@retry(wait=wait_exponential(min=1, max=20, multiplier=0.5), stop=stop_after_attempt(3),retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)))
       

def _get(session, url, *, params=None, timeout=(5, 60)):
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r



def fetch_newsapi(query='"technology readiness"', language="en", page_size=20) -> pd.DataFrame:
    key = os.getenv("NEWSAPI_KEY", "")
    if not key:
        print("[INFO] NEWSAPI_KEY yok, atlanıyor.")
        return pd.DataFrame()

    s = _make_session()  # aynı session + adapter ayarları
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": key,
        # opsiyonel: sonuçları daha temiz almak için
        # "sortBy": "relevancy"
    }
    data = _get(s, url, params=params, timeout=(5, 30)).json()  # <-- session ilk argüman
    rows = [{
        "source": "newsapi",
        "title": a.get("title"),
        "description": a.get("description"),
        "content": a.get("content"),
        "url": a.get("url"),
        "publishedAt": a.get("publishedAt"),
        "source_name": (a.get("source") or {}).get("name"),
    } for a in data.get("articles", [])]

    return pd.DataFrame(rows)

import re
import time

def build_techport_trl_dataset(
    csv_in: str,
    csv_out: str,
    url_col: str = "Project API URL",
    max_rows: int | None = None,
    sleep: float = 0.03,
    retries: int = 2,
    timeout: int = 30,
):
    """
    TRL + taxonomy + description + funding + activity + orgType + start/end date
    Model-ready sade dataset.
    """
    import os, time, re, requests
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

    # 1) CSV yükle
    df = pd.read_csv(csv_in)

    def _extract_url(x: str):
        if pd.isna(x):
            return None
        m = re.search(r"(https://techport\.nasa\.gov/api/projects/\d+)", str(x))
        return m.group(1) if m else None

    df["project_api_url"] = df[url_col].map(_extract_url)

    urls = df["project_api_url"].dropna().tolist()
    if max_rows is not None:
        urls = urls[:max_rows]

    print(f"Toplam URL: {len(urls)}")

    # HTTP session
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})

    # ---------------------------------------------------------
    #  TEK PROJE GETİR
    # ---------------------------------------------------------
    def _fetch_one(url: str) -> dict:
        for _ in range(retries):
            try:
                r = s.get(url, timeout=timeout)
                if r.status_code == 200:
                    root = r.json() or {}
                    p = root.get("project", {}) or {}
                    prog = p.get("program", {}) or {}

                    # --- taxonomy (primaryTx, tree, ek tx'ler vs. hepsini topla) ---
                    tx_nodes: list[dict] = []

                    def _collect(node):
                        # dict → direkt ekle
                        if isinstance(node, dict):
                            tx_nodes.append(node)
                        # liste / list-of-lists → içini gez
                        elif isinstance(node, list):
                            for x in node:
                                _collect(x)

                    _collect(p.get("taxonomyNodes"))
                    _collect(p.get("primaryTaxonomyNodes"))
                    _collect(p.get("additionalTaxonomyNodes"))
                    _collect(p.get("allNodesTree"))

                    _collect(p.get("primaryTx"))
                    _collect(p.get("primaryTxTree"))
                    _collect(p.get("additionalTxs"))
                    _collect(p.get("additionalTxTree"))

                    tx_codes = [n.get("code", "") for n in tx_nodes if isinstance(n, dict)]
                    tx_titles = [n.get("title", "") for n in tx_nodes if isinstance(n, dict)]

                    primary_tx_code = tx_codes[0] if tx_codes else None
                    primary_tx_title = tx_titles[0] if tx_titles else None
                    taxonomy_list = "; ".join(
                        f"{c}: {t}".strip(": ")
                        for c, t in zip(tx_codes, tx_titles)
                        if c or t
                    )

                    # --- metinsel alanlar ---
                    benefits_text = p.get("benefits") or p.get("benefitsText")

                    objectives_text = (
                        p.get("technologyObjectives")
                        or p.get("objectives")
                        or p.get("objectivesText")
                        or p.get("description")  # hiçbiri yoksa description'a düş
                    )

                    # --- funding & flags ---
                    raw_total_funding = (
                        p.get("totalFunding")
                        or p.get("totalProjectCost")
                        or p.get("totalProjectedCost")
                    )
                    # veri yoksa NaN yerine string "None"
                    total_funding = (
                        raw_total_funding
                        if raw_total_funding is not None
                        else "None"
                    )

                    is_active = p.get("isActive")
                    if is_active is None:
                        # bazı projelerde program seviyesinde geliyor
                        is_active = prog.get("isActive")

                    # --- organizasyon tipi (tek kategorik alan) ---
                    lead_org = p.get("leadOrganization") or {}
                    lead_org_type = lead_org.get("organizationTypePretty")

                    return {
                        "project_api_url": url,
                        "projectId": p.get("projectId") or p.get("id"),

                        # TRL + tarihler
                        "startTrl": p.get("trlBegin"),
                        "currentTrl": p.get("trlCurrent"),
                        "endTrl": p.get("trlEnd"),
                        "startDate": p.get("startDate"),
                        "endDate": p.get("endDate"),

                        # açıklama
                        "desc_api": p.get("description"),

                        # taxonomy
                        "primaryTxCode": primary_tx_code,
                        "primaryTxTitle": primary_tx_title,
                        "taxonomy_list": taxonomy_list,

                        # metinler
                        "benefits_text": benefits_text,
                        "objectives_text": objectives_text,

                        # sayı + flag
                        "totalFunding": total_funding,
                        "isActive": is_active,

                        # kategorik
                        "lead_org_type": lead_org_type,
                    }

                # 403 / 404 → tekrar denemenin anlamı yok
                if r.status_code in (403, 404):
                    break

            except requests.RequestException:
                time.sleep(1.0)

        # hata durumunda boş satır
        return {
            "project_api_url": url,
            "projectId": None,
            "startTrl": None,
            "currentTrl": None,
            "endTrl": None,
            "startDate": None,
            "endDate": None,
            "desc_api": None,
            "primaryTxCode": None,
            "primaryTxTitle": None,
            "taxonomy_list": None,
            "benefits_text": None,
            "objectives_text": None,
            "totalFunding": None,
            "isActive": None,
            "lead_org_type": None,
        }

    # ---------------------------------------------------------
    #  TOPLU ÇEKİM + YAZ
    # ---------------------------------------------------------
    rows = []
    for url in tqdm(urls, desc="TRL fetch", unit="proj"):
        rows.append(_fetch_one(url))
        time.sleep(sleep)

    trl_df = pd.DataFrame(rows)
    out = df.merge(trl_df, on="project_api_url", how="left")

    out.to_csv(csv_out, index=False)
    print(f"\nYazıldı: {csv_out} (satır: {len(out)})")

    return None

import re
from html import unescape
import pandas as pd  # zaten varsa tekrar import etmen sorun değil

def clean_techport_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    TechPort dataframe'ini sadeleştirir:
      - HTML / &nbsp / (AKA ...) temizliği
      - description/benefits/objectives kolonlarını temizler
      - bazı kolonları drop eder
      - start/endDate kolonlarından yıl/ay türetir
    """

    def _clean_text(s):
        if pd.isna(s):
            return s
        txt = unescape(str(s))
        txt = re.sub(r"<[^>]+>", " ", txt)                 # HTML tag'leri
        txt = re.sub(r"\(AKA[^)]*\)", "", txt,
                     flags=re.IGNORECASE)                  # (AKA …)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    cols_to_clean = [
        "Project Description",
        "desc_api",
        "benefits_text",
        "objectives_text",
    ]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)

    cols_to_drop = [
        "taxonomy_list",
        "Primary Taxonomy",   # CSV'deki kolon adı
        "totalFunding",
        "Project API URL",
        "projectId",
        "isActive",
        "project_api_url",
        "Project Last Updated",
        "objectives_text",
        "Responsible NASA Program",
        "Project URL",
        "TechPort ID"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # tarihleri datetime + yıl/ay
    for col in ["startDate", "endDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "startDate" in df.columns:
        df["start_year"] = df["startDate"].dt.year
        df["start_month"] = df["startDate"].dt.month

    if "endDate" in df.columns:
        df["end_year"] = df["endDate"].dt.year
        df["end_month"] = df["endDate"].dt.month
    
    times_to_drop = ["startDate", "endDate"]
    df = df.drop(columns=[c for c in times_to_drop if c in df.columns])

    order = [
    "TechPort ID",      # 1
    "Project Title",    # 2
    "Project Description",  # 3
    "desc_api",         # 4
    "benefits_text",    # 5
    "primaryTxCode",    # 6
    "primaryTxTitle",   # 7
    "startTrl",         # 8
    "currentTrl",       # 9
    "endTrl",           # 10
    "start_year",       # 11
    "start_month",      # 12
    "end_year",         # 13
    "end_month",        # 14
    "lead_org_type",    # 15
]

    # Sadece gerçekten var olan kolonları al
    order = [c for c in order if c in df.columns]
    df = df[order]

    return df