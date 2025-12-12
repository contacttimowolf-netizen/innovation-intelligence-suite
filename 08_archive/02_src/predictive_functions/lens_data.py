import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("LENS_API_TOKEN")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


def test_lens_access():
    """Hem Scholarly hem Patent API için erişim testi yapar."""
    
    endpoints = {
        "scholarly": "https://api.lens.org/scholarly/search",
        "patent": "https://api.lens.org/patent/search"
    }

    body = {"query": {"match_all": {}}}

    results = {}
    for name, url in endpoints.items():
        try:
            r = requests.post(url, json=body, headers=headers)
            results[name] = r.status_code
        except Exception as e:
            results[name] = str(e)

    return results


def lens_search(endpoint: str, query_body: dict, size: int = 10):
    """Gerçek küçük veri çekme — güvenli"""
    
    url = f"https://api.lens.org/{endpoint}/search"
    query_body["size"] = size
    r = requests.post(url, json=query_body, headers=headers)

    if r.status_code in (200, 201):
        return r.json()
    else:
        raise ValueError(f"API Error {r.status_code}: {r.text}")

import pandas as pd

def lens_results_to_df(res):
    import pandas as pd

    rows = []

    for rec in res.get("data", []):
        # ---- basit alanlar ----
        lens_id        = rec.get("lens_id")
        jurisdiction   = rec.get("jurisdiction")
        doc_number     = rec.get("doc_number")
        kind           = rec.get("kind")
        date_published = rec.get("date_published")

        # ---- application number ----
        app_ref = rec.get("application_reference") or {}
        application_number = app_ref.get("doc_number")

        # ---- invention title ----
        inv_title = None
        inv = rec.get("invention_title")
        if isinstance(inv, list) and inv:
            inv_title = inv[0].get("text")

        # ---- applicants (listeyi stringe çevir) ----
        applicants = None
        parties = rec.get("parties") or {}
        appl_list = []
        for a in parties.get("applicants", []):
            name = None
            if isinstance(a, dict):
                # extracted_name > value varsa onu al
                name = (a.get("extracted_name") or {}).get("value") or a.get("name")
            if name:
                appl_list.append(name)
        if appl_list:
            applicants = "; ".join(appl_list)

        # ---- patent status ----
        patent_status = None
        ps = rec.get("patent_status")
        if isinstance(ps, dict):
            patent_status = ps.get("status")

        # ---- priority date ----
        priority_date = None
        ec = rec.get("earliest_claim") or {}
        priority_date = ec.get("date")

        # ---- abstract (metin) ----
        abstract = None
        abs_ = rec.get("abstract")
        if isinstance(abs_, list) and abs_:
            abstract = abs_[0].get("text")
        elif isinstance(abs_, dict):
            abstract = abs_.get("text")

        rows.append(
            {
                "lens_id": lens_id,
                "jurisdiction": jurisdiction,
                "doc_number": doc_number,
                "kind": kind,
                "date_published": date_published,
                "application_number": application_number,
                "invention_title": inv_title,
                "applicants": applicants,
                "patent_status": patent_status,
                "priority_date": priority_date,
                "abstract": abstract,
            }
        )

    return pd.DataFrame(rows)