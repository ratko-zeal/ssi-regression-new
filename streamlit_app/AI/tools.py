"""
Function tools exposed to the model.
- get_growth_scores
- get_current_view_state
- get_indicator_scores
- prioritize_focus
- file_search (stub)
- web_search (stub, gated by config flag)
"""
from typing import Dict, List, Optional
import json
import pandas as pd
from pathlib import Path
from .config import BUILD_DIR, AS_OF_DATE, VERSION, ENABLE_WEB_SEARCH
from .logger import log_turn
from .knowledge import search_knowledge

BUILD = Path(BUILD_DIR)
FINAL_PATH = BUILD / "final_scores.csv"
DISPLAY_PATH = BUILD / "display_domain_scores.csv"
INDICATOR_PATH = BUILD / "indicator_scores.csv"
WEIGHTS_PATH = BUILD / "indicator_weights.csv"
META_PATH = BUILD / "meta.json"

# Simple in-memory session store
SESSION_STATE: Dict[str, Dict] = {}

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required build file: {path}")
    return pd.read_csv(path)

def get_growth_scores(country: str, year: Optional[int] = None, domains: Optional[List[str]] = None) -> Dict:
    """Return final score and display domain scores for a country."""
    final_tbl = _load_csv(FINAL_PATH)
    disp_tbl = _load_csv(DISPLAY_PATH)
    # Lookup
    row = final_tbl.loc[final_tbl["COUNTRY"].str.lower() == country.lower()]
    if row.empty:
        return {"error": {"code":"NOT_FOUND","message": f"Country '{country}' not found"}}
    final_score = float(row["FINAL_SCORE"].iloc[0])

    dom = disp_tbl.loc[disp_tbl["COUNTRY"].str.lower() == country.lower()]
    if domains:
        dom = dom[dom["DOMAIN"].isin(domains)]
    display = {r["DOMAIN"]: float(r["DISPLAY_SCORE"]) for _, r in dom.iterrows()}

    # Meta
    meta = json.loads(Path(META_PATH).read_text()) if META_PATH.exists() else {"as_of_date": AS_OF_DATE, "version": VERSION or None}

    return {
        "country": country,
        "as_of_date": meta.get("as_of_date"),
        "version": meta.get("version"),
        "final_score": final_score,
        "display_domain_scores": display
    }

def get_current_view_state(session_id: str) -> Dict:
    """Return current UI filters for the session."""
    return SESSION_STATE.get(session_id, {"country": None, "domains": None, "year": None})

def set_current_view_state(session_id: str, country: Optional[str], domains: Optional[List[str]], year: Optional[int]=None):
    SESSION_STATE[session_id] = {"country": country, "domains": domains, "year": year}

def get_indicator_scores(country: str, indicators: Optional[List[str]] = None) -> Dict:
    """Return normalized indicator scores for the country (numbers are OK to show)."""
    ind_tbl = _load_csv(INDICATOR_PATH)
    row = ind_tbl.loc[ind_tbl["COUNTRY"].str.lower() == country.lower()]
    if row.empty:
        return {"error": {"code":"NOT_FOUND","message": f"Country '{country}' not found"}}
    row = row.iloc[0].to_dict()
    row.pop("COUNTRY", None)
    if indicators:
        row = {k:v for k,v in row.items() if k in indicators}
    # Ensure floats
    scores = {k: float(v) for k,v in row.items()}
    return {"country": country, "indicator_scores": scores}

def prioritize_focus(country: str, indicators: List[str]) -> Dict:
    """
    Rank indicators to focus on (no raw weights revealed).
    Impact score = WEIGHT * (100 - current_score).
    Returns ordered list with tiers and rationales.
    """
    ind_tbl = _load_csv(INDICATOR_PATH)
    w_tbl = _load_csv(WEIGHTS_PATH)
    row = ind_tbl.loc[ind_tbl["COUNTRY"].str.lower() == country.lower()]
    if row.empty:
        return {"error": {"code":"NOT_FOUND","message": f"Country '{country}' not found"}}
    row = row.iloc[0].to_dict()
    row.pop("COUNTRY", None)

    weights = dict(zip(w_tbl["INDICATOR"], w_tbl["WEIGHT"]))

    records = []
    for ind in indicators:
        if ind not in row:
            continue
        score = float(row[ind])
        weight = float(weights.get(ind, 0.0))
        impact = weight * (100.0 - score)
        records.append({"indicator": ind, "score": score, "weight": weight, "impact": impact})

    if not records:
        return {"error":{"code":"NO_DATA","message":"No matching indicators with scores/weights."}}

    # Sort by impact desc, tie-break by weight desc then lower score
    records.sort(key=lambda x: (-x["impact"], -x["weight"], x["score"]))

    # Tiering by impact quantiles
    impacts = [r["impact"] for r in records]
    if len(impacts) >= 3:
        q66 = sorted(impacts)[int(len(impacts)*2/3)]
        q33 = sorted(impacts)[int(len(impacts)*1/3)]
    else:
        q66 = q33 = impacts[len(impacts)//2]

    def tier(imp):
        if imp >= q66: return "High"
        if imp <= q33: return "Low"
        return "Medium"

    out = []
    for r in records:
        t = tier(r["impact"])
        rationale = []
        if r["weight"] > 0: rationale.append("high leverage")
        if r["score"] < 60: rationale.append("large improvement gap")
        if not rationale: rationale.append("meaningful but moderate influence")
        out.append({"indicator": r["indicator"], "tier": t, "rationale": ", ".join(rationale)})

    return {"country": country, "recommendations": out}

# --- Stubs ---
def file_search(query: str) -> Dict:
    results = search_knowledge(query)
    if not results:
        return {"citations": [], "snippets": []}
    snippets = []
    citations = []
    for res in results:
        snippets.append(res["snippet"])
        citations.append(res["path"])
    return {"citations": citations, "snippets": snippets}

def web_search(query: str) -> Dict:
    if not ENABLE_WEB_SEARCH:
        return {"error": {"code":"DISABLED","message":"Web search is disabled by configuration."}}
    # Placeholder for real web search integration.
    return {"citations": [], "snippets": []}
