"""Build AI-facing tables consumed by the Responses tool layer."""
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from .config import DATA_DIR, BUILD_DIR, AS_OF_DATE, VERSION

BUILD_DIR_PATH = Path(BUILD_DIR)
BUILD_DIR_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = {
    "final_scores": BUILD_DIR_PATH / "final_scores.csv",
    "display_domain_scores": BUILD_DIR_PATH / "display_domain_scores.csv",
    "indicator_scores": BUILD_DIR_PATH / "indicator_scores.csv",
    "indicator_weights": BUILD_DIR_PATH / "indicator_weights.csv",
    "meta": BUILD_DIR_PATH / "meta.json",
}

def build() -> Dict[str, Any]:
    data_dir = Path(DATA_DIR)
    # Required files
    final_csv = data_dir / "final_scores.csv"
    indicator_csv = data_dir / "indicator_scores.csv"
    domains_csv = data_dir / "domains.csv"
    weights_csv = data_dir / "indicator_impact_summary.csv"  # blended weights
    # Optional
    # domain_weights_csv = data_dir / "domain_impact_summary.csv"

    # Load
    final_df = pd.read_csv(final_csv)
    final_df.columns = [c.upper() for c in final_df.columns]
    indicator_df = pd.read_csv(indicator_csv)
    indicator_df.columns = [c.upper() for c in indicator_df.columns]
    domains_map = pd.read_csv(domains_csv)
    domains_map.columns = [c.upper() for c in domains_map.columns]

    weights_df = pd.read_csv(weights_csv)
    weights_df.columns = [c.upper() for c in weights_df.columns]
    # Expect columns: INDICATOR, BLENDED_WEIGHT (allow variations)
    weight_col = None
    for cand in ["BLENDED_WEIGHT","WEIGHT","IMPACT_WEIGHT","WEIGHTS","BLENDED INDICATOR IMPACT (%)","BLENDED_INDICATOR_IMPACT_(%)"]:
        if cand in weights_df.columns:
            weight_col = cand
            break
    if weight_col is None:
        raise ValueError("Weights file must include a numeric blended weight column (e.g., BLENDED_WEIGHT).")

    # --- Final scores table ---
    if "COUNTRY" not in final_df.columns:
        raise ValueError("final_scores.csv must include COUNTRY column.")
    # Find final score column
    score_col = None
    for cand in ["FINAL_SCORE_0_100","FINAL_BLENDED_0_100","FINAL_LOG_0_100","FINAL_PERCAP_0_100","FINAL_DOMAINAVG_0_100"]:
        if cand in final_df.columns:
            score_col = cand
            break
    if score_col is None:
        raise ValueError("No final score column found in final_scores.csv (expect Final_Score_0_100 or similar).")
    final_tbl = final_df[["COUNTRY", score_col]].rename(columns={score_col:"FINAL_SCORE"})
    final_tbl.to_csv(OUTPUT_FILES["final_scores"].with_suffix(".csv"), index=False)

    # --- Display domain scores (rank-adjusted) ---
    domain_cols = [c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]
    if not domain_cols:
        raise ValueError("No DOMAIN_SCORE__* columns found in final_scores.csv for domain display scores.")
    # Compute rank-percentiles per domain across countries
    disp_rows = []
    for dcol in domain_cols:
        col_series = final_df[dcol].astype(float)
        # Higher raw value => higher percentile; convert to 0-100 "display" score
        pct = col_series.rank(method="min", ascending=True, pct=True) * 100.0
        domain_name = dcol.replace("DOMAIN_SCORE__", "")
        for country, val in zip(final_df["COUNTRY"], pct):
            disp_rows.append({"COUNTRY": country, "DOMAIN": domain_name, "DISPLAY_SCORE": float(val)})
    display_tbl = pd.DataFrame(disp_rows)
    display_tbl.to_csv(OUTPUT_FILES["display_domain_scores"].with_suffix(".csv"), index=False)

    # --- Indicator scores ---
    if "COUNTRY" not in indicator_df.columns:
        raise ValueError("indicator_scores.csv must include COUNTRY column.")
    # Wide format: COUNTRY + indicator columns
    indicator_tbl = indicator_df.copy()
    indicator_tbl.to_csv(OUTPUT_FILES["indicator_scores"].with_suffix(".csv"), index=False)

    # --- Indicator weights ---
    # Expect INDICATOR names to match indicator_tbl columns (except COUNTRY)
    if "INDICATOR" not in weights_df.columns:
        raise ValueError("indicator_impact_summary.csv must include INDICATOR column.")
    weights_df = weights_df[["INDICATOR", weight_col]].rename(columns={weight_col:"WEIGHT"})
    weights_df.to_csv(OUTPUT_FILES["indicator_weights"].with_suffix(".csv"), index=False)

    # --- Metadata ---
    meta = {"as_of_date": AS_OF_DATE, "version": VERSION or None}
    OUTPUT_FILES["meta"].write_text(pd.Series(meta).to_json(), encoding="utf-8")

    return {"paths": {k: str(v) for k, v in OUTPUT_FILES.items()}, "meta": meta}

def validate():
    import json
    out = {k: p.exists() for k,p in OUTPUT_FILES.items()}
    # Quick integrity checks
    errors = []
    if not out["final_scores"]:
        errors.append("final_scores table missing")
    if not out["display_domain_scores"]:
        errors.append("display_domain_scores table missing")
    if not out["indicator_scores"]:
        errors.append("indicator_scores table missing")
    if not out["indicator_weights"]:
        errors.append("indicator_weights table missing")
    if errors:
        raise AssertionError("; ".join(errors))
    return out

if __name__ == "__main__":
    info = build()
    print(info)
    print(validate())
