# build_maturity_index.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error

# ========= CONFIG =========
DOMAINS_PATH = "domains.csv"
SCORES_PATH  = "Input_scores.csv"

OUTDIR_ROOT = "outputs"  # all outputs will live here
TGT_NORM = "# of High Growth Company - Normalized"
TGT_RAW  = "# of High Growth Companies"

# Final blend weights (sum to 1.0)
BLEND_W_LOG      = 0.60
BLEND_W_PERCAP   = 0.30   # ignored if HG_per_100k not available
BLEND_W_DOMAIN   = 0.10

# Optional winsorization (clipping) of indicators before modeling/scoring
ENABLE_WINSORIZE = True
WINSOR_LO        = 0.01    # 1st percentile
WINSOR_HI        = 0.99    # 99th percentile

# ========= IO HELPERS =========
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def out_paths(model: str, target: str) -> Dict[str, str]:
    base = os.path.join(OUTDIR_ROOT, model, target)
    ensure_dir(base)
    return {
        "ind":    os.path.join(base, "indicator_weights.csv"),
        "dom":    os.path.join(base, "domain_weights.csv"),
        "scores": os.path.join(base, "scores.csv"),
    }

# ========= LOAD & PREP =========
def load_data(domains_path: str, scores_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(domains_path):
        raise FileNotFoundError(f"Missing {domains_path}")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing {scores_path}")

    domains = pd.read_csv(domains_path)
    if "Unnamed: 2" in domains.columns:
        domains = domains.drop(columns=["Unnamed: 2"])
    domains["INDICATOR"] = domains["INDICATOR"].astype(str).str.strip()
    domains["DOMAIN"]    = domains["DOMAIN"].astype(str).str.strip()

    scores = pd.read_csv(scores_path)
    return domains, scores

def detect_id_cols(scores: pd.DataFrame) -> List[str]:
    lower_map = {c.lower(): c for c in scores.columns}
    want = ["country", "population"]
    return [lower_map[w] for w in want if w in lower_map]

def build_feature_matrix(domains: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One column per indicator; average duplicate-named columns if present."""
    indicators = [c for c in domains["INDICATOR"].tolist() if c in scores.columns]
    if not indicators:
        raise ValueError("No indicator columns from domains.csv were found in Input_scores.csv.")
    parts = []
    for name in sorted(set(indicators)):
        same = [c for c in scores.columns if c == name]
        if len(same) == 1:
            parts.append(scores[[name]])
        else:
            parts.append(scores[same].mean(axis=1).to_frame(name=name))
    X = pd.concat(parts, axis=1)
    return X

def make_targets(scores: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """Create HG_per_100k and HG_log_raw targets alongside existing columns."""
    # POPULATION numeric
    pop_col = None
    for c in id_cols:
        if c.lower() == "population":
            pop_col = c
            break
    if pop_col is not None:
        pop = (scores[pop_col].astype(str)
               .str.replace(",", "", regex=False)
               .replace({"": np.nan})
               .astype(float))
    else:
        pop = pd.Series(np.nan, index=scores.index, name="POPULATION_num")

    # per 100k (keep for comparison)
    if TGT_RAW in scores.columns:
        hg_per_100k = np.where(
            (scores[TGT_RAW].notna()) & (pop > 0),
            (scores[TGT_RAW] / pop) * 100_000,
            np.nan,
        )
        scores["HG_per_100k"] = hg_per_100k
        # log target for primary modeling
        scores["HG_log_raw"]  = np.log1p(scores[TGT_RAW].astype(float))
    else:
        scores["HG_per_100k"] = np.nan
        scores["HG_log_raw"]  = np.nan

    return scores

# ========= WINSORIZE HELPERS =========
def compute_quantile_bounds(X: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Return a dataframe with two rows: 'lo' and 'hi' quantiles for each column."""
    qlo = X.quantile(lo, interpolation="linear")
    qhi = X.quantile(hi, interpolation="linear")
    return pd.DataFrame({"lo": qlo, "hi": qhi}).T  # index = ['lo','hi']

def apply_bounds_clip(X: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Clip each column using previously computed bounds (columns not in bounds pass through)."""
    Xc = X.copy()
    for c in Xc.columns:
        if c in bounds.columns:
            lo = bounds.loc["lo", c]
            hi = bounds.loc["hi", c]
            if pd.notna(lo) and pd.notna(hi) and hi >= lo:
                Xc[c] = Xc[c].clip(lower=lo, upper=hi)
    return Xc

# ========= MODEL PREP & FIT =========
def preprocess_for_target(X_all: pd.DataFrame, y: pd.Series):
    """
    Median-impute, optional winsorize (fit on train, apply to all),
    drop all-NaN & zero-variance cols, z-score using training stats.
    """
    mask = y.notna()
    X_train = X_all.loc[mask].copy().apply(pd.to_numeric, errors="coerce")

    # drop all-NaN columns on training rows
    X_train = X_train.loc[:, X_train.notna().any(axis=0)]

    # Optional winsorization (fit quantiles on train, apply to train)
    bounds = None
    if ENABLE_WINSORIZE:
        bounds = compute_quantile_bounds(X_train, WINSOR_LO, WINSOR_HI)
        X_train = apply_bounds_clip(X_train, bounds)

    # median impute
    med = X_train.median(axis=0, skipna=True)
    X_train = X_train.fillna(med)

    # drop zero-variance
    std = X_train.std(axis=0, ddof=0)
    keep = std[std > 0].index
    X_train = X_train[keep]

    # standardize
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0).replace(0, 1.0)
    Xz  = (X_train - mu) / std

    kept_cols = list(Xz.columns)
    return mask, Xz, y.loc[mask], mu, std, med, kept_cols, bounds

def fit_models(Xz: pd.DataFrame, y: pd.Series):
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5, scoring="neg_mean_squared_error")
    enet  = ElasticNetCV(
        l1_ratio=[0.1,0.3,0.5,0.7,0.9],
        alphas=np.logspace(-3, 1, 20),
        cv=5, max_iter=10000, random_state=42
    )
    ridge.fit(Xz, y)
    enet.fit(Xz, y)
    return ridge, enet

def extract_weights(model, kept_cols: List[str], std: pd.Series, domain_map: Dict[str, str]):
    coef_std  = pd.Series(model.coef_.ravel(), index=kept_cols)
    coef_orig = coef_std / std[kept_cols].values

    w = pd.DataFrame({
        "INDICATOR": kept_cols,
        "coef_std": coef_std.values,
        "coef_original_scale": coef_orig.values
    })
    w["DOMAIN"] = w["INDICATOR"].map(domain_map)
    w["weight_indicator"] = w["coef_original_scale"]
    denom = w["weight_indicator"].abs().sum()
    w["weight_indicator_norm"] = 0.0 if (denom == 0 or pd.isna(denom)) else w["weight_indicator"] / denom

    dom_w = (w.groupby("DOMAIN", dropna=False)["weight_indicator_norm"]
               .sum().reset_index()
               .rename(columns={"weight_indicator_norm": "weight_domain"})
               .sort_values("weight_domain", ascending=False))
    return w[["INDICATOR","DOMAIN","coef_std","coef_original_scale",
              "weight_indicator","weight_indicator_norm"]], dom_w

def minmax_0_100(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0

def score_and_save(model, X_all, kept, mu, std, med, ids_df, mname, tname, bounds=None):
    """Standardize ALL rows using training stats, predict, and write per-country index 0–100."""
    paths = out_paths(mname, tname)

    X_num = X_all[kept].apply(pd.to_numeric, errors="coerce")
    if bounds is not None:
        # apply same clipping bounds to ALL rows
        X_num = apply_bounds_clip(X_num, bounds)
    X_imp = X_num.fillna(med[kept])
    X_all_std = (X_imp - mu[kept]) / std[kept]

    preds_all = pd.Series(model.predict(X_all_std), index=X_all_std.index)
    idx_all   = minmax_0_100(preds_all)

    out_scores = ids_df.copy() if ids_df is not None else pd.DataFrame(index=X_all_std.index)
    out_scores[f"{mname}_{tname}_model_pred"]  = preds_all
    out_scores[f"{mname}_{tname}_index_0_100"] = idx_all
    out_scores.to_csv(paths["scores"], index=False)

    return paths

# ========= FINAL OUTPUTS =========
def build_unweighted_domain_scores(domains: pd.DataFrame, X_all: pd.DataFrame) -> pd.DataFrame:
    """Unweighted domain scores: mean of domain indicators per country, min-max to 0–100 across countries."""
    dmap = domains[domains["INDICATOR"].isin(X_all.columns)].copy()
    if dmap.empty:
        return pd.DataFrame(index=X_all.index)

    out = pd.DataFrame(index=X_all.index)
    for d in sorted(dmap["DOMAIN"].dropna().unique().tolist()):
        cols = dmap.loc[dmap["DOMAIN"] == d, "INDICATOR"].unique().tolist()
        cols = [c for c in cols if c in X_all.columns]
        if not cols:
            continue
        raw = X_all[cols].astype(float)
        dom_mean = raw.mean(axis=1, skipna=True)
        out[f"DOMAIN_SCORE__{d}"] = minmax_0_100(dom_mean)
    return out

def build_indicator_scores_0_100(X_all: pd.DataFrame) -> pd.DataFrame:
    """Indicator-level 0–100 (no weights), min-max per indicator across countries."""
    df = pd.DataFrame(index=X_all.index)
    for col in X_all.columns:
        df[col] = minmax_0_100(pd.to_numeric(X_all[col], errors="coerce"))
    return df

def pick_country_col(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() == "country":
            return c
    return None

# ========= MAIN =========
def main():
    ensure_dir(OUTDIR_ROOT)
    final_dir = os.path.join(OUTDIR_ROOT, "final")
    ensure_dir(final_dir)

    domains, scores = load_data(DOMAINS_PATH, SCORES_PATH)
    id_cols = detect_id_cols(scores)
    X_all = build_feature_matrix(domains, scores)
    scores = make_targets(scores, id_cols)

    # IDs df for output joins
    ids_df = scores[id_cols].copy() if id_cols else None

    # Domain map for weights
    domain_map = domains.set_index("INDICATOR")["DOMAIN"].to_dict()

    # Targets to run (PRIMARY first: HG_log_raw)
    targets = {}
    if "HG_log_raw" in scores.columns and scores["HG_log_raw"].notna().any():
        targets["HG_log_raw"] = scores["HG_log_raw"]
    if "HG_per_100k" in scores.columns:
        targets["HG_per_100k"] = scores["HG_per_100k"]
    if TGT_NORM in scores.columns:
        targets["HG_Normalized"] = scores[TGT_NORM]

    manifest_rows = []
    metrics_rows  = []

    # Fit models and write per-model outputs
    for tname, y in targets.items():
        mask, Xz, y_fit, mu, std, med, kept, bounds = preprocess_for_target(X_all, y)
        if len(y_fit) < 5:
            print(f"[WARN] Not enough rows with target '{tname}' to fit models; skipping.")
            continue

        ridge, enet = fit_models(Xz, y_fit)

        for mname, model in [("ridge", ridge), ("enet", enet)]:
            # metrics (on training rows)
            y_pred = pd.Series(model.predict(Xz), index=Xz.index)
            r2   = r2_score(y_fit, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_fit, y_pred)))

            metrics = dict(
                r2=r2,
                rmse=rmse,
                n_samples=int(len(y_fit)),
                n_features=int(len(kept)),
                alpha_=getattr(model, "alpha_", None),
                l1_ratio_=getattr(model, "l1_ratio_", None) if mname == "enet" else None
            )

            # weights
            ind_w, dom_w = extract_weights(model, kept, std, domain_map)

            # save weights and scores
            paths = out_paths(mname, tname)
            ind_w.to_csv(paths["ind"], index=False)
            dom_w.to_csv(paths["dom"], index=False)
            score_and_save(model, X_all, kept, mu, std, med, ids_df, mname, tname, bounds=bounds)

            # manifest & metrics
            manifest_rows.append(dict(
                model_target=f"{mname}_{tname}",
                indicator_weights_csv=paths["ind"],
                domain_weights_csv=paths["dom"],
                scores_csv=paths["scores"]
            ))
            metrics_rows.append(dict(model_target=f"{mname}_{tname}", **metrics))

    # write manifests
    pd.DataFrame(manifest_rows).to_csv(os.path.join(OUTDIR_ROOT, "_manifest_outputs.csv"), index=False)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(OUTDIR_ROOT, "_metrics.csv"), index=False)

    # ---------- FINAL REQUESTED OUTPUTS ----------
    # 1) Unweighted domain scores (0–100 per domain, per country)
    #    (apply the SAME winsor bounds effect by leaving as-is; these are already normalized per domain)
    unweighted_domain = build_unweighted_domain_scores(domains, X_all)

    # Build a simple Domain Avg index (equal domain weights → then min-max across countries)
    dom_cols = [c for c in unweighted_domain.columns if c.startswith("DOMAIN_SCORE__")]
    domain_avg = None
    if dom_cols:
        domain_avg_raw = unweighted_domain[dom_cols].mean(axis=1, skipna=True)
        domain_avg = minmax_0_100(domain_avg_raw)
        unweighted_domain["Final_DomainAvg_0_100"] = domain_avg

    # 2) Indicator scores (0–100 per indicator, per country)
    indicator_scores = build_indicator_scores_0_100(X_all)

    # 3) Load component model indices we want to blend
    def try_load(model, target):
        p = os.path.join(OUTDIR_ROOT, model, target, "scores.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    ridge_log   = try_load("ridge", "HG_log_raw")
    ridge_pcap  = try_load("ridge", "HG_per_100k")

    # pick country col helper
    c_scores = pick_country_col(scores)

    # Prepare base with COUNTRY
    final_df = unweighted_domain.copy()
    if pick_country_col(final_df) is None and c_scores:
        final_df[c_scores] = scores[c_scores]

    cfinal = pick_country_col(final_df)

    # Bring components
    def merge_component(df_base, comp_df, rename_to):
        if comp_df is None:
            return df_base, None
        c = pick_country_col(comp_df)
        idx_cols = [x for x in comp_df.columns if x.endswith("_index_0_100")]
        if not idx_cols:
            return df_base, None
        ic = idx_cols[0]
        if cfinal and c:
            df_base = df_base.merge(comp_df[[c, ic]].rename(columns={ic: rename_to}),
                                    left_on=cfinal, right_on=c, how="left")
            if c != cfinal and c in df_base.columns:
                df_base = df_base.drop(columns=[c])
        else:
            df_base[rename_to] = comp_df[ic].values
        return df_base, rename_to

    final_df, name_log   = merge_component(final_df, ridge_log,  "Final_Log_0_100")
    final_df, name_pcap  = merge_component(final_df, ridge_pcap, "Final_PerCap_0_100")

    # Domain avg already computed
    if "Final_DomainAvg_0_100" not in final_df.columns and domain_avg is not None:
        final_df["Final_DomainAvg_0_100"] = domain_avg

    # Build blended score
    w_log    = BLEND_W_LOG
    w_pcap   = BLEND_W_PERCAP if name_pcap in final_df.columns else 0.0
    w_domain = BLEND_W_DOMAIN if "Final_DomainAvg_0_100" in final_df.columns else 0.0

    # Normalize weights to sum to 1 among available components
    w_sum = w_log + w_pcap + w_domain
    if w_sum == 0:
        raise RuntimeError("No components available to blend a final score.")
    w_log, w_pcap, w_domain = w_log / w_sum, w_pcap / w_sum, w_domain / w_sum

    # Weighted sum of already 0–100 components → remains 0–100 scale
    final_df["Final_Blended_0_100"] = 0.0
    if "Final_Log_0_100" in final_df.columns:
        final_df["Final_Blended_0_100"] += w_log * final_df["Final_Log_0_100"]
    if "Final_PerCap_0_100" in final_df.columns:
        final_df["Final_Blended_0_100"] += w_pcap * final_df["Final_PerCap_0_100"]
    if "Final_DomainAvg_0_100" in final_df.columns:
        final_df["Final_Blended_0_100"] += w_domain * final_df["Final_DomainAvg_0_100"]

    # For backward compatibility with your Streamlit app:
    final_df["Final_Score_0_100"] = final_df["Final_Blended_0_100"]

    # Reorder: COUNTRY first (if present), then domain cols, then all finals
    col_order = []
    if cfinal:
        col_order.append(cfinal)
    col_order.extend(sorted([c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]))
    for extra in ["Final_Log_0_100", "Final_PerCap_0_100", "Final_DomainAvg_0_100", "Final_Blended_0_100", "Final_Score_0_100"]:
        if extra in final_df.columns:
            col_order.append(extra)
    col_order = [c for c in col_order if c in final_df.columns]
    final_df = final_df[col_order]

    # Save final outputs
    final_dir = os.path.join(OUTDIR_ROOT, "final")
    ensure_dir(final_dir)
    final_scores_path    = os.path.join(final_dir, "final_scores.csv")
    final_indic_path     = os.path.join(final_dir, "indicator_scores.csv")
    final_df.to_csv(final_scores_path, index=False)

    indicator_scores_out = indicator_scores.copy()
    if c_scores and c_scores not in indicator_scores_out.columns:
        indicator_scores_out.insert(0, c_scores, scores[c_scores])
    indicator_scores_out.to_csv(final_indic_path, index=False)

    print("Done.")
    print(f"Manifest: {os.path.join(OUTDIR_ROOT, '_manifest_outputs.csv')}")
    print(f"Metrics:  {os.path.join(OUTDIR_ROOT, '_metrics.csv')}")
    print(f"Final scores:       {final_scores_path}")
    print(f"Indicator scores:   {final_indic_path}")

if __name__ == "__main__":
    main()