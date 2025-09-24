import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ecosystem Maturity Index", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FINAL_SCORES_PATH = DATA_DIR / "final_scores.csv"
INDICATOR_SCORES_PATH = DATA_DIR / "indicator_scores.csv"
META_PATH = DATA_DIR / "meta.csv"  # optional

# ---------- Helpers ----------
@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df

def pick_col(df, name_lower):
    return next((c for c in df.columns if c.lower() == name_lower), None)

def maturity_bucket(score, scheme, p33=33.33, p66=66.67):
    if pd.isna(score):
        return "Unknown"
    if scheme == "Fixed thresholds (50/75)":
        if score < 50: return "Nascent"
        if score < 75: return "Advancing"
        return "Mature"
    else:  # Percentiles
        if score < p33: return "Nascent"
        if score < p66: return "Advancing"
        return "Mature"

def as_percentile_thresholds(series):
    return np.nanpercentile(series, [33.33, 66.67])

def domain_cols(df):
    return [c for c in df.columns if c.startswith("DOMAIN_SCORE__")]

def domain_name(col):
    return col.replace("DOMAIN_SCORE__","")

# ---------- Load data ----------
final_df = load_csv(FINAL_SCORES_PATH)
ind_df   = load_csv(INDICATOR_SCORES_PATH)
meta_df  = load_csv(META_PATH)

if final_df is None or ind_df is None:
    st.error("Missing required data. Please place `final_scores.csv` and `indicator_scores.csv` in `streamlit_app/data/`.")
    st.stop()

COUNTRY = pick_col(final_df, "country")
if COUNTRY is None:
    st.error("`final_scores.csv` must include a COUNTRY column.")
    st.stop()

# Merge optional meta
if meta_df is not None:
    c_meta = pick_col(meta_df, "country")
    if c_meta:
        final_df = final_df.merge(meta_df, left_on=COUNTRY, right_on=c_meta, how="left")
        # avoid duplicate country column
        if c_meta != COUNTRY and c_meta in final_df.columns:
            final_df = final_df.drop(columns=[c_meta])

regions_present = meta_df is not None and pick_col(final_df, "region") is not None
REGION = pick_col(final_df, "region") if regions_present else None

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

# Region
if regions_present:
    region_vals = sorted([r for r in final_df[REGION].dropna().unique().tolist()])
    selected_regions = st.sidebar.multiselect("Region", region_vals, default=region_vals)
else:
    selected_regions = None

# Countries
if regions_present:
    base_choices = final_df.loc[final_df[REGION].isin(selected_regions), COUNTRY].dropna().unique()
else:
    base_choices = final_df[COUNTRY].dropna().unique()

base_choices = sorted(base_choices.tolist())
select_all = st.sidebar.checkbox("Select All Countries", value=True)
if select_all:
    selected_countries = base_choices
else:
    selected_countries = st.sidebar.multiselect("Select Countries", base_choices, default=base_choices[:10])

# Bucket scheme
scheme = st.sidebar.radio("Maturity bucket scheme", ["Fixed thresholds (50/75)", "Percentiles (P33/P66)"], index=0)

# Filtered data
df = final_df.copy()
if regions_present:
    df = df[df[REGION].isin(selected_regions)]
df = df[df[COUNTRY].isin(selected_countries)]

# Percentile cutoffs (computed on filtered universe to match view)
p33, p66 = as_percentile_thresholds(final_df["Final_Score_0_100"])
if scheme == "Percentiles (P33/P66)":
    p33, p66 = as_percentile_thresholds(df["Final_Score_0_100"])

df["Maturity"] = df["Final_Score_0_100"].apply(lambda x: maturity_bucket(x, scheme, p33, p66))

# ---------- Title ----------
st.title("Ecosystem Maturity Index")

# ---------- Top row: Leaderboard & Donut ----------
col1, col2 = st.columns([2.2, 1.2])

with col1:
    st.subheader("Leaderboard")
    rank_cols = [COUNTRY, "Final_Score_0_100"]
    lb = df[rank_cols].dropna().sort_values("Final_Score_0_100", ascending=False)
    fig = px.bar(
        lb,
        x=COUNTRY,
        y="Final_Score_0_100",
        hover_data=rank_cols,
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Final Score (0–100)", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Maturity")
    pie = df.groupby("Maturity", dropna=False)[COUNTRY].count().reset_index(name="Count")
    if not pie.empty:
        fig_pie = px.pie(
            pie, names="Maturity", values="Count", hole=0.55
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data available for donut.")

st.divider()

# ---------- Domain Breakdown ----------
st.subheader("Domain Breakdown")

dom_cols = domain_cols(final_df)
if not dom_cols:
    st.info("No domain columns (DOMAIN_SCORE__*) found in final_scores.csv")
else:
    # Stacked horizontal bars for selected countries
    melt = df[[COUNTRY]+dom_cols].melt(id_vars=COUNTRY, var_name="DomainCol", value_name="Score")
    melt["Domain"] = melt["DomainCol"].apply(domain_name)
    fig_stack = px.bar(
        melt, y=COUNTRY, x="Score", color="Domain", orientation="h",
        category_orders={COUNTRY: df.sort_values("Final_Score_0_100", ascending=True)[COUNTRY].tolist()},
    )
    fig_stack.update_layout(showlegend=True, barmode="stack", xaxis_title="Score (0–100)", yaxis_title=None, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_stack, use_container_width=True)

    # Radar: selected country vs global avg
    st.markdown("**Radar: Selected Country vs Global Average**")
    pick = st.selectbox("Country for radar", sorted(df[COUNTRY].unique().tolist()))
    dom_avgs = final_df[dom_cols].mean(axis=0, skipna=True)

    r_labels = [domain_name(c) for c in dom_cols]
    sel_vals = df.loc[df[COUNTRY]==pick, dom_cols].iloc[0].values.tolist() if (pick in df[COUNTRY].values) else [np.nan]*len(dom_cols)
    avg_vals = dom_avgs.values.tolist()

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=sel_vals, theta=r_labels, fill='toself', name=pick))
    radar.add_trace(go.Scatterpolar(r=avg_vals, theta=r_labels, fill='toself', name='Global Avg'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(radar, use_container_width=True)

st.divider()

# ---------- Indicator Deep Dive (lite) ----------
st.subheader("Indicator Deep-Dive")

# Build domain -> indicators map from the columns we see in indicator_scores (we don't know domains there)
# We infer with names present in final_df domain map via overlap (optional):
# Fallback: just show the top-variance indicators overall.
ind_df_work = ind_df.copy()
icountry = pick_col(ind_df_work, "country")
if icountry is None:
    st.info("`indicator_scores.csv` lacks COUNTRY column; showing all indicators without country names.")
else:
    # filter same subset as df
    ind_df_work = ind_df_work[ind_df_work[icountry].isin(df[COUNTRY])]
    # rename to same country column if names differ
    if icountry != COUNTRY:
        ind_df_work = ind_df_work.rename(columns={icountry: COUNTRY})

# Most varying indicators across selected countries (helps pick interesting signals)
value_cols = [c for c in ind_df_work.columns if c != COUNTRY]
if not value_cols:
    st.info("No indicator columns found in indicator_scores.csv.")
else:
    variances = ind_df_work[value_cols].var(numeric_only=True).sort_values(ascending=False)
    topN = st.slider("Top indicators by variance", min_value=5, max_value=min(40, len(variances)), value=min(15, len(variances)))
    top_inds = variances.head(topN).index.tolist()

    # Table view (wide format)
    show = ind_df_work[[COUNTRY] + top_inds].sort_values(COUNTRY)
    st.dataframe(show, use_container_width=True, height=400)

# ---------- Footer ----------
with st.expander("About this dashboard"):
    st.markdown("""
**Data:**  
- `final_scores.csv`: Final score per country (0–100) and per-domain unweighted scores (0–100).  
- `indicator_scores.csv`: Per-indicator 0–100 scores (unweighted).  

**Maturity buckets:** configurable in the sidebar (fixed thresholds or percentiles).
""")