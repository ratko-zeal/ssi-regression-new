import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ecosystem Maturity Index", layout="wide")

# ---------- Paths ----------
DATA_DIR = Path(__file__).parent / "data"
FINAL_SCORES_PATH = DATA_DIR / "final_scores.csv"
INDICATOR_SCORES_PATH = DATA_DIR / "indicator_scores.csv"
REGIONS_PATH = DATA_DIR / "country_regions.csv"  # Country, Region

# ---------- Helpers ----------
@st.cache_data
def load_csv(p: Path):
    if not p.exists():
        return None
    return pd.read_csv(p)

def pick_col(df: pd.DataFrame, name_lower: str):
    return next((c for c in df.columns if c.lower() == name_lower), None)

def domain_cols(df):
    return [c for c in df.columns if c.startswith("DOMAIN_SCORE__")]

def domain_name(col):
    return col.replace("DOMAIN_SCORE__","")

def maturity_bucket(score, scheme, p33=33.33, p66=66.67):
    if pd.isna(score):
        return "Unknown"
    if scheme == "Fixed thresholds (50/75)":
        if score < 50: return "Nascent"
        if score < 75: return "Advancing"
        return "Mature"
    else:
        if score < p33: return "Nascent"
        if score < p66: return "Advancing"
        return "Mature"

def percentile_cutoffs(series):
    return np.nanpercentile(series, [33.33, 66.67])

# ---------- Load ----------
final_df = load_csv(FINAL_SCORES_PATH)
ind_df   = load_csv(INDICATOR_SCORES_PATH)
reg_df   = load_csv(REGIONS_PATH)  # optional, but you said you added it

if final_df is None or ind_df is None:
    st.error("Please add `final_scores.csv` and `indicator_scores.csv` in `streamlit_app/data/`.")
    st.stop()

COUNTRY = pick_col(final_df, "country")
if COUNTRY is None:
    st.error("`final_scores.csv` must include a COUNTRY column.")
    st.stop()

# Attach regions
if reg_df is not None:
    c_reg = pick_col(reg_df, "country")
    r_reg = pick_col(reg_df, "region")
    if c_reg and r_reg:
        final_df = final_df.merge(reg_df[[c_reg, r_reg]], left_on=COUNTRY, right_on=c_reg, how="left")
        if c_reg != COUNTRY and c_reg in final_df.columns:
            final_df = final_df.drop(columns=[c_reg])
        REGION = r_reg
    else:
        REGION = None
else:
    REGION = None

# Score variants present
score_variants = [c for c in [
    "Final_Score_0_100",
    "Final_Blended_0_100",
    "Final_Log_0_100",
    "Final_PerCap_0_100",
    "Final_DomainAvg_0_100"
] if c in final_df.columns]

if not score_variants:
    st.error("No final score columns found (e.g., Final_Score_0_100).")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("Filters")

# Choose which final score to visualize
score_to_show = st.sidebar.selectbox(
    "Score to visualize",
    score_variants,
    index=score_variants.index("Final_Score_0_100") if "Final_Score_0_100" in score_variants else 0
)

# Bucket scheme
scheme = st.sidebar.radio(
    "Maturity bucket scheme",
    ["Fixed thresholds (50/75)", "Percentiles (P33/P66)"],
    index=0
)

# Region filter from country_regions.csv
if REGION:
    all_regions = sorted([r for r in final_df[REGION].dropna().unique().tolist()])
    selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
    df_filtered_reg = final_df[final_df[REGION].isin(selected_regions)]
else:
    selected_regions = None
    df_filtered_reg = final_df

# Country multiselect with "Select All"
base_countries = sorted(df_filtered_reg[COUNTRY].dropna().unique().tolist())
select_all = st.sidebar.checkbox("Select All Countries", value=True)
if select_all:
    selected_countries = base_countries
else:
    selected_countries = st.sidebar.multiselect("Select Countries", base_countries, default=base_countries[:12])

# Apply country filter
df = df_filtered_reg[df_filtered_reg[COUNTRY].isin(selected_countries)].copy()

# Cutoffs for maturity (compute on the displayed universe)
p33, p66 = percentile_cutoffs(df[score_to_show])
df["Maturity"] = df[score_to_show].apply(lambda x: maturity_bucket(x, scheme, p33, p66))

# ---------- Title ----------
st.title("Ecosystem Maturity Index")

# ---------- Leaderboard & Donut ----------
col1, col2 = st.columns([2.3, 1.2])

with col1:
    st.subheader("Leaderboard")
    rank = df[[COUNTRY, score_to_show]].dropna().sort_values(score_to_show, ascending=False)
    fig = px.bar(rank, x=COUNTRY, y=score_to_show)
    fig.update_layout(xaxis_title=None, yaxis_title="Final Score (0–100)", margin=dict(l=0,r=0,t=10,b=0))
    fig.update_xaxes(tickangle=-75)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Maturity")
    pie = df.groupby("Maturity", dropna=False)[COUNTRY].count().reset_index(name="Count")
    if len(pie) > 0:
        fig_p = px.pie(pie, names="Maturity", values="Count", hole=0.55,
                       color="Maturity",
                       color_discrete_map={"Mature":"#ffb896","Advancing":"#abd7f9","Nascent":"#81c3f6"})
        fig_p.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("No data for donut.")

st.divider()

# ---------- Domain Breakdown ----------
st.subheader("Domain Breakdown")

dom_cols = domain_cols(final_df)
if not dom_cols:
    st.info("No DOMAIN_SCORE__* columns in final_scores.csv.")
else:
    melt = df[[COUNTRY] + dom_cols].melt(id_vars=COUNTRY, var_name="DomainCol", value_name="Score")
    melt["Domain"] = melt["DomainCol"].apply(domain_name)
    order = df.sort_values(score_to_show, ascending=True)[COUNTRY].tolist()
    fig_stack = px.bar(
        melt, y=COUNTRY, x="Score", color="Domain", orientation="h",
        category_orders={COUNTRY: order}
    )
    fig_stack.update_layout(barmode="stack", xaxis_title="Score (0–100)", yaxis_title=None, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("**Radar: Selected Country vs Global Average**")
    pick = st.selectbox("Country for radar", sorted(df[COUNTRY].unique().tolist()))
    dom_avgs = final_df[dom_cols].mean(axis=0, skipna=True)
    r_labels = [domain_name(c) for c in dom_cols]
    sel_vals = df.loc[df[COUNTRY] == pick, dom_cols].iloc[0].values.tolist() if pick in df[COUNTRY].values else [np.nan]*len(dom_cols)
    avg_vals = dom_avgs.values.tolist()

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=sel_vals, theta=r_labels, fill='toself', name=pick))
    radar.add_trace(go.Scatterpolar(r=avg_vals, theta=r_labels, fill='toself', name='Global Avg'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                        showlegend=True, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(radar, use_container_width=True)

st.divider()

# ---------- Indicator Deep-Dive ----------
st.subheader("Indicator Deep-Dive")

icountry = pick_col(ind_df, "country")
work = ind_df.copy()
if icountry and icountry != COUNTRY:
    work = work.rename(columns={icountry: COUNTRY})

# Align with selected countries
work = work[work[COUNTRY].isin(df[COUNTRY])]

indicator_cols = [c for c in work.columns if c != COUNTRY]
if indicator_cols:
    # Most varying indicators across selected countries
    variances = work[indicator_cols].var(numeric_only=True).sort_values(ascending=False)
    topN = st.slider("Top indicators by variance", min_value=5, max_value=min(40, len(variances)), value=min(15, len(variances)))
    top_inds = variances.head(topN).index.tolist()
    table = work[[COUNTRY] + top_inds].sort_values(COUNTRY)
    st.dataframe(table, use_container_width=True, height=400)
else:
    st.info("No indicator columns found in indicator_scores.csv.")

# ---------- Footer ----------
with st.expander("About"):
    st.markdown(f"""
**Score shown:** `{score_to_show}`.  
**Maturity buckets:** set by *{scheme}* (change in sidebar).  
**Files used**  
- `final_scores.csv` — Final score variants (0–100) and per-domain unweighted `DOMAIN_SCORE__*`.  
- `indicator_scores.csv` — 0–100 by indicator.  
- `country_regions.csv` — Country → Region mapping for filters.  
""")