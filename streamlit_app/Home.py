import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ecosystem Maturity Index", layout="wide")

# ---------------- Paths ----------------
DATA_DIR = Path(__file__).parent / "data"
FINAL_SCORES_PATH     = DATA_DIR / "final_scores.csv"
INDICATOR_SCORES_PATH = DATA_DIR / "indicator_scores.csv"
REGIONS_PATH          = DATA_DIR / "country_regions.csv"     # Country, Region
DOMAINS_MAP_PATH      = DATA_DIR / "domains.csv"             # INDICATOR, DOMAIN (optional)
INPUT_SCORES_PATH     = DATA_DIR / "Input_scores.csv"        # optional, for raw HG counts

# ---------------- Helpers ----------------
@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def pick_col(df: pd.DataFrame, name_lower: str):
    return next((c for c in df.columns if c.lower() == name_lower), None)

def domain_cols(df):
    return [c for c in df.columns if c.startswith("DOMAIN_SCORE__")]

def domain_name(col):
    return col.replace("DOMAIN_SCORE__","")

def maturity_from_percent_ranks(n: int):
    """Return (mature_end_idx, advancing_end_idx) for 15% / 35% scheme."""
    m_end = max(1, int(np.ceil(0.15 * n)))
    a_end = max(m_end, int(np.ceil(0.50 * n)))  # 15% + 35% = 50%
    return m_end, a_end

def try_find_hg_col(df: pd.DataFrame):
    if df is None:
        return None
    candidates = [
        "# of High Growth Companies",
        "High Growth Companies",
        "HG_COMPANIES",
        "High Growth Startups",
        "seed"  # legacy
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

# ---------------- Load Data ----------------
final_df = load_csv(FINAL_SCORES_PATH)
ind_df   = load_csv(INDICATOR_SCORES_PATH)
reg_df   = load_csv(REGIONS_PATH)
dom_map  = load_csv(DOMAINS_MAP_PATH)   # optional
input_df = load_csv(INPUT_SCORES_PATH)  # optional

if final_df is None or ind_df is None:
    st.error("Please add `final_scores.csv` and `indicator_scores.csv` in `streamlit_app/data/`.")
    st.stop()

COUNTRY = pick_col(final_df, "country")
if COUNTRY is None:
    st.error("`final_scores.csv` must include a `Country` column.")
    st.stop()

# Choose score column automatically (no sidebar selector anymore)
score_candidates = [
    "Final_Score_0_100", "Final_Blended_0_100", "Final_Log_0_100",
    "Final_PerCap_0_100", "Final_DomainAvg_0_100"
]
score_to_show = next((c for c in score_candidates if c in final_df.columns), None)
if score_to_show is None:
    st.error("No final score columns found (e.g., `Final_Score_0_100`).")
    st.stop()

# Attach regions
REGION = None
if reg_df is not None:
    c_reg = pick_col(reg_df, "country")
    r_reg = pick_col(reg_df, "region")
    if c_reg and r_reg:
        final_df = final_df.merge(reg_df[[c_reg, r_reg]], left_on=COUNTRY, right_on=c_reg, how="left")
        if c_reg != COUNTRY and c_reg in final_df.columns:
            final_df.drop(columns=[c_reg], inplace=True)
        REGION = r_reg

# ---------------- Sidebar (regions only + Global) ----------------
st.sidebar.header("Regions")
regions_list = sorted(final_df[REGION].dropna().unique().tolist()) if REGION else []
default_regions = regions_list[:]  # preselect all when Global is off

global_on = st.sidebar.checkbox("Global", value=True)
if REGION and not global_on:
    selected_regions = st.sidebar.multiselect("Select Regions", regions_list, default=default_regions)
else:
    selected_regions = regions_list  # ignored if global_on True

# Apply region filter if Global is OFF
if REGION and not global_on:
    df = final_df[final_df[REGION].isin(selected_regions)].copy()
else:
    df = final_df.copy()

# ---------------- Title ----------------
st.title("Ecosystem Maturity Index")

# ---------------- GRAPH 1: Leaderboard with 15/35/50 maturity bands ----------------
st.subheader("Ecosystem Maturity Breakdown")

rank = df[[COUNTRY, score_to_show]].dropna().sort_values(score_to_show, ascending=False).reset_index(drop=True)
n = len(rank)
if n == 0:
    st.info("No countries to display. Adjust filters.")
else:
    m_end, a_end = maturity_from_percent_ranks(n)

    fig = px.bar(rank, x=COUNTRY, y=score_to_show)

    # Background bands (paper coords)
    fig.add_vrect(x0=0.0, x1=(m_end / n), xref="paper", fillcolor="#ffb896", opacity=0.35, line_width=0, layer="below")
    fig.add_vrect(x0=(m_end / n), x1=(a_end / n), xref="paper", fillcolor="#abd7f9", opacity=0.35, line_width=0, layer="below")
    fig.add_vrect(x0=(a_end / n), x1=1.0, xref="paper", fillcolor="#81c3f6", opacity=0.35, line_width=0, layer="below")

    # Titles for bands
    ymax = rank[score_to_show].max()
    fig.add_annotation(x=(0.5 * m_end / n), y=ymax * 1.03, text="<b>Mature</b>", showarrow=False)
    fig.add_annotation(x=((m_end + a_end) / 2 / n), y=ymax * 1.03, text="<b>Advancing</b>", showarrow=False)
    fig.add_annotation(x=((a_end + n) / 2 / n), y=ymax * 1.03, text="<b>Nascent</b>", showarrow=False)

    fig.update_layout(xaxis_title=None, yaxis_title=score_to_show.replace("_"," "),
                      margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(tickangle=-75)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- GRAPH 2: Bubble chart (HG vs Score) with grouping/guides ----------------
# ---------------- GRAPH 2: Bubble chart (HG vs Score) with grouping/guides + log options ----------------
st.subheader("High Growth Companies Against Score")

# Find HG column
hg_col_final = try_find_hg_col(final_df)
hg_col_input = try_find_hg_col(input_df)
hg_source = "final" if hg_col_final else ("input" if hg_col_input else None)

if hg_source is None:
    st.info("No `# of High Growth Companies` column found (expected in `final_scores.csv` or `Input_scores.csv`).")
else:
    if hg_source == "final":
        bubble_df = df[[COUNTRY, score_to_show, hg_col_final]].dropna()
        hg_col = hg_col_final
    else:
        merged = df.merge(input_df[[pick_col(input_df, "country"), hg_col_input]],
                          left_on=COUNTRY, right_on=pick_col(input_df, "country"), how="left")
        if pick_col(input_df, "country") != COUNTRY:
            merged.drop(columns=[pick_col(input_df, "country")], inplace=True)
        bubble_df = merged[[COUNTRY, score_to_show, hg_col_input]].dropna()
        hg_col = hg_col_input

    if len(bubble_df) == 0:
        st.info("Not enough data to draw the bubble chart with the selected filters.")
    else:
        # Color by maturity buckets computed on the shown data (15/35/50)
        rank2 = bubble_df[[COUNTRY, score_to_show]].sort_values(score_to_show, ascending=False).reset_index(drop=True)
        n2 = len(rank2)
        m_end2, a_end2 = maturity_from_percent_ranks(n2)
        rank2["Maturity"] = ["Mature"]*m_end2 + ["Advancing"]*(a_end2-m_end2) + ["Nascent"]*(n2-a_end2)
        bubble_df = bubble_df.merge(rank2[[COUNTRY, "Maturity"]], on=COUNTRY, how="left")

        # ---- NEW: scaling options ----
        scale_mode = st.radio(
            "Scale",
            ["Log X & Log Y (compact)", "Log Y only", "Linear"],
            index=0,
            horizontal=True,
            key="bubble_scale_mode"
        )

        # sqrt bubble sizes → less dominance at the very top
        bubble_df["_size"] = np.sqrt(np.maximum(bubble_df[hg_col].values, 1.0))

        # avoid zeros on log-x by adding a tiny epsilon (for plotting only)
        eps = 0.1
        bubble_df["_x"] = bubble_df[score_to_show].astype(float) + eps

        # draw
        fig_sc = px.scatter(
            bubble_df,
            x="_x" if scale_mode != "Linear" else score_to_show,
            y=hg_col,
            size="_size",
            color="Maturity",
            color_discrete_map={"Mature":"#ffb896","Advancing":"#abd7f9","Nascent":"#81c3f6"},
            hover_name=COUNTRY,
            size_max=60,
            labels={score_to_show: "SS Index", "_x": "SS Index", hg_col: "High Growth Companies"}
        )

        # set axis scales
        if scale_mode == "Log X & Log Y (compact)":
            fig_sc.update_xaxes(type="log")
            fig_sc.update_yaxes(type="log")
        elif scale_mode == "Log Y only":
            fig_sc.update_yaxes(type="log")

        # optional guides
        show_guides = st.checkbox("Show grouping guides", value=True)
        if show_guides:
            # Horizontal dashed lines at common tiers
            for yv in [3, 10, 100, 1000]:
                fig_sc.add_hline(y=yv, line=dict(color="#699bc6", width=1, dash="dash"))
            # Vertical dashed lines at score thresholds (50, 75)
            for xv in [50, 75]:
                x_plot = xv + (eps if scale_mode != "Linear" else 0)
                if bubble_df["_x" if scale_mode != "Linear" else score_to_show].between(
                    bubble_df["_x" if scale_mode != "Linear" else score_to_show].min(),
                    bubble_df["_x" if scale_mode != "Linear" else score_to_show].max()
                ).any():
                    fig_sc.add_vline(x=x_plot, line=dict(color="#699bc6", width=1, dash="dash"))

        # tidy
        fig_sc.update_traces(marker_line_width=0)
        fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ---------------- DOMAIN BREAKDOWN: Radar only (up to 5 countries) ----------------
st.subheader("Domain Deep-dive (Radar)")

dom_cols = domain_cols(final_df)
if not dom_cols:
    st.info("No `DOMAIN_SCORE__*` columns found in `final_scores.csv`.")
else:
    sel_countries = st.multiselect(
        "Pick up to 5 countries",
        sorted(df[COUNTRY].unique().tolist()),
        default=[],
        max_selections=5
    )

    if not sel_countries:
        st.caption("Select countries above to view the radar.")
    else:
        r_labels = [domain_name(c) for c in dom_cols]
        radar = go.Figure()
        for ctry in sel_countries:
            vals = df.loc[df[COUNTRY] == ctry, dom_cols]
            if vals.empty:
                continue
            radar.add_trace(go.Scatterpolar(
                r=vals.iloc[0].tolist(), theta=r_labels, fill='toself', name=ctry
            ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            showlegend=True, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(radar, use_container_width=True)

st.divider()

# ---------------- DOMAIN BREAKDOWN: Tabs with up to 3 indicators per domain ----------------
st.subheader("Domain Deep-dive (Indicators by Domain)")

dom_map = load_csv(DOMAINS_MAP_PATH)  # reload to ensure availability at runtime
if dom_map is None or pick_col(dom_map, "indicator") is None or pick_col(dom_map, "domain") is None:
    st.info("Add `domains.csv` (columns: `INDICATOR`, `DOMAIN`) to enable this section.")
else:
    ind_map_indicator = pick_col(dom_map, "indicator")
    ind_map_domain    = pick_col(dom_map, "domain")

    icountry = pick_col(ind_df, "country")
    work = ind_df.copy()
    if icountry and icountry != COUNTRY:
        work = work.rename(columns={icountry: COUNTRY})
    work = work.merge(df[[COUNTRY]], on=COUNTRY, how="inner")

    available_inds = set([c for c in work.columns if c != COUNTRY])
    dom_map_use = dom_map[[ind_map_indicator, ind_map_domain]].copy()
    dom_map_use[ind_map_indicator] = dom_map_use[ind_map_indicator].astype(str)
    dom_map_use[ind_map_domain]    = dom_map_use[ind_map_domain].astype(str)
    dom_map_use = dom_map_use[dom_map_use[ind_map_indicator].isin(available_inds)]
    domains_list = sorted(dom_map_use[ind_map_domain].dropna().unique().tolist())

    tabs = st.tabs(domains_list if domains_list else ["Domains"])
    if not domains_list:
        with tabs[0]:
            st.info("No domain/indicator mapping matches your indicator file.")
    else:
        chosen_countries = st.multiselect(
            "Select countries for domain bars (up to 5)",
            sorted(df[COUNTRY].unique().tolist()),
            default=[],
            max_selections=5,
            key="dom_bar_countries"
        )

        for i, dname in enumerate(domains_list):
            with tabs[i]:
                d_inds = dom_map_use.loc[dom_map_use[ind_map_domain] == dname, ind_map_indicator].unique().tolist()
                if not d_inds:
                    st.info(f"No indicators mapped for domain: {dname}")
                    continue

                chosen_inds = st.multiselect(
                    "Choose up to 3 indicators:",
                    options=d_inds,
                    default=d_inds[:3] if len(d_inds) >= 3 else d_inds,
                    max_selections=3,
                    key=f"inds_{dname}"
                )

                if not chosen_inds or not chosen_countries:
                    st.caption("Select up to 3 indicators and at least one country.")
                    continue

                plot_df = work[work[COUNTRY].isin(chosen_countries)][[COUNTRY] + chosen_inds].copy()
                melt_df = plot_df.melt(id_vars=[COUNTRY], value_vars=chosen_inds,
                                       var_name="Indicator", value_name="Score")
                order = (melt_df.groupby(COUNTRY)["Score"].sum().sort_values(ascending=False).index.tolist())
                fig_bars = px.bar(
                    melt_df, y=COUNTRY, x="Score", color=COUNTRY,
                    barmode="group", text="Score", facet_col="Indicator",
                    category_orders={COUNTRY: order}
                )
                fig_bars.update_traces(texttemplate="%{text:.1f}", textposition="outside", width=0.6)
                fig_bars.update_layout(
                    bargroupgap=0.12, yaxis_title="", xaxis_title="",
                    margin=dict(r=100, l=5, t=10, b=0), legend_title_text=""
                )
                fig_bars.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig_bars.for_each_xaxis(lambda xaxis: xaxis.update(title=""))
                st.plotly_chart(fig_bars, use_container_width=True)

# ---------------- Footer ----------------
with st.expander("About"):
    st.markdown(f"""
**Score shown:** `{score_to_show}` (auto-selected).  
**Maturity bands:** top 15% = Mature, next 35% = Advancing, rest = Nascent.  
**Files used**  
- `final_scores.csv` — Final score variants (0–100) and per-domain `DOMAIN_SCORE__*`.  
- `indicator_scores.csv` — 0–100 by indicator.  
- `country_regions.csv` — Country → Region mapping.  
- `domains.csv` — Indicator → Domain mapping (optional, for domain tabs).  
- `Input_scores.csv` — Optional; only used to fetch `# of High Growth Companies` for the bubble chart.
""")