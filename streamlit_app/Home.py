import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Ecosystem Maturity Index", layout="wide")

# --- Constants for Column Names (for robustness and clarity) ---
COL_COUNTRY = "Country"
COL_REGION = "Region"
COL_INDICATOR = "INDICATOR"
COL_DOMAIN = "DOMAIN"
# Auto-select the main score column to use throughout the app
SCORE_CANDIDATES = [
    "Final_Score_0_100", "Final_Blended_0_100", "Final_Log_0_100",
    "Final_PerCap_0_100", "Final_DomainAvg_0_100"
]

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    """
    Loads all data, performs validation, merges, and pre-processes.
    Returns a dictionary of dataframes and key configuration values.
    """
    # Define paths
    data_dir = Path(__file__).parent / "data"
    paths = {
        "final": data_dir / "final_scores.csv",
        "indicator": data_dir / "indicator_scores.csv",
        "regions": data_dir / "country_regions.csv",
        "domains": data_dir / "domains.csv",
        "input": data_dir / "Input_scores.csv"
    }

    # Load data
    final_df = pd.read_csv(paths["final"]) if paths["final"].exists() else None
    ind_df = pd.read_csv(paths["indicator"]) if paths["indicator"].exists() else None
    reg_df = pd.read_csv(paths["regions"]) if paths["regions"].exists() else None
    dom_map = pd.read_csv(paths["domains"]) if paths["domains"].exists() else None
    input_df = pd.read_csv(paths["input"]) if paths["input"].exists() else None

    # --- Validation ---
    if final_df is None or ind_df is None:
        st.error("Fatal Error: `final_scores.csv` and `indicator_scores.csv` must be present in `streamlit_app/data/`.")
        st.stop()
    if COL_COUNTRY not in final_df.columns:
        st.error(f"Fatal Error: `final_scores.csv` must include a '{COL_COUNTRY}' column.")
        st.stop()

    score_to_show = next((c for c in SCORE_CANDIDATES if c in final_df.columns), None)
    if score_to_show is None:
        st.error(f"Fatal Error: No final score column found in `final_scores.csv`. Expected one of: {', '.join(SCORE_CANDIDATES)}")
        st.stop()

    # --- Pre-processing ---
    # 1. Attach regions
    if reg_df is not None and COL_COUNTRY in reg_df.columns and COL_REGION in reg_df.columns:
        final_df = final_df.merge(reg_df[[COL_COUNTRY, COL_REGION]], on=COL_COUNTRY, how="left")
    else:
        # If no region file, create a placeholder column to prevent errors
        final_df[COL_REGION] = "Global"

    # 2. Assign fixed maturity categories
    def assign_maturity(score):
        if score >= 55:
            return "Mature"
        elif score >= 25:
            return "Advancing"
        else:
            return "Nascent"

    final_df['Maturity'] = final_df[score_to_show].apply(assign_maturity)

    return {
        "final_df": final_df,
        "ind_df": ind_df,
        "dom_map": dom_map,
        "input_df": input_df,
        "score_to_show": score_to_show
    }

# --- Helper function to find High Growth Companies column ---
def try_find_hg_col(df: pd.DataFrame):
    if df is None: return None
    candidates = ["# of High Growth Companies", "High Growth Companies", "HG_COMPANIES", "seed"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None

# --- Main App ---
data = load_data()
final_df = data["final_df"]
ind_df = data["ind_df"]
dom_map = data["dom_map"]
input_df = data["input_df"]
score_to_show = data["score_to_show"]

# --- Sidebar ---
st.sidebar.header("Filters")
regions_list = sorted(final_df[COL_REGION].dropna().unique().tolist())

# Use checkbox for Global view
global_on = st.sidebar.checkbox("Global View", value=True)
if global_on:
    selected_regions = regions_list
else:
    selected_regions = st.sidebar.multiselect("Select Regions", regions_list, default=regions_list)

# Apply region filter
df = final_df[final_df[COL_REGION].isin(selected_regions)].copy()

# --- Page Title ---
st.title("Ecosystem Maturity Index")

# --- GRAPH 1: Leaderboard with Fixed Maturity Bands ---
st.subheader("Ecosystem Maturity Breakdown")

rank = df[[COL_COUNTRY, score_to_show, 'Maturity']].dropna().sort_values(score_to_show, ascending=False).reset_index(drop=True)
if rank.empty:
    st.info("No countries to display. Adjust filters.")
else:
    n = len(rank)
    fig = px.bar(rank, x=COL_COUNTRY, y=score_to_show, color='Maturity',
                 color_discrete_map={"Mature":"#ffb896", "Advancing":"#abd7f9", "Nascent":"#81c3f6"})

    # --- Replicate old dashboard's background bands using robust index method ---
    # Find the last index (position) for each maturity group in the sorted dataframe
    mature_indices = rank[rank['Maturity'] == 'Mature'].index
    advancing_indices = rank[rank['Maturity'] == 'Advancing'].index

    # Define the end position for each band
    pos_mature_end = (mature_indices.max() + 1) / n if not mature_indices.empty else 0
    pos_advancing_end = (advancing_indices.max() + 1) / n if not advancing_indices.empty else pos_mature_end

    # Add background rectangles
    # xref='paper' means the x-coordinates are fractions of the plot width (0.0 to 1.0)
    if pos_mature_end > 0:
        fig.add_vrect(x0=0, x1=pos_mature_end, fillcolor="#ffb896", opacity=0.2, line_width=0, layer="below")
        fig.add_annotation(x=(pos_mature_end / 2), y=rank[score_to_show].max(), yref='y', xref='paper',
                           text="<b>Mature</b>", showarrow=False, yshift=10)

    if pos_advancing_end > pos_mature_end:
        fig.add_vrect(x0=pos_mature_end, x1=pos_advancing_end, fillcolor="#abd7f9", opacity=0.2, line_width=0, layer="below")
        fig.add_annotation(x=((pos_mature_end + pos_advancing_end) / 2), y=rank[score_to_show].max(), yref='y', xref='paper',
                           text="<b>Advancing</b>", showarrow=False, yshift=10)

    if pos_advancing_end < 1:
        fig.add_vrect(x0=pos_advancing_end, x1=1, fillcolor="#81c3f6", opacity=0.2, line_width=0, layer="below")
        fig.add_annotation(x=((pos_advancing_end + 1) / 2), y=rank[score_to_show].max(), yref='y', xref='paper',
                           text="<b>Nascent</b>", showarrow=False, yshift=10)

    fig.update_layout(
        xaxis_title=None, yaxis_title=score_to_show.replace("_", " "),
        margin=dict(l=0, r=0, t=30, b=0), showlegend=False
    )
    fig.update_xaxes(tickangle=-75)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- GRAPH 2: Bubble chart (HG vs Score) - Log/Log view only ---
st.subheader("High Growth Companies vs. Index Score")

# Find HG column from either final_scores or Input_scores
hg_col = try_find_hg_col(final_df) or try_find_hg_col(input_df)
if hg_col is None:
    st.info("No High Growth Companies column found in data files.")
else:
    if hg_col in final_df.columns:
        bubble_df = df.copy()
    else: # Merge from input_df if needed
        bubble_df = df.merge(input_df[[COL_COUNTRY, hg_col]], on=COL_COUNTRY, how="left")

    bubble_df = bubble_df[[COL_COUNTRY, score_to_show, 'Maturity', hg_col]].dropna()

    if bubble_df.empty:
        st.info("Not enough data to draw the bubble chart with the selected filters.")
    else:
        # Use sqrt for bubble sizes to reduce dominance of outliers
        bubble_df["_size"] = np.sqrt(bubble_df[hg_col].clip(lower=1))
        # Add a small epsilon for log scaling of the x-axis
        bubble_df["_x_log"] = bubble_df[score_to_show] + 0.1

        fig_sc = px.scatter(
            bubble_df,
            x="_x_log",
            y=hg_col,
            size="_size",
            color="Maturity",
            color_discrete_map={"Mature":"#ffb896", "Advancing":"#abd7f9", "Nascent":"#81c3f6"},
            hover_name=COL_COUNTRY,
            size_max=60,
            log_x=True,
            log_y=True,
            labels={"_x_log": "Index Score (Log Scale)", hg_col: "High Growth Companies (Log Scale)"}
        )

        # Optional guides
        if st.checkbox("Show grouping guides", value=True):
            for y_val in [10, 100, 1000]: fig_sc.add_hline(y=y_val, line_dash="dash", line_color="#699bc6", line_width=1)
            for x_val in [25, 55]: fig_sc.add_vline(x=x_val, line_dash="dash", line_color="#699bc6", line_width=1)

        fig_sc.update_traces(marker_line_width=0.5, marker_line_color='black')
        fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Maturity")
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# --- UNIFIED DEEP-DIVE SECTION ---
st.subheader("Country Comparison Deep-Dive")

# --- Single, unified multiselect for all deep-dive charts ---
top_countries = df.sort_values(score_to_show, ascending=False)[COL_COUNTRY].head(3).tolist()
comparison_countries = st.multiselect(
    "Select up to 5 countries to compare:",
    options=sorted(df[COL_COUNTRY].unique().tolist()),
    default=top_countries,
    max_selections=5,
    key="country_comparator"
)

# --- Radar Chart ---
st.markdown("##### Domain Profile (Radar)")
domain_cols = [c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]

if not domain_cols:
    st.info("No `DOMAIN_SCORE__*` columns found in `final_scores.csv`.")
elif not comparison_countries:
    st.caption("Select one or more countries above to view the radar chart.")
else:
    radar_labels = [c.replace("DOMAIN_SCORE__", "") for c in domain_cols]
    radar_fig = go.Figure()
    for country in comparison_countries:
        vals = df.loc[df[COL_COUNTRY] == country, domain_cols].iloc[0].tolist()
        radar_fig.add_trace(go.Scatterpolar(r=vals, theta=radar_labels, fill='toself', name=country))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# --- Indicator Tabs ---
st.markdown("##### Indicator Drill-Down")

if dom_map is None or ind_df is None:
    st.info("Add `domains.csv` and `indicator_scores.csv` to enable this section.")
elif not comparison_countries:
    st.caption("Select one or more countries to view indicator scores.")
else:
    # Prepare indicator data for selected countries
    work_df = ind_df.merge(df[[COL_COUNTRY]], on=COL_COUNTRY, how="inner")
    domains_list = sorted(dom_map[COL_DOMAIN].dropna().unique().tolist())
    tabs = st.tabs(domains_list)

    for i, dname in enumerate(domains_list):
        with tabs[i]:
            d_inds = dom_map.loc[dom_map[COL_DOMAIN] == dname, COL_INDICATOR].unique().tolist()
            d_inds_available = [ind for ind in d_inds if ind in work_df.columns]

            if not d_inds_available:
                st.info(f"No indicator data available for domain: {dname}")
                continue

            chosen_inds = st.multiselect(
                "Choose up to 3 indicators:", options=d_inds_available,
                default=d_inds_available[:3], max_selections=3, key=f"inds_{dname}"
            )

            if not chosen_inds:
                st.caption("Select at least one indicator to plot.")
                continue

            plot_df = work_df[work_df[COL_COUNTRY].isin(comparison_countries)][[COL_COUNTRY] + chosen_inds]
            melt_df = plot_df.melt(id_vars=[COL_COUNTRY], var_name="Indicator", value_name="Score")
            
            # Sort countries by the sum of scores for consistent ordering
            order = melt_df.groupby(COL_COUNTRY)["Score"].sum().sort_values(ascending=False).index

            fig_bars = px.bar(
                melt_df, y=COL_COUNTRY, x="Score", color=COL_COUNTRY,
                barmode="group", text="Score", facet_col="Indicator",
                category_orders={COL_COUNTRY: order}
            )
            fig_bars.update_traces(texttemplate="%{text:.0f}", textposition="outside", width=0.6)
            fig_bars.update_layout(
                bargroupgap=0.15, yaxis_title="", xaxis_title="", legend_title_text="", showlegend=False,
                margin=dict(r=5, l=5, t=30, b=5)
            )
            fig_bars.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_bars.for_each_xaxis(lambda axis: axis.update(title="", range=[0, 105]))
            st.plotly_chart(fig_bars, use_container_width=True)

# --- Footer ---
with st.expander("About This Dashboard"):
    st.markdown(f"""
    - **Score shown:** `{score_to_show}` (auto-selected from available columns).
    - **Maturity bands:**
        - **Mature:** Score >= 55
        - **Advancing:** Score >= 25 and < 55
        - **Nascent:** Score < 25
    - **Files Used:** `final_scores.csv`, `indicator_scores.csv`, `country_regions.csv`, `domains.csv`, and `Input_scores.csv`.
    """)