import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Ecosystem Maturity Index", layout="wide")

# --- Constants for Column Names (ALL CAPS to match CSVs) ---
COL_COUNTRY = "COUNTRY"
COL_REGION = "REGION"
COL_INDICATOR = "INDICATOR"
COL_DOMAIN = "DOMAIN"
SCORE_CANDIDATES = [
    "Final_Score_0_100", "Final_Blended_0_100", "Final_Log_0_100",
    "Final_PerCap_0_100", "Final_DomainAvg_0_100"
]

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    """
    Loads all data, performs validation, merges, and pre-processes.
    """
    data_dir = Path(__file__).parent / "data"
    paths = {
        "final": data_dir / "final_scores.csv",
        "indicator": data_dir / "indicator_scores.csv",
        "regions": data_dir / "country_regions.csv",
        "domains": data_dir / "domains.csv",
        "input": data_dir / "Input_scores.csv"
    }

    final_df = pd.read_csv(paths["final"]) if paths["final"].exists() else None
    ind_df = pd.read_csv(paths["indicator"]) if paths["indicator"].exists() else None
    reg_df = pd.read_csv(paths["regions"]) if paths["regions"].exists() else None
    dom_map = pd.read_csv(paths["domains"]) if paths["domains"].exists() else None
    input_df = pd.read_csv(paths["input"]) if paths["input"].exists() else None

    if final_df is None or ind_df is None:
        st.error("Fatal Error: `final_scores.csv` and `indicator_scores.csv` must be present.")
        st.stop()
    if COL_COUNTRY not in final_df.columns:
        st.error(f"Fatal Error: `final_scores.csv` must include a '{COL_COUNTRY}' column.")
        st.stop()

    score_to_show = next((c for c in SCORE_CANDIDATES if c in final_df.columns), None)
    if score_to_show is None:
        st.error(f"Fatal Error: No final score column found.")
        st.stop()

    if reg_df is not None and COL_COUNTRY in reg_df.columns and COL_REGION in reg_df.columns:
        final_df = final_df.merge(reg_df[[COL_COUNTRY, COL_REGION]], on=COL_COUNTRY, how="left")
    else:
        final_df[COL_REGION] = "Global"

    # --- UPDATED: New maturity ranges as requested ---
    def assign_maturity(score):
        if score >= 55:
            return "Mature"
        elif score > 20: # Changed from 25
            return "Advancing"
        else: # Now <= 20
            return "Nascent"

    final_df['Maturity'] = final_df[score_to_show].apply(assign_maturity)

    return {
        "final_df": final_df, "ind_df": ind_df, "dom_map": dom_map,
        "input_df": input_df, "score_to_show": score_to_show
    }

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
final_df, ind_df, dom_map, input_df, score_to_show = (
    data["final_df"], data["ind_df"], data["dom_map"],
    data["input_df"], data["score_to_show"]
)

# --- NEW: Define a consistent color palette for comparison charts ---
COMPARISON_COLORS = ["#054b81", "#FF7433", "#59b0F2", "#29A0B1", "#686868"]

# --- Sidebar ---
st.sidebar.header("Filters")
regions_list = sorted(final_df[COL_REGION].dropna().unique().tolist())
global_on = st.sidebar.checkbox("Global View", value=True)
selected_regions = regions_list if global_on else st.sidebar.multiselect(
    "Select Regions", regions_list, default=regions_list
)
df = final_df[final_df[COL_REGION].isin(selected_regions)].copy()

# --- Page Title ---
st.title("Ecosystem Maturity Index")

# --- GRAPH 1 & 2: Leaderboard and Pie Chart ---
st.subheader("Ecosystem Maturity Breakdown")
col1, col2 = st.columns([3, 1])

with col1:
    rank = df[[COL_COUNTRY, score_to_show, 'Maturity']].dropna().sort_values(
        score_to_show, ascending=False
    ).reset_index(drop=True)
    
    if rank.empty:
        st.info("No countries to display. Adjust filters.")
    else:
        fig = px.bar(rank, x=COL_COUNTRY, y=score_to_show, color_discrete_sequence=['#054b81'])
        n = len(rank)
        
        # --- THIS LOGIC IS NOW UPDATED ---
        # Find the last index (position) for each maturity group
        mature_indices = rank.index[rank['Maturity'] == 'Mature'].tolist()
        advancing_indices = rank.index[rank['Maturity'] == 'Advancing'].tolist()

        last_mature_idx = mature_indices[-1] if mature_indices else -1
        last_advancing_idx = advancing_indices[-1] if advancing_indices else last_mature_idx

        # We now use the bar indices directly for the coordinates (e.g., from bar -0.5 to 10.5)
        # This is more reliable than using 'paper' for categorical axes.
        
        # Add Mature background area
        if last_mature_idx > -1:
            fig.add_vrect(x0=-0.5, x1=last_mature_idx + 0.5, 
                          fillcolor="#ff7433", opacity=0.5, line_width=0, layer="below")
            fig.add_annotation(x=(last_mature_idx - 0.5) / 2, y=rank[score_to_show].max(),
                               text="<b>Mature</b>", showarrow=False, yshift=10)

        # Add Advancing background area
        if last_advancing_idx > last_mature_idx:
            fig.add_vrect(x0=last_mature_idx + 0.5, x1=last_advancing_idx + 0.5, 
                          fillcolor="#59b0F2", opacity=0.5, line_width=0, layer="below")
            fig.add_annotation(x=(last_mature_idx + last_advancing_idx) / 2 + 0.5, y=rank[score_to_show].max(),
                               text="<b>Advancing</b>", showarrow=False, yshift=10)
        
        # Add Nascent background area
        if last_advancing_idx < n - 1:
            fig.add_vrect(x0=last_advancing_idx + 0.5, x1=n - 0.5, 
                          fillcolor="#0865AC", opacity=0.5, line_width=0, layer="below")
            fig.add_annotation(x=(last_advancing_idx + n) / 2, y=rank[score_to_show].max(),
                               text="<b>Nascent</b>", showarrow=False, yshift=10)

        # Make the plot background transparent so the vrects show through
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)') 
        
        fig.update_layout(xaxis_title=None, yaxis_title=score_to_show.replace("_", " "),
                          margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        fig.update_xaxes(tickangle=-75)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # --- THIS IS THE FIX ---
    # Wrap the chart in a container to prevent ID conflicts
    with st.container():
        if not rank.empty:
            pie_data = rank.groupby('Maturity')[COL_COUNTRY].count().reset_index()
            fig_pie = px.pie(
                pie_data, values=COL_COUNTRY, names='Maturity',
                hole=0.5,
                color='Maturity',
                color_discrete_map={'Mature': '#ff7433', 'Advancing': '#59b0F2', 'Nascent': '#0865AC'}
            )
            fig_pie.update_layout(
                annotations=[dict(text='Maturity', x=0.5, y=0.5, font_size=20, showarrow=False)],
                showlegend=True, margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- GRAPH 3: Bubble Chart with Maturity Backgrounds ---
st.subheader("High Growth Companies vs. Index Score")
hg_col = try_find_hg_col(final_df) or try_find_hg_col(input_df)
if hg_col is None:
    st.info("No High Growth Companies column found in data files.")
else:
    if 'COUNTRY' not in (input_df.columns if input_df is not None else []) and 'Country' in (input_df.columns if input_df is not None else []):
        input_df = input_df.rename(columns={'Country': 'COUNTRY'})
    bubble_df = df.merge(input_df[[COL_COUNTRY, hg_col]], on=COL_COUNTRY, how="left") if hg_col in (input_df.columns if input_df is not None else []) else df
    bubble_df = bubble_df[[COL_COUNTRY, score_to_show, 'Maturity', hg_col]].dropna()

    if bubble_df.empty:
        st.info("Not enough data for the bubble chart.")
    else:
        bubble_df["_size"] = np.sqrt(bubble_df[hg_col].clip(lower=1))
        bubble_df["_x_log"] = bubble_df[score_to_show] + 0.1

        fig_sc = px.scatter(
            bubble_df, x="_x_log", y=hg_col, size="_size", color="Maturity",
            color_discrete_map={'Mature': '#ff7433', 'Advancing': '#59b0F2', 'Nascent': '#0865AC'},
            hover_name=COL_COUNTRY, size_max=60, log_x=True, log_y=True,
            labels={"_x_log": "Index Score (Log Scale)", hg_col: "High Growth Companies (Log Scale)"}
        )
        
        # --- UPDATED: Checkbox to show maturity plot areas ---
        if st.checkbox("Show Maturity Areas", value=True):
            min_score, max_score = bubble_df["_x_log"].min(), bubble_df["_x_log"].max()
            fig_sc.add_vrect(x0=min_score, x1=20, fillcolor="#0865AC", opacity=0.2, line_width=0, layer="below")
            fig_sc.add_vrect(x0=20, x1=55, fillcolor="#59b0F2", opacity=0.2, line_width=0, layer="below")
            fig_sc.add_vrect(x0=55, x1=max_score, fillcolor="#ff7433", opacity=0.2, line_width=0, layer="below")

        fig_sc.update_traces(marker_line_width=0.5, marker_line_color='black')
        fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Maturity")
        st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# --- UNIFIED DEEP-DIVE SECTION (No changes here) ---
st.subheader("Country Comparison Deep-Dive")
top_countries = df.sort_values(score_to_show, ascending=False)[COL_COUNTRY].head(3).tolist()
comparison_countries = st.multiselect(
    "Select up to 5 countries to compare:",
    options=sorted(df[COL_COUNTRY].unique().tolist()),
    default=top_countries, max_selections=5, key="country_comparator"
)
# --- Radar Chart ---
st.markdown("##### Domain Profile (Radar)")
domain_cols = [c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]

if not domain_cols:
    st.info("No `DOMAIN_SCORE__*` columns found.")
elif not comparison_countries:
    st.caption("Select countries to view the radar chart.")
else:
    # Create a color map for the selected countries
    color_map = dict(zip(comparison_countries, COMPARISON_COLORS))
    
    radar_labels = [c.replace("DOMAIN_SCORE__", "") for c in domain_cols]
    radar_fig = go.Figure()
    
    for country in comparison_countries:
        vals = df.loc[df[COL_COUNTRY] == country, domain_cols].iloc[0].tolist()
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, 
            theta=radar_labels, 
            fill='toself', 
            name=country,
            line=dict(color=color_map.get(country)) # Apply color here
        ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(radar_fig, use_container_width=True)

st.markdown("##### Indicator Drill-Down")
if dom_map is None or ind_df is None:
    st.info("Add `domains.csv` and `indicator_scores.csv` to enable this section.")
elif not comparison_countries:
    st.caption("Select countries to view indicator scores.")
else:
    if 'COUNTRY' not in ind_df.columns and 'Country' in ind_df.columns:
        ind_df = ind_df.rename(columns={'Country': 'COUNTRY'})
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
            chosen_inds = st.multiselect("Choose up to 3 indicators:", options=d_inds_available,
                                         default=d_inds_available[:3], max_selections=3, key=f"inds_{dname}")
            if not chosen_inds:
                st.caption("Select at least one indicator to plot.")
                continue
            plot_df = work_df[work_df[COL_COUNTRY].isin(comparison_countries)][[COL_COUNTRY] + chosen_inds]
            melt_df = plot_df.melt(id_vars=[COL_COUNTRY], var_name="Indicator", value_name="Score")
            order = melt_df.groupby(COL_COUNTRY)["Score"].sum().sort_values(ascending=False).index
            fig_bars = px.bar(melt_df, y=COL_COUNTRY, x="Score", color=COL_COUNTRY,
                              barmode="group", text="Score", facet_col="Indicator",
                              category_orders={COL_COUNTRY: order})
            fig_bars.update_traces(texttemplate="%{text:.0f}", textposition="outside", width=0.6)
            fig_bars.update_layout(bargroupgap=0.15, yaxis_title="", xaxis_title="", legend_title_text="",
                                   showlegend=False, margin=dict(r=5, l=5, t=30, b=5))
            fig_bars.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_bars.for_each_xaxis(lambda axis: axis.update(title="", range=[0, 105]))
            st.plotly_chart(fig_bars, use_container_width=True)

# --- Footer ---
with st.expander("About This Dashboard"):
    st.markdown(f"""
    - **Score shown:** `{score_to_show}`
    - **Maturity bands:**
        - **Mature:** Score >= 55
        - **Advancing:** Score > 20 and < 55
        - **Nascent:** Score <= 20
    - **Files Used:** `final_scores.csv`, `indicator_scores.csv`, `country_regions.csv`, `domains.csv`, and `Input_scores.csv`.
    """)

# --- Chatbot (Lean MVP) ---
import os
from typing import Literal

st.divider()
with st.expander("Chat (beta)", expanded=False):

    # 1) Session state for chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi! I can answer questions about the Ecosystem Maturity Index. Ask me about countries, domains, indicators, or say 'plot top 10'."}
        ]

    # 2) Small “tool” layer the model can request (we invoke these in Python)
    def tool_summarize_view():
        return {
            "rows": len(df),
            "score": data["score_to_show"],
            "regions": sorted(df[COL_REGION].dropna().unique().tolist()),
            "countries": len(df[COL_COUNTRY].unique())
        }

    def tool_topk(metric: str, k: int = 10, region: str | None = None):
        _metric = metric if metric in df.columns else data["score_to_show"]
        d = df if region is None else df[df[COL_REGION] == region]
        res = d[[COL_COUNTRY, _metric]].dropna().sort_values(_metric, ascending=False).head(k)
        return res.to_dict(orient="records")

    def tool_plot_topk(metric: str, k: int = 10):
        _metric = metric if metric in df.columns else data["score_to_show"]
        top = df[[COL_COUNTRY, _metric]].dropna().sort_values(_metric, ascending=False).head(k)
        if top.empty:
            st.info("Nothing to plot for that request.")
            return
        fig = px.bar(top, x=COL_COUNTRY, y=_metric, color_discrete_sequence=['#054b81'])
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False, xaxis_title=None, yaxis_title=_metric)
        fig.update_xaxes(tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)

    # 3) A tiny intent router (no external libs)
    def parse_intent(text: str) -> dict:
        t = text.lower().strip()
        if t.startswith("plot"):
            # examples: "plot top 10", "plot top 5 by Final_Score_0_100"
            import re
            k = 10
            m_k = re.search(r"top\s+(\d+)", t)
            if m_k: k = int(m_k.group(1))
            # try to extract metric by column name mention
            metric = data["score_to_show"]
            for c in df.columns:
                if c.lower() in t:
                    metric = c
                    break
            return {"intent": "plot_topk", "metric": metric, "k": k}
        if "top" in t:
            import re
            k = 10
            m_k = re.search(r"top\s+(\d+)", t)
            if m_k: k = int(m_k.group(1))
            metric = data["score_to_show"]
            for c in df.columns:
                if c.lower() in t:
                    metric = c
                    break
            # optional region mention
            region = None
            for r in df[COL_REGION].dropna().unique():
                if r.lower() in t:
                    region = r
                    break
            return {"intent": "topk", "metric": metric, "k": k, "region": region}
        if "summary" in t or "summarize" in t or "overview" in t:
            return {"intent": "summarize"}
        return {"intent": "chat"}  # default: general chat

    # 4) Simple model response helper (provider-agnostic)
    def llm_reply(system: str, user: str) -> str:
        # Minimal fallback if no API key: a local canned response
        if not os.getenv("OPENAI_API_KEY"):
            return "I’m running in local mode. I can run the tools (filters, top-k, simple plots). For richer natural language answers, set OPENAI_API_KEY."
        # OpenAI example (swap with your preferred provider)
        from openai import OpenAI
        client = OpenAI()
        msg = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        return msg.choices[0].message.content

    # 5) Render history
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 6) Handle new input
    if prompt := st.chat_input("Ask about the current view (respects sidebar filters)…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # establish system context with safe metadata (not raw CSV)
        system = (
            "You are a helpful data copilot for a Streamlit dashboard. "
            "Only talk about columns the app has: "
            f"{list(final_df.columns)}. Current score column: {data['score_to_show']}. "
            "When asked for results, prefer concise bullet points. "
            "If a user asks for a plot, the app will render it; you should just explain what will be shown."
        )

        intent = parse_intent(prompt)

        with st.chat_message("assistant"):
            if intent["intent"] == "summarize":
                info = tool_summarize_view()
                reply = (
                    f"Current view has **{info['rows']} rows** across **{info['countries']} countries**, "
                    f"score column is **{info['score']}**. Regions visible: {', '.join(info['regions'])}."
                )
                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

            elif intent["intent"] == "topk":
                rows = tool_topk(intent["metric"], intent["k"], intent.get("region"))
                if not rows:
                    out = "No results for that request."
                else:
                    lines = [f"{i+1}. {r[COL_COUNTRY]} — {list(r.values())[1]}" for i, r in enumerate(rows)]
                    out = "**Top results:**\n\n" + "\n".join(lines)
                st.markdown(out)
                st.session_state.chat_messages.append({"role": "assistant", "content": out})

            elif intent["intent"] == "plot_topk":
                # text response + actual chart
                out = f"Plotting top {intent['k']} by **{intent['metric']}** for the current filtered view."
                st.markdown(out)
                tool_plot_topk(intent["metric"], intent["k"])
                st.session_state.chat_messages.append({"role": "assistant", "content": out})

            else:
                # general chat – optional LLM call if key set
                reply = llm_reply(system, prompt)
                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})