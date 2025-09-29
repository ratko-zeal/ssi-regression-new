import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from AI.ai_helper import get_ai_response

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

    # --- ADD THIS BLOCK to standardize all loaded dataframes ---
    for df in [final_df, ind_df, reg_df, dom_map, input_df]:
        standardize_columns(df)

    if final_df is None or ind_df is None:
        st.error("Fatal Error: `final_scores.csv` and `indicator_scores.csv` must be present.")
        st.stop()

    if reg_df is not None and COL_COUNTRY in reg_df.columns and COL_REGION in reg_df.columns:
        final_df = final_df.merge(reg_df[[COL_COUNTRY, COL_REGION]], on=COL_COUNTRY, how="left")
    else:
        final_df[COL_REGION] = "Global"

    if COL_COUNTRY not in final_df.columns:
        st.error(f"Fatal Error: `final_scores.csv` must include a '{COL_COUNTRY}' column.")
        st.stop()

    score_to_show = next((c for c in SCORE_CANDIDATES if c in final_df.columns), None)
    if score_to_show is None:
        st.error(f"Fatal Error: No final score column found.")
        st.stop()

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

def standardize_columns(df: pd.DataFrame):
    """
    Standardizes column names to uppercase for consistent access.
    e.g., 'Country' or 'country' becomes 'COUNTRY'.
    """
    if df is None:
        return

    rename_map = {}
    # Create a map of {current_col: standardized_col}
    for col in df.columns:
        std_col = col.upper().strip()
        # Map all predefined constant columns to their standard form
        if std_col in [COL_COUNTRY, COL_REGION, COL_INDICATOR, COL_DOMAIN]:
            rename_map[col] = std_col

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


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

# 1. Initialize session state for the filter logic
if 'initial_load_complete' not in st.session_state:
    st.session_state.initial_load_complete = False
    st.session_state.last_selected_regions = []
    st.session_state.comparison_countries = [] # Start with an empty selection

# --- Region Filter ---
regions_list = sorted(final_df[COL_REGION].dropna().unique().tolist())
global_on = st.sidebar.checkbox("Global View", value=True)
selected_regions = regions_list if global_on else st.sidebar.multiselect(
    "Select Regions", regions_list, default=regions_list
)
df = final_df[final_df[COL_REGION].isin(selected_regions)].copy()
available_countries = sorted(df[COL_COUNTRY].unique().tolist())

# --- Country Filter Logic ---
# This logic now runs only AFTER the initial page load
if st.session_state.initial_load_complete:
    region_has_changed = (st.session_state.last_selected_regions != selected_regions)
    if region_has_changed:
        # If countries were already selected, filter them to what's available
        # If the list is empty, this condition is false and nothing happens.
        new_selection = [
            c for c in st.session_state.comparison_countries if c in available_countries
        ]
        st.session_state.comparison_countries = new_selection

# Display the country multiselect widget
comparison_countries = st.sidebar.multiselect(
    "Select Countries",
    options=available_countries,
    default=st.session_state.comparison_countries,
    key="country_comparator"
)

# Update session state for the next run
st.session_state.comparison_countries = comparison_countries
st.session_state.last_selected_regions = selected_regions
st.session_state.initial_load_complete = True

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
        # --- UPDATED: Add a column for highlighting and apply new colors ---
        rank['Highlight'] = np.where(rank[COL_COUNTRY].isin(comparison_countries), 'Selected', 'Other')
        fig = px.bar(
            rank, 
            x=COL_COUNTRY, 
            y=score_to_show,
            color='Highlight',
            color_discrete_map={
                'Selected': '#ff7433', # New highlight color
                'Other': '#054b81'      # Original color for other bars
            },
            # --- THIS IS THE FIX ---
            # Enforce the original sort order of the countries
            category_orders={COL_COUNTRY: rank[COL_COUNTRY].tolist()}
        )
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

        # The tickfont color cannot be set individually, so we remove the line causing the error.
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
    bubble_df = df.merge(input_df[[COL_COUNTRY, hg_col]], on=COL_COUNTRY, how="left") if hg_col in (input_df.columns if input_df is not None else []) else df
    bubble_df = bubble_df[[COL_COUNTRY, score_to_show, 'Maturity', hg_col]].dropna()

    if bubble_df.empty:
        st.info("Not enough data for the bubble chart.")
    else:
        bubble_df["_size"] = np.sqrt(bubble_df[hg_col].clip(lower=1))
        bubble_df["_x_log"] = bubble_df[score_to_show] + 0.1

        maturity_color_map = {'Mature': '#ff7433', 'Advancing': '#59b0F2', 'Nascent': '#0865AC'}      

        fig_sc = px.scatter(
            bubble_df, x="_x_log", y=hg_col, size="_size", color="Maturity",
            color_discrete_map=maturity_color_map,
            hover_name=COL_COUNTRY, size_max=60, log_x=True, log_y=True,
            labels={"_x_log": "Index Score (Log Scale)", hg_col: "High Growth Companies (Log Scale)"}
        )
        
        # --- UPDATED: Checkbox to show maturity plot areas ---
        if st.checkbox("Show Maturity Areas", value=True):
            min_score, max_score = bubble_df["_x_log"].min(), bubble_df["_x_log"].max()
            fig_sc.add_vrect(x0=min_score, x1=20, fillcolor="#0865AC", opacity=0.2, line_width=0, layer="below")
            fig_sc.add_vrect(x0=20, x1=55, fillcolor="#59b0F2", opacity=0.2, line_width=0, layer="below")
            fig_sc.add_vrect(x0=55, x1=max_score, fillcolor="#ff7433", opacity=0.2, line_width=0, layer="below")

        # --- NEW: Loop through chart data to apply conditional highlighting ---
        # This approach preserves the color-by-maturity while highlighting selections.    
        for trace in fig_sc.data:
            # Get the country names for the points in the current trace (e.g., all 'Mature' countries)
            countries_in_trace = trace.hovertext
            # Get the original fill color for this trace (e.g., the color for 'Mature')
            original_color = maturity_color_map[trace.name]

            # --- UPDATED LOGIC ---
            # Create a list of border colors: black for selected, or the original color for others
            line_colors = ['black' if c in comparison_countries else original_color for c in countries_in_trace]

            # Set border width: 2 for selected countries, 1 for others (2x thicker)
            line_widths = [2 if c in comparison_countries else 1 for c in countries_in_trace]

            # Update the trace with the new marker line properties
            trace.marker.line.color = line_colors
            trace.marker.line.width = line_widths
            trace.marker.opacity = 0.6 # Apply a consistent transparency to all bubbles


        fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Maturity")
        
        # --- ADD THIS CODE right before st.plotly_chart(fig_sc, ...) ---

        # Check if any countries are selected
        if comparison_countries:
            # Create a nicely formatted string of the selected countries
            selected_countries_str = ", ".join(f"<b>{c}</b>" for c in comparison_countries)
            # Display the list using st.markdown
            st.markdown(f"**Highlighted Countries:** {selected_countries_str}", unsafe_allow_html=True)
        else:
            # Show a default message if no countries are selected
            st.caption("Select countries in the sidebar to highlight them.")

        # This is where you would display the chart
        st.plotly_chart(fig_sc, use_container_width=True)
    

st.divider()

# --- UNIFIED DEEP-DIVE SECTION (No changes here) ---
st.subheader("Country Comparison Deep-Dive")
# --- Radar Chart ---
st.markdown("##### Domain Profile (Radar)")
domain_cols = [c for c in final_df.columns if c.startswith("DOMAIN_SCORE__")]

if not domain_cols:
    st.info("No `DOMAIN_SCORE__*` columns found.")
elif not comparison_countries:
    st.caption("Select countries to view the radar chart.")
else:
    # --- ADD THIS BLOCK to calculate rank-percentile scores for the radar chart ---
    # This avoids the "USA problem" where one outlier squishes the 0-100 scale.
    radar_df = df[[COL_COUNTRY] + domain_cols].copy()
    for col in domain_cols:
        # Calculate rank (highest is rank 1), then convert to percentile
        # .rank(pct=True) gives percentile from 0-1, so we multiply by 100
        radar_df[col] = radar_df[col].rank(method='min', ascending=False).apply(lambda x: 100 * (1 - (x - 1) / len(radar_df)))

    # Create a color map for the selected countries
    color_map = dict(zip(comparison_countries, COMPARISON_COLORS))
    
    radar_labels = [c.replace("DOMAIN_SCORE__", "") for c in domain_cols]
    
    radar_labels.append(radar_labels[0])
    
    radar_fig = go.Figure()
    
    for country in comparison_countries:
        vals = df.loc[df[COL_COUNTRY] == country, domain_cols].iloc[0].tolist()
        # Grab the percentile-ranked values for the plot
        vals = radar_df.loc[radar_df[COL_COUNTRY] == country, domain_cols].iloc[0].tolist()
        vals.append(vals[0])

        radar_fig.add_trace(go.Scatterpolar(
            r=vals, 
            theta=radar_labels, 
            fill='none', 
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
            
            color_map = dict(zip(comparison_countries, COMPARISON_COLORS))

            fig_bars = px.bar(
                melt_df, y=COL_COUNTRY, x="Score", color=COL_COUNTRY,
                barmode="group", text="Score", facet_col="Indicator",
                category_orders={COL_COUNTRY: order},
                color_discrete_map=color_map # This line applies your custom colors
            )
            fig_bars.update_traces(texttemplate="%{text:.0f}", textposition="outside", width=0.6)
            fig_bars.update_layout(bargroupgap=0.15, yaxis_title="", xaxis_title="", legend_title_text="",
                                   showlegend=False, margin=dict(r=5, l=5, t=30, b=5))
            fig_bars.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig_bars.for_each_xaxis(lambda axis: axis.update(title="", range=[0, 105]))
            st.plotly_chart(fig_bars, use_container_width=True)

# --- World Map Visualization ---
st.subheader("Global Perspective")

# 1. Prepare the data for the map
map_df = final_df.copy()

# 2. Create the expanded "Map Category" column for multi-level coloring
# This logic now prioritizes "Selected", then separates "Other", and finally uses "Maturity"
conditions = [
    map_df[COL_COUNTRY].isin(comparison_countries), # Highest priority: Is the country specifically selected?
    ~map_df[COL_REGION].isin(selected_regions)   # Next: Is the country outside the selected region?
]
choices = [
    "Selected", # Assign to selected countries
    "Other"     # Assign to countries not in the filtered region
]
# The default choice will be the country's actual maturity level
map_df["Map Category"] = np.select(conditions, choices, default=map_df['Maturity'])


# 3. Define the new, expanded color scheme
color_map = {
    # Specific highlight for selected countries
    "Selected": "#FFD700",  # A distinct gold color

    # Maturity colors (consistent with other charts)
    'Mature': '#ff7433',
    'Advancing': '#59b0F2',
    'Nascent': '#0865AC',

    # Neutral color for all other countries
    "Other": "#d4d4d8"
}

# 4. Create the choropleth map with the new coloring
fig_map = px.choropleth(
    map_df,
    locations=COL_COUNTRY,
    locationmode='country names',
    color="Map Category",          # Color countries based on our new 5-level category
    color_discrete_map=color_map,
    hover_name=COL_COUNTRY,
    hover_data={
        "Map Category": False,
        COL_COUNTRY: False,
        score_to_show: ':.0f'
    }
)

# 5. Style the map layout
fig_map.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    legend_title_text='Status / Maturity', # Updated legend title
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ),
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showframe=False,
        showcoastlines=False,
        projection_type='natural earth'
    )
)

# --- NEW: Loop through the map data to add a border highlight ---
for trace in fig_map.data:
    # Get the country names for the shapes in this trace
    countries_in_trace = trace.locations

    # Create a list of border colors: black for selected, transparent for others
    # We use the trace's own color for non-selected borders to make them blend in
    border_colors = ['black' if c in comparison_countries else 'rgba(0,0,0,0)' for c in countries_in_trace]

    # Create a list of border widths: thick for selected, thin for others
    border_widths = [2 if c in comparison_countries else 0.5 for c in countries_in_trace]

    # Update the trace with the new marker line properties
    trace.marker.line.color = border_colors
    trace.marker.line.width = border_widths


# 6. Display the map in your Streamlit app
st.plotly_chart(fig_map, use_container_width=True)

# --- AI Chat Section ---
st.subheader("🤖 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the maturity index..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- THIS BLOCK IS UPDATED ---
        # 1. Check if the secret is available
        if "openai" in st.secrets and "api_key" in st.secrets.openai:
            api_key = st.secrets.openai.api_key
            
            # 2. Create a placeholder for the "Thinking..." message
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            
            # 3. Call the AI helper function
            response_generator = get_ai_response(
                prompt=prompt,
                api_key=api_key,
                final_df=final_df,
                ind_df=ind_df,
                dom_map=dom_map,
                chat_history=st.session_state.messages
            )
            
            # 4. Stream the response into the placeholder, replacing "Thinking..."
            full_response = placeholder.write_stream(response_generator)
            
            # 5. Add the complete response to the session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Show an error if the key is not found
            st.error("OpenAI API key not found. Please add it to your Streamlit secrets to enable the AI assistant.")
            st.stop()

## --- Powered By Logos ---
st.sidebar.markdown("---")
st.sidebar.markdown("##### Powered by")

script_dir = Path(__file__).resolve().parent

# Display a single, centered logo directly in the sidebar
st.sidebar.image(
    # 👈 Change this filename if you want to use the other logo
    str(script_dir / "logos" / "Seedstars - Logo.png"),
    use_container_width=True
)

# --- CTA Section ---
# This creates a visual separator from the filters above
st.sidebar.markdown("---")

# Body text of the CTA
st.sidebar.markdown(
    "Explore entrepreneurial ecosystems mapping from ANDE"
)

# The actual button that links to your page
st.sidebar.link_button("View the Ecosystem Maps", "https://andeglobal.org/ecosystem-maps/") # 👈 Replace with your URL