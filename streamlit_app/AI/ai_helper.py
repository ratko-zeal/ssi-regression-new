# ai_helper.py
import os
import re
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Public API
# ----------------------------
def render_ai_sidebar_chat(
    *,
    final_df: pd.DataFrame,
    ind_df: Optional[pd.DataFrame],
    dom_map: Optional[pd.DataFrame],
    view_df: pd.DataFrame,              # already-filtered df from your sidebar (the "current view")
    score_to_show: str,                 # e.g., "Final_Score_0_100"
    col_country: str = "COUNTRY",
    col_region: str = "REGION",
    domain_prefix: str = "DOMAIN_SCORE__",
    allow_global_prefix: bool = True,   # allow "global:" to use full dataset
    sidebar_title: str = "Chat (beta)"
) -> None:
    """
    Renders a compact chatbot into the Streamlit sidebar that:
      1) "Sees" the current view (view_df)
      2) Can optionally access the full data (final_df) via "global:" prefix
      3) Answers "why is the <domain> score for <country> so low?"
    """
    _ensure_state_keys()

    # Build domain column maps
    domain_cols = [c for c in final_df.columns if c.startswith(domain_prefix)]
    domain_name_map = {c: c.replace(domain_prefix, "") for c in domain_cols}
    domain_name_to_col = {v.lower(): k for k, v in domain_name_map.items()}

    # Build country registry and regex for extraction
    all_countries = sorted(final_df[col_country].dropna().unique().tolist())
    country_regex = r"|".join([re.escape(c.lower()) for c in all_countries])

    # Prepare a tiny context + signature for provenance
    ctx, signature = _make_view_context(
        view_df=view_df,
        score_col=score_to_show,
        is_global=False,
        regions_selected=sorted(view_df[col_region].dropna().unique().tolist()) if col_region in view_df.columns else ["Global"],
        col_country=col_country
    )

    st.sidebar.divider()
    with st.sidebar.expander(sidebar_title, expanded=False):
        # Header showing current scope
        scope_line = f"**Scope:** {len(view_df)} rows • {view_df[col_country].nunique()} countries • Score: `{score_to_show}`"
        st.sidebar.caption(scope_line)

        # Render recent history
        for role, content in st.session_state.ai_chat_history[-10:]:
            st.sidebar.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {content}")

        # Input controls
        user_q = st.sidebar.text_area("Type your question…", key="ai_chat_input", height=80)
        col_sb1, col_sb2 = st.sidebar.columns([1, 1], gap="small")
        send = col_sb1.button("Send", use_container_width=True, key="ai_btn_send")
        clear = col_sb2.button("Clear", use_container_width=True, key="ai_btn_clear")

        if clear:
            st.session_state.ai_chat_history = []

        if send and user_q.strip():
            st.session_state.ai_chat_history.append(("user", user_q.strip()))

            # Detect optional "global:" prefix
            text = user_q.strip()
            force_global = allow_global_prefix and text.lower().startswith("global:")
            use_view = not force_global
            question_core = text[7:].strip() if force_global else text

            # Light NLP extraction
            country = _extract_country(question_core, all_countries, country_regex)
            domain = _extract_domain(question_core, domain_name_to_col)

            if country and domain:
                facts = _explain_low_domain(
                    country=country,
                    domain_name=domain,
                    use_view=use_view,
                    view_df=view_df,
                    final_df=final_df,
                    ind_df=ind_df,
                    dom_map=dom_map,
                    col_country=col_country,
                    col_region=col_region,
                    domain_name_to_col=domain_name_to_col
                )
                if "error" in facts:
                    reply = f"Sorry — {facts['error']}"
                else:
                    reply = _llm_explain(question_core, facts)
                    # Scope note
                    if use_view:
                        reply += f"\n\n_Scope: current filters (sig: `{signature}`). Score: **{score_to_show}**._"
                    else:
                        reply += f"\n\n_Scope: global (all data). Score: **{score_to_show}**._"
            else:
                reply = _help_message(score_to_show)

            st.session_state.ai_chat_history.append(("assistant", reply))
            st.experimental_rerun()


# ----------------------------
# Internal helpers
# ----------------------------
def _ensure_state_keys():
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            ("assistant", "Hi! Ask me about countries, domains, and why scores are high/low.\n"
                          "Example: *why is the Access domain score for Croatia so low?*\n"
                          "Use `global:` prefix to widen scope.")
        ]

def _make_view_context(view_df, score_col, is_global, regions_selected, col_country):
    ctx = {
        "is_global": is_global,
        "regions": regions_selected,
        "score": score_col,
        "n_rows": int(len(view_df)),
        "countries": sorted(view_df[col_country].dropna().unique().tolist())[:50],
    }
    sig = hashlib.sha1(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:10]
    return ctx, sig

def _extract_country(text: str, all_countries: list[str], country_regex: str) -> Optional[str]:
    t = text.lower()
    m = re.search(rf"\b({country_regex})\b", t)
    if not m:
        return None
    hit = m.group(1)
    for c in all_countries:
        if c.lower() == hit:
            return c
    return None

def _extract_domain(text: str, domain_name_to_col: dict[str, str]) -> Optional[str]:
    t = text.lower()
    for dname in domain_name_to_col.keys():
        if re.search(rf"\b{re.escape(dname)}\b", t):
            return dname
    # loose fallback like "domain score for Access"
    m = re.search(r"domain\s+score\s+for\s+([a-zA-Z][a-zA-Z\s]+)", t)
    if m:
        guess = m.group(1).strip().lower()
        for dname in domain_name_to_col.keys():
            if dname in guess or guess in dname:
                return dname
    return None

def _indicators_for_domain(domain_name: str, dom_map: Optional[pd.DataFrame], col_domain: str = "DOMAIN", col_indicator: str = "INDICATOR") -> list[str]:
    if dom_map is None or col_domain not in dom_map.columns or col_indicator not in dom_map.columns:
        return []
    return dom_map.loc[dom_map[col_domain] == domain_name, col_indicator].dropna().unique().tolist()

def _explain_low_domain(
    *,
    country: str,
    domain_name: str,
    use_view: bool,
    view_df: pd.DataFrame,
    final_df: pd.DataFrame,
    ind_df: Optional[pd.DataFrame],
    dom_map: Optional[pd.DataFrame],
    col_country: str,
    col_region: str,
    domain_name_to_col: dict[str, str]
) -> dict:
    notes = []
    domain_col = domain_name_to_col.get(domain_name.lower())
    if not domain_col:
        return {"error": f"Unknown domain '{domain_name}'."}

    if country not in final_df[col_country].values:
        return {"error": f"Country '{country}' not found in data."}

    row_full = final_df.loc[final_df[col_country] == country]
    if row_full.empty or domain_col not in row_full.columns or pd.isna(row_full.iloc[0][domain_col]):
        return {"error": f"No domain score found for {country} in '{domain_name}'."}
    country_score = float(row_full.iloc[0][domain_col])

    # view stats
    if use_view and domain_col in view_df.columns and not view_df.empty:
        view_series = view_df[domain_col].dropna()
        view_avg = float(view_series.mean()) if not view_series.empty else None
        rank_df = view_df[[col_country, domain_col]].dropna().sort_values(domain_col, ascending=False)
        n_view = len(rank_df)
        rank_pos = int(rank_df.index[rank_df[col_country] == country][0]) + 1 if country in rank_df[col_country].values else None
    else:
        view_avg, n_view, rank_pos = None, None, None

    # global stats
    global_series = final_df[domain_col].dropna()
    global_avg = float(global_series.mean()) if not global_series.empty else None

    # indicators
    weakest = []
    missing_inds = []
    inds = _indicators_for_domain(domain_name, dom_map)
    if inds and ind_df is not None:
        _ind = ind_df.copy()
        if col_country not in _ind.columns and "Country" in _ind.columns:
            _ind = _ind.rename(columns={"Country": col_country})
        inds_present = [i for i in inds if i in _ind.columns]
        if inds_present:
            ind_view = _ind.merge(view_df[[col_country]], on=col_country, how="inner") if use_view and not view_df.empty else None
            ind_country = _ind.loc[_ind[col_country] == country]
            if not ind_country.empty:
                row = ind_country.iloc[0]
                rows = []
                for ind in inds_present:
                    val = row[ind] if ind in row.index else np.nan
                    if pd.isna(val):
                        missing_inds.append(ind); continue
                    vmean = float(ind_view[ind].dropna().mean()) if (ind_view is not None and ind in (ind_view.columns if ind_view is not None else [])) else None
                    gmean = float(_ind[ind].dropna().mean())
                    gap_v = (val - vmean) if vmean is not None else None
                    gap_g = (val - gmean) if gmean is not None else None
                    rows.append((ind, float(val), vmean, gmean, gap_v, gap_g))
                rows.sort(key=lambda x: (x[4] if x[4] is not None else x[5] if x[5] is not None else 0))
                weakest = rows[:5]

    if view_avg is not None and country_score < view_avg - 1e-9:
        notes.append(f"{country} is below view average ({country_score:.1f} < {view_avg:.1f}).")
    if global_avg is not None and country_score < global_avg - 1e-9:
        notes.append(f"{country} is below global average ({country_score:.1f} < {global_avg:.1f}).")
    if not notes:
        notes.append("The score is not notably below the selected comparison averages.")

    return {
        "country": country,
        "domain_display": domain_name,
        "domain_col": domain_col,
        "country_score": country_score,
        "view_avg": view_avg,
        "global_avg": global_avg,
        "rank_in_view": rank_pos,
        "n_countries_view": n_view,
        "weakest_indicators": weakest,      # list of (indicator, val, vmean, gmean, gap_v, gap_g)
        "missing_indicators": missing_inds,
        "notes": notes
    }

def _llm_explain(prompt: str, facts: dict) -> str:
    # Build a compact factual context
    lines = []
    lines.append(f"Country: {facts.get('country')}")
    lines.append(f"Domain: {facts.get('domain_display')}")
    lines.append(f"Score: {facts.get('country_score')}")
    if facts.get("view_avg") is not None:   lines.append(f"ViewAvg: {facts['view_avg']}")
    if facts.get("global_avg") is not None: lines.append(f"GlobalAvg: {facts['global_avg']}")
    if facts.get("rank_in_view") is not None:
        lines.append(f"RankInView: {facts['rank_in_view']} of {facts.get('n_countries_view')}")
    if facts.get("weakest_indicators"):
        lines.append("Weakest indicators (val | view | global):")
        for (ind, val, vmean, gmean, _, _) in facts["weakest_indicators"]:
            vtxt = f"{vmean:.1f}" if vmean is not None else "—"
            lines.append(f"- {ind}: {val:.1f} | {vtxt} | {gmean:.1f}")
    if facts.get("missing_indicators"):
        lines.append("Missing indicators: " + ", ".join(facts["missing_indicators"]))
    if facts.get("notes"):
        lines.append("Notes: " + " ".join(facts["notes"]))
    ctx = "\n".join(lines)

    # Local fallback (no API key): return concise bullets
    if not os.getenv("OPENAI_API_KEY"):
        bullets = []
        bullets.append(f"- **{facts.get('domain_display')}** score for **{facts.get('country')}** is **{facts.get('country_score'):.1f}**.")
        if facts.get("view_avg") is not None:
            bullets.append(f"- View average: **{facts['view_avg']:.1f}**.")
        if facts.get("global_avg") is not None:
            bullets.append(f"- Global average: **{facts['global_avg']:.1f}**.")
        if facts.get("rank_in_view") is not None:
            bullets.append(f"- Rank in view: **{facts['rank_in_view']} / {facts.get('n_countries_view')}**.")
        if facts.get("weakest_indicators"):
            bullets.append("- Likely drivers (lower relative indicators):")
            for (ind, val, vmean, gmean, _, _) in facts["weakest_indicators"]:
                vtxt = f"{vmean:.1f}" if vmean is not None else "—"
                bullets.append(f"  • {ind}: {val:.1f} vs view {vtxt}, global {gmean:.1f}")
        if facts.get("missing_indicators"):
            bullets.append(f"- Missing indicator values: {', '.join(facts['missing_indicators'])}.")
        bullets.extend([f"- {n}" for n in facts.get("notes", [])])
        return "\n".join(bullets)

    try:
        from openai import OpenAI
        client = OpenAI()
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a helpful data copilot. Use ONLY the provided facts. Be concise, bullet points preferred."},
                {"role": "user", "content": f"User prompt: {prompt}\n\nFacts:\n{ctx}"}
            ]
        )
        return rsp.choices[0].message.content
    except Exception as e:
        return f"(LLM unavailable) {str(e)}"

def _help_message(score_to_show: str) -> str:
    return (
        "I can explain domain scores and surface likely drivers (indicators).\n\n"
        "Examples:\n"
        f"- why is the Access domain score for Croatia so low?\n"
        f"- compare Croatia vs Slovenia on Talent\n"
        f"- show top 10 countries by {score_to_show}\n"
        "- global: average Final_Score_0_100"
    )