# TOOL_GUIDE

## get_growth_scores(country, year?, domains?)
- **Use for:** any numeric about final score or domain display scores.
- **Returns:** {country, as_of_date, version, final_score, display_domain_scores{domain:score}}
- **Pitfalls:** year currently ignored (no time dimension). Domain scores are rank-adjusted display values.

## get_current_view_state(session_id)
- **Use for:** “about what I see” questions.
- **Returns:** {country, domains, year}

## get_indicator_scores(country, indicators?)
- **Use for:** numeric indicator values (normalized). OK to display to users.
- **Returns:** {country, indicator_scores{indicator:score}}
- **Pitfalls:** Ensure indicator names match exactly.

## prioritize_focus(country, indicators[])
- **Use for:** “which indicators should we focus on?”
- **Logic:** Internal impact = blended_weight × (100 - current_score)
- **Returns:** ordered recommendations with tier (High/Medium/Low) + rationale (no raw weights).

## file_search(query)
- **Use for:** pulling qualitative context from `streamlit_app/knowledge` (PDF/TXT/MD supported).
- **Returns:** snippets + filesystem citations.

## web_search(query)
- **Use for:** external facts only if user explicitly asks. Include citations. Disabled by default.
