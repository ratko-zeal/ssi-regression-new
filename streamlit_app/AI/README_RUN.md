# README — Minimal E2E Slice

## Prereqs
- Python 3.10+
- Install dependencies (from the repo root):

  ```bash
  pip install -r streamlit_app/requirements.txt
  ```

## Data locations
- Place the exported CSVs in `streamlit_app/data/`:
  - `final_scores.csv`
  - `indicator_scores.csv`
  - `domains.csv`
  - `indicator_impact_summary.csv`
- Derived AI tables are written to `streamlit_app/data/build/`.
- Optional knowledge articles (PDF/Markdown/TXT) can be dropped into `streamlit_app/knowledge/` to power the `file_search` tool.

You can override these defaults via `DATA_DIR` / `BUILD_DIR` environment variables.

## Build AI-facing tables
Run from the `streamlit_app` directory:

```bash
cd streamlit_app
python -m AI.builder
```

This produces:
- `data/build/final_scores.csv`
- `data/build/display_domain_scores.csv`
- `data/build/indicator_scores.csv`
- `data/build/indicator_weights.csv`
- `data/build/meta.json`

## Run smoke test

```bash
cd streamlit_app
python -m AI.smoke_test "What is the final score and weakest domains for Kenya?" --session abc123
```

Set environment variables as needed (optional):

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-5"
export ENABLE_WEB_SEARCH=false
```

## Integration
- Import `answer_question` from `AI.handler` to embed in your service or Streamlit app.
- Use `AI.tools.set_current_view_state(session_id, country, domains, year)` to keep UI state in sync.
