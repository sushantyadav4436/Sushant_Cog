# Fact-Check Agent

Upload a PDF and get a claim-by-claim report: **Verified**, **Inaccurate** (outdated or wrong figure, with the correct fact) or **False** (no evidence found).

## How it works
1. `pypdf` extracts the text.
2. Claude extracts self-contained, checkable claims (stats, dates, financial and technical figures) as JSON.
3. For each claim, Claude runs live web searches (Anthropic web search tool), then returns a verdict, the real fact and source URLs.
4. Streamlit shows the results, False and Inaccurate first, with a JSON download.

## Run locally
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. On share.streamlit.io choose the repo, main file `app.py`.
3. In Advanced settings > Secrets add: `ANTHROPIC_API_KEY = "sk-ant-..."`
4. Optional env var `CLAUDE_MODEL` overrides the default model.

## Notes and limits
- Image-only (scanned) PDFs are not supported; there is no OCR.
- Web search must be enabled for your Anthropic organization.
- Claims capped at 30 per run to control cost. Verdicts are model judgements backed by cited sources; review the links for high-stakes use.
