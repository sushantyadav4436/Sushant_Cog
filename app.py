"""Fact-Check Agent: upload a PDF, extract checkable claims, verify them against
the live web, and flag each as Verified / Inaccurate / False.

Pipeline: pypdf (text) -> Claude (claim extraction, JSON) -> Claude + web_search
tool per claim (verdict + real fact + sources) -> Streamlit report.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_CLAIMS_DEFAULT = 15
SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

EXTRACT_SYSTEM = (
    "You extract verifiable factual claims from marketing or report text. "
    "Return ONLY a JSON array, no prose, no code fences. Each item: "
    '{"claim": "<self-contained sentence with entity, figure and time frame>", '
    '"type": "stat|date|financial|technical", "context": "<short quote from the text>"}. '
    "Include only claims that can be checked against public sources: statistics, "
    "percentages, dates, prices, revenue and market-size figures, version numbers, "
    "technical specs. Skip opinions and vague statements. Make each claim "
    "understandable on its own."
)

VERIFY_SYSTEM = (
    "You are a rigorous fact-checker. Today's date is {today}. Use web search to check the claim "
    "against current, authoritative sources (official sites, filings, standards bodies, "
    "reputable news or research). Then reply with ONLY a JSON object, no prose, no code fences:\n"
    '{{"verdict": "Verified|Inaccurate|False", "real_fact": "<the correct, current fact with '
    'figure and date>", "explanation": "<1-2 sentences>", "sources": ["<url>", "..."]}}\n'
    "Verdict rules: Verified = the claim matches current data. Inaccurate = the claim was true "
    "once or is close but the figure, date or scope is wrong or outdated; give the correct value "
    "in real_fact. False = after searching you find no evidence the claim is true, or sources "
    "contradict it outright; say what is actually true if you can. Never invent sources."
)


def get_client() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)
    if not key:
        st.error("Missing ANTHROPIC_API_KEY. Add it in Streamlit secrets or the environment.")
        st.stop()
    return Anthropic(api_key=key)


def read_pdf(file) -> str:
    reader = PdfReader(file)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages).strip()


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def extract_claims(client: Anthropic, text: str, limit: int) -> list[dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": f"Extract up to {limit} claims.\n\nTEXT:\n{text[:60000]}"}],
    )
    data = _parse_json(_text_of(resp))
    return data[:limit] if isinstance(data, list) else []


def verify_claim(client: Anthropic, claim: dict) -> dict:
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=VERIFY_SYSTEM.format(today=date.today().isoformat()),
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": f"Claim: {claim['claim']}\nContext: {claim.get('context', '')}"}],
        )
        out = _parse_json(_text_of(resp)) or {}
    except Exception as exc:  # network, rate limit, etc.
        out = {"verdict": "False", "real_fact": "", "explanation": f"Verification error: {exc}", "sources": []}
    verdict = out.get("verdict", "False")
    if verdict not in ("Verified", "Inaccurate", "False"):
        verdict = "False"
    return {**claim, "verdict": verdict, "real_fact": out.get("real_fact", ""),
            "explanation": out.get("explanation", ""), "sources": out.get("sources", [])}


BADGE = {"Verified": ("#12B78F", "Verified"), "Inaccurate": ("#F5A623", "Inaccurate"), "False": ("#E5586B", "False")}


def render(results: list[dict]) -> None:
    counts = {k: sum(r["verdict"] == k for r in results) for k in BADGE}
    c1, c2, c3 = st.columns(3)
    c1.metric("Verified", counts["Verified"])
    c2.metric("Inaccurate", counts["Inaccurate"])
    c3.metric("False", counts["False"])
    order = {"False": 0, "Inaccurate": 1, "Verified": 2}
    for r in sorted(results, key=lambda r: order[r["verdict"]]):
        color, label = BADGE[r["verdict"]]
        with st.container(border=True):
            st.markdown(f"<span style='background:{color};color:white;padding:2px 10px;border-radius:10px;"
                        f"font-weight:600'>{label}</span> &nbsp; **{r['claim']}**", unsafe_allow_html=True)
            if r["verdict"] != "Verified" and r["real_fact"]:
                st.markdown(f"**Correct fact:** {r['real_fact']}")
            if r["explanation"]:
                st.caption(r["explanation"])
            for url in r["sources"][:4]:
                st.markdown(f"- [{url}]({url})")
    st.download_button("Download report (JSON)", json.dumps(results, indent=2), "factcheck_report.json")


def main() -> None:
    st.set_page_config(page_title="Fact-Check Agent", page_icon="✅", layout="wide")
    st.title("Fact-Check Agent")
    st.write("Upload a PDF. The agent extracts stats, dates and figures, checks each against the live web, "
             "and flags claims as **Verified**, **Inaccurate** (outdated or off) or **False** (no evidence).")
    max_claims = st.sidebar.slider("Max claims to check", 5, 30, MAX_CLAIMS_DEFAULT)
    file = st.file_uploader("PDF", type="pdf")
    if file and st.button("Run fact-check", type="primary"):
        client = get_client()
        with st.status("Reading PDF...", expanded=True) as status:
            text = read_pdf(file)
            if len(text) < 50:
                status.update(label="No extractable text (scanned PDF?)", state="error")
                st.stop()
            status.update(label="Extracting claims...")
            claims = extract_claims(client, text, max_claims)
            if not claims:
                status.update(label="No checkable claims found", state="error")
                st.stop()
            st.write(f"Found {len(claims)} claims. Verifying against the live web...")
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda c: verify_claim(client, c), claims))
            status.update(label="Done", state="complete")
        render(results)


if __name__ == "__main__":
    main()
