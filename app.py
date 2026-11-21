import os
import re
import json
import time
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

MODEL = os.getenv("gemini-3.8-flash", "gemini-2.5-flash")
MAX_CLAIMS = 15          # keep low to stay inside free-tier limits
PAUSE_SECONDS = 7        # free tier allows only a few requests per minute


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def extract_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("No JSON found in model output")
    end = max(text.rfind("]"), text.rfind("}"))
    return json.loads(text[start:end + 1])


def call_model(client, prompt, use_search=False, retries=3):
    config = None
    if use_search:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(15 * (attempt + 1))  # back off on rate limits


def extract_claims(client, text):
    prompt = f"""Extract up to {MAX_CLAIMS} specific, self-contained, checkable factual
claims from the document below (statistics, dates, financial or technical figures).
Each claim must make sense on its own without the document.
Return ONLY a JSON array of strings, nothing else.

DOCUMENT:
{text[:30000]}"""
    resp = call_model(client, prompt)
    claims = parse_json(resp.text)
    return [c for c in claims if isinstance(c, str)][:MAX_CLAIMS]


def get_sources(resp):
    urls = []
    try:
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks or []
        for ch in chunks:
            if ch.web and ch.web.uri and ch.web.uri not in urls:
                urls.append(ch.web.uri)
    except Exception:
        pass
    return urls


def verify_claim(client, claim):
    prompt = f"""Fact-check this claim using Google Search. Today's date matters:
flag figures that are outdated.

CLAIM: {claim}

Return ONLY a JSON object with these keys:
"verdict": one of "Verified", "Inaccurate", "False"
  - Verified: matches current, reliable sources
  - Inaccurate: partly right but the figure/date is outdated or wrong
  - False: no evidence found, or contradicted by sources
"real_fact": the correct fact (one or two sentences)
"explanation": short reason for the verdict"""
    resp = call_model(client, prompt, use_search=True)
    try:
        data = parse_json(resp.text)
    except Exception:
        data = {"verdict": "False", "real_fact": "", "explanation": "Could not parse model answer."}
    verdict = str(data.get("verdict", "False")).strip().capitalize()
    if verdict not in ("Verified", "Inaccurate", "False"):
        verdict = "False"
    return {
        "claim": claim,
        "verdict": verdict,
        "real_fact": data.get("real_fact", ""),
        "explanation": data.get("explanation", ""),
        "sources": get_sources(resp),
    }


st.set_page_config(page_title="Fact-Check Agent", page_icon="🔍")
st.title("🔍 Fact-Check Agent (Gemini)")
st.caption("Upload a PDF to get a claim-by-claim report: Verified, Inaccurate or False.")

api_key = get_api_key()
if not api_key:
    st.error("GEMINI_API_KEY is not set. Add it as an environment variable or in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)
pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf and st.button("Run fact-check"):
    text = extract_text(pdf)
    if not text:
        st.error("No text found. Scanned/image-only PDFs aren't supported (no OCR).")
        st.stop()

    with st.spinner("Extracting claims..."):
        claims = extract_claims(client, text)
    st.write(f"Found **{len(claims)}** claims to check.")

    results = []
    bar = st.progress(0.0)
    for i, claim in enumerate(claims, 1):
        with st.spinner(f"Checking {i}/{len(claims)}: {claim[:80]}..."):
            try:
                results.append(verify_claim(client, claim))
            except Exception as e:
                results.append({"claim": claim, "verdict": "False", "real_fact": "",
                                "explanation": f"Error: {e}", "sources": []})
        bar.progress(i / len(claims))
        time.sleep(PAUSE_SECONDS)

    order = {"False": 0, "Inaccurate": 1, "Verified": 2}
    results.sort(key=lambda r: order[r["verdict"]])
    icons = {"False": "❌", "Inaccurate": "⚠️", "Verified": "✅"}

    for r in results:
        with st.expander(f"{icons[r['verdict']]} {r['verdict']}: {r['claim']}"):
            if r["real_fact"]:
                st.markdown(f"**Real fact:** {r['real_fact']}")
            if r["explanation"]:
                st.markdown(f"**Why:** {r['explanation']}")
            for url in r["sources"]:
                st.markdown(f"- {url}")

    st.download_button("Download JSON", json.dumps(results, indent=2),
                       file_name="fact_check_report.json", mime="application/json")
    st.caption("Verdicts are model judgements backed by cited sources. Review the links for anything high-stakes.")
