"""
app.py - News Credibility & Fact Verification Engine
Editorial-grade interface for statistical attribution and semantic fact-checking.
"""

import os
import re
import urllib.parse
import importlib
import streamlit as st

# Automatically load backend .env environment variables
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

# Hot-reload backend modules to avoid stale memory caches
import detector
try:
    importlib.reload(detector)
except Exception:
    pass

from detector import (
    analyze_text,
    verify_claim_with_gemini,
    is_valid_api_key_format,
    get_gemini_api_key,
    parse_gemini_verdict
)
from scraper import extract_article

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="TruthLens · News Credibility Platform",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Editorial Theme & Design System (Anti-AI Aesthetic)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Hide Streamlit sidebar and controls completely */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Base typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #F1F5F9;
    }

    /* Container constraints */
    .block-container {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Brand Header */
    .brand-masthead {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.5rem;
        margin-bottom: 1.8rem;
    }

    .brand-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0 0 0.5rem 0;
    }

    .brand-description {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.55;
        margin: 0;
    }

    /* Input Controls */
    .stTextArea textarea {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        padding: 12px 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }

    .stTextInput input {
        background-color: #0F172A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
    }
    .stTextInput input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }

    /* Primary Action Button */
    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: background-color 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }

    /* Preset & Secondary Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #CBD5E1 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.75rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #334155 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #94A3B8;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Editorial Analysis Card */
    .analysis-container {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px 24px;
        margin-top: 1.8rem;
        margin-bottom: 1.5rem;
    }

    /* Credibility Header */
    .credibility-status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 14px;
        margin-bottom: 18px;
        flex-wrap: wrap;
        gap: 10px;
    }

    .status-badge-real {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 4px;
    }

    .status-badge-fake {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.35);
        color: #FB7185;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 4px;
    }

    .status-badge-unverified {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FBBF24;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 4px;
    }

    /* Metric Grid Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }

    .metric-box {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px 14px;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Fact Check Brief Box */
    .fact-check-brief {
        background: #111827;
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 16px 18px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    .fact-check-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #38BDF8;
        margin-bottom: 8px;
    }

    .fact-check-body {
        font-size: 0.95rem;
        color: #E2E8F0;
        line-height: 1.65;
        font-weight: 400;
        margin: 0;
    }

    /* Lexical Marker Chips */
    .chip-real {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 3px 3px 3px 0;
        font-weight: 500;
    }

    .chip-fake {
        display: inline-block;
        background: rgba(244, 63, 94, 0.12);
        color: #FDA4AF;
        border: 1px solid rgba(244, 63, 94, 0.25);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 3px 3px 3px 0;
        font-weight: 500;
    }

    /* Citation Links */
    .citation-bar {
        display: flex;
        gap: 12px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .citation-link {
        font-size: 0.8rem;
        color: #94A3B8;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        transition: color 0.15s ease;
    }
    .citation-link:hover {
        color: #38BDF8;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="brand-masthead">
    <div class="brand-badge">VERITAS · CREDIBILITY INTELLIGENCE</div>
    <h1 class="brand-title">News Authenticity & Fact Verification</h1>
    <p class="brand-description">
        Cross-reference news content against trained lexical models, stylometric signals, and real-world factual context.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

SAMPLE_REAL = (
    "WASHINGTON (Reuters) - The Federal Reserve held benchmark interest rates steady on Wednesday, "
    "noting that inflation has continued to ease over the past year while economic activity expanded "
    "at a solid pace. Officials emphasized that future monetary policy decisions will remain data-dependent."
)

SAMPLE_FAKE = (
    "SHOCKING BOMBSHELL: Whistleblowers reveal secret government project spraying toxic airborne chemicals "
    "to control civilian populations! Mainstream media enforces total blackout while doctors are silenced! "
    "Share this urgent bulletin before authorities remove it everywhere!!"
)

# ---------------------------------------------------------
# Input Tabs
# ---------------------------------------------------------
tab_text, tab_url = st.tabs(["Text Input", "Article URL"])

with tab_text:
    # Editorial Sample Selectors
    col_s1, col_s2, col_s3 = st.columns([1.2, 1.2, 0.6])
    with col_s1:
        if st.button("Sample: Wire Report", use_container_width=True):
            st.session_state["news_text"] = SAMPLE_REAL
            st.rerun()
    with col_s2:
        if st.button("Sample: Viral Claim", use_container_width=True):
            st.session_state["news_text"] = SAMPLE_FAKE
            st.rerun()
    with col_s3:
        if st.button("Clear", use_container_width=True):
            st.session_state["news_text"] = ""
            st.rerun()

    user_input = st.text_area(
        label="Article Text or Headline:",
        value=st.session_state["news_text"],
        height=160,
        placeholder="Paste article body, headline, or claim to evaluate...",
        label_visibility="collapsed"
    )

with tab_url:
    st.markdown("<p style='font-size: 0.88rem; color: #94A3B8; margin-bottom: 8px;'>Extract and analyze publicly accessible news articles via URL:</p>", unsafe_allow_html=True)
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        url_input = st.text_input("Article URL", placeholder="https://example.com/article-path", label_visibility="collapsed")
    with col_u2:
        fetch_clicked = st.button("Extract Content", use_container_width=True)

    if fetch_clicked:
        if url_input.strip():
            with st.spinner("Extracting article content..."):
                scrape_res = extract_article(url_input.strip())
                if scrape_res.get("success"):
                    title = scrape_res.get("title", "")
                    body = scrape_res.get("text", "")
                    combined = f"{title}\n\n{body}".strip()
                    st.session_state["news_text"] = combined
                    st.success(f"Extracted '{title[:60]}...' ({scrape_res.get('word_count', 0)} words)")
                    st.rerun()
                else:
                    st.error(scrape_res.get("error", "Failed to retrieve article content."))
        else:
            st.warning("Please provide a valid web URL.")

# ---------------------------------------------------------
# Analysis Trigger
# ---------------------------------------------------------
analyze_clicked = st.button("Evaluate Authenticity", type="primary", use_container_width=True)

if analyze_clicked:
    target_content = user_input.strip()
    if not target_content:
        st.warning("Please enter or load article text to analyze.")
    else:
        with st.spinner("Running linguistic analysis & verifying claim..."):
            # 1. Run local ML & stylometrics pipeline
            res = analyze_text(target_content)

            # 2. Check for backend Gemini key and run fact-check
            gemini_key = get_gemini_api_key()
            gemini_available = bool(gemini_key and is_valid_api_key_format(gemini_key))
            
            fact_verdict = None
            fact_explanation = None
            if gemini_available:
                ai_verdict, ai_err = verify_claim_with_gemini(gemini_key, target_content)
                if ai_verdict:
                    fact_verdict, fact_explanation = parse_gemini_verdict(ai_verdict)

            # Determine badge styling
            cred_score = res["credibility_score"]
            if cred_score >= 60.0:
                status_class = "status-badge-real"
                status_text = "AUTHENTIC PROFILE"
                status_desc = "Lexical patterns align with verified journalistic standards."
            elif cred_score <= 40.0:
                status_class = "status-badge-fake"
                status_text = "FABRICATION RISK"
                status_desc = "Content exhibits statistical markers of fabricated or sensationalist media."
            else:
                status_class = "status-badge-unverified"
                status_text = "UNSUBSTANTIATED / MIXED"
                status_desc = "Mixed linguistic signals; requires independent verification."

            # Render Analysis Container
            st.markdown(f"""
            <div class="analysis-container">
                <div class="credibility-status-row">
                    <div>
                        <span class="{status_class}">{status_text}</span>
                        <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 6px;">{status_desc}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Credibility Index</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC;">{cred_score:.1f} <span style="font-size: 0.9rem; color: #64748B;">/ 100</span></div>
                    </div>
                </div>

                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Model Attribution</div>
                        <div class="metric-value">{res['prob_real']:.1f}% <span style="font-size: 0.8rem; font-weight: 500; color: #94A3B8;">Real</span> / {res['prob_fake']:.1f}% <span style="font-size: 0.8rem; font-weight: 500; color: #94A3B8;">Fake</span></div>
                        <div class="metric-sub">TF-IDF Vector Logistic Weight</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Sensationalism Score</div>
                        <div class="metric-value">{res['stylometrics']['sensationalism_score']:.1f} <span style="font-size: 0.8rem; font-weight: 500; color: #94A3B8;">/ 100</span></div>
                        <div class="metric-sub">{res['stylometrics']['exclamation_count']} exclamations · {res['stylometrics']['all_caps_count']} caps words</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Vocabulary Scope</div>
                        <div class="metric-value">{res['stylometrics']['word_count']} <span style="font-size: 0.8rem; font-weight: 500; color: #94A3B8;">Words</span></div>
                        <div class="metric-sub">{res['stylometrics']['sentence_count']} structural sentences</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Fact-Check Debrief (2 to 3 lines)
            if fact_verdict and fact_explanation:
                v_lower = fact_verdict.lower()
                if "fake" in v_lower:
                    fact_badge = '<span class="status-badge-fake">VERDICT: FALSE</span>'
                elif "real" in v_lower:
                    fact_badge = '<span class="status-badge-real">VERDICT: TRUE</span>'
                else:
                    fact_badge = '<span class="status-badge-unverified">VERDICT: UNVERIFIED</span>'

                st.markdown(f"""
                <div class="fact-check-brief">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                        <span class="fact-check-header">Factual Correctness Assessment</span>
                        {fact_badge}
                    </div>
                    <p class="fact-check-body">{fact_explanation}</p>
                </div>
                """, unsafe_allow_html=True)

            # Lexical Attribution Columns
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.markdown("<p style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Factual Lexical Markers</p>", unsafe_allow_html=True)
                if res["top_real_words"]:
                    real_chips = "".join([f'<span class="chip-real">{item["token"]}</span>' for item in res["top_real_words"][:7]])
                    st.markdown(f"<div>{real_chips}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>No strong factual markers identified.</p>", unsafe_allow_html=True)

            with col_l2:
                st.markdown("<p style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Sensationalist / Clickbait Markers</p>", unsafe_allow_html=True)
                if res["top_fake_words"]:
                    fake_chips = "".join([f'<span class="chip-fake">{item["token"]}</span>' for item in res["top_fake_words"][:7]])
                    st.markdown(f"<div>{fake_chips}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>No high-risk sensationalist markers found.</p>", unsafe_allow_html=True)

            # External Reference Citations
            encoded_query = urllib.parse.quote(target_content[:90])
            st.markdown(f"""
                <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 14px; margin-top: 16px;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">External Verification Archives</div>
                    <div class="citation-bar">
                        <a href="https://toolbox.google.com/factcheck/explorer/search/list:57?hl=en&num=10&query={encoded_query}" target="_blank" class="citation-link">
                            Google Fact Check Database ↗
                        </a>
                        <span style="color: rgba(255,255,255,0.15);">·</span>
                        <a href="https://www.snopes.com/search/{encoded_query}/" target="_blank" class="citation-link">
                            Snopes Archive Index ↗
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Methodology Overview (Subtle, Professional Disclosure)
# ---------------------------------------------------------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

with st.expander("Technical Architecture & Methodology"):
    st.markdown("""
    **Verification Methodology:**
    - **Vector Representation**: Normalized TF-IDF matrix projection capturing term frequencies and inverse document frequencies across standard news corpora.
    - **Classification Architecture**: L2-regularized logistic classification calibrated against lexical patterns to assign class probabilities.
    - **Stylometric Layer**: Automated heuristics inspecting uppercase word density, punctuation abuse, and clickbait trigger distribution.
    - **Factual Verification Layer**: Cross-checks claims via Google Gemini REST API using semantic reasoning and international wire archives.
    """)