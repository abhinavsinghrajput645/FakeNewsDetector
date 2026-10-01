"""
app.py - Responsive, Explainable & Hybrid Fake News Detection Web Application
Built with Python, Streamlit, Scikit-Learn, and Google Gemini AI.
"""

import os
import urllib.parse
import streamlit as st
import joblib

# Automatically load backend .env environment variables if present
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
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Responsive & Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Hide Streamlit sidebar and its toggle button */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Responsive fonts and typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main container max width and padding */
    .block-container {
        max-width: 840px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 2.8rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 24px 18px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .app-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.45;
        max-width: 620px;
        margin: 0 auto;
    }

    /* Result Card Styles */
    .result-card-real {
        background: rgba(16, 185, 129, 0.12);
        border: 1.5px solid #10B981;
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .result-card-fake {
        background: rgba(239, 68, 68, 0.12);
        border: 1.5px solid #EF4444;
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .result-card-mixed {
        background: rgba(234, 179, 8, 0.12);
        border: 1.5px solid #EAB308;
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .result-title-real {
        font-size: 1.65rem;
        font-weight: 800;
        color: #34D399;
        margin-bottom: 4px;
    }

    .result-title-fake {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F87171;
        margin-bottom: 4px;
    }

    .result-title-mixed {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FACC15;
        margin-bottom: 4px;
    }

    .confidence-text {
        font-size: 1.02rem;
        color: #E2E8F0;
        font-weight: 500;
    }

    /* Word Tags */
    .tag-real {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 3px 4px;
        font-weight: 500;
    }

    .tag-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 3px 4px;
        font-weight: 500;
    }

    /* Gemini AI Card */
    .gemini-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    .gemini-title {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .highlight-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #E2E8F0;
    }

    /* Responsive Mobile Adjustments */
    @media (max-width: 640px) {
        .app-title {
            font-size: 1.45rem;
        }
        .app-subtitle {
            font-size: 0.85rem;
        }
        .result-title-real, .result-title-fake, .result-title-mixed {
            font-size: 1.3rem;
        }
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Vectorizer Loader (Cached for fast performance)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_components():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "fake_news_model.pkl")
    tfidf_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

try:
    model, tfidf = load_ml_components()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">📰 Fake News Detection System</div>
    <div class="app-subtitle">
        Enter a news article or web link to check whether it is authentic or fabricated using NLP Machine Learning and AI Fact-Checking.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Gemini AI Configuration (Environment or Direct UI Input)
# ---------------------------------------------------------
env_gemini_key = get_gemini_api_key()
with st.expander("✨ Gemini AI Fact-Checking Settings (Optional)", expanded=False):
    if env_gemini_key and is_valid_api_key_format(env_gemini_key):
        st.success("✅ Gemini API Key detected from environment (.env). AI Fact-Checking is active!")
        custom_key = st.text_input("Override Gemini API Key (optional):", type="password", placeholder="Leave blank to use .env key")
        active_gemini_key = custom_key.strip() if custom_key.strip() else env_gemini_key
    else:
        st.info("ℹ️ To enable AI-powered semantic claim verification, enter a free Gemini API key below (or set GEMINI_API_KEY in `.env`).")
        custom_key = st.text_input("Enter Google Gemini API Key:", type="password", placeholder="Paste your API key here (AIza...)")
        active_gemini_key = custom_key.strip()
        st.caption("Get a free API key at [Google AI Studio](https://aistudio.google.com/). The local ML model works offline without a key.")

is_gemini_active = bool(active_gemini_key and is_valid_api_key_format(active_gemini_key))


# ---------------------------------------------------------
# Preset Test Examples
# ---------------------------------------------------------
EXAMPLE_REAL = (
    "WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday, "
    "stating that inflation has continued to ease over the past year while economic activity "
    "expanded at a solid pace. Officials noted that future decisions will depend on incoming data."
)

EXAMPLE_FAKE = (
    "SHOCKING BOMBSHELL!! Whistleblowers reveal secret government project spraying dangerous chemicals "
    "to control citizens! Mainstream media is completely silent! Doctors banned from speaking the truth! "
    "Share this urgent video before it gets deleted everywhere!!"
)

st.markdown("**Quick Examples to Try:**")
col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 1])

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

with col_ex1:
    if st.button("🟢 Load Real News", use_container_width=True):
        st.session_state["news_text"] = EXAMPLE_REAL
        st.rerun()

with col_ex2:
    if st.button("🔴 Load Fake News", use_container_width=True):
        st.session_state["news_text"] = EXAMPLE_FAKE
        st.rerun()

with col_ex3:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state["news_text"] = ""
        st.rerun()


# ---------------------------------------------------------
# Optional Web URL Scraper
# ---------------------------------------------------------
with st.expander("🔗 Or fetch article directly from a Web URL"):
    url_input = st.text_input("Enter Article URL:", placeholder="https://example.com/news-story")
    if st.button("Fetch Article Content"):
        if url_input.strip():
            with st.spinner("Extracting article content from website..."):
                article_data = extract_article(url_input.strip())
                if not article_data.get("success"):
                    st.error(article_data.get("error", "Failed to fetch article."))
                else:
                    text_content = article_data.get("text", "")
                    title = article_data.get("title", "")
                    combined_text = f"{title}\n\n{text_content}".strip() if title else text_content
                    st.session_state["news_text"] = combined_text
                    domain_msg = f" from **{article_data.get('domain')}**" if article_data.get('domain') else ""
                    st.success(f"Article loaded{domain_msg} ({article_data.get('word_count', 0)} words)!")
                    st.rerun()
        else:
            st.warning("Please enter a valid URL.")


# ---------------------------------------------------------
# News Article Input & Analysis
# ---------------------------------------------------------
user_input = st.text_area(
    "News Article Text:",
    value=st.session_state["news_text"],
    height=180,
    placeholder="Paste news headline or paragraph here to analyze..."
)

if st.button("⚡ Analyze News Article", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please paste or type a news article first.")
    else:
        with st.spinner("Analyzing linguistic patterns with Machine Learning..."):
            res = analyze_text(user_input)

            # Prediction Card
            label = res["prediction"]
            if label == "REAL NEWS":
                st.markdown(f"""
                <div class="result-card-real">
                    <div class="result-title-real">✅ LIKELY REAL NEWS</div>
                    <div class="confidence-text">
                        Credibility Score: <strong>{res['credibility_score']:.1f}%</strong> (Model Confidence: {res['confidence']:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif label == "FAKE NEWS":
                st.markdown(f"""
                <div class="result-card-fake">
                    <div class="result-title-fake">❌ LIKELY FAKE NEWS</div>
                    <div class="confidence-text">
                        Suspicion Confidence: <strong>{res['confidence']:.1f}%</strong> (Credibility Score: {res['credibility_score']:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-mixed">
                    <div class="result-title-mixed">⚠️ UNVERIFIED / MIXED SIGNALS</div>
                    <div class="confidence-text">
                        Credibility Score: <strong>{res['credibility_score']:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability Breakdown Bars
            col_bar1, col_bar2 = st.columns([1, 1])
            with col_bar1:
                st.caption(f"Real Probability: **{res['prob_real']:.1f}%**")
                st.progress(res["prob_real"] / 100.0)
            with col_bar2:
                st.caption(f"Fake Probability: **{res['prob_fake']:.1f}%**")
                st.progress(res["prob_fake"] / 100.0)

            # Influential Keyword Signals
            st.markdown("#### 🔍 Why did the model make this decision?")
            col_w1, col_w2 = st.columns(2)

            with col_w1:
                st.markdown("**🟢 Words pointing to Real News:**")
                if res["top_real_words"]:
                    tags_html = "".join([f'<span class="tag-real">{w["token"]}</span>' for w in res["top_real_words"][:8]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No strong real indicator words found.*")

            with col_w2:
                st.markdown("**🔴 Words pointing to Fake News:**")
                if res["top_fake_words"]:
                    tags_html = "".join([f'<span class="tag-fake">{w["token"]}</span>' for w in res["top_fake_words"][:8]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No strong fake indicator words found.*")

            # Word-level highlight preview
            if res.get("highlighted_html"):
                with st.expander("📝 View Interactive Word-Level Highlighting"):
                    st.markdown(
                        f'<div class="highlight-container">{res["highlighted_html"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption("🟢 Green highlights indicate words associated with authentic reporting. 🔴 Red highlights indicate words associated with fabricated/sensational reporting.")

            # Stylometric or Sensationalism Alerts
            sty = res.get("stylometrics", {})
            if sty.get("sensationalism_score", 0) > 30 or sty.get("exclamation_count", 0) > 2 or sty.get("all_caps_count", 0) > 1:
                alerts = []
                if sty.get("exclamation_count", 0) > 1:
                    alerts.append(f"{sty['exclamation_count']} exclamation marks")
                if sty.get("all_caps_count", 0) > 1:
                    alerts.append(f"{sty['all_caps_count']} ALL-CAPS words")
                if sty.get("clickbait_hits"):
                    alerts.append(f"clickbait triggers ({', '.join(sty['clickbait_hits'][:3])})")
                details = ", ".join(alerts) if alerts else "sensational language"
                st.warning(f"⚠️ **Sensationalism Alert**: Detected {details}. Real journalistic news typically maintains neutral, objective tone and standard punctuation.")

            # Model Bias Notice
            if res.get("bias_warning"):
                st.info(f"ℹ️ **Linguistic Note**: {res['bias_warning']}")

            # Fact-Check Search Links
            st.markdown("---")
            st.markdown("#### 🌐 Verify with Established Fact-Checkers")
            query = urllib.parse.quote(user_input[:80])
            col_fc1, col_fc2 = st.columns(2)
            with col_fc1:
                st.link_button("🔍 Search on Google Fact Check", f"https://toolbox.google.com/factcheck/explorer/search/list:57?hl=en&num=10&query={query}", use_container_width=True)
            with col_fc2:
                st.link_button("🔎 Search on Snopes.com", f"https://www.snopes.com/search/{query}/", use_container_width=True)

            # Gemini AI Real-World Semantic Fact-Check
            if is_gemini_active:
                st.markdown("---")
                st.markdown("#### ✨ Gemini AI Semantic Fact-Check")
                with st.spinner("Consulting Gemini AI for real-world factual correctness..."):
                    ai_verdict, ai_err = verify_claim_with_gemini(user_input, api_key=active_gemini_key)
                    if ai_verdict:
                        verdict_tag, explanation = parse_gemini_verdict(ai_verdict)
                        v_lower = verdict_tag.lower()
                        if "fake" in v_lower:
                            badge_bg = "rgba(239, 68, 68, 0.2)"
                            badge_border = "rgba(239, 68, 68, 0.5)"
                            badge_color = "#F87171"
                            badge_icon = "🔴"
                        elif "real" in v_lower:
                            badge_bg = "rgba(34, 197, 94, 0.2)"
                            badge_border = "rgba(34, 197, 94, 0.5)"
                            badge_color = "#4ADE80"
                            badge_icon = "🟢"
                        else:
                            badge_bg = "rgba(234, 179, 8, 0.2)"
                            badge_border = "rgba(234, 179, 8, 0.5)"
                            badge_color = "#FACC15"
                            badge_icon = "🟡"

                        st.markdown(f"""
                        <div class="gemini-card">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                                <div class="gemini-title" style="margin-bottom: 0;">🤖 Gemini AI Fact-Check</div>
                                <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; font-weight: 700; padding: 4px 14px; border-radius: 9999px; font-size: 0.88rem; letter-spacing: 0.03em;">
                                    {badge_icon} {verdict_tag.upper()}
                                </span>
                            </div>
                            <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                                Factual Correctness Summary:
                            </div>
                            <div style="color: #F1F5F9; line-height: 1.65; font-size: 0.98rem; font-weight: 400;">
                                {explanation}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ AI Verification Notice: {ai_err}")
            else:
                st.caption("💡 *Tip: Configure your Gemini API key in the '✨ Gemini AI Fact-Checking Settings' expander above or in `.env` to enable instant AI real-world fact checking.*")


# ---------------------------------------------------------
# Educational & Explanatory Accordions (Viva / Presentation)
# ---------------------------------------------------------
st.markdown("---")

with st.expander("💡 How does this project work? (Easy Explanation)"):
    st.markdown("""
    This project uses **Natural Language Processing (NLP)** and **Machine Learning**:
    
    1. **Text Preprocessing**: The text is converted to lowercase and cleaned of punctuation.
    2. **TF-IDF Vectorization** (*Term Frequency - Inverse Document Frequency*): 
       Converts words into numbers. Words that appear frequently in fake news (or real news) get special numerical weights.
    3. **Logistic Regression Classifier**: 
       A binary classification model trained on thousands of labeled news articles. It calculates whether the combination of words in the article leans closer to **Real (1)** or **Fake (0)**.
    4. **Debiased Credibility Scoring**:
       Evaluates stylometrics (sensationalism, punctuation, clickbait triggers) to avoid false positives on neutral facts.
    5. **Hybrid AI Verification**:
       Leverages Google Gemini LLM to cross-verify claims against real-world knowledge.
    """)

with st.expander("🛡️ 4 Quick Tips to Spot Fake News Yourself"):
    st.markdown("""
    - **1. Check the Source**: Look at the website domain. Is it an established news agency or an unfamiliar blog?
    - **2. Read Beyond the Headline**: Headlines are often exaggerated clickbait to get views. Read the full body.
    - **3. Check the Author & Date**: Are there real author credentials? Is an old story being shared as current?
    - **4. Cross-Verify**: If breaking news is genuine, multiple major media outlets will be reporting on it simultaneously.
    """)

# Footer
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 36px; padding-top: 18px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
    Fake News Detection System • Built with Python, Streamlit, Scikit-Learn & Google Gemini AI • MIT License
</div>
""", unsafe_allow_html=True)