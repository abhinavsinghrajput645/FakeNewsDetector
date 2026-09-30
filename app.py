"""
app.py - Clean, Beginner-Friendly & Responsive Fake News Detector
Built with Streamlit, Scikit-Learn, and TF-IDF.
"""

import os
import re
import urllib.parse
import streamlit as st
import joblib

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
    /* Responsive fonts and padding */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main container max width and padding */
    .block-container {
        max-width: 820px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 20px 16px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 22px;
    }

    .app-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Result Card Styles */
    .result-card-real {
        background: rgba(16, 185, 129, 0.12);
        border: 1.5px solid #10B981;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .result-card-fake {
        background: rgba(239, 68, 68, 0.12);
        border: 1.5px solid #EF4444;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .result-title-real {
        font-size: 1.6rem;
        font-weight: 800;
        color: #34D399;
        margin-bottom: 4px;
    }

    .result-title-fake {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F87171;
        margin-bottom: 4px;
    }

    .confidence-text {
        font-size: 1rem;
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

    /* Responsive Mobile Adjustments */
    @media (max-width: 640px) {
        .app-title {
            font-size: 1.45rem;
        }
        .app-subtitle {
            font-size: 0.85rem;
        }
        .result-title-real, .result-title-fake {
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
# Simple Helper Functions (Easy to explain in an interview)
# ---------------------------------------------------------
def clean_input_text(text: str) -> str:
    """Preprocesses input text by lowercasing and removing extra spaces."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_text_from_url(url: str):
    """Simple scraper to fetch article text from a web link."""
    try:
        import requests
        from bs4 import BeautifulSoup

        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get main paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 35]
        if not paragraphs:
            return None, "Could not find article paragraphs on this page."
        
        full_text = "\n\n".join(paragraphs[:10])
        return full_text, None
    except Exception as e:
        return None, f"Could not load URL: {str(e)}"


def predict_news(text: str):
    """
    Predicts if news is Real or Fake and finds the top influential words.
    - Class 1 = Real News
    - Class 0 = Fake News
    """
    cleaned = clean_input_text(text)
    vector = tfidf.transform([cleaned])

    # Model prediction
    pred = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    prob_fake = probs[0] * 100
    prob_real = probs[1] * 100

    if pred == 1:
        label = "REAL NEWS"
        confidence = prob_real
        is_real = True
    else:
        label = "FAKE NEWS"
        confidence = prob_fake
        is_real = False

    # Extract word influence (Attribution: TF-IDF value * Model Weight)
    coef = model.coef_[0]
    feature_names = tfidf.get_feature_names_out()
    nonzeros = vector.nonzero()[1]

    word_scores = []
    for idx in nonzeros:
        word = feature_names[idx]
        score = vector[0, idx] * coef[idx]
        word_scores.append((word, score))

    # Top words pointing to Real (positive score) and Fake (negative score)
    real_words = [w for w, s in sorted(word_scores, key=lambda x: x[1], reverse=True) if s > 0][:6]
    fake_words = [w for w, s in sorted(word_scores, key=lambda x: x[1]) if s < 0][:6]

    # Simple sensationalism check (exclamation marks, all caps)
    exclamations = text.count("!")
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2 and w.isalpha()]
    
    return {
        "label": label,
        "is_real": is_real,
        "confidence": confidence,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "real_words": real_words,
        "fake_words": fake_words,
        "exclamations": exclamations,
        "caps_count": len(caps_words)
    }


# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------

# Header Banner
st.markdown("""
<div class="app-header">
    <div class="app-title">📰 Fake News Detection System</div>
    <div class="app-subtitle">
        Enter a news article or web link to check whether it is authentic or fabricated using Machine Learning.
    </div>
</div>
""", unsafe_allow_html=True)

# Preset examples for fast testing
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

# Interactive Preset Buttons
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

# Optional URL input
with st.expander("🔗 Or fetch article directly from a Web URL"):
    url_input = st.text_input("Enter Article URL:", placeholder="https://example.com/news-story")
    if st.button("Fetch Article Content"):
        if url_input.strip():
            with st.spinner("Fetching article from website..."):
                scraped_text, err = fetch_text_from_url(url_input.strip())
                if err:
                    st.error(err)
                else:
                    st.session_state["news_text"] = scraped_text
                    st.success("Article loaded successfully!")
                    st.rerun()
        else:
            st.warning("Please enter a valid URL.")

# Text Area for Input
user_input = st.text_area(
    "News Article Text:",
    value=st.session_state["news_text"],
    height=180,
    placeholder="Paste news headline or paragraph here to analyze..."
)

# Analyze Button
if st.button("⚡ Analyze News Article", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please paste or type a news article first.")
    else:
        with st.spinner("Analyzing text patterns with Machine Learning..."):
            res = predict_news(user_input)

            # Result Banner
            if res["is_real"]:
                st.markdown(f"""
                <div class="result-card-real">
                    <div class="result-title-real">✅ LIKELY REAL NEWS</div>
                    <div class="confidence-text">
                        Model Confidence: <strong>{res['confidence']:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-fake">
                    <div class="result-title-fake">❌ LIKELY FAKE NEWS</div>
                    <div class="confidence-text">
                        Model Confidence: <strong>{res['confidence']:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Progress Bar
            col_bar1, col_bar2 = st.columns([1, 1])
            with col_bar1:
                st.caption(f"Real Probability: **{res['prob_real']:.1f}%**")
                st.progress(res["prob_real"] / 100.0)
            with col_bar2:
                st.caption(f"Fake Probability: **{res['prob_fake']:.1f}%**")
                st.progress(res["prob_fake"] / 100.0)

            # Key Word Signals (Beginner-Friendly Explanation)
            st.markdown("#### 🔍 Why did the model make this decision?")
            col_w1, col_w2 = st.columns(2)

            with col_w1:
                st.markdown("**🟢 Words pointing to Real News:**")
                if res["real_words"]:
                    tags_html = "".join([f'<span class="tag-real">{w}</span>' for w in res["real_words"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No strong real indicator words found.*")

            with col_w2:
                st.markdown("**🔴 Words pointing to Fake News:**")
                if res["fake_words"]:
                    tags_html = "".join([f'<span class="tag-fake">{w}</span>' for w in res["fake_words"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No strong fake indicator words found.*")

            # Sensationalism notice if any
            if res["exclamations"] > 2 or res["caps_count"] > 1:
                st.info(f"⚠️ **Sensationalism Alert**: Found {res['exclamations']} exclamation marks and {res['caps_count']} ALL-CAPS words. Real news articles typically use neutral, objective punctuation.")

            # Instant Fact-Check Links
            st.markdown("---")
            st.markdown("#### 🌐 Verify with Trusted Fact-Checkers")
            query = urllib.parse.quote(user_input[:80])
            col_fc1, col_fc2 = st.columns(2)
            with col_fc1:
                st.link_button("🔍 Search on Google Fact Check", f"https://toolbox.google.com/factcheck/explorer/search/list:57?hl=en&num=10&query={query}", use_container_width=True)
            with col_fc2:
                st.link_button("🔎 Search on Snopes.com", f"https://www.snopes.com/search/{query}/", use_container_width=True)


# ---------------------------------------------------------
# Educational & Explanatory Accordions (Great for Project Viva/Presentation)
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
    """)

with st.expander("🛡️ 4 Quick Tips to Spot Fake News Yourself"):
    st.markdown("""
    - **1. Check the Source**: Look at the website domain. Is it an established news agency or an unfamiliar blog?
    - **2. Read Beyond the Headline**: Headlines are often exaggerated clickbait to get views. Read the full body.
    - **3. Check the Author & Date**: Are there real author credentials? Is an old story being shared as current?
    - **4. Cross-Verify**: If breaking news is genuine, multiple major media outlets will be reporting on it simultaneously.
    """)