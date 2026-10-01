"""
detector.py - NLP Engine, Explainability Layer, and Stylometric Analysis
for Fake News Detection.
"""

import os
import re
import html
import joblib
import requests
import numpy as np

# Load backend environment variables from .env if present
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

# Cached model and Gemini states
_MODEL = None
_TFIDF = None
_WORKING_GEMINI_MODEL = None


# Curated list of sensationalist / clickbait triggers
CLICKBAIT_TRIGGERS = {
    "shocking", "bombshell", "unbelievable", "you won't believe", "exposed",
    "secret", "conspiracy", "urgent", "must see", "alert", "proof", "revealed",
    "mainstream media won't tell you", "miracle", "hidden truth", "they don't want you to know",
    "mind-blowing", "scandalous", "epic", "huge leak", "devastating", "banned"
}

def load_models(model_path="fake_news_model.pkl", tfidf_path="tfidf_vectorizer.pkl"):
    """Loads and caches the model and vectorizer."""
    global _MODEL, _TFIDF
    if _MODEL is None or _TFIDF is None:
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            tfidf_path = os.path.join(base_dir, tfidf_path)
        
        _MODEL = joblib.load(model_path)
        _TFIDF = joblib.load(tfidf_path)
    return _MODEL, _TFIDF


def clean_text(text: str) -> str:
    """Preprocesses text for model ingestion."""
    if not text:
        return ""
    # Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


COMMON_ACRONYMS = {
    "NASA", "NATO", "WHO", "CDC", "FDA", "FBI", "CIA", "DOJ", "EPA", "DOD",
    "FAA", "FCC", "FTC", "SEC", "NIH", "COVID", "USA", "UK", "EU", "UN",
    "GDP", "AI", "ML", "CEO", "CFO", "CTO", "UFO", "USAF", "STEM", "ESA",
    "BBC", "CNN", "NPR", "GOP", "DNC", "POTUS", "SCOTUS", "IRS", "ICE", "TSA"
}

def compute_stylometrics(text: str) -> dict:
    """
    Extracts stylometric signals:
    - Capitalization density (ALL-CAPS words, excluding standard acronyms)
    - Punctuation abuse (multiple ?, !)
    - Sensationalism/clickbait keyword triggers
    - Reading metrics (word count, sentence count)
    """
    if not text or not text.strip():
        return {
            "all_caps_ratio": 0.0,
            "all_caps_count": 0,
            "caps_words_sample": [],
            "exclamation_count": 0,
            "question_count": 0,
            "excessive_punct_count": 0,
            "clickbait_hits": [],
            "sensationalism_score": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0
        }

    raw_words = text.split()
    total_words = len(raw_words)
    
    # Check for ALL CAPS words, ignoring common acronyms and single letters
    caps_words = [
        re.sub(r"[^\w]", "", w)
        for w in raw_words
        if w.isupper() and len(re.sub(r"[^\w]", "", w)) >= 2
        and re.sub(r"[^\w]", "", w).isalpha()
        and re.sub(r"[^\w]", "", w) not in COMMON_ACRONYMS
    ]
    all_caps_count = len(caps_words)
    # Ratio only counted if more than 1 non-acronym caps word exists
    all_caps_ratio = (all_caps_count / max(total_words, 1)) * 100 if all_caps_count >= 2 else 0.0

    # Punctuation abuse
    exclamation_count = text.count("!")
    question_count = text.count("?")
    excessive_punct = len(re.findall(r"[!?]{2,}", text))

    # Clickbait keyword search
    text_lower = text.lower()
    clickbait_hits = [phrase for phrase in CLICKBAIT_TRIGGERS if phrase in text_lower]

    # Sentence count
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Average word length
    clean_words = [re.sub(r"[^\w]", "", w) for w in raw_words if re.sub(r"[^\w]", "", w)]
    avg_word_length = sum(len(w) for w in clean_words) / max(len(clean_words), 1)

    # Composite Sensationalism Score (0 to 100)
    # Weighted combination of clickbait hits, punctuation abuse, and ALL CAPS ratio
    sensational_points = (
        (len(clickbait_hits) * 15.0) +
        (min(excessive_punct, 5) * 8.0) +
        (min(all_caps_ratio, 15.0) * 3.0) +
        (min(exclamation_count / max(sentence_count, 1), 5.0) * 8.0)
    )
    sensationalism_score = round(min(max(sensational_points, 0.0), 100.0), 1)

    return {
        "all_caps_ratio": round(all_caps_ratio, 2),
        "all_caps_count": all_caps_count,
        "caps_words_sample": caps_words[:5],
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "excessive_punct_count": excessive_punct,
        "clickbait_hits": clickbait_hits,
        "sensationalism_score": sensationalism_score,
        "word_count": total_words,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2)
    }



def is_valid_api_key_format(key: str) -> bool:
    """Validates Gemini API keys (supports traditional AIza... and newer formats)."""
    if not key:
        return False
    clean = key.encode("ascii", "ignore").decode("ascii").strip()
    if " " in clean or len(clean) < 20:
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]{20,}$", clean))


def get_gemini_api_key(explicit_key: str = None) -> str:
    """Returns valid API key from explicit argument or GEMINI_API_KEY environment variable."""
    key = explicit_key or os.environ.get("GEMINI_API_KEY", "")
    return key.encode("ascii", "ignore").decode("ascii").strip() if key else ""


def get_gemini_models_list(api_key: str):
    """Queries Google API to discover which models are enabled for this API key."""
    clean_key = api_key.encode("ascii", "ignore").decode("ascii").strip()
    if not is_valid_api_key_format(clean_key):
        return None

    headers = {"x-goog-api-key": clean_key, "Content-Type": "application/json"}
    for version in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{version}/models?key={clean_key}"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                models = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                if models:
                    # Filter for standard text/flash models, excluding audio/image/experimental omni previews
                    flash_models = [
                        m for m in models
                        if "flash" in m.lower()
                        and not any(x in m.lower() for x in ["tts", "image", "omni", "preview", "customtools"])
                    ]
                    # Put gemini-3.8-flash and stable flash models first
                    priority_order = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]
                    ordered = [m for m in priority_order if m in flash_models]
                    for m in sorted(flash_models, reverse=True):
                        if m not in ordered:
                            ordered.append(m)
                    chosen = ordered if ordered else models
                    return [(version, m) for m in chosen]
        except Exception:
            continue
    return None


def verify_claim_with_gemini(arg1: str = "", arg2: str = None, api_key: str = None, text: str = None):
    """
    LLM Fact-Checking using Google Gemini REST API.
    Auto-detects active model and falls back through candidate models.
    Supports verify_claim_with_gemini(text, api_key) or verify_claim_with_gemini(api_key, text).
    Uses explicit api_key or falls back to GEMINI_API_KEY environment variable.
    """
    global _WORKING_GEMINI_MODEL

    # Disambiguate arguments
    raw_text = text
    raw_key = api_key

    if raw_text is None and raw_key is None:
        if arg2 is not None:
            # Two positional arguments: determine which is key and which is text
            if is_valid_api_key_format(arg1) and (len(arg2) > 60 or " " in arg2):
                raw_key, raw_text = arg1, arg2
            else:
                raw_text, raw_key = arg1, arg2
        else:
            raw_text = arg1

    target_text = (raw_text or "").strip()
    clean_key = get_gemini_api_key(raw_key)

    if not clean_key:
        return None, "Gemini API key is missing. Set GEMINI_API_KEY in your .env or backend environment."
    if not is_valid_api_key_format(clean_key):
        return None, "Invalid API key format. Please enter a valid Gemini API key from Google AI Studio."
    if not target_text:
        return None, "Article text to fact check is empty."

    prompt = (
        "You are an expert, objective fact-checking assistant. "
        "Analyze the factual correctness of the following news claim or article snippet:\n\n"
        f"\"{target_text[:1500]}\"\n\n"
        "Provide your analysis concisely in exactly this structure:\n"
        "- **Verdict**: [Real News | Fake News | Unverified]\n"
        "- **Factual Reality**: Provide 2 to 3 concise, clear sentences explaining whether this news is correct or false, what actually happened, and noting credible sources like Reuters, AP, or international defense authorities."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
    }
    headers = {"x-goog-api-key": clean_key, "Content-Type": "application/json"}

    candidates = []
    if _WORKING_GEMINI_MODEL:
        candidates.append(_WORKING_GEMINI_MODEL)

    detected_list = get_gemini_models_list(clean_key)
    if detected_list:
        for pair in detected_list:
            if pair not in candidates:
                candidates.append(pair)

    fallback_models = [
        ("v1beta", "gemini-3.8-flash"),
        ("v1", "gemini-3.8-flash"),
        ("v1beta", "gemini-flash-latest"),
        ("v1beta", "gemini-3.7-flash"),
        ("v1beta", "gemini-3.5-flash"),
        ("v1beta", "gemini-2.5-flash"),
        ("v1beta", "gemini-pro"),
    ]
    for pair in fallback_models:
        if pair not in candidates:
            candidates.append(pair)

    last_error = None
    for version, model_name in candidates:
        url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={clean_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                _WORKING_GEMINI_MODEL = (version, model_name)
                data = resp.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                return text_out, None
            elif resp.status_code == 404:
                last_error = f"Model {model_name} on {version} returned 404."
                continue
            elif resp.status_code == 400:
                return None, "Invalid API key or malformed request. Please check your Gemini key."
            elif resp.status_code in [429, 503]:
                last_error = f"Model {model_name} temporarily unavailable (HTTP {resp.status_code})."
                continue
            else:
                last_error = f"Gemini API returned status code {resp.status_code}."
        except requests.exceptions.Timeout:
            last_error = "Connection to Gemini timed out."
        except Exception as e:
            last_error = f"Error contacting Gemini API: {str(e)}"

    return None, f"Could not connect to Gemini ({last_error}). Please verify your API key."


def parse_gemini_verdict(raw_text: str):
    """Parses Gemini response into a clean verdict and a 2-3 lines correctness explanation."""
    if not raw_text:
        return "Unverified", "No factual assessment provided."

    # Extract verdict
    v_match = re.search(r"\*?\*?Verdict\*?\*?:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    verdict = v_match.group(1).replace("**", "").replace("[", "").replace("]", "").strip() if v_match else "Unverified"

    if "fake" in verdict.lower():
        verdict = "Fake News"
    elif "real" in verdict.lower():
        verdict = "Real News"
    elif "unverified" in verdict.lower() or "satire" in verdict.lower():
        verdict = "Unverified / Satire"

    # Extract explanation / factual reality
    exp_match = re.search(r"\*?\*?(?:Factual Reality|Explanation|Summary)\*?\*?:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
    else:
        # Fallback: remove verdict line and join remaining lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not re.search(r"verdict", line, re.IGNORECASE)]
        explanation = " ".join(lines)

    # Clean leading asterisks or bullet dashes from explanation
    explanation = re.sub(r"^[-*•\s]+", "", explanation).strip()
    return verdict, explanation


def analyze_text(
    text: str,
    model_path="fake_news_model.pkl",
    tfidf_path="tfidf_vectorizer.pkl",
    verify_with_gemini: bool = False,
    gemini_api_key: str = None
) -> dict:
    """
    Performs comprehensive analysis:
    - ML inference (Real vs Fake probability)
    - Feature-level explainability (word attributions)
    - Stylometric signals
    - Composite Credibility Index
    - Optional Gemini AI semantic fact-check verification
    """
    model, tfidf = load_models(model_path, tfidf_path)
    
    if not text or not text.strip():
        return {
            "prediction": "EMPTY",
            "is_real": False,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "confidence": 0.0,
            "credibility_score": 0.0,
            "word_contributions": [],
            "top_real_words": [],
            "top_fake_words": [],
            "highlighted_html": "",
            "stylometrics": compute_stylometrics(""),
            "bias_warning": None,
            "gemini_verification": None
        }

    stylometrics = compute_stylometrics(text)
    cleaned = clean_text(text)
    
    # TF-IDF transform
    vector = tfidf.transform([cleaned])
    
    # Probabilities
    # Model classes: [0, 1] -> 0 = FAKE NEWS, 1 = REAL NEWS
    probabilities = model.predict_proba(vector)[0]
    prob_fake = float(probabilities[0])
    prob_real = float(probabilities[1])
    raw_prediction = int(model.predict(vector)[0])
    
    # Feature attributions
    # contribution = tfidf_value * coefficient
    coefs = model.coef_[0]
    feature_names = tfidf.get_feature_names_out()
    
    nonzero_indices = vector.nonzero()[1]
    contributions = []
    token_score_map = {}

    for idx in nonzero_indices:
        tfidf_val = vector[0, idx]
        weight = coefs[idx]
        score = float(tfidf_val * weight)
        token = feature_names[idx]
        
        contributions.append({
            "token": token,
            "score": score,
            "weight": float(weight),
            "tfidf": float(tfidf_val)
        })
        token_score_map[token] = score

    # Sort contributions
    # Positive score = pushes towards Real
    # Negative score = pushes towards Fake
    contributions.sort(key=lambda x: abs(x["score"]), reverse=True)
    
    top_real_words = [c for c in contributions if c["score"] > 0][:10]
    top_fake_words = [c for c in contributions if c["score"] < 0][:10]

    # Bias detection: check if "reuters" or weekday shortcut is dominating the prediction
    has_reuters = "reuters" in token_score_map
    has_weekday = any(w in token_score_map for w in ["monday", "tuesday", "wednesday", "thursday", "friday"])
    bias_warning = None
    if has_reuters or has_weekday:
        bias_warning = "Dataset Shortcut Detected: Article contains wire/dateline markers (e.g. 'Reuters' or weekday stamps) which heavily bias Kaggle-trained models toward 'Real'."

    # Calculate Debias-Adjusted Credibility Score (0 - 100%)
    # Model baseline probability
    model_real_pct = prob_real * 100.0

    # Aggregate fake and real lexical weights
    total_fake_pull = sum(abs(c["score"]) for c in top_fake_words)
    total_real_pull = sum(abs(c["score"]) for c in top_real_words)
    sensationalism = stylometrics["sensationalism_score"]
    clickbait_count = len(stylometrics["clickbait_hits"])

    # Distinguish between active fake indicators vs lack of Reuters/politics vocabulary
    if model_real_pct >= 50.0:
        # Model considers it real; check if sensationalism degrades it
        credibility = model_real_pct - (sensationalism * 0.45)
    else:
        # Model considers it fake. Is it genuinely fake, or neutral text suffering from model bias?
        if sensationalism < 15 and clickbait_count == 0 and (total_fake_pull < 1.0 or sensationalism == 0):
            # Neutral factual text without sensationalism or known fake markers
            # Adjust for the trained model's negative intercept (-1.24) and domain gap
            neutral_boost = 55.0 - (total_fake_pull * 15.0)
            credibility = max(model_real_pct, neutral_boost)
            if not bias_warning:
                bias_warning = "Domain Notice: Neutral, non-sensational text without political wire markers. Adjusted for model's wire-service training bias."
        else:
            # Active sensationalism or strong fake lexical pull
            credibility = model_real_pct - (sensationalism * 0.3)

    # Hard penalties for egregious clickbait
    if sensationalism > 60 or clickbait_count >= 2:
        credibility = min(credibility, 25.0)
    elif sensationalism > 35:
        credibility = min(credibility, 45.0)

    # Clamping
    credibility_score = round(float(np.clip(credibility, 2.0, 98.0)), 1)
    
    # Final label based on combined credibility index
    if credibility_score >= 60.0:
        final_prediction = "REAL NEWS"
        is_real = True
        confidence = credibility_score
    elif credibility_score <= 40.0:
        final_prediction = "FAKE NEWS"
        is_real = False
        confidence = round(100.0 - credibility_score, 1)
    else:
        final_prediction = "UNVERIFIED / MIXED"
        is_real = False
        confidence = round(max(credibility_score, 100.0 - credibility_score), 1)

    # Build Explainable HTML Highlighting
    highlighted_html = generate_highlighted_html(text, token_score_map)

    # Optional Gemini verification
    gemini_result = None
    if verify_with_gemini:
        verdict, err = verify_claim_with_gemini(text, api_key=gemini_api_key)
        gemini_result = {
            "success": verdict is not None,
            "verdict": verdict,
            "error": err
        }

    return {
        "prediction": final_prediction,
        "is_real": is_real,
        "prob_real": round(prob_real * 100.0, 1),
        "prob_fake": round(prob_fake * 100.0, 1),
        "confidence": round(confidence, 1),
        "credibility_score": credibility_score,
        "word_contributions": contributions[:25],
        "top_real_words": top_real_words,
        "top_fake_words": top_fake_words,
        "highlighted_html": highlighted_html,
        "stylometrics": stylometrics,
        "bias_warning": bias_warning,
        "gemini_verification": gemini_result
    }


def generate_highlighted_html(text: str, token_score_map: dict) -> str:
    """
    Renders text into HTML with subtle green/red highlighting based on
    word attribution toward Real vs Fake.
    """
    # Tokenize preserving spaces and punctuations
    tokens = re.findall(r"(\w+|[^\w\s]|\s+)", text)
    html_parts = []
    
    # Determine maximum absolute score for normalization
    max_score = max([abs(s) for s in token_score_map.values()], default=1.0)
    if max_score == 0:
        max_score = 1.0

    for tok in tokens:
        clean_tok = tok.lower().strip()
        if clean_tok in token_score_map:
            score = token_score_map[clean_tok]
            # Normalized intensity between 0.15 and 0.8
            intensity = min(max(abs(score) / max_score, 0.2), 0.85)
            
            if score > 0:
                # Real News indicator -> Green tint
                bg_color = f"rgba(34, 197, 94, {intensity:.2f})"
                tooltip = f"Real Indicator (+{score:.3f})"
            else:
                # Fake News indicator -> Red tint
                bg_color = f"rgba(239, 68, 68, {intensity:.2f})"
                tooltip = f"Suspicious Indicator ({score:.3f})"
                
            escaped = html.escape(tok)
            span = (
                f'<span style="background-color: {bg_color}; padding: 1px 4px; border-radius: 4px; '
                f'font-weight: 500; cursor: help;" title="{tooltip}">{escaped}</span>'
            )
            html_parts.append(span)
        else:
            html_parts.append(html.escape(tok))

    return "".join(html_parts)


def get_top_model_features(n=20, model_path="fake_news_model.pkl", tfidf_path="tfidf_vectorizer.pkl"):
    """
    Retrieves the top n most predictive features for REAL and FAKE classes.
    """
    model, tfidf = load_models(model_path, tfidf_path)
    coef = model.coef_[0]
    feat = np.array(tfidf.get_feature_names_out())

    # Top Real (highest positive coefficients)
    top_real_idx = np.argsort(coef)[-n:][::-1]
    top_real = [{"feature": feat[i], "weight": round(float(coef[i]), 3)} for i in top_real_idx]

    # Top Fake (lowest negative coefficients)
    top_fake_idx = np.argsort(coef)[:n]
    top_fake = [{"feature": feat[i], "weight": round(float(coef[i]), 3)} for i in top_fake_idx]

    return {
        "top_real": top_real,
        "top_fake": top_fake,
        "vocabulary_size": len(feat)
    }
