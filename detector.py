"""
detector.py - NLP Engine, Explainability Layer, and Stylometric Analysis
for Fake News Detection.
"""

import os
import re
import html
import joblib
import numpy as np

# Cache instances
_MODEL = None
_TFIDF = None

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


def analyze_text(text: str, model_path="fake_news_model.pkl", tfidf_path="tfidf_vectorizer.pkl") -> dict:
    """
    Performs comprehensive analysis:
    - ML inference (Real vs Fake probability)
    - Feature-level explainability (word attributions)
    - Stylometric signals
    - Composite Credibility Index
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
            "bias_warning": None
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
        if sensationalism < 15 and clickbait_count == 0 and total_fake_pull < 0.6:
            # Neutral factual text without sensationalism or known fake markers
            # Adjust for the trained model's negative intercept (-1.24) and domain gap
            neutral_boost = 55.0 - (total_fake_pull * 20.0)
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
        "bias_warning": bias_warning
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
