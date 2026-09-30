"""
evaluate.py - Evaluation and Bias Audit Utility for Fake News Classifiers
Evaluates accuracy on benchmark datasets and tests model vulnerability to wire-service shortcut bias.
"""

import argparse
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset_utils import get_benchmark_dataframe, scrub_wire_bias

def audit_wire_bias(model, vectorizer):
    """
    Tests whether the model changes its verdict on neutral facts
    solely because 'Reuters' or a weekday stamp is added or removed.
    """
    neutral_facts = [
        "Astronomers observed a supernova explosion in a distant galaxy using infrared telescopes.",
        "The municipality approved construction of a new water filtration facility to upgrade aging pipes.",
        "Geologists recorded a magnitude 4.2 earthquake along the offshore fault line with no reported injuries.",
        "The national research council announced an annual grant program for clean energy technologies."
    ]

    print("\n" + "=" * 60)
    print("[BIAS AUDIT] Testing Shortcut Vulnerability (Reuters/Dateline)")
    print("=" * 60)

    flipped_count = 0
    for fact in neutral_facts:
        wire_version = f"WASHINGTON (Reuters) - {fact}"
        
        vec_plain = vectorizer.transform([fact.lower()])
        vec_wire = vectorizer.transform([wire_version.lower()])
        
        prob_plain = model.predict_proba(vec_plain)[0][1] * 100
        prob_wire = model.predict_proba(vec_wire)[0][1] * 100

        pred_plain = "REAL" if prob_plain >= 50 else "FAKE"
        pred_wire = "REAL" if prob_wire >= 50 else "FAKE"

        print(f"\nFact: \"{fact[:65]}...\"")
        print(f" -> Plain text:     {pred_plain} (Real Prob: {prob_plain:.1f}%)")
        print(f" -> With 'Reuters': {pred_wire} (Real Prob: {prob_wire:.1f}%)")
        
        if pred_plain != pred_wire or (prob_wire - prob_plain) > 25:
            flipped_count += 1
            print("    [!] Vulnerability detected: Adding 'Reuters' altered confidence significantly!")

    print("\n" + "-" * 60)
    print(f"Audit Summary: {flipped_count}/{len(neutral_facts)} facts strongly skewed by wire-service markers.")
    print("=" * 60)


def evaluate_model(model_path="fake_news_model.pkl", tfidf_path="tfidf_vectorizer.pkl"):
    """Evaluates the model on the benchmark dataset."""
    print("=" * 60)
    print(f"[EVALUATE] Loading Model: {model_path}")
    print(f"[EVALUATE] Loading Vectorizer: {tfidf_path}")
    print("=" * 60)

    model = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)

    # Benchmark evaluation
    df = get_benchmark_dataframe(clean_bias=False)
    combined = (df["title"] + " " + df["text"]).tolist()
    X = np.array(combined, dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)

    X_clean = [t.lower() for t in X]
    X_vec = vectorizer.transform(X_clean)

    y_pred = model.predict(X_vec)
    acc = accuracy_score(y, y_pred)

    print("\n--- Benchmark Dataset Performance ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=["Fake News (0)", "Real News (1)"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Audit wire bias
    if hasattr(model, "predict_proba"):
        audit_wire_bias(model, vectorizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model and audit bias.")
    parser.add_argument("--model", type=str, default="fake_news_model.pkl")
    parser.add_argument("--tfidf", type=str, default="tfidf_vectorizer.pkl")
    args = parser.parse_args()

    evaluate_model(args.model, args.tfidf)
