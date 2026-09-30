"""
train.py - Reproducible Training and Evaluation Pipeline for Fake News Classifiers
Supports model selection (Logistic Regression, Passive Aggressive, Linear SVC),
wire-service debiasing, and performance reporting.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_utils import get_benchmark_dataframe, load_custom_dataset, scrub_wire_bias

def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Configures a sublinear TF-IDF vectorizer with unigrams and bigrams."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
        min_df=1
    )


def get_classifier(model_type: str):
    """Factory for model classifiers."""
    model_type = model_type.lower()
    if model_type == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif model_type == "passive_aggressive":
        # Calibrate to obtain predict_proba
        base = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        return CalibratedClassifierCV(base, method="sigmoid")
    elif model_type == "linear_svc":
        base = LinearSVC(C=1.0, random_state=42, dual=False)
        return CalibratedClassifierCV(base, method="sigmoid")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'logistic', 'passive_aggressive', or 'linear_svc'.")


def train_pipeline(data_path: str = None, model_type: str = "logistic", clean_bias: bool = True, output_dir: str = "."):
    """Executes the complete training, evaluation, and serialization pipeline."""
    print("=" * 60)
    print(f"[TRAIN] Training Fake News Classifier [Model: {model_type}, Debias Wire: {clean_bias}]")
    print("=" * 60)

    if data_path and os.path.exists(data_path):
        print(f"Loading custom dataset from: {data_path}")
        df = load_custom_dataset(data_path, clean_bias=clean_bias)
        X = np.array(df["full_text"].tolist(), dtype=object)
        y = np.array(df["label"].tolist(), dtype=int)
    else:
        print("Using curated benchmark dataset...")
        df = get_benchmark_dataframe(clean_bias=clean_bias)
        # Augment with titles and texts
        combined = (df["title"] + " " + df["text"]).tolist()
        X = np.array(combined, dtype=object)
        y = np.array(df["label"].tolist(), dtype=int)

    print(f"Dataset Size: {len(X)} samples (Real: {sum(y == 1)}, Fake: {sum(y == 0)})")

    # Train / Test split
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = build_vectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)} features")

    # Classifier training
    print(f"Training {model_type} classifier...")
    clf = get_classifier(model_type)
    clf.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake News (0)", "Real News (1)"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Output paths
    os.makedirs(output_dir, exist_ok=True)
    model_out = os.path.join(output_dir, "fake_news_model_new.pkl")
    tfidf_out = os.path.join(output_dir, "tfidf_vectorizer_new.pkl")

    joblib.dump(clf, model_out)
    joblib.dump(vectorizer, tfidf_out)
    print(f"\nArtifacts saved cleanly to:")
    print(f" - Model: {model_out}")
    print(f" - Vectorizer: {tfidf_out}")
    print("=" * 60)
    return clf, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate fake news detection models.")
    parser.add_argument("--data", type=str, default=None, help="Path to external CSV dataset.")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "passive_aggressive", "linear_svc"])
    parser.add_argument("--clean-bias", action="store_true", default=True, help="Scrub wire service markers.")
    parser.add_argument("--out", type=str, default=".", help="Directory to save model artifacts.")

    args = parser.parse_args()
    train_pipeline(data_path=args.data, model_type=args.model, clean_bias=args.clean_bias, output_dir=args.out)
