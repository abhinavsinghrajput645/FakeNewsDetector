"""
api.py - FastAPI REST API for Fake News Detection, Explainability, and Article Scraping.
Provides endpoints for programmatic inference, browser extensions, and integrations.
"""

from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from detector import analyze_text, get_top_model_features, load_models
from scraper import extract_article

app = FastAPI(
    title="Fake News Detection & Explainability API",
    description="Production-grade REST API providing ML-based fake news prediction, word-level XAI attribution, stylometrics, and live article scraping.",
    version="2.0.0"
)

# Enable CORS for browser extensions and third-party web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw text or article body to analyze.")
    url: Optional[str] = Field(None, description="Web URL of the news article to scrape and analyze.")


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="Web URL of the article to extract.")


@app.on_event("startup")
def startup_event():
    """Warm up model and vectorizer in memory on server launch."""
    try:
        load_models()
        print("[API] Models loaded successfully.")
    except Exception as e:
        print(f"[API ERROR] Failed to preload models: {e}")


@app.get("/")
def root():
    return {
        "name": "Fake News Detection & Explainability API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "endpoints": ["/health", "/predict", "/features", "/scrape"]
    }


@app.get("/health")
def health_check():
    """Health check endpoint with model status and vocabulary metadata."""
    try:
        model, tfidf = load_models()
        vocab_size = len(tfidf.vocabulary_)
        model_name = type(model).__name__
        status = "healthy"
    except Exception as e:
        status = f"unhealthy: {str(e)}"
        vocab_size = 0
        model_name = "unknown"

    return {
        "status": status,
        "model_type": model_name,
        "vocabulary_size": vocab_size,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Analyzes an article submitted via raw text or web URL.
    Returns credibility index, ML probabilities, word attributions, and stylometrics.
    """
    article_meta = None
    target_text = req.text or ""

    if req.url:
        scrape_result = extract_article(req.url)
        if not scrape_result.get("success"):
            raise HTTPException(status_code=400, detail=scrape_result.get("error", "Failed to extract article from URL."))
        
        target_text = f"{scrape_result.get('title', '')}\n\n{scrape_result.get('text', '')}".strip()
        article_meta = {
            "title": scrape_result.get("title"),
            "author": scrape_result.get("author"),
            "domain": scrape_result.get("domain"),
            "word_count": scrape_result.get("word_count")
        }

    if not target_text or not target_text.strip():
        raise HTTPException(status_code=422, detail="Either 'text' or a valid 'url' with readable text must be provided.")

    try:
        result = analyze_text(target_text)
        if article_meta:
            result["article_metadata"] = article_meta
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/features")
def features_endpoint(n: int = Query(20, ge=5, le=100)):
    """Returns top predictive real and fake lexical features with model coefficients."""
    try:
        return get_top_model_features(n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model features: {str(e)}")


@app.post("/scrape")
def scrape_endpoint(req: ScrapeRequest):
    """Fetches article content from a web URL without running classification."""
    result = extract_article(req.url)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
