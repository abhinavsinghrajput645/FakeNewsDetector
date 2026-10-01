"""
api.py - FastAPI REST API for Fake News Detection, Explainability, and Article Scraping.
Provides endpoints for programmatic inference, browser extensions, and integrations.
"""

from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from detector import (
    analyze_text,
    get_top_model_features,
    load_models,
    verify_claim_with_gemini,
    is_valid_api_key_format,
    get_gemini_api_key
)
from scraper import extract_article

# Load environment variables
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model and vectorizer in memory on server launch."""
    try:
        load_models()
        print("[API] Models loaded successfully.")
    except Exception as e:
        print(f"[API ERROR] Failed to preload models: {e}")
    yield

app = FastAPI(
    title="Fake News Detection & Explainability API",
    description="Production-grade REST API providing ML-based fake news prediction, word-level XAI attribution, stylometrics, Gemini AI fact-checking, and live article scraping.",
    version="2.1.0",
    lifespan=lifespan
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
    verify_with_gemini: Optional[bool] = Field(False, description="Run optional Google Gemini semantic fact-checking.")
    gemini_api_key: Optional[str] = Field(None, description="Optional override Gemini API key. Defaults to backend GEMINI_API_KEY from environment or .env.")


class FactCheckRequest(BaseModel):
    text: str = Field(..., description="Statement or claim to verify using Google Gemini.")
    gemini_api_key: Optional[str] = Field(None, description="Optional override Gemini API key.")


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="Web URL of the article to extract.")




@app.get("/")
def root():
    return {
        "name": "Fake News Detection & Explainability API",
        "version": "2.1.0",
        "docs_url": "/docs",
        "endpoints": ["/health", "/gemini/status", "/predict", "/verify", "/features", "/scrape"]
    }


@app.get("/gemini/status")
def gemini_status():
    """Checks whether Gemini API key is configured in the backend environment or .env."""
    key = get_gemini_api_key()
    configured = bool(key and is_valid_api_key_format(key))
    masked_key = f"{key[:6]}...{key[-4:]}" if (configured and len(key) >= 10) else None
    return {
        "gemini_configured": configured,
        "masked_key": masked_key,
        "help": "Set GEMINI_API_KEY in your .env file or backend environment to enable AI fact checking."
    }


@app.post("/verify")
def verify_claim_endpoint(req: FactCheckRequest):
    """Directly verifies a claim using Google Gemini AI fact-checking reasoning."""
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text claim cannot be empty.")
    
    verdict, err = verify_claim_with_gemini(req.text, api_key=req.gemini_api_key)
    if err and not verdict:
        raise HTTPException(status_code=400, detail=err)
    return {
        "success": True,
        "claim": req.text,
        "gemini_reasoning": verdict
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
    Returns credibility index, ML probabilities, word attributions, stylometrics,
    and optional Gemini AI fact-checking reasoning.
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
        result = analyze_text(
            target_text,
            verify_with_gemini=req.verify_with_gemini or False,
            gemini_api_key=req.gemini_api_key
        )
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
