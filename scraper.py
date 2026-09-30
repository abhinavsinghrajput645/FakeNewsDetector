"""
scraper.py - Web Article Scraper and Content Extractor
Fetches live articles from URLs, strips boilerplate, and prepares text for NLP analysis.
"""

import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def extract_article(url: str, timeout: int = 10) -> dict:
    """
    Fetches and extracts article text and metadata from a web URL.
    Returns:
        dict: {
            "success": bool,
            "title": str,
            "text": str,
            "author": str,
            "domain": str,
            "word_count": int,
            "error": str or None
        }
    """
    if not url or not url.strip():
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": "",
            "domain": "",
            "word_count": 0,
            "error": "URL is empty."
        }

    # Ensure valid scheme
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        domain = urlparse(url).netloc
    except Exception:
        domain = ""

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": "",
            "domain": domain,
            "word_count": 0,
            "error": "Connection timed out while trying to reach the website."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": "",
            "domain": domain,
            "word_count": 0,
            "error": f"Failed to fetch URL: {str(e)}"
        }

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove noisy elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript"]):
            element.decompose()

        # Extract title
        title = ""
        # 1. OpenGraph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        # 2. Document title
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        # 3. H1
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract author
        author = ""
        meta_author = soup.find("meta", attrs={"name": re.compile(r"author", re.I)})
        if meta_author and meta_author.get("content"):
            author = meta_author["content"].strip()

        # Extract main text
        # Check for article tag first
        article_tag = soup.find("article")
        source_container = article_tag if article_tag else soup

        # Extract paragraphs
        paragraphs = []
        for p in source_container.find_all("p"):
            p_text = p.get_text(separator=" ", strip=True)
            # Filter out boilerplate sentences / short copyright notices
            if len(p_text.split()) >= 6:
                paragraphs.append(p_text)

        extracted_text = "\n\n".join(paragraphs)

        # Fallback if no paragraphs found
        if not extracted_text:
            extracted_text = source_container.get_text(separator="\n", strip=True)
            # Remove excessive whitespace
            extracted_text = re.sub(r"\n\s*\n+", "\n\n", extracted_text)

        word_count = len(extracted_text.split())

        if word_count < 15:
            return {
                "success": False,
                "title": title,
                "text": extracted_text,
                "author": author,
                "domain": domain,
                "word_count": word_count,
                "error": "Could not extract sufficient article text from this page. Content may be paywalled or rendered with client-side JavaScript."
            }

        return {
            "success": True,
            "title": title,
            "text": extracted_text,
            "author": author,
            "domain": domain,
            "word_count": word_count,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "title": "",
            "text": "",
            "author": "",
            "domain": domain,
            "word_count": 0,
            "error": f"Error parsing article HTML: {str(e)}"
        }
