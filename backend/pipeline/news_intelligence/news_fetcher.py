"""
News Fetcher — fetches recent articles about a horse from NewsAPI
and Google News RSS. No scraping, no authentication beyond NewsAPI key.

Returns raw article text ready for the LLM summariser.
"""

import os
import time
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Optional
from backend.utils.config import config
from backend.utils.cache import cache_get, cache_set
from backend.utils.logger import logger


# ── Google News RSS (free, no API key) ───────────────────────────────────────

def fetch_google_news_rss(query: str, max_articles: int = 5) -> list[dict]:
    """
    Fetches articles from Google News RSS feed.
    Free, no API key, rate-limited by Google so cache aggressively.
    """
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en&gl=GB&ceid=GB:en"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title":       entry.get("title", ""),
                "summary":     entry.get("summary", ""),
                "published":   entry.get("published", ""),
                "source":      entry.get("source", {}).get("title", "Google News"),
                "url":         entry.get("link", ""),
            })
        return articles
    except Exception as e:
        logger.warning(f"Google News RSS failed for '{query}': {e}")
        return []


# ── NewsAPI (free tier: 100 req/day) ─────────────────────────────────────────

def fetch_newsapi(query: str, days_back: int = 14, max_articles: int = 5) -> list[dict]:
    """
    Fetches articles from NewsAPI.org.
    Free tier: 100 requests/day, articles up to 1 month old.
    Set NEWS_API_KEY in .env to enable.
    """
    if not config.NEWS_API_KEY:
        return []

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":          query,
        "from":       from_date,
        "sortBy":     "relevancy",
        "language":   "en",
        "pageSize":   max_articles,
        "apiKey":     config.NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", [])[:max_articles]:
            articles.append({
                "title":     art.get("title", ""),
                "summary":   art.get("description", ""),
                "published": art.get("publishedAt", ""),
                "source":    art.get("source", {}).get("name", "NewsAPI"),
                "url":       art.get("url", ""),
            })
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI failed for '{query}': {e}")
        return []


# ── Master fetch function ─────────────────────────────────────────────────────

def fetch_horse_news(horse_name: str, trainer_name: str = "",
                     days_back: int = 14, use_cache: bool = True) -> list[dict]:
    """
    Fetches all available news for a horse from all sources.
    Results are cached for 1 hour to avoid hammering APIs.

    Args:
        horse_name   : e.g. "Thunderstrike"
        trainer_name : e.g. "John Gosden" (improves search precision)
        days_back    : how many days of news to look back
        use_cache    : cache results to avoid redundant API calls

    Returns:
        List of article dicts with title, summary, source, published, url
    """
    cache_key = f"news:{horse_name.lower().replace(' ', '_')}:{days_back}"
    if use_cache:
        cached = cache_get(cache_key, ttl_seconds=3600)
        if cached:
            logger.info(f"News cache hit for {horse_name}")
            return cached

    # Build search queries — horse name alone + with trainer for precision
    queries = [f'"{horse_name}" horse racing']
    if trainer_name:
        queries.append(f'"{horse_name}" {trainer_name}')

    all_articles = []
    seen_titles  = set()

    for query in queries:
        # Google News RSS — free, always try first
        rss_articles = fetch_google_news_rss(query, max_articles=4)
        for art in rss_articles:
            if art["title"] not in seen_titles:
                seen_titles.add(art["title"])
                all_articles.append(art)
        time.sleep(0.3)  # polite delay

        # NewsAPI — only if key is set
        api_articles = fetch_newsapi(query, days_back=days_back, max_articles=3)
        for art in api_articles:
            if art["title"] not in seen_titles:
                seen_titles.add(art["title"])
                all_articles.append(art)

    # Limit total articles sent to LLM
    all_articles = all_articles[:config.NEWS_MAX_ARTICLES]

    if use_cache:
        cache_set(cache_key, all_articles)

    logger.info(f"Fetched {len(all_articles)} articles for '{horse_name}'")
    return all_articles


def format_articles_for_llm(articles: list[dict]) -> str:
    """Formats articles into a single text block for the LLM prompt."""
    if not articles:
        return "No recent news articles found for this horse."

    parts = []
    for i, art in enumerate(articles, 1):
        parts.append(
            f"[Article {i}] {art['source']} — {art['published']}\n"
            f"Title: {art['title']}\n"
            f"Summary: {art['summary']}\n"
        )
    return "\n---\n".join(parts)