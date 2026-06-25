"""
News Intelligence Pipeline — master entry point.

Usage:
    from backend.pipeline.news_intelligence import enrich_race_with_news

    horses = [
        {"name": "Thunderstrike", "trainer": "J. Gosden", ...},
        ...
    ]
    race_info = "Royal Ascot, 1m4f, Good going"
    enriched  = enrich_race_with_news(horses, race_info)

    # Each horse now has:
    #   horse["news_intel"]    — raw LLM output
    #   horse["news_features"] — {injury_risk, travel_risk, fatigue_risk} for model
    #   horse["news_display"]  — formatted for React dashboard
"""

import time
from backend.pipeline.news_intelligence.news_fetcher import (
    fetch_horse_news, format_articles_for_llm
)
from backend.pipeline.news_intelligence.llm_summarizer import summarise_horse_news
from backend.pipeline.news_intelligence.sentiment import batch_news_intel
from backend.utils.logger import logger


def enrich_horse_with_news(horse: dict, race_info: str,
                            use_cache: bool = True) -> dict:
    """
    Fetch and analyse news for a single horse.
    Adds news_intel, news_features, news_display keys to the horse dict.
    """
    name    = horse.get("name", horse.get("horseName", "Unknown"))
    trainer = horse.get("trainer", horse.get("trainerName", ""))

    # 1. Fetch articles
    articles = fetch_horse_news(name, trainer, days_back=14, use_cache=use_cache)

    # 2. Format for LLM
    articles_text = format_articles_for_llm(articles)

    # 3. Claude API summarisation + risk extraction
    intel = summarise_horse_news(
        horse_name=name,
        articles_text=articles_text,
        race_info=race_info,
        use_cache=use_cache,
    )

    horse["news_intel"]    = intel
    horse["news_articles"] = articles
    return horse


def enrich_race_with_news(horses: list, race_info: str,
                           use_cache: bool = True,
                           delay_between_horses: float = 0.5) -> list:
    """
    Enrich all horses in a race with news intelligence.
    Processes sequentially with a small delay to be polite to APIs.

    Args:
        horses                  : list of horse dicts
        race_info               : e.g. "Royal Ascot R3, 1m4f, Good going"
        use_cache               : use cached results where available
        delay_between_horses    : seconds between API calls

    Returns:
        List of enriched horse dicts with news_intel, news_features, news_display
    """
    logger.info(f"Enriching {len(horses)} horses with news intelligence...")
    enriched = []

    for i, horse in enumerate(horses):
        name = horse.get("name", horse.get("horseName", f"Horse {i+1}"))
        logger.info(f"  [{i+1}/{len(horses)}] Fetching news for {name}...")

        try:
            horse = enrich_horse_with_news(horse, race_info, use_cache=use_cache)
        except Exception as e:
            logger.error(f"News intelligence failed for {name}: {e}")
            horse["news_intel"]    = {
                "horse_name": name, "summary": "News fetch failed.",
                "injury_risk": 0.0, "travel_risk": 0.0, "fatigue_risk": 0.0,
                "sentiment": 0.0, "confidence": 0.0, "flags": [], "positive_signals": []
            }

        enriched.append(horse)
        if i < len(horses) - 1:
            time.sleep(delay_between_horses)

    # Format for model features + dashboard display
    enriched = batch_news_intel(enriched)

    total_flags = sum(len(h.get("news_intel", {}).get("flags", [])) for h in enriched)
    logger.info(f"News intelligence complete. {total_flags} risk flags detected across {len(enriched)} horses.")
    return enriched