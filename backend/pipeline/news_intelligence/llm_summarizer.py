"""
LLM Summariser — uses Claude API to extract structured intelligence
from raw news articles about a horse.

Outputs:
  - summary      : 2-3 sentence human-readable summary for the dashboard
  - injury_risk  : float 0-1 (feeds directly into model as a feature)
  - travel_risk  : float 0-1
  - fatigue_risk : float 0-1
  - sentiment    : float -1 to +1 (negative = bad news, positive = good news)
  - flags        : list of specific concerns extracted from text
  - confidence   : how confident the LLM is in its assessment
"""

import json
import anthropic
from backend.utils.config import config
from backend.utils.cache import cache_get, cache_set
from backend.utils.logger import logger


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional horse racing intelligence analyst.
Your job is to read news articles about a racehorse and extract structured
risk signals that will feed into a prediction model.

You must respond with ONLY valid JSON — no preamble, no explanation, no markdown.
"""

def build_user_prompt(horse_name: str, race_info: str, articles_text: str) -> str:
    return f"""Analyse these news articles about the racehorse "{horse_name}" 
racing in: {race_info}

ARTICLES:
{articles_text}

Extract the following and return as JSON only:

{{
  "summary": "2-3 sentence summary of the horse's current condition and news",
  "injury_risk": <float 0.0-1.0, where 1.0 = confirmed serious injury>,
  "travel_risk": <float 0.0-1.0, where 1.0 = long international travel just completed>,
  "fatigue_risk": <float 0.0-1.0, where 1.0 = raced very recently or overraced>,
  "sentiment": <float -1.0 to 1.0, where -1=very negative news, 0=neutral, 1=very positive>,
  "flags": [<list of specific concern strings, e.g. "reported tendon inflammation", "travelled from USA">],
  "confidence": <float 0.0-1.0, how confident you are given the available articles>,
  "positive_signals": [<list of positive signals e.g. "strong workout reported", "trainer confident">]
}}

Risk scoring guide:
- injury_risk > 0.5: any mention of injury, lameness, vet visits, soreness, setbacks
- injury_risk > 0.8: confirmed injury, withdrawal risk, won't run at full capacity
- travel_risk > 0.3: international travel, long trip, arrived yesterday
- travel_risk > 0.7: intercontinental flight within 48 hours of race
- fatigue_risk > 0.4: raced within last 7 days, or 3+ races in 30 days
- fatigue_risk > 0.7: clearly tired horse, trainer mentioned needing a rest

If no articles are found or articles are irrelevant, return all risks as 0.0 and confidence as 0.1.
"""


# ── Main summariser ───────────────────────────────────────────────────────────

def summarise_horse_news(horse_name: str, articles_text: str,
                          race_info: str = "upcoming race",
                          use_cache: bool = True) -> dict:
    """
    Calls Claude API to extract structured intelligence from news articles.

    Args:
        horse_name    : name of the horse
        articles_text : formatted article text from news_fetcher.format_articles_for_llm()
        race_info     : brief race description e.g. "Royal Ascot, 1m4f, Good going"
        use_cache     : cache results for 2 hours

    Returns:
        dict with summary, injury_risk, travel_risk, fatigue_risk,
             sentiment, flags, confidence, positive_signals
    """
    cache_key = f"llm:{horse_name.lower().replace(' ', '_')}"
    if use_cache:
        cached = cache_get(cache_key, ttl_seconds=7200)
        if cached:
            logger.info(f"LLM cache hit for {horse_name}")
            return cached

    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — returning neutral risk scores")
        return _neutral_result(horse_name, reason="No API key configured")

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": build_user_prompt(horse_name, race_info, articles_text)
            }]
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        # Clamp all float values to valid ranges
        result["injury_risk"]  = float(max(0.0, min(1.0, result.get("injury_risk", 0.0))))
        result["travel_risk"]  = float(max(0.0, min(1.0, result.get("travel_risk", 0.0))))
        result["fatigue_risk"] = float(max(0.0, min(1.0, result.get("fatigue_risk", 0.0))))
        result["sentiment"]    = float(max(-1.0, min(1.0, result.get("sentiment", 0.0))))
        result["confidence"]   = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
        result["flags"]            = result.get("flags", [])
        result["positive_signals"] = result.get("positive_signals", [])
        result["horse_name"]       = horse_name

        logger.info(
            f"News intelligence for {horse_name}: "
            f"injury={result['injury_risk']:.2f} "
            f"travel={result['travel_risk']:.2f} "
            f"fatigue={result['fatigue_risk']:.2f} "
            f"sentiment={result['sentiment']:.2f}"
        )

        if use_cache:
            cache_set(cache_key, result)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {horse_name}: {e}\nRaw: {raw[:200]}")
        return _neutral_result(horse_name, reason="LLM response parse error")
    except Exception as e:
        logger.error(f"Claude API error for {horse_name}: {e}")
        return _neutral_result(horse_name, reason=str(e))


def _neutral_result(horse_name: str, reason: str = "") -> dict:
    """Returns neutral (zero-risk) scores when LLM is unavailable."""
    return {
        "horse_name":       horse_name,
        "summary":          f"No news intelligence available for {horse_name}. {reason}".strip(),
        "injury_risk":      0.0,
        "travel_risk":      0.0,
        "fatigue_risk":     0.0,
        "sentiment":        0.0,
        "confidence":       0.1,
        "flags":            [],
        "positive_signals": [],
    }