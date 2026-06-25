"""
Integration test for the news intelligence pipeline.
Run: python tests/integration/test_news_pipeline.py

Tests:
  1. News fetcher (Google News RSS — no API key needed)
  2. LLM summariser (Claude API — needs ANTHROPIC_API_KEY in .env)
  3. Sentiment/risk formatter
  4. Full pipeline end-to-end on a sample race
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.pipeline.news_intelligence.news_fetcher import (
    fetch_horse_news, format_articles_for_llm
)
from backend.pipeline.news_intelligence.llm_summarizer import summarise_horse_news
from backend.pipeline.news_intelligence.sentiment import (
    news_intel_to_model_features, news_intel_to_dashboard, risk_to_label
)
from backend.pipeline.news_intelligence import enrich_race_with_news
from backend.utils.logger import logger


# ── Sample race ───────────────────────────────────────────────────────────────

SAMPLE_HORSES = [
    {"name": "Frankel",      "trainer": "John Gosden"},
    {"name": "Enable",       "trainer": "John Gosden"},
    {"name": "Stradivarius", "trainer": "John Gosden"},
]

RACE_INFO = "Royal Ascot, 1m4f, Good going, Group 1"


def test_news_fetcher():
    print("\n── test_news_fetcher (Google News RSS) ──")
    articles = fetch_horse_news("Frankel", "John Gosden", days_back=30, use_cache=False)
    print(f"  Fetched {len(articles)} articles for Frankel")
    for art in articles[:2]:
        print(f"  [{art['source']}] {art['title'][:80]}")
    assert isinstance(articles, list)
    print("  ✅ PASSED")
    return articles


def test_article_formatter(articles):
    print("\n── test_article_formatter ──")
    text = format_articles_for_llm(articles)
    assert isinstance(text, str)
    print(f"  Formatted {len(articles)} articles into {len(text)} chars")
    print(f"  Preview: {text[:120]}...")
    print("  ✅ PASSED")
    return text


def test_llm_summariser(articles_text):
    print("\n── test_llm_summariser (Claude API) ──")
    intel = summarise_horse_news(
        horse_name="Frankel",
        articles_text=articles_text,
        race_info=RACE_INFO,
        use_cache=False,
    )
    print(f"  Summary    : {intel['summary'][:100]}...")
    print(f"  injury_risk: {intel['injury_risk']:.2f}  {risk_to_label('injury', intel['injury_risk'])}")
    print(f"  travel_risk: {intel['travel_risk']:.2f}  {risk_to_label('travel', intel['travel_risk'])}")
    print(f"  fatigue_risk:{intel['fatigue_risk']:.2f}  {risk_to_label('fatigue', intel['fatigue_risk'])}")
    print(f"  sentiment  : {intel['sentiment']:.2f}")
    print(f"  confidence : {intel['confidence']:.2f}")
    print(f"  flags      : {intel['flags']}")
    print(f"  positives  : {intel['positive_signals']}")

    assert 0.0 <= intel["injury_risk"]  <= 1.0
    assert 0.0 <= intel["travel_risk"]  <= 1.0
    assert 0.0 <= intel["fatigue_risk"] <= 1.0
    assert -1.0 <= intel["sentiment"]   <= 1.0
    print("  ✅ PASSED")
    return intel


def test_sentiment_formatter(intel):
    print("\n── test_sentiment_formatter ──")
    features = news_intel_to_model_features(intel)
    display  = news_intel_to_dashboard(intel)

    print(f"  Model features : {features}")
    print(f"  Overall risk   : {display['overall_risk']}%")
    print(f"  Sentiment      : {display['sentiment_label']}")
    print(f"  Risk flags     : {[f['label'] for f in display['risk_flags']]}")

    assert "injury_risk"  in features
    assert "travel_risk"  in features
    assert "fatigue_risk" in features
    assert "overall_risk" in display
    print("  ✅ PASSED")


def test_full_pipeline():
    print("\n── test_full_pipeline (3 horses) ──")
    enriched = enrich_race_with_news(SAMPLE_HORSES, RACE_INFO, use_cache=True)

    assert len(enriched) == 3
    print(f"\n  {'Horse':<20} {'Injury':>8} {'Travel':>8} {'Fatigue':>8} {'Sentiment':>10} {'Flags'}")
    print(f"  {'-'*70}")
    for h in enriched:
        intel = h.get("news_intel", {})
        flags = ", ".join(intel.get("flags", [])[:2]) or "none"
        print(f"  {h['name']:<20} "
              f"{intel.get('injury_risk', 0):.2f}     "
              f"{intel.get('travel_risk', 0):.2f}     "
              f"{intel.get('fatigue_risk', 0):.2f}     "
              f"{intel.get('sentiment', 0):>+.2f}       "
              f"{flags[:40]}")

    print("\n  ✅ FULL PIPELINE PASSED")
    return enriched


if __name__ == "__main__":
    os.makedirs("./data/cache", exist_ok=True)

    articles     = test_news_fetcher()
    articles_text = test_article_formatter(articles)
    intel        = test_llm_summariser(articles_text)
    test_sentiment_formatter(intel)
    enriched     = test_full_pipeline()

    print("\n" + "═"*60)
    print("ALL NEWS INTELLIGENCE TESTS PASSED")
    print("═"*60)
    print("\nNext step: python scripts/backtest.py")