import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./data/models")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")

    # Model hyperparameters (CPU-optimised defaults)
    TAB_TRANSFORMER_DIM: int = 32
    TAB_TRANSFORMER_DEPTH: int = 3
    TAB_TRANSFORMER_HEADS: int = 4
    BATCH_SIZE: int = 256
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3

    # News
    NEWS_MAX_ARTICLES: int = 5
    NEWS_LOOKBACK_DAYS: int = 14

config = Config()
