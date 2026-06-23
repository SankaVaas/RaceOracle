"""
Step 1 of the RaceOracle pipeline — run this BEFORE train_model.py.

Usage:
    python scripts/ingest_data.py path/to/your/horse_races.csv

Reads the real CSV, engineers all features, saves to data/processed/horse_races.csv
Also saves data/processed/course_map.json so train_model.py knows the cardinality.
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline.data_ingestion.loader import ingest_real_data
from backend.utils.logger import logger

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_data.py <path_to_csv>")
        print("Example: python scripts/ingest_data.py data/raw/horse_races.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)

    df, course_map = ingest_real_data(csv_path, "./data/processed/horse_races.csv")

    # Save course map so train_model.py can read the correct cardinality
    meta = {
        "n_courses":    len(course_map),
        "course_map":   course_map,
        "n_rows":       len(df),
        "n_races":      df["race_id"].nunique(),
        "n_horses":     df["horse_name"].nunique(),
        "date_min":     str(df["race_date"].min()),
        "date_max":     str(df["race_date"].max()),
        "win_rate":     float(df["won"].mean()),
    }
    os.makedirs("./data/processed", exist_ok=True)
    with open("./data/processed/dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved dataset_meta.json → n_courses={len(course_map)}")
    logger.info("Next step: python scripts/train_model.py")