import json, os, hashlib, time
from backend.utils.config import config

def _key_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(config.CACHE_DIR, f"{h}.json")

def cache_get(key: str, ttl_seconds: int = 3600):
    path = _key_path(key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        entry = json.load(f)
    if time.time() - entry["ts"] > ttl_seconds:
        os.remove(path)
        return None
    return entry["data"]

def cache_set(key: str, data):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = _key_path(key)
    with open(path, "w") as f:
        json.dump({"ts": time.time(), "data": data}, f)
