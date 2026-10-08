"""Shows whether you are talking to Qdrant CLOUD or the LOCAL folder.  Run: python -m tests.check_qdrant"""
from urllib.parse import urlparse
from core import config
from core.store import get_client

if config.QDRANT_URL:
    print("MODE: CLOUD ->", urlparse(config.QDRANT_URL).netloc, "| api key set:", bool(config.QDRANT_API_KEY))
else:
    print("MODE: LOCAL folder (QDRANT_URL is empty -> .env not found or not filled in)")
c = get_client()
if c.collection_exists(config.COLLECTION):
    print("collection", repr(config.COLLECTION), "points:", c.count(config.COLLECTION).count)
else:
    print("collection", repr(config.COLLECTION), "does not exist yet")
