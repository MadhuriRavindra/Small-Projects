"""Reddit opinions via the OFFICIAL API (praw). Two steps, so indexing never needs the API:

  1) python -m ingest.reddit_loader --fetch [--only LAP007 LAP027] [--posts 5]   -> data/web/reddit_cache.jsonl
  2) python -m ingest.index --sources reddit                                    -> reads the cache

Needs REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env (create a 'script' app at
reddit.com/prefs/apps). Check Reddit's current Data API terms before any public/commercial use.
Usernames are NOT stored. Only the text, score, subreddit and link are kept.
"""
import argparse
import datetime
import json
import os
import time

from core import config
from core.matching import aliases_for
from core.schema import Document
from ingest.catalog import load_catalog
from ingest.chunking import chunk_text

CACHE = config.DATA_DIR / "web" / "reddit_cache.jsonl"
SUBREDDITS = "LaptopsIndia+IndianGaming+IndiaTech+SuggestALaptop+laptops"
MIN_CHARS, MIN_SCORE = 80, 2


def _ok(text: str, score: int) -> bool:
    t = (text or "").strip()
    return len(t) >= MIN_CHARS and score >= MIN_SCORE and t not in ("[deleted]", "[removed]")


def collect(reddit, catalog, only=None, posts_per_product=5, comments_per_post=6, delay=1.0) -> list[dict]:
    """reddit = praw.Reddit(...) (or any object with the same .subreddit().search() shape)."""
    rows, sub = [], reddit.subreddit(SUBREDDITS)
    for c in catalog:
        if only and c["product_id"] not in only:
            continue
        query = f"{c['brand']} {c['model']}"
        aliases = aliases_for(c["brand"], c["model"])
        for post in sub.search(query, sort="relevance", time_filter="all", limit=posts_per_product):
            title = post.title or ""
            if not any(a in " ".join(f"{title} {post.selftext or ''}".lower().split()) for a in aliases):
                continue  # search matched loosely; keep only posts that really name the laptop
            link = f"https://www.reddit.com{post.permalink}"
            base = dict(product_id=c["product_id"], subreddit=str(post.subreddit), url=link, title=title)
            if _ok(post.selftext, int(post.score)):
                rows.append({**base, "kind": "post", "id": post.id, "score": int(post.score), "text": post.selftext})
            post.comment_sort = "top"
            post.comments.replace_more(limit=0)
            for cm in list(post.comments)[:comments_per_post]:
                if _ok(cm.body, int(cm.score)):
                    rows.append({**base, "kind": "comment", "id": cm.id, "score": int(cm.score), "text": cm.body})
        time.sleep(delay)
    return rows


def save_cache(rows: list[dict], path=CACHE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} items to {path}")


def load_reddit_documents(path=CACHE) -> list[Document]:
    if not path.exists():
        print(f"  no Reddit cache at {path} - run: python -m ingest.reddit_loader --fetch")
        return []
    by_id = {c["product_id"]: c for c in load_catalog()}
    today = datetime.date.today().isoformat()
    docs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        prod = by_id.get(r["product_id"])
        if not prod:
            continue
        for i, ch in enumerate(chunk_text(f"Reddit r/{r['subreddit']} - {r['title']}\n{r['text']}")):
            docs.append(Document(
                doc_id=f"{r['product_id']}#reddit#{r['id']}#{i}",
                text=f"{prod['brand']} {prod['model']} - {ch}",
                source_type="reddit",
                source_name=f"Reddit r/{r['subreddit']}",
                product_id=r["product_id"],
                fetched_at=today,
                meta={"brand": prod["brand"], "model": prod["model"], "url": r["url"],
                      "score": r["score"], "subreddit": r["subreddit"]},
            ))
    print(f"  {len(docs)} Reddit chunks from {path.name}")
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="call the Reddit API and refresh the cache")
    ap.add_argument("--only", nargs="*", help="product ids, e.g. LAP007 LAP027")
    ap.add_argument("--posts", type=int, default=5)
    a = ap.parse_args()
    if not a.fetch:
        ap.error("add --fetch to call the API (indexing reads the cache automatically)")
    import praw
    from dotenv import load_dotenv
    load_dotenv()
    reddit = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"], client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                         user_agent=os.environ.get("REDDIT_USER_AGENT", "laptop-advisor by u/yourname"))
    reddit.read_only = True
    save_cache(collect(reddit, load_catalog(), only=set(a.only) if a.only else None, posts_per_product=a.posts))
