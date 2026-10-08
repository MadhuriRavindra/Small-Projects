"""Web pages -> chunked Documents.

Two ways to feed it:
  A) data/web/urls.csv        columns: url,product_id   (product_id optional; blank = auto-detect per chunk)
  B) data/web/pages/*.html|*.txt|*.md   pages you saved yourself (best for sites that block bots)
Respects robots.txt, identifies itself, waits between requests. Check each site's terms of use first.
"""
import datetime
import hashlib
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from core import config
from core.matching import build_alias_index, match_text
from core.schema import Document
from ingest.catalog import load_catalog
from ingest.chunking import chunk_text

WEB_DIR = config.DATA_DIR / "web"
USER_AGENT = "LaptopAdvisorBot/0.1 (personal learning project)"
DELAY_S = 2.0


def extract_text(html: str) -> str:
    try:
        import trafilatura
        out = trafilatura.extract(html, include_comments=False, include_tables=True)
        if out:
            return out
    except Exception:
        pass
    import re
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def allowed_by_robots(url: str) -> bool:
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True  # robots.txt unreachable (e.g. 404) -> treated as no restrictions


def fetch(url: str) -> str | None:
    import requests
    if not allowed_by_robots(url):
        print(f"  SKIP {url}: disallowed by robots.txt")
        return None
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None


def _docs_from_text(text, label, ref, forced_pid, catalog_by_id, alias_index, today, kind_id) -> list[Document]:
    docs = []
    for i, ch in enumerate(chunk_text(text)):
        pids = {forced_pid} if forced_pid else match_text(ch, alias_index)
        for pid in pids:
            if pid not in catalog_by_id:
                continue
            prod = catalog_by_id[pid]
            docs.append(Document(
                doc_id=f"{pid}#web#{kind_id}#{i}",
                text=f"{prod['brand']} {prod['model']} - {ch}",
                source_type="web",
                source_name=f"Web: {label}",
                product_id=pid,
                fetched_at=today,
                meta={"brand": prod["brand"], "model": prod["model"], "url": ref},
            ))
    return docs


def load_web_documents(web_dir: Path = WEB_DIR, delay: float = DELAY_S) -> list[Document]:
    catalog = load_catalog()
    by_id = {c["product_id"]: c for c in catalog}
    alias_index = build_alias_index(catalog)
    today = datetime.date.today().isoformat()
    docs: list[Document] = []

    urls_csv = web_dir / "urls.csv"
    if urls_csv.exists():
        df = pd.read_csv(urls_csv, dtype=str).fillna("")
        for n, r in enumerate(df.itertuples()):
            url = r.url.strip()
            if not url:
                continue
            html = fetch(url)
            if n < len(df) - 1:
                time.sleep(delay)
            if not html:
                continue
            text = extract_text(html)
            got = _docs_from_text(text, urlparse(url).netloc, url, (r.product_id.strip() or None),
                                  by_id, alias_index, today, hashlib.sha1(url.encode()).hexdigest()[:8])
            print(f"  {url}: {len(got)} chunks")
            docs += got

    pages_dir = web_dir / "pages"
    if pages_dir.exists():
        for path in sorted(pages_dir.iterdir()):
            if path.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = extract_text(raw) if path.suffix.lower() in {".html", ".htm"} else raw
            forced = None
            ids = match_text(path.stem.replace("_", " ").replace("-", " "), alias_index)
            if len(ids) == 1:
                forced = next(iter(ids))
            got = _docs_from_text(text, path.name, f"file:{path.name}", forced, by_id, alias_index, today,
                                  hashlib.sha1(path.name.encode()).hexdigest()[:8])
            print(f"  {path.name}: {len(got)} chunks" + (f" (forced to {forced})" if forced else " (auto-matched)"))
            docs += got
    if not docs:
        print(f"  no web sources found (add {urls_csv} or files in {pages_dir})")
    return docs
