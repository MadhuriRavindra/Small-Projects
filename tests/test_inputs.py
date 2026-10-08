"""Offline test of the new inputs (PDF, web, Reddit) + evidence retrieval. Run: python -m tests.test_inputs"""
import json, os, shutil, tempfile, threading, functools, http.server
from pathlib import Path
from types import SimpleNamespace
os.environ["EMBEDDER"] = "fake"; os.environ["QDRANT_URL"] = ""
from core import config
shutil.rmtree(config.QDRANT_LOCAL_PATH, ignore_errors=True)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from core.matching import build_alias_index, match_text
from ingest.catalog import load_catalog
from ingest.csv_loader import load_laptops_csv
from ingest.index import index_documents
from ingest.pdf_loader import load_pdf_documents
from ingest.web_loader import load_web_documents
from ingest import reddit_loader
from core.retriever import search_evidence
from core import answer

catalog = load_catalog(); idx = build_alias_index(catalog)
pid = {f"{c['brand']} {c['model']}": c["product_id"] for c in catalog}
LEGION = pid["Lenovo Legion 5 15ARP9"]; PRO_M3 = pid["Apple MacBook Pro 14 M3"]; PRO_M3PRO = pid["Apple MacBook Pro 14 M3 Pro"]

# ---- matching ----
assert match_text("Legion 5 15ARP9 vs MacBook Pro 14 M3 Pro", idx) == {LEGION, PRO_M3PRO}   # M3 Pro must NOT also match plain M3
assert match_text("my Legion 5 runs hot", idx) == {LEGION}
assert match_text("nothing about laptops here", idx) == set()

tmp = Path(tempfile.mkdtemp())

# ---- PDF ----
pdfs = tmp / "pdfs"; pdfs.mkdir()
def make_pdf(name, pages):
    c = canvas.Canvas(str(pdfs / name), pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(40, y, line); y -= 16
        c.showPage()
    c.save()
make_pdf("lenovo_legion_5_15arp9_spec.pdf", [
    "FAKE TEST SPEC SHEET\nThe Legion 5 has a 80Wh battery and a 165Hz QHD display.\nCooling: dual fans with vapor chamber.",
    "Ports: 2x USB-C, 3x USB-A, HDMI 2.1, Ethernet.\nWarranty: 1 year onsite."])
make_pdf("macbook_pro_14_m3_pro.pdf", ["FAKE TEST: MacBook Pro M3 Pro supports up to 3 external displays."])
make_pdf("random_brochure.pdf", ["Not about any known laptop name."])
make_pdf("macbook_air_13_m2_scan.pdf", [""])   # no text layer -> reported, no chunks
docs = load_pdf_documents(pdfs)
assert {d.product_id for d in docs} == {LEGION, PRO_M3PRO}, {d.product_id for d in docs}   # scan + brochure add nothing
assert all(d.source_type == "pdf" and d.meta["page"] >= 1 for d in docs)
(pdfs / "pdf_map.csv").write_text(f"filename,product_id\nrandom_brochure.pdf,{PRO_M3}\n")
docs2 = load_pdf_documents(pdfs)
assert PRO_M3 in {d.product_id for d in docs2}   # explicit map rescues unmatched file

# ---- web (local server + saved page) ----
web = tmp / "web"; (web / "pages").mkdir(parents=True)
(web / "review.html").write_text("<html><body><article><h1>Legion 5 review</h1><p>" + ("The Legion 5 15ARP9 keeps temperatures in check even in long gaming sessions. " * 3) + "</p></article></body></html>")
(web / "pages" / "macbook_pro_14_m3_pro_notes.txt").write_text("The MacBook Pro 14 M3 Pro battery easily lasts a work day, owners say.")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
(web / "urls.csv").write_text(f"url,product_id\nhttp://127.0.0.1:{srv.server_port}/review.html,\n")
wdocs = load_web_documents(web, delay=0)
srv.shutdown()
assert {d.product_id for d in wdocs} == {LEGION, PRO_M3PRO}, {d.product_id for d in wdocs}
assert any("temperatures" in d.text for d in wdocs)

# ---- Reddit (mocked praw: same shape as the real one) ----
class Comments(list):
    def replace_more(self, limit=0): pass
def comment(i, body, score): return SimpleNamespace(id=i, body=body, score=score)
def post(i, title, body, score, comments):
    return SimpleNamespace(id=i, title=title, selftext=body, score=score, permalink=f"/r/LaptopsIndia/comments/{i}/x/",
                           subreddit="LaptopsIndia", comment_sort=None, comments=Comments(comments))
POSTS = [
    post("p1", "Legion 5 15ARP9 - 3 months review", "Using the Legion 5 15ARP9 daily for coding and gaming. Fans get loud but temps stay under 80C." * 2, 120,
         [comment("c1", "Battery lasts about 3 hours on the Legion 5 for light work, gaming is plugged in only, as expected.", 40),
          comment("c2", "lol", 50), comment("c3", "[deleted]", 30), comment("c4", "Great keyboard on this one, long typing sessions are comfortable and no flex at all.", 1)]),
    post("p2", "Best laptop for students?", "Looking for suggestions under 50k for college use, any ideas from the community here?", 30, []),
]
class Sub:
    def search(self, q, **kw): return POSTS
reddit = SimpleNamespace(subreddit=lambda name: Sub())
rows = reddit_loader.collect(reddit, catalog, only={LEGION}, delay=0)
kinds = sorted((r["kind"], r["id"]) for r in rows)
assert kinds == [("comment", "c1"), ("post", "p1")], kinds       # short/low-score/deleted/off-topic dropped
assert all("user" not in r and "author" not in r for r in rows)   # no usernames stored
cache = tmp / "reddit.jsonl"; reddit_loader.save_cache(rows, cache)
rdocs = reddit_loader.load_reddit_documents(cache)
assert rdocs and all(d.source_type == "reddit" and d.meta["subreddit"] == "LaptopsIndia" for d in rdocs)

# ---- index everything together, then retrieve evidence ----
client = index_documents(load_laptops_csv(config.CSV_PATH) + docs2 + wdocs + rdocs, recreate=True)
ev = search_evidence("gaming battery and cooling", [LEGION, PRO_M3PRO], per_product=3, client=client)
types_legion = {e["source_type"] for e in ev if e["product_id"] == LEGION}
assert types_legion == {"pdf", "web", "reddit"}, types_legion       # one of each source type surfaces
assert all(e["source_type"] != "csv" for e in ev)
assert any(e["source_type"] == "reddit" and e["score"] >= 2 for e in ev)   # upvotes preserved, not overwritten
assert search_evidence("x", [], client=client) == []

# product search must still return ONLY product cards (no chunks) after multi-source indexing
from core.retriever import Requirements, search
res = search(Requirements(query="gaming laptop battery cooling"), top_k=50, client=client)
assert res and all(r["source_type"] == "csv" for r in res)

# ---- whole cycle: evidence reaches the answer prompt, with source labels ----
class LLM:
    def __init__(self): self.calls = []
    def _create(self, **kw):
        self.calls.append(kw)
        if "JSON filters" in kw["messages"][0]["content"]:
            out = json.dumps({"query": "gaming laptop good cooling battery", "budget_max": 125000, "needs_dedicated_gpu": True,
                              "brands": ["Lenovo"], "intent": "recommend"})
        else:
            out = "OK"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])
    @property
    def chat(self): return SimpleNamespace(completions=SimpleNamespace(create=self._create))
llm = LLM()
a, hits, req, note, evidence = answer.recommend("gaming laptop with good cooling and battery, Lenovo", top_k=5, qdrant_client=client, llm_client=llm)
prompt = llm.calls[-1]["messages"][1]["content"]
assert "Evidence snippets" in prompt and "PDF spec sheet" in prompt and "Reddit" in prompt, prompt[-1500:]
assert "upvotes" in prompt
assert evidence and any(e["product_id"] == LEGION for e in evidence)
print("ALL CHECKS PASSED")
