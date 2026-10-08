"""Offline test of the whole cycle with a FAKE LLM (no Groq call). Run: python -m tests.test_answer"""
import os, shutil, json
from types import SimpleNamespace
os.environ["EMBEDDER"] = "fake"; os.environ["QDRANT_URL"] = ""
from core import config
shutil.rmtree(config.QDRANT_LOCAL_PATH, ignore_errors=True)
from ingest.csv_loader import load_laptops_csv
from ingest.index import index_documents
from core import answer

qc = index_documents(load_laptops_csv(config.CSV_PATH), recreate=True)

class FakeLLM:
    def __init__(self): self.calls = []
    def _create(self, **kw):
        self.calls.append(kw)
        sysmsg = kw["messages"][0]["content"]
        if "JSON filters" in sysmsg:
            out = json.dumps({"query": "coding, lightweight", "budget_max": "70000", "min_ram_gb": 16,
                              "needs_dedicated_gpu": False, "categories": ["Student", "toaster"], "brands": None})
        else:
            out = "FAKE ANSWER based on:\n" + kw["messages"][1]["content"][:200]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])
    @property
    def chat(self): return SimpleNamespace(completions=SimpleNamespace(create=self._create))

llm = FakeLLM()
ans, hits, req, note, evidence = answer.recommend("laptop for coding under 70k, light", qdrant_client=qc, llm_client=llm)
assert req.budget_max == 70000.0 and req.min_ram_gb == 16 and req.categories == ["student"], req
assert req.needs_dedicated_gpu is None
assert hits and all(h["price_inr"] <= 70000 and h["ram_gb"] >= 16 and h["category"] == "student" for h in hits)
assert "FAKE ANSWER" in ans and len(llm.calls) == 2
# static system prompt first -> cache friendly
assert llm.calls[1]["messages"][0]["content"] == answer.ANSWER_SYSTEM

# relaxing: impossible combo (Apple + 20k budget) must still degrade gracefully
req2 = answer.Requirements(query="laptop", budget_max=20000, brands=["Apple"])
h2, used, note2 = answer.retrieve_with_fallback(req2, 3, client=qc)
assert h2 == [] and note2 and "Nothing" in note2
req3 = answer.Requirements(query="laptop", budget_max=60000, brands=["Apple"])
h3, used3, note3 = answer.retrieve_with_fallback(req3, 3, client=qc)
assert h3 and note3 and all(h["price_inr"] <= 60000 for h in h3), (h3, note3)

# broken LLM output must not crash parsing
bad = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("rate limit")))))
r = answer.parse_requirements("cheap laptop", client=bad)
assert r.query == "cheap laptop"

class NoJsonMode(FakeLLM):
    def _create(self, **kw):
        if "response_format" in kw:
            raise RuntimeError("json mode not supported")
        return super()._create(**kw)
nj = NoJsonMode()
r = answer.parse_requirements("laptop for coding under 70k", client=nj)
assert r.budget_max == 70000.0, r
assert all(c.get("reasoning_effort") == "low" for c in nj.calls)   # default models are gpt-oss

# ---- greetings / intents ----
quiet = FakeLLM()
for greeting in ["Hi", "hello!", "Hey there", "good morning", "what can you do?"]:
    a, h, r, n, e = answer.recommend(greeting, qdrant_client=qc, llm_client=quiet)
    assert a == answer.INTRO and h == [] and e == [], greeting
assert quiet.calls == [], "greetings must not call the LLM"
assert answer.recommend("thanks", qdrant_client=qc, llm_client=quiet)[0] == answer.THANKS
assert answer.recommend("bye", qdrant_client=qc, llm_client=quiet)[0] == answer.BYE
assert answer.small_talk_reply("hi, laptop under 50k") is None      # real request must NOT be swallowed

class IntentLLM(FakeLLM):
    def __init__(self, intent): super().__init__(); self.intent = intent
    def _create(self, **kw):
        self.calls.append(kw)
        out = json.dumps({"query": "laptop", "intent": self.intent})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])
a, h, r, n, e = answer.recommend("I need a laptop", qdrant_client=qc, llm_client=IntentLLM("clarify"))
assert a == answer.CLARIFY and h == []
a, h, r, n, e = answer.recommend("what's the weather", qdrant_client=qc, llm_client=IntentLLM("other"))
assert a == answer.OFF_TOPIC and h == []
a, h, r, n, e = answer.recommend("laptop", qdrant_client=qc, llm_client=IntentLLM("garbage"))
assert r.intent == "recommend" and h        # unknown intent falls back to recommending
print("ALL CHECKS PASSED")
