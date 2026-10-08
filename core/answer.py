"""Step 1: user text -> Requirements (LLM).  Step 3: retrieved laptops -> comparison answer (LLM).

Prompt caching on Groq is automatic for supported models: keep the STATIC text first
(system prompts below) and the changing text (user request, retrieved laptops) last.
"""
import json
import re
from dataclasses import replace

from core import config
from core.retriever import Requirements, search, search_evidence

CATEGORIES = ["budget", "student", "ultraportable", "business", "gaming", "creator"]
BRANDS = ["Lenovo", "HP", "Dell", "ASUS", "Acer", "Apple", "Samsung", "MSI", "Honor", "Infinix"]

# ---------- conversation handling (no LLM needed for these) ----------
INTRO = ("Hello! I'm your Laptop Advisor for the Indian market. How can I help you today?\n\n"
         "Tell me your **budget** and what you'll mainly use the laptop for (coding, college, gaming, video editing...), "
         "plus anything you care about, like weight, battery life, or brand. I'll compare the best options for you.")
THANKS = "You're welcome! If you want to compare more laptops or change your budget, just tell me."
BYE = "Goodbye! Come back anytime you want help choosing a laptop."
OFF_TOPIC = ("I can only help with choosing laptops. Tell me your budget and what you'll use it for, "
             "and I'll find and compare options.")
CLARIFY = ("Happy to help you pick one! Two quick questions so I can recommend well:\n\n"
           "1. What's your **budget** (in rupees)?\n2. What will you **mainly use it for** (coding, college, gaming, editing, office work)?")

_GREET = re.compile(r"^\s*(hi+|hello+|hey+|hola|namaste|namaskar|good\s*(morning|afternoon|evening)|greetings|yo|sup|hii+)"
                    r"(\s+(there|advisor|bot|claude|everyone|sir|madam))?\s*[!.?]*\s*$", re.I)
_THANKS = re.compile(r"^\s*(thanks?|thank\s*you|thx|ty|cheers|great|awesome|nice|cool)(\s+(a lot|so much|you))?\s*[!.]*\s*$", re.I)
_BYE = re.compile(r"^\s*(bye+|goodbye|good\s*night|see\s*you|cya|tata)\s*[!.]*\s*$", re.I)
_ABOUT = re.compile(r"\b(who are you|what (can|do) you do|how (do|does) (you|this) work|what are you|help me use)\b", re.I)


def small_talk_reply(text: str) -> str | None:
    """Instant, free replies for greetings / thanks / goodbyes / 'what can you do'. None = a real request."""
    t = text.strip()
    if _GREET.match(t) or _ABOUT.search(t):
        return INTRO
    if _THANKS.match(t):
        return THANKS
    if _BYE.match(t):
        return BYE
    return None


# ---------- static prompts (keep identical between requests so they can be cached) ----------
PARSE_SYSTEM = f"""You convert a shopper's laptop request into JSON filters for a search engine.
Return ONLY one JSON object with exactly these keys. Use null when the user did not state it.

- "query": short string of the soft preferences (use cases, features, feel). Example: "coding, lightweight, long battery". Never null; use "laptop" if nothing else.
- "budget_max": number in Indian rupees (INR) or null.
- "budget_min": number in INR or null.
- "min_ram_gb": integer or null.
- "min_storage_gb": integer or null.
- "max_weight_kg": number or null.
- "needs_dedicated_gpu": true if the user wants gaming, 3D, or a dedicated GPU; otherwise null.
- "categories": list chosen only from {CATEGORIES}, only if the user clearly asks for that type; otherwise null.
- "brands": list chosen only from {BRANDS}, only if the user names brands; otherwise null.
- "min_rating": number or null.
- "intent": one of:
    "recommend" = the user wants laptop suggestions AND gave at least one detail (budget, use case, brand, a spec, a preference), or this is a follow-up that adjusts an earlier laptop request (e.g. "make it lighter", "only Lenovo", "cheaper").
    "clarify"   = the user wants a laptop but gave NO usable detail at all (e.g. "I need a laptop", "suggest something") and there is no earlier request to build on.
    "other"     = not about buying a laptop (weather, jokes, coding help, general chat).

Rules:
- Only set hard filters (budget, RAM, GPU, weight...) when the user states or clearly implies them. Do not guess.
- 70k = 70000. 1 lakh = 100000. "under 70k" -> budget_max 70000. "around 60k" -> budget_min 51000, budget_max 69000.
- If earlier user messages are given, merge them: the latest message overrides conflicting earlier details.
- Greetings are handled elsewhere; if one reaches you, use "other".
- Output JSON only. No explanation."""

ANSWER_SYSTEM = """You are a laptop buying advisor for shoppers in India. Prices are in Indian rupees (INR).

You are given the shopper's request and a list of candidate laptops retrieved from a database.
Rules:
1. Use ONLY the laptops and facts provided. Never invent specs, prices, or models. If something is not in the data, say it is not available.
2. Recommend the best 1 to 3 laptops. Start with a one-line top pick, then compare the others briefly.
3. Tie every point to the shopper's stated needs (budget, use case, weight, battery, etc.).
4. Mention real trade-offs (weight, battery, noise, build) using the pros and cons provided.
5. If no laptop fits perfectly, say so plainly and explain which requirement was relaxed.
6. Evidence snippets may come from different sources: PDF = manufacturer spec sheets (facts), Reddit and web = opinions and reviews. When you use a snippet, say where it comes from in plain words (for example "a Reddit owner reports..." or "the spec sheet lists..."). Treat opinions as opinions: one comment is not a consensus, so say "some owners" unless several snippets agree. If a snippet conflicts with the database, point out the difference. Do not quote snippets at length.
7. Prices and specs in this database are approximate sample data. Remind the shopper in one short line to verify the current price and configuration on the seller's site before buying.
8. Keep the answer under 300 words. Use short paragraphs or a compact list. Plain language, no hype."""


# ---------- helpers ----------
def _is_reasoning(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def _complete(client, model, system, user, json_mode=False, temperature=0.2, max_tokens=1500):
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,  # reasoning models spend part of this on thinking, so keep it generous
    )
    if _is_reasoning(model):
        kwargs["reasoning_effort"] = "low"  # faster + fewer tokens; plenty for this task
    if json_mode:
        try:
            resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            return resp.choices[0].message.content or ""
        except Exception:
            pass  # some models reject JSON mode: retry as plain text, we extract the JSON ourselves
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def get_llm_client():
    from groq import Groq
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (local) or Streamlit secrets (deployed).")
    return Groq(api_key=config.GROQ_API_KEY)


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    n = _num(v)
    return None if n is None else int(n)


def _list(v, allowed):
    if not v:
        return None
    if isinstance(v, str):
        v = [v]
    lookup = {a.lower(): a for a in allowed}
    out = [lookup[str(x).lower()] for x in v if str(x).lower() in lookup]
    return out or None


def coerce_requirements(data: dict, fallback_query: str) -> Requirements:
    gpu = data.get("needs_dedicated_gpu")
    intent = str(data.get("intent") or "recommend").lower()
    if intent not in ("recommend", "clarify", "other"):
        intent = "recommend"
    req = Requirements(
        query=str(data.get("query") or fallback_query),
        budget_max=_num(data.get("budget_max")),
        budget_min=_num(data.get("budget_min")),
        min_ram_gb=_int(data.get("min_ram_gb")),
        min_storage_gb=_int(data.get("min_storage_gb")),
        max_weight_kg=_num(data.get("max_weight_kg")),
        needs_dedicated_gpu=True if gpu is True else None,
        categories=_list(data.get("categories"), CATEGORIES),
        brands=_list(data.get("brands"), BRANDS),
        min_rating=_num(data.get("min_rating")),
    )
    req.intent = intent
    return req


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


# ---------- step 1 ----------
def parse_requirements(user_text: str, history: list[str] | None = None, client=None) -> Requirements:
    client = client or get_llm_client()
    context = ""
    if history:
        context = "Earlier user messages:\n" + "\n".join(f"- {h}" for h in history[-3:]) + "\n\n"
    user = f"{context}Latest message:\n{user_text}"
    try:
        raw = _complete(client, config.GROQ_PARSE_MODEL, PARSE_SYSTEM, user, json_mode=True, max_tokens=1000, temperature=0)
        return coerce_requirements(_extract_json(raw), user_text)
    except Exception:
        # never block the user: fall back to pure semantic search on their words
        return Requirements(query=user_text)


# ---------- step 2 (with graceful relaxing) ----------
def retrieve_with_fallback(req: Requirements, top_k: int = 5, client=None):
    hits = search(req, top_k=top_k, client=client)
    if hits:
        return hits, req, None
    relaxed = replace(req, brands=None, categories=None, min_rating=None, min_storage_gb=None, max_weight_kg=None)
    hits = search(relaxed, top_k=top_k, client=client)
    if hits:
        return hits, relaxed, "No exact match, so I relaxed brand/type/weight/storage conditions and kept budget, RAM and GPU."
    budget_only = Requirements(query=req.query, budget_max=(req.budget_max * 1.1) if req.budget_max else None,
                               budget_min=req.budget_min)
    hits = search(budget_only, top_k=top_k, client=client)
    if hits:
        return hits, budget_only, "No exact match, so I kept only your budget (with a 10% margin) and ranked by your preferences."
    return [], req, "Nothing in the database matches this budget."


# ---------- step 3 ----------
def _fmt_hit(h: dict) -> str:
    return (f"- id={h['product_id']} | {h['brand']} {h['model']} | Rs {h['price_inr']:,.0f} | {h['category']} | "
            f"CPU {h['cpu']} | {h['ram_gb']}GB RAM | {h['storage_gb']}GB SSD | GPU {h['gpu']} | "
            f"{h['screen_in']}in {h['resolution']} {h['refresh_hz']}Hz | {h['weight_kg']}kg | {h['battery_wh']}Wh | "
            f"rating {h['rating']} | pros: {h['pros']} | cons: {h['cons']}")


def _fmt_evidence(e: dict) -> str:
    kind = {"pdf": "PDF spec sheet", "reddit": "Reddit", "web": "Web review"}.get(e["source_type"], e["source_type"])
    where = e.get("file") or e.get("subreddit") or e.get("url") or e.get("source_name", "")
    extra = f", p.{e['page']}" if e.get("page") else (f", {e['score']} upvotes" if e["source_type"] == "reddit" and e.get("score") else "")
    body = " ".join(str(e["text"]).split())
    body = body.split(" - ", 1)[1] if " - " in body[:80] else body   # drop the "Brand Model - " prefix
    return f"[{e['product_id']}] {kind} ({where}{extra}): {body[:400]}"


def generate_answer(user_text: str, req: Requirements, hits: list[dict], note: str | None = None,
                    client=None, evidence: list[dict] | None = None) -> str:
    if not hits:
        return ("I couldn't find a laptop in my database that fits that. Try a higher budget or fewer must-haves, "
                "and I'll look again.")
    client = client or get_llm_client()
    filters = {k: v for k, v in req.__dict__.items() if v not in (None, "", []) and k != "intent"}
    user = (f"Shopper's request: {user_text}\n"
            f"Filters applied: {json.dumps(filters, ensure_ascii=False)}\n"
            + (f"Note: {note}\n" if note else "")
            + "Candidate laptops (best match first):\n" + "\n".join(_fmt_hit(h) for h in hits)
            + ("\nEvidence snippets (by laptop id):\n" + "\n".join(_fmt_evidence(e) for e in evidence)
               if evidence else "\nNo supporting documents were found; answer from the database only."))
    return _complete(client, config.GROQ_MODEL, ANSWER_SYSTEM, user, max_tokens=1500, temperature=0.3)


def recommend(user_text: str, history: list[str] | None = None, top_k: int = 5, qdrant_client=None, llm_client=None):
    """Whole cycle. Returns (answer_text, hits, requirements, note, evidence)."""
    quick = small_talk_reply(user_text)          # greetings etc.: no LLM call, no search
    if quick:
        return quick, [], Requirements(query=user_text, intent="other"), None, []
    llm_client = llm_client or get_llm_client()
    req = parse_requirements(user_text, history, client=llm_client)
    if req.intent == "other":
        return OFF_TOPIC, [], req, None, []
    if req.intent == "clarify":
        return CLARIFY, [], req, None, []
    hits, used_req, note = retrieve_with_fallback(req, top_k=top_k, client=qdrant_client)
    evidence = []
    try:
        evidence = search_evidence(user_text, [h["product_id"] for h in hits[:4]], per_product=3, client=qdrant_client)
    except Exception:
        pass  # evidence is a bonus: never fail the answer because of it
    answer = generate_answer(user_text, used_req, hits, note, client=llm_client, evidence=evidence)
    return answer, hits, used_req, note, evidence
