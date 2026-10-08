import os

import streamlit as st

# Bridge Streamlit Cloud secrets -> environment variables BEFORE importing core.config
for _k in ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_PARSE_MODEL"):
    try:
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
    except Exception:
        pass

from core.answer import get_llm_client, recommend  # noqa: E402
from core.store import get_client  # noqa: E402

st.set_page_config(page_title="Laptop Advisor India", page_icon="💻", layout="wide")


@st.cache_resource
def _qdrant():
    return get_client()


@st.cache_resource
def _llm():
    return get_llm_client()


ICONS = {"pdf": "📄 Spec sheet", "reddit": "💬 Reddit", "web": "🌐 Web review"}


def _render_sources(evidence):
    with st.expander(f"Sources used ({len(evidence)})"):
        for e in evidence:
            label = ICONS.get(e["source_type"], e["source_type"])
            name = f"{e.get('brand', '')} {e.get('model', '')}".strip()
            where = e.get("file") or (f"r/{e['subreddit']}" if e.get("subreddit") else e.get("source_name", ""))
            page = f" p.{e['page']}" if e.get("page") else ""
            link = e.get("url") if str(e.get("url", "")).startswith("http") else None
            head = f"**{label}** - {name} - {where}{page}"
            st.markdown(f"{head}  [open]({link})" if link else head)
            body = " ".join(str(e["text"]).split())
            body = body.split(" - ", 1)[1] if " - " in body[:80] else body
            st.caption(body[:350] + ("..." if len(body) > 350 else ""))


st.title("💻 Laptop Advisor (India)")
st.caption("Tell me what you need, and I'll compare laptops for you. "
           "Prices and specs are approximate sample data - always verify on the seller's site.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Laptops to compare", 3, 8, 5)
    show_debug = st.checkbox("Show how I understood the request", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try:**\n- Laptop for coding under 70k, light to carry\n- Gaming laptop under 1 lakh with RTX\n"
                "- Best MacBook for a student\n- Cheapest laptop with 16GB RAM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("table") is not None:
            st.dataframe(m["table"], use_container_width=True, hide_index=True)
        if m.get("sources"):
            _render_sources(m["sources"])

user_text = st.chat_input("e.g. Laptop for coding and travel under 80k")
if user_text:
    history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and comparing..."):
                answer, hits, req, note, evidence = recommend(user_text, history, top_k=top_k,
                                                    qdrant_client=_qdrant(), llm_client=_llm())
        except Exception as e:  # keep the app alive on rate limits / network errors
            st.error(f"Something went wrong: {e}")
            st.stop()

        st.markdown(answer)
        table = None
        if hits:
            import pandas as pd
            table = pd.DataFrame([{
                "Laptop": f"{h['brand']} {h['model']}", "Price (Rs)": int(h["price_inr"]),
                "RAM": f"{h['ram_gb']}GB", "Storage": f"{h['storage_gb']}GB", "GPU": h["gpu"],
                "Weight (kg)": h["weight_kg"], "Battery (Wh)": h["battery_wh"], "Rating": h["rating"],
            } for h in hits])
            st.dataframe(table, use_container_width=True, hide_index=True)
        if evidence:
            _render_sources(evidence)
        if show_debug:
            with st.expander("How I understood your request"):
                st.json({k: v for k, v in req.__dict__.items() if v not in (None, "", [])})  # includes intent
                if note:
                    st.info(note)
    st.session_state.messages.append({"role": "assistant", "content": answer, "table": table, "sources": evidence})
