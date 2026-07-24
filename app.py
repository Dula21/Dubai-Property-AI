"""
app.py — Dubai Property Intelligence (v2: grounded retrieval)

Retrieval strategy, in order:
  1. Deterministic match: fuzzy-match any area name mentioned in the query
     against the real DLD area names (difflib-adjacent, stdlib, zero cost).
     If matched, pull the EXACT pre-computed aggregate stats for that area
     (and property type, if mentioned) from area_aggregates.csv. This path
     never lets the LLM compute or guess a number — the math is already
     done in ingest.py from real transactions.
  2. Market-level routing: if no area is named but the query is clearly
     asking about overall market direction (contains "market" / "econom..."),
     return the pre-built overall market summary. This runs BEFORE the
     Chroma fallback so a broad question like "is the economy slowing
     down" can't be hijacked by an unrelated area chunk that merely scores
     low on embedding distance.
  3. Semantic fallback: if neither of the above matches, query the
     ChromaDB store (semantic search over the same aggregate chunks) in
     case the user phrased an area differently.
  4. Honest fallback: if nothing finds anything relevant, tell the user
     directly that the dataset doesn't cover it. The LLM is not called
     with an empty/irrelevant context — it is never given the chance to
     fill the gap from training knowledge.

Data covers DLD sale transactions from 2026-01-01 to 2026-07-19 only.
"""


"""
app.py — Dubai Property Intelligence (v2: grounded retrieval)
"""

import os
import re
import difflib

from dotenv import load_dotenv
import pandas as pd
import gradio as gr
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model config — llama-3.1-8b-instant was deprecated by Groq (announced
# June 17, 2026, shutdown Aug 16, 2026). Migrated to gpt-oss-20b, Groq's
# recommended replacement: fastest and cheapest currently-supported model
# on the platform. It IS a reasoning model, unlike the old Llama Instant,
# so `reasoning_effort` controls how much chain-of-thought it generates
# before answering. "low" keeps latency/cost close to the old model's
# profile; bump to "medium" only if testing shows "low" misses on the
# grounding rules (blurring property types, mis-citing articles, etc).
MODEL_NAME = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"  # "low" | "medium" | "high"

AGG_PATH = "area_aggregates.csv"
SUMMARY_PATH = "market_intelligence.txt"
CHROMA_DIR = "chroma_db"

aggregates = pd.read_csv(AGG_PATH)
KNOWN_AREAS = sorted(aggregates["AREA_EN"].unique().tolist())

try:
    with open(SUMMARY_PATH) as f:
        market_summary = f.read()
except FileNotFoundError:
    market_summary = "No overall market summary available."

_chroma_collection = None
def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client_db = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client_db.get_collection("dld_aggregates", embedding_function=embed_fn)
    except Exception as e:
        print(f"[warn] ChromaDB unavailable, semantic fallback disabled: {e}")
        _chroma_collection = False
    return _chroma_collection


_legal_collection = None
def get_legal_collection():
    global _legal_collection
    if _legal_collection is not None:
        return _legal_collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client_db = chromadb.PersistentClient(path=CHROMA_DIR)
        _legal_collection = client_db.get_collection("dld_laws", embedding_function=embed_fn)
    except Exception as e:
        print(f"[warn] Legal collection unavailable, legal grounding disabled: {e}")
        _legal_collection = False
    return _legal_collection


AREA_WORDS = {tuple(sorted(a.split())): a for a in KNOWN_AREAS}
_STOPWORDS = {"THE", "OF", "AND", "IN", "AT", "PARK", "DUBAI"}

_ALIAS_CANDIDATES = {
    "JVC": "JUMEIRAH VILLAGE CIRCLE",
    "JVT": "JUMEIRAH VILLAGE TRIANGLE",
    "JLT": "JUMEIRAH LAKES TOWERS",
}
AREA_ALIASES = {k: v for k, v in _ALIAS_CANDIDATES.items() if v in KNOWN_AREAS}


def find_area_match(query: str, cutoff: float = 0.6):
    q = query.upper()
    q_words = set(re.findall(r"[A-Z]+", q))

    for abbr, area in AREA_ALIASES.items():
        if abbr in q_words:
            return area

    for area in KNOWN_AREAS:
        if area in q or area.replace(" ", "") in q.replace(" ", ""):
            return area

    candidates = []
    for words, area in AREA_WORDS.items():
        meaningful = [w for w in words if w not in _STOPWORDS]
        if meaningful and all(w in q_words for w in meaningful):
            candidates.append(area)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return max(candidates, key=len)

    return None


def get_area_context(area: str) -> str:
    rows = aggregates[aggregates["AREA_EN"] == area]
    if rows.empty:
        return ""

    total_count = int(rows["transaction_count"].sum())
    weighted_avg = round((rows["avg_price_per_sqft"] * rows["transaction_count"]).sum() / total_count, 1)

    lines = [
        f"Verified DLD transaction data for area '{area}' (Jan 1 - Jul 19, 2026):",
        f"OVERALL (all property types combined): {total_count} sales, "
        f"count-weighted average price = AED {weighted_avg}/sqft.",
        "Breakdown by property type:",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"- {r['PROP_SB_TYPE_EN']}: {int(r['transaction_count'])} sales, "
            f"avg AED {r['avg_price_per_sqft']}/sqft, median AED {r['median_price_per_sqft']}/sqft, "
            f"range AED {r['min_price_per_sqft']}-{r['max_price_per_sqft']}/sqft "
            f"({r['earliest_date']} to {r['latest_date']})"
        )
    return "\n".join(lines)


def get_semantic_context(query: str, k: int = 3, max_distance: float = 1.1):
    collection = get_chroma_collection()
    if not collection:
        return ""
    try:
        results = collection.query(query_texts=[query], n_results=k, include=["documents", "distances", "metadatas"])
    except Exception as e:
        print(f"[warn] chroma query failed: {e}")
        return ""

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    q_words = set(re.findall(r"[A-Z]+", query.upper()))

    kept = []
    for doc, dist, meta in zip(docs, dists, metas):
        if dist > max_distance:
            continue
        area = (meta or {}).get("area", "")
        area_words = {w for w in re.findall(r"[A-Z]+", area.upper()) if w not in _STOPWORDS}
        if area_words and not (area_words & q_words):
            continue
        kept.append(doc)

    return "\n".join(kept)


LEGAL_TRIGGER_WORDS = {
    "LAW", "LAWS", "LEGAL", "REGULATION", "REGULATIONS", "RIGHTS", "RERA",
    "OWNERSHIP", "MORTGAGE", "TENANT", "TENANTS", "TENANCY", "LANDLORD",
    "LANDLORDS", "LEASE", "CONTRACT", "DISPUTE", "ESCROW", "BROKER",
    "BROKERAGE", "EVICTION", "FOREIGNER", "FREEHOLD", "ARTICLE", "DECREE",
    "TITLE", "DEED", "REGISTRATION", "FINE", "FEE", "FEES", "PENALTY",
}


def is_legal_query(query: str) -> bool:
    q_words = set(re.findall(r"[A-Z]+", query.upper()))
    return bool(q_words & LEGAL_TRIGGER_WORDS)


def get_legal_context(query: str, k: int = 3, max_distance: float = 1.1):
    collection = get_legal_collection()
    if not collection:
        return ""
    try:
        results = collection.query(query_texts=[query], n_results=k, include=["documents", "distances", "metadatas"])
    except Exception as e:
        print(f"[warn] legal chroma query failed: {e}")
        return ""

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    q_words = {w for w in re.findall(r"[A-Z]+", query.upper()) if w not in _STOPWORDS}

    kept = []
    for doc, dist in zip(docs, dists):
        if dist > max_distance:
            continue
        doc_words = {w for w in re.findall(r"[A-Z]+", doc.upper()) if w not in _STOPWORDS}
        if not (doc_words & q_words):
            continue
        kept.append(doc)

    return "\n".join(kept)


_MARKET_LEVEL_PATTERN = re.compile(r"\bmarket\b|\beconom", re.IGNORECASE)


def is_market_level_query(query: str) -> bool:
    return bool(_MARKET_LEVEL_PATTERN.search(query))


def retrieve_context(query: str):
    blocks = []

    area = find_area_match(query)
    if area:
        ctx = get_area_context(area)
        if ctx:
            blocks.append(f"[TRANSACTION DATA]\n{ctx}")

    if is_market_level_query(query) and market_summary and "No overall market summary" not in market_summary:
        blocks.append(f"[MARKET SUMMARY]\n{market_summary}")

    if is_legal_query(query):
        legal_ctx = get_legal_context(query)
        if legal_ctx:
            blocks.append(f"[LEGAL PROVISIONS]\n{legal_ctx}")

    if blocks:
        return "\n\n".join(blocks), True

    semantic_ctx = get_semantic_context(query)
    if semantic_ctx:
        return f"[TRANSACTION DATA]\n{semantic_ctx}", True

    return "", False


FALLBACK_MESSAGE = (
    "I don't have data covering that. This tool is grounded in two sources: "
    "real DLD sale transactions (2026-01-01 to 2026-07-19, "
    f"{len(KNOWN_AREAS)} Dubai areas) and the DLD real-estate legislation "
    "compilation (ownership, mortgage, escrow, tenancy, brokers, foreign-ownership "
    "zones). I'd rather tell you that honestly than guess. "
    "For anything outside these - including visa/Golden Visa rules, other emirates, "
    "or recent amendments not yet indexed - please confirm with a RERA-registered "
    "agent or a property lawyer."
)


SYSTEM_TEMPLATE = """You are a senior Dubai real estate advisor speaking with a client who may
know nothing about the property market. Talk the way an experienced advisor actually
would - plain language, confident, conversational - not like an automated report.

Ground every number and every legal statement ONLY in the DATA CONTEXT below. The context
may contain multiple labeled blocks:
- [TRANSACTION DATA]: real DLD sale transactions (2026-01-01 to 2026-07-19).
- [MARKET SUMMARY]: overall market direction, for broad questions.
- [LEGAL PROVISIONS]: excerpts from the DLD real-estate legislation compilation, each
  already tagged with its Law/Decree number and Article number - cite that number
  directly (e.g. "under Article (12) of Law No. (6) of 2019") rather than paraphrasing
  without attribution.

Rules:
- Never invent a figure or a legal claim, and never fall back on outside/training
  knowledge for either - if it's not in the DATA CONTEXT, say so plainly.
- If the DATA CONTEXT includes a line starting with "OVERALL", that figure is already
  correctly computed for you (count-weighted across property types) - state it directly,
  don't recompute or second-guess it.
- The breakdown lines (e.g. "Flat: 954 sales, avg AED ...") are PER PROPERTY TYPE, not
  area-wide totals. When you cite one of these numbers, always name the property type
  explicitly (e.g. "954 flat sales", not just "954 sales") - do not let a per-type figure
  read as if it covers the whole area unless it's the line starting with "OVERALL".
- The compiled legislation is a specific edition and may not reflect amendments passed
  after it was last updated. For anything the [LEGAL PROVISIONS] context doesn't clearly
  cover - visa/Golden Visa rules, other emirates, or any point you're not fully certain
  is still current - say so and suggest confirming with a RERA-registered agent or
  property lawyer. Do not answer legal questions from general/training knowledge, even if
  you believe you know the answer.
- Use the transaction count and coverage window as evidence for your answer, mentioned in
  passing - not as a formal label pair.
- Prices are in AED. Do not convert currencies unless asked.

DATA CONTEXT:
{context}
"""

def chat_function(message, history):
    context, found = retrieve_context(message)

    if not found:
        return FALLBACK_MESSAGE

    system_prompt = SYSTEM_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        else:
            human, assistant = turn
            messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.2,
            reasoning_effort=REASONING_EFFORT,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System error while generating the answer: {e}"


def build_demo():
    return gr.ChatInterface(
        fn=chat_function,
        title="🇦🇪 Dubai Property Intelligence",
        description=(
            "Grounded in real DLD sale transactions (Jan 1 - Jul 19, 2026). "
            "Answers only from retrieved transaction data - if the data doesn't "
            "cover your question, it will say so instead of guessing."
        ),
        examples=[
            "What is the average price per sqft in Dubai Marina?",
            "How many flats sold in JVC and at what median price?",
            "Is the overall Dubai market growing or stabilizing?",
        ],
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()