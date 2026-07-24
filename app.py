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

import os
import re
import difflib

from dotenv import load_dotenv
import pandas as pd
import gradio as gr
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Load pre-built data (produced once by ingest.py, committed to the repo)
# ---------------------------------------------------------------------------

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
    """Lazy-load chroma so the app still boots even if the store is missing."""
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


# ---------------------------------------------------------------------------
# Step 1: deterministic area matching
# ---------------------------------------------------------------------------

AREA_WORDS = {tuple(sorted(a.split())): a for a in KNOWN_AREAS}
_STOPWORDS = {"THE", "OF", "AND", "IN", "AT", "PARK", "DUBAI"}

# Common abbreviations people actually type. Only kept if the canonical
# name genuinely exists in this dataset - checked at import time below.
_ALIAS_CANDIDATES = {
    "JVC": "JUMEIRAH VILLAGE CIRCLE",
    "JVT": "JUMEIRAH VILLAGE TRIANGLE",
    "JLT": "JUMEIRAH LAKES TOWERS",
}
AREA_ALIASES = {k: v for k, v in _ALIAS_CANDIDATES.items() if v in KNOWN_AREAS}


def find_area_match(query: str, cutoff: float = 0.6):
    """Try to find a known DLD area name mentioned in the query.

    Deliberately conservative: an unmatched query must return None rather
    than a wrong guess. A previous looser version matched unrelated
    queries (e.g. "Atlantis The Royal residences") to the nearest-sounding
    area purely by short-string fuzzy overlap - that produced confidently
    wrong, ungrounded answers, which is the exact failure mode this whole
    pipeline exists to prevent.
    """
    q = query.upper()
    q_words = set(re.findall(r"[A-Z]+", q))

    # 1. Known abbreviations, matched as a whole word only.
    for abbr, area in AREA_ALIASES.items():
        if abbr in q_words:
            return area

    # 2. Exact substring match (handles "dubai marina", "Marina", etc.
    #    for single-distinctive-word areas this is safe; for multi-word
    #    areas it requires the full phrase to appear).
    for area in KNOWN_AREAS:
        if area in q or area.replace(" ", "") in q.replace(" ", ""):
            return area

    # 3. Full-token-overlap fallback: every word in the area's name must
    #    appear somewhere in the query (order-independent). This catches
    #    reordering or extra words around the area name, but will NOT
    #    match on partial/loose similarity - eliminates the false-positive
    #    class found in testing.
    candidates = []
    for words, area in AREA_WORDS.items():
        meaningful = [w for w in words if w not in _STOPWORDS]
        if meaningful and all(w in q_words for w in meaningful):
            candidates.append(area)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Ambiguous - prefer the most specific (longest) name rather than guess randomly.
        return max(candidates, key=len)

    return None


def get_area_context(area: str) -> str:
    rows = aggregates[aggregates["AREA_EN"] == area]
    if rows.empty:
        return ""

    # Pre-compute the overall (all property types combined) figures here,
    # in code, not in the prompt. Two real failures were observed when this
    # was left to the LLM: it once summed the per-type counts wrong (1020
    # instead of 1024), and separately claimed an overall average "cannot
    # be calculated" from data that was already sufficient to compute one.
    # A count-weighted average is exact from these aggregates - no need for
    # row-level area data - so it is stated directly rather than left as
    # arithmetic for the model to attempt.
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


# ---------------------------------------------------------------------------
# Step 2: market-level query routing (no area named)
# ---------------------------------------------------------------------------

# Matches "market" (trend/status/growing/stabilizing/direction/etc.) or
# "economy"/"economic" without needing to enumerate every phrasing. This is
# deliberately broad because it only runs AFTER area matching has already
# failed to find anything, so a real area-specific question ("average price
# in Dubai Marina") never reaches this check - area matches always win.
_MARKET_LEVEL_PATTERN = re.compile(r"\bmarket\b|\beconom", re.IGNORECASE)


def is_market_level_query(query: str) -> bool:
    return bool(_MARKET_LEVEL_PATTERN.search(query))


# ---------------------------------------------------------------------------
# Step 3: semantic fallback via ChromaDB
# ---------------------------------------------------------------------------

def get_semantic_context(query: str, k: int = 3, max_distance: float = 1.1):
    """Semantic fallback - deliberately fails CLOSED, not open.

    Bug found in production: an unrelated query ("what's the weather in
    Tokyo") matched a chunk closely enough on raw embedding distance alone
    to reach the LLM. Raw distance thresholds are hard to calibrate without
    live access to the embedding model, so this now also requires the
    matched chunk's own area name to share at least one real word with the
    query - i.e. semantic search only rescues phrasing variants of a real
    area mention, it does not open the door to arbitrary topics.
    """
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
            # Distance was low enough, but the query shares no real word
            # with the matched area - treat as a false positive, not a hit.
            continue
        kept.append(doc)

    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Retrieval orchestration
# ---------------------------------------------------------------------------

def retrieve_context(query: str):
    """Returns (context_text, found: bool)."""
    area = find_area_match(query)
    if area:
        ctx = get_area_context(area)
        if ctx:
            return ctx, True

    # Market-level questions (no area named) must route to the pre-built
    # overall summary BEFORE the Chroma fallback runs - otherwise a query
    # like "is the economy slowing down" can be hijacked by an unrelated
    # area chunk that merely scores low on embedding distance (the same
    # failure class already fixed for "weather in Tokyo" below).
    if is_market_level_query(query):
        return market_summary, True

    semantic_ctx = get_semantic_context(query)
    if semantic_ctx:
        return semantic_ctx, True

    return "", False


FALLBACK_MESSAGE = (
    "I don't have DLD transaction data covering that in the current dataset. "
    "The dataset spans sale transactions from 2026-01-01 to 2026-07-19 across "
    f"{len(KNOWN_AREAS)} Dubai areas. I'd rather tell you that honestly than guess a number."
)


# ---------------------------------------------------------------------------
# Chat function
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE =SYSTEM_TEMPLATE = """You are a senior Dubai real estate advisor speaking with a client who
may know nothing about the property market. Talk the way an experienced advisor actually
would - plain language, confident, conversational - not like an automated report. Never
repeat field labels like "Sample size:" or "Date range:" as a checklist; weave the same
facts naturally into your sentences instead (e.g. "based on just over 93,000 sales recorded
through mid-July" rather than a labeled line).

Ground every number in the DATA CONTEXT below, which comes from real DLD (Dubai Land
Department) sale transactions. Never invent a figure and never fall back on outside or
training knowledge for market numbers.

Rules:
- If the DATA CONTEXT includes a line starting with "OVERALL", that figure is already
  correctly computed for you (count-weighted across property types) - state it directly,
  don't recompute or second-guess it.
- Use the transaction count and coverage window as evidence for your answer, mentioned in
  passing - not as a formal label pair.
- If the question is about something the DATA CONTEXT doesn't cover - property law, visa or
  Golden Visa rules, legal process, tax treatment, contracts - say plainly that this tool is
  grounded only in DLD transaction data and doesn't cover legal or regulatory matters, and
  suggest confirming those specifics with a RERA-registered agent or property lawyer. Do not
  answer legal questions from general/training knowledge, even if you believe you know the
  answer.
- Prices are in AED. Do not convert currencies unless asked.

DATA CONTEXT:
{context}

OVERALL MARKET SUMMARY (for broad market-direction questions only):
{market_summary}
"""

def chat_function(message, history):
    context, found = retrieve_context(message)

    if not found:
        return FALLBACK_MESSAGE

    system_prompt = SYSTEM_TEMPLATE.format(context=context, market_summary=market_summary)
    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        if isinstance(turn, dict):
            # Gradio's messages-format history dicts can carry extra keys
            # (e.g. "metadata") for its own UI features. Groq's API rejects
            # any unrecognized field, so only forward role/content.
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
            model="llama-3.1-8b-instant",
            temperature=0.2,  # low temperature: this is a data-lookup task, not creative writing
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System error while generating the answer: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_demo():
    return gr.ChatInterface(
        fn=chat_function,
        title="🇦🇪 Dubai Property Intelligence",
        description=(
            "Grounded in real DLD sale transactions (Jan 1 - Jul 19, 2026). "
            "Answers only from retrieved transaction data - if the data doesn't "
            "cover your question, it will say so instead of guessing."
        ),
        #theme="soft",
        examples=[
            "What is the average price per sqft in Dubai Marina?",
            "How many flats sold in JVC and at what median price?",
            "Is the overall Dubai market growing or stabilizing?",
        ],
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()