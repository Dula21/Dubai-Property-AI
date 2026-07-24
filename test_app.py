"""
test_app.py — Run this BEFORE merging dev into main.

    pytest test_app.py -v

If anything fails, do not push to main. HF Spaces rebuilds automatically
on push to main/master, so a broken push goes live immediately with no
review step in between - this file is the review step.

These tests encode real bugs found during development (sqm/sqft
conversion, casing duplicates, false-positive area matching) so they
can never silently regress.
"""

import os
import re
import glob
import pandas as pd
import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used-in-these-tests")


# ---------------------------------------------------------------------------
# 1. Required files exist before deploy
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "app.py",
    "ingest.py",
    "requirements.txt",
    "area_aggregates.csv",
    "market_intelligence.txt",
]

@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_file_exists(filename):
    assert os.path.exists(filename), f"{filename} is missing - app.py will crash on load"


def test_chroma_db_dir_exists():
    # Not fatal (app.py degrades gracefully without it) but should be flagged.
    if not os.path.isdir("chroma_db"):
        pytest.skip("chroma_db/ missing - semantic fallback will be disabled. "
                    "Fine for a quick fix, but re-run ingest.py before a real deploy.")


# ---------------------------------------------------------------------------
# 2. No secrets committed (the notebook leaked a Groq key once already)
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [r"gsk_[A-Za-z0-9]{20,}"]  # Groq API key format

@pytest.mark.parametrize("path", glob.glob("*.py") + glob.glob("*.ipynb") + glob.glob("*.md"))
def test_no_hardcoded_api_keys(path):
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    for pattern in SECRET_PATTERNS:
        match = re.search(pattern, content)
        assert not match, f"Hardcoded API key found in {path}: {match.group()[:10]}... REVOKE AND REMOVE before pushing."


# ---------------------------------------------------------------------------
# 3. Data quality: the bugs we already found must not come back
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aggregates():
    return pd.read_csv("area_aggregates.csv")


def test_no_casing_duplicate_areas(aggregates):
    """Bug found: 'Business Bay' and 'BUSINESS BAY' counted as separate areas."""
    areas = aggregates["AREA_EN"].unique()
    upper_map = {}
    for a in areas:
        upper_map.setdefault(a.upper().strip(), []).append(a)
    dupes = {k: v for k, v in upper_map.items() if len(v) > 1}
    assert not dupes, f"Casing duplicates found (re-run ingest.py's normalize step): {dupes}"


def test_price_per_sqft_in_realistic_range(aggregates):
    """Bug found: sqm/sqft conversion missing gave ~21,700 AED/sqft instead of ~1,850."""
    # Generous sanity bounds for Dubai residential, not a tight assertion -
    # this exists to catch a missing/duplicated unit conversion, not to
    # judge whether the market moved.
    assert aggregates["median_price_per_sqft"].between(200, 15000).mean() > 0.95, (
        "More than 5% of area medians fall outside a sane AED/sqft range - "
        "check the sqm->sqft conversion in ingest.py hasn't been lost or doubled."
    )


def test_dubai_marina_present_and_sane(aggregates):
    marina = aggregates[aggregates["AREA_EN"] == "DUBAI MARINA"]
    assert not marina.empty, "Dubai Marina missing from aggregates - check ingestion ran on the full CSV"
    flats = marina[marina["PROP_SB_TYPE_EN"] == "Flat"]
    assert not flats.empty
    median = flats.iloc[0]["median_price_per_sqft"]
    assert 1000 <= median <= 4000, f"Marina flat median price/sqft = {median}, outside sane range"


# ---------------------------------------------------------------------------
# 4. Retrieval logic: exact cases already validated manually, now automated
# ---------------------------------------------------------------------------

import app as appmod  # noqa: E402  (import after env var setup above)

AREA_MATCH_CASES = [
    ("what is the average price per sqft in Dubai Marina", "DUBAI MARINA"),
    ("how much do flats cost in Marina", "DUBAI MARINA"),
    ("JVC price trends", "JUMEIRAH VILLAGE CIRCLE"),
    ("average price in Jumeirah Village Circle for flats", "JUMEIRAH VILLAGE CIRCLE"),
    ("price per sqft in Business Bay", "BUSINESS BAY"),
    ("whats happening in Palm Jumeirah", "PALM JUMEIRAH"),
    ("dubai hills price", "DUBAI HILLS"),
    ("investment park first prices", "DUBAI INVESTMENT PARK FIRST"),
    # Negative cases - must NOT match, this is where the false positives were
    ("tell me about Atlantis The Royal residences", None),
    ("what is bitcoin price today", None),
    ("best restaurants in dubai", None),
    ("investment park prices", None),  # ambiguous between FIRST/SECOND
]

@pytest.mark.parametrize("query,expected_area", AREA_MATCH_CASES)
def test_area_matching(query, expected_area):
    result = appmod.find_area_match(query)
    assert result == expected_area, f"Query {query!r} matched {result!r}, expected {expected_area!r}"


def test_fallback_never_calls_llm_on_unmatched_query(monkeypatch):
    """The whole point of this pipeline: no match -> no LLM call -> no hallucination risk."""
    def fail_if_called(*a, **k):
        raise AssertionError("LLM was called for an unmatched query - this must never happen")

    monkeypatch.setattr(appmod.client.chat.completions, "create", fail_if_called)
    result = appmod.chat_function("what's the weather in Tokyo", [])
    assert result == appmod.FALLBACK_MESSAGE


def test_grounded_query_context_contains_real_numbers(monkeypatch):
    """When a match is found, the system prompt sent to the LLM must contain
    the real retrieved numbers, not a placeholder or an empty context."""
    captured = {}

    class FakeMsg:
        content = "fake response"

    class FakeChoice:
        message = FakeMsg()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_create(**kwargs):
        sys_msg = kwargs["messages"][0]["content"]
        captured["system_prompt"] = sys_msg
        return FakeResponse()

    monkeypatch.setattr(appmod.client.chat.completions, "create", fake_create)
    appmod.chat_function("average price per sqft in Dubai Marina", [])

    assert "DUBAI MARINA" in captured["system_prompt"]
    assert "sales" in captured["system_prompt"]
    assert "ONLY in the DATA CONTEXT" in captured["system_prompt"] or "ONLY using the DATA CONTEXT" in captured["system_prompt"]


def test_area_context_includes_precomputed_overall_total():
    """Bug found in production: the LLM once summed per-type transaction
    counts wrong (1020 instead of 1024) when asked to add them up itself.
    The total must now be computed in code and handed to the model, not
    left as arithmetic in the prompt."""
    ctx = appmod.get_area_context("DUBAI MARINA")
    assert "1024 sales" in ctx, "Pre-computed total transaction count missing or wrong"


def test_area_context_includes_precomputed_overall_average():
    """Bug found in production: asked for the overall average price/sqft
    across all property types, the LLM incorrectly claimed this 'cannot be
    calculated' from the given per-type data - it can, via a count-weighted
    average, and that number must now be pre-computed rather than left for
    the model to derive (or wrongly claim is impossible)."""
    ctx = appmod.get_area_context("DUBAI MARINA")
    assert "OVERALL" in ctx
    assert "2086.6" in ctx, "Pre-computed weighted overall average missing or wrong"


# ---------------------------------------------------------------------------
# 5. App boots without crashing
# ---------------------------------------------------------------------------

def test_app_module_imports_cleanly():
    assert hasattr(appmod, "chat_function")
    assert hasattr(appmod, "build_demo")


def test_known_areas_loaded():
    assert len(appmod.KNOWN_AREAS) > 100, "Suspiciously few areas loaded - check area_aggregates.csv"


# ---------------------------------------------------------------------------
# 6. Regression: Gradio history dicts must be sanitized before hitting Groq
# ---------------------------------------------------------------------------

def test_gradio_metadata_keys_stripped_before_groq_call(monkeypatch):
    """Bug found in production: Gradio's messages-format history includes
    extra keys (e.g. 'metadata', 'options') for its own UI features. Groq's
    API rejects any message with unrecognized fields (400 error). Every
    message forwarded to Groq must contain only role/content."""
    captured = {}

    class FakeMsg:
        content = "ok"

    class FakeChoice:
        message = FakeMsg()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        for m in kwargs["messages"]:
            assert set(m.keys()) <= {"role", "content"}, (
                f"Message has extra keys Groq will reject: {m.keys()}"
            )
        return FakeResponse()

    monkeypatch.setattr(appmod.client.chat.completions, "create", fake_create)

    history_with_metadata = [
        {"role": "user", "content": "average price in Business Bay", "metadata": None, "options": None},
        {"role": "assistant", "content": "Business Bay flats average AED 2426.9/sqft", "metadata": {"title": None}},
    ]
    result = appmod.chat_function("what about Dubai Marina", history_with_metadata)
    assert result == "ok"


# ---------------------------------------------------------------------------
# 7. Regression: chroma semantic fallback must fail closed on unrelated queries
# ---------------------------------------------------------------------------

def test_semantic_fallback_rejects_unrelated_query_despite_low_distance(monkeypatch):
    """Bug found in production: 'what's the weather in Tokyo' matched a
    chunk closely enough on raw embedding distance alone that the LLM got
    called on a totally unrelated topic. Real chroma/embeddings can't be
    exercised in this test environment, so the chroma collection itself is
    mocked here to return a deceptively low distance for an unrelated
    chunk - the word-overlap safety net must still reject it."""

    class FakeCollection:
        def query(self, query_texts, n_results, include=None):
            return {
                "documents": [["Area: DUBAI MARINA. Property type: Flat. ..."]],
                "distances": [[0.3]],  # deceptively low - would pass a naive threshold
                "metadatas": [[{"area": "DUBAI MARINA", "prop_type": "Flat"}]],
            }

    monkeypatch.setattr(appmod, "get_chroma_collection", lambda: FakeCollection())

    ctx = appmod.get_semantic_context("what's the weather in Tokyo")
    assert ctx == "", (
        "Semantic fallback returned context for an unrelated query - "
        "word-overlap safety net did not fire"
    )


def test_semantic_fallback_accepts_related_low_distance_match(monkeypatch):
    """Sanity check the safety net doesn't over-correct: a genuinely
    on-topic query with word overlap should still pass through."""

    class FakeCollection:
        def query(self, query_texts, n_results, include=None):
            return {
                "documents": [["Area: DUBAI MARINA. Property type: Flat. ..."]],
                "distances": [[0.3]],
                "metadatas": [[{"area": "DUBAI MARINA", "prop_type": "Flat"}]],
            }

    monkeypatch.setattr(appmod, "get_chroma_collection", lambda: FakeCollection())

    ctx = appmod.get_semantic_context("marina prices")
    assert "DUBAI MARINA" in ctx


# ---------------------------------------------------------------------------
# 8. Regression: general market-trend questions must be reachable
# ---------------------------------------------------------------------------

MARKET_LEVEL_QUERIES = [
    "Is the overall Dubai market growing or stabilizing?",
    "what is the market trend",
    "is the economy slowing down",
]

@pytest.mark.parametrize("query", MARKET_LEVEL_QUERIES)
def test_market_level_queries_are_answerable(query):
    """Bug found in production: market_intelligence.txt was loaded correctly
    but structurally unreachable - retrieve_context only ever returned
    found=True via an area match or a chroma hit, so any question about
    overall market direction (no area named) always fell through to the
    honest-fallback message, despite real market-summary data existing to
    answer it."""
    ctx, found = appmod.retrieve_context(query)
    assert found, f"Market-level query incorrectly fell through to fallback: {query!r}"
    assert "DUBAI MARKET REPORT" in ctx


def test_area_queries_still_take_priority_over_market_summary():
    """The market-summary route must not swallow area-specific questions."""
    ctx, found = appmod.retrieve_context("average price per sqft in Dubai Marina")
    assert found
    assert "DUBAI MARINA" in ctx
    assert "OVERALL (all property types combined)" in ctx or "OVERALL" in ctx


def test_unrelated_query_still_rejected_despite_market_route():
    """Adding the market-summary route must not loosen the honest fallback
    for genuinely unrelated queries."""
    ctx, found = appmod.retrieve_context("best restaurants in dubai")
    assert not found


# ---------------------------------------------------------------------------
# 9. Legal grounding: retrieval, fail-closed safety net, combined domains
# ---------------------------------------------------------------------------

_FAKE_LEGAL_DOC = (
    "Law No. (6) of 2019, Article (12): A Unit Owner may sell or dispose of "
    "his Unit in any legal manner, and may mortgage his Unit to any bank or "
    "financing institution licensed to operate in the Emirate."
)


class _FakeLegalCollection:
    def query(self, query_texts, n_results, include=None):
        return {
            "documents": [[_FAKE_LEGAL_DOC]],
            "distances": [[0.4]],
            "metadatas": [[{"law": "Law No. (6) of 2019", "article_num": 12, "page": 10}]],
        }


def test_legal_context_accepts_related_query(monkeypatch):
    monkeypatch.setattr(appmod, "get_legal_collection", lambda: _FakeLegalCollection())
    ctx = appmod.get_legal_context("can I sell or mortgage my unit")
    assert "Article (12)" in ctx
    assert "Law No. (6) of 2019" in ctx


def test_legal_context_rejects_unrelated_query_despite_low_distance(monkeypatch):
    """Same fail-closed principle already required for price semantic
    search: a low embedding distance alone must not be trusted, since a
    long legal Article has enough words that spurious overlap is a real
    risk without live testing against the actual embedding model."""
    monkeypatch.setattr(appmod, "get_legal_collection", lambda: _FakeLegalCollection())
    ctx = appmod.get_legal_context("what's the weather like today")
    assert ctx == ""


def test_combined_price_and_legal_retrieval(monkeypatch):
    """A single question spanning both domains (price + legal rights) must
    retrieve and label both, not force a choice between them."""
    monkeypatch.setattr(appmod, "get_legal_collection", lambda: _FakeLegalCollection())
    ctx, found = appmod.retrieve_context("can I mortgage my unit in Dubai Marina")
    assert found
    assert "[TRANSACTION DATA]" in ctx
    assert "DUBAI MARINA" in ctx
    assert "[LEGAL PROVISIONS]" in ctx
    assert "Article (12)" in ctx


def test_legal_query_outside_book_scope_falls_back_honestly():
    """Golden Visa / immigration questions are not in the indexed
    legislation (verified: the compiled book covers only DLD/RERA
    real-estate law - ownership, mortgage, escrow, tenancy, brokers,
    foreign-ownership zones - not federal immigration matters). With no
    legal collection mocked, this must fall through to the honest
    fallback rather than answering from training knowledge."""
    ctx, found = appmod.retrieve_context("what is the golden visa process")
    assert not found


def test_is_legal_query_detection():
    assert appmod.is_legal_query("what are my ownership rights as a foreigner")
    assert appmod.is_legal_query("can my landlord evict me without notice")
    assert not appmod.is_legal_query("average price per sqft in Dubai Marina")


# ---------------------------------------------------------------------------
# 10. Legal ingestion parser (ingest_laws.py) - run only if book.pdf present
# ---------------------------------------------------------------------------

LAW_PDF_PATH = "book.pdf"

@pytest.mark.skipif(not os.path.exists(LAW_PDF_PATH), reason="book.pdf not present in this environment")
def test_law_ingestion_finds_no_unknown_law():
    """Bug found during development: the book's opening law never got a
    title assigned, because title-detection only ran on a NUMBERING DROP,
    and there's nothing to drop from at the very start of the document."""
    import ingest_laws
    articles = ingest_laws.parse_articles(LAW_PDF_PATH)
    unknown = [a for a in articles if a["law"] == "Unknown Law"]
    assert not unknown, f"{len(unknown)} articles have no law title assigned"


@pytest.mark.skipif(not os.path.exists(LAW_PDF_PATH), reason="book.pdf not present in this environment")
def test_law_ingestion_title_matches_real_citations_not_preamble_noise():
    """Bug found during development: a short rolling window for title
    detection could scroll past the real title and land on a law CITED in
    the preamble instead (e.g. detecting 'Law No. (16) of 2007' - a law
    referenced in "After perusal of..." text - instead of the real title
    'Law No. (4) of 2019'). The fix takes the FIRST title-like match in an
    unbounded buffer since the previous heading, since the real title
    always appears before any preamble citations."""
    import ingest_laws
    articles = ingest_laws.parse_articles(LAW_PDF_PATH)
    laws = {a["law"] for a in articles}
    # These are verified real law titles from the table of contents -
    # if title detection regresses to grabbing preamble citations instead,
    # these specific known-correct titles will disappear.
    for expected in ["Law No. (6) of 2019", "Law No. (4) of 2019", "Decree No. (31) of 2016"]:
        assert expected in laws, f"Expected law title {expected!r} not found - title detection may have regressed"


@pytest.mark.skipif(not os.path.exists(LAW_PDF_PATH), reason="book.pdf not present in this environment")
def test_law_ingestion_article_count_reasonable():
    import ingest_laws
    articles = ingest_laws.parse_articles(LAW_PDF_PATH)
    assert len(articles) > 400, "Suspiciously few articles parsed - check column extraction"