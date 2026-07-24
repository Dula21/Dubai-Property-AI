"""
smoke_test_model.py — Manual live check of actual LLM output quality.

Run this by hand after changing MODEL_NAME or REASONING_EFFORT in app.py.
NOT part of pytest / test_app.py: this makes real Groq API calls (small
cost, real latency) and requires a human to read the answers - it checks
whether the MODEL follows the grounding rules, which no automated test in
test_app.py currently covers (those all mock the LLM call and only check
what's sent TO the model, never what comes back).

    python smoke_test_model.py

For each case below, read the printed answer and check the "watch for"
note - these are the exact failure modes found in production with the
old llama-3.1-8b-instant model, so they're the highest-risk spots for a
new model (or new reasoning_effort setting) to regress on.
"""

import app as appmod

CASES = [
    {
        "query": "average price per sqft in Dubai Marina",
        "watch_for": (
            "Must state the OVERALL count-weighted average (1024 sales, "
            "AED 2086.6/sqft) directly, without re-deriving or hedging on it."
        ),
    },
    {
        "query": "how many flats sold in JVC and at what median price?",
        "watch_for": (
            "Must label the count as FLAT sales specifically, not blur it "
            "into an area-wide total (this was the original per-type vs "
            "overall confusion bug)."
        ),
    },
    {
        "query": "is the overall Dubai market growing or stabilizing?",
        "watch_for": (
            "Must answer from [MARKET SUMMARY] context, not fall back to "
            "the honest 'I don't have data' message."
        ),
    },
    {
        "query": "can I sell or mortgage my unit?",
        "watch_for": (
            "Must cite the specific Article and Law number from "
            "[LEGAL PROVISIONS] (e.g. 'Article (12) of Law No. (6) of 2019'), "
            "not paraphrase without attribution."
        ),
    },
    {
        "query": "what is the golden visa process for property owners?",
        "watch_for": (
            "Must say this is outside the indexed legislation and suggest "
            "a RERA-registered agent/lawyer - must NOT answer from general "
            "training knowledge about Golden Visas."
        ),
    },
    {
        "query": "tell me about Atlantis The Royal residences",
        "watch_for": (
            "Must fall back honestly (no area match should fire) - this was "
            "the original false-positive fuzzy-matching bug."
        ),
    },
]


def main():
    print(f"Model: {appmod.MODEL_NAME}  |  reasoning_effort: {appmod.REASONING_EFFORT}\n")
    print("=" * 80)
    for case in CASES:
        answer = appmod.chat_function(case["query"], [])
        print(f"Q: {case['query']}")
        print(f"WATCH FOR: {case['watch_for']}")
        print(f"A: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()
