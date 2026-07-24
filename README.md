---
title: Dubai Property Advisor
emoji: 🏙️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.36.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🏗️ Dubai Property Intelligence (DPI)
### **Grounded Real Estate Q&A over Real DLD Transaction Data**

## 🌟 Project Overview
Dubai Property Intelligence answers questions about Dubai residential sale
prices using **real, transaction-level data from the Dubai Land Department
(DLD)** — not general knowledge or estimates. The dataset currently covers
genuine sale transactions from **January 1, 2026 to July 19, 2026** across
228 areas.

If a question falls outside what the dataset covers (an area not present,
a date range outside the window, a topic unrelated to Dubai property
transactions), the app says so directly instead of guessing.

## 🧭 Version history (honest account)
- **v1** (Colab prototype): loaded the DLD *residential price index* file
  (an aggregate monthly index, not per-area transaction data), computed
  month-over-month growth with pandas, and injected a short static summary
  into every LLM query as context. This was a lightweight form of
  context-injection RAG, but it could only speak to overall market
  direction — not area-specific questions like "price per sqft in X."
- **v2** (this version): rebuilt on the actual DLD **transaction-level**
  export (124k+ rows, `TRANS_VALUE`, `ACTUAL_AREA`, `AREA_EN`, etc.),
  with a proper retrieval pipeline: pandas aggregation for exact numeric
  answers, ChromaDB for semantic fallback, and a strict "answer only from
  retrieved data, or say you don't know" instruction to the LLM.
- The move from the original Colab notebook to a standalone `app.py` was
  driven by Hugging Face Spaces compatibility: the notebook used
  Colab-specific display/share-link calls, and Gradio's `ChatInterface`
  history format needed updating for stability on Spaces. The underlying
  data logic was preserved and upgraded, not discarded.
- **Model migration (July 2026)**: Groq deprecated `llama-3.1-8b-instant`
  (announced June 17, 2026, shutdown August 16, 2026). Migrated to
  `openai/gpt-oss-20b` at `reasoning_effort="low"` — Groq's recommended,
  fastest, and cheapest currently-supported replacement. Verified against
  the same production bug cases the original model was tested on (overall
  vs. per-type figures, legal Article citation, false-positive area
  matching, honest fallback on out-of-scope topics) before deploying;
  see `smoke_test_model.py`.

## 🚀 How answers are grounded
1. **Deterministic match**: the query is checked against the 228 real DLD
   area names (including common abbreviations like "JVC") using exact and
   whole-word matching. On a match, the **exact pre-computed statistics**
   for that area (count, average/median/min/max price per sqft, date
   range) are pulled from `area_aggregates.csv` — this path never lets the
   LLM invent or calculate a number itself.
2. **Market-level routing**: broad questions about overall market
   direction (no area named) route to a pre-built market summary.
3. **Legal grounding**: questions touching ownership, mortgages, tenancy,
   escrow, brokers, or foreign-ownership rules are checked against a
   ChromaDB collection built from the DLD real-estate legislation
   compilation, chunked by individual Article and tagged with its Law/Decree
   number for precise citation (e.g. "Article (12) of Law No. (6) of 2019").
   A price question and a legal question in the same message are both
   retrieved and answered together, clearly labeled by domain.
4. **Semantic fallback**: if no area is directly named, a ChromaDB vector
   search (local `all-MiniLM-L6-v2` embeddings, no external API) looks for
   the closest matching aggregate chunk.
5. **Honest fallback**: if nothing relevant is found in any domain, the
   app says so directly and points to a RERA-registered agent or property
   lawyer for anything outside its scope — the LLM is never called with
   empty or irrelevant context.

## ⚖️ Legal grounding: scope and limitations
- The indexed legislation covers **DLD/RERA real-estate law only**:
  jointly-owned property, mortgages, escrow accounts, landlord-tenant
  relations, brokers register, foreign-ownership zones, rental disputes.
- It does **not** cover visa/Golden Visa rules (a federal immigration
  matter, outside DLD's regulatory scope), other emirates, or criminal law.
  Questions on these topics get the honest fallback, not a guess.
- The compiled edition is dated; it may not reflect amendments passed
  after its last update. The app's system prompt instructs the model to
  flag this rather than present retrieved text as necessarily current,
  and every legal answer suggests confirming with a RERA-registered agent
  or property lawyer.
- Extraction required column-aware parsing (the source book uses a
  two-column layout that garbles under naive linear text extraction) and
  careful heading detection to distinguish real Article headings from
  the frequent inline cross-references in the legal text (e.g. "pursuant
  to Article (37) of Law No...."). See `ingest_laws.py` for the verified
  approach.

## ⚠️ Known data-quality fixes applied during ingestion
- `ACTUAL_AREA` / `PROCEDURE_AREA` in the raw DLD export are recorded in
  **square meters**, despite not being labeled as such. All price-per-sqft
  figures are computed with the sqm→sqft conversion (×10.7639) applied.
  Verified against Dubai Marina: raw math gave a nonsensical ~AED
  21,700/sqft; corrected figure is ~AED 1,850/sqft, which matches known
  market levels.
- Only rows where `GROUP_EN == "Sales"` are used for price statistics.
  `Mortgage` rows can repeat a single portfolio loan value across multiple
  unit rows, and `Gifts` aren't market transactions — both would distort
  price aggregates if included.
- Area names had inconsistent casing in the raw export (e.g. "Business
  Bay" vs "BUSINESS BAY" counted as different areas); normalized before
  aggregation.
- Top/bottom 0.5% of price-per-sqft values are excluded from aggregate
  statistics as likely data-entry outliers (row-level data is kept as-is;
  only the aggregates used for answers exclude these).

## 🛠️ Tech Stack
* **Core Logic:** Python, Pandas
* **Retrieval:** ChromaDB (local, free) + deterministic area matching
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API cost)
* **LLM:** GPT-OSS 20B (`openai/gpt-oss-20b`, `reasoning_effort="low"`) via Groq Cloud API (free tier)
* **Interface:** Gradio
* **Hosting:** Hugging Face Spaces (free CPU tier)

## ⚙️ Setup
1. Run `python ingest.py /path/to/dld_transactions.csv` once, wherever you
   have internet access (the embedding model downloads from Hugging Face
   Hub on first run). This produces `cleaned_sales.csv`,
   `area_aggregates.csv`, `market_intelligence.txt`, and `chroma_db/`
   (the `dld_aggregates` collection).
2. Run `python ingest_laws.py /path/to/book.pdf` once, same environment.
   This adds a second collection, `dld_laws`, to the same `chroma_db/`.
3. Commit `area_aggregates.csv`, `market_intelligence.txt`, and
   `chroma_db/` (both collections) alongside `app.py` to your Space
   repo — `app.py` only *loads* them at startup, it never recomputes
   embeddings itself.
4. Set the `GROQ_API_KEY` secret in your Space settings.

## 📈 Limitations to be upfront about
- Coverage is limited to the date range and areas present in the ingested
  CSV. Re-run `ingest.py` with a fresh export to extend coverage.
- Aggregate statistics, not individual listing lookups — this tells you
  what DLD-recorded sales looked like for an area/property type, not a
  live "what should I list my unit for" valuation.

---
*Developed by **Dulasi Nethma** | BSc (Hons) in IT | Aspiring AI Engineer*