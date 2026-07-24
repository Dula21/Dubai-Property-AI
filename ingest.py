"""
ingest.py — Dubai Property Intelligence: DLD data pipeline

Run this ONCE whenever you have a fresh DLD transactions CSV export.
It must run somewhere with internet access (your HF Space, Colab, or your
own machine) because the ChromaDB step downloads a small embedding model
from Hugging Face Hub on first use.

Produces (all committed to the repo so app.py never has to recompute them):
  - cleaned_sales.csv        -> full cleaned row-level sales data
  - area_aggregates.csv      -> per (area, property sub-type) stats
  - chroma_db/                -> persisted vector store for semantic search
  - market_intelligence.txt  -> overall market summary (kept for continuity
                                 with the original notebook's macro-trend use)

Run:
    pip install pandas chromadb sentence-transformers
    python ingest.py /path/to/transactions.csv
"""

import sys
import difflib
import pandas as pd

SQM_TO_SQFT = 10.7639

# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Only genuine sale transactions carry a real market price.
    # Mortgage rows can repeat the SAME total value across multiple unit rows
    # (portfolio mortgages) and Gifts aren't market transactions at all.
    sales = df[
        (df["GROUP_EN"] == "Sales")
        & (df["TRANS_VALUE"] > 0)
        & (df["ACTUAL_AREA"] > 0)
    ].copy()

    # Some area names appear with inconsistent casing in the raw export
    # (e.g. "Business Bay" and "BUSINESS BAY" as separate strings), which
    # would silently split one area's transactions into two groups and
    # understate its true sample size. Normalize before anything else.
    sales["AREA_EN"] = sales["AREA_EN"].str.strip().str.upper()

    # CRITICAL FIX: ACTUAL_AREA / PROCEDURE_AREA are recorded in square
    # METERS despite the column names, not square feet. Verified against
    # known Dubai Marina price levels (median comes out ~AED 1,850/sqft
    # only after this conversion; ~10.7x too high without it).
    sales["area_sqft"] = sales["ACTUAL_AREA"] * SQM_TO_SQFT
    sales["price_per_sqft"] = sales["TRANS_VALUE"] / sales["area_sqft"]

    # Drop extreme data-entry outliers (top/bottom 0.5%) so aggregates
    # aren't distorted by obvious mis-entries. Row-level data is kept as-is;
    # only the AGGREGATE stats exclude these.
    low, high = sales["price_per_sqft"].quantile([0.005, 0.995])
    sales["is_outlier"] = (sales["price_per_sqft"] < low) | (sales["price_per_sqft"] > high)

    sales["INSTANCE_DATE"] = pd.to_datetime(sales["INSTANCE_DATE"])
    return sales


# ---------------------------------------------------------------------------
# 2. Aggregate per area + property sub-type
# ---------------------------------------------------------------------------

def build_aggregates(sales: pd.DataFrame) -> pd.DataFrame:
    clean = sales[~sales["is_outlier"]]

    agg = (
        clean.groupby(["AREA_EN", "PROP_SB_TYPE_EN"])
        .agg(
            transaction_count=("price_per_sqft", "count"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            min_price_per_sqft=("price_per_sqft", "min"),
            max_price_per_sqft=("price_per_sqft", "max"),
            avg_trans_value=("TRANS_VALUE", "mean"),
            earliest_date=("INSTANCE_DATE", "min"),
            latest_date=("INSTANCE_DATE", "max"),
        )
        .reset_index()
    )
    numeric_cols = agg.select_dtypes(include="number").columns
    agg[numeric_cols] = agg[numeric_cols].round(1)
    return agg


# ---------------------------------------------------------------------------
# 3. Build text chunks for ChromaDB (one per area+type aggregate group)
# ---------------------------------------------------------------------------

def build_chunks(agg: pd.DataFrame):
    """Returns (ids, documents, metadatas) ready for chroma_collection.add()."""
    ids, docs, metas = [], [], []
    for i, row in agg.iterrows():
        # Skip groups too small to be a meaningful statistic.
        if row["transaction_count"] < 3:
            continue
        doc = (
            f"Area: {row['AREA_EN']}. Property type: {row['PROP_SB_TYPE_EN']}. "
            f"Based on {int(row['transaction_count'])} DLD sale transactions "
            f"from {row['earliest_date'].date()} to {row['latest_date'].date()}: "
            f"average price per sqft = AED {row['avg_price_per_sqft']}, "
            f"median price per sqft = AED {row['median_price_per_sqft']}, "
            f"range = AED {row['min_price_per_sqft']} to AED {row['max_price_per_sqft']}. "
            f"Average transaction value = AED {row['avg_trans_value']}."
        )
        ids.append(f"agg-{i}")
        docs.append(doc)
        metas.append({
            "area": row["AREA_EN"],
            "prop_type": row["PROP_SB_TYPE_EN"],
            "transaction_count": int(row["transaction_count"]),
        })
    return ids, docs, metas


# ---------------------------------------------------------------------------
# 4. Persist to ChromaDB (needs internet for the embedding model download)
# ---------------------------------------------------------------------------

def build_chroma_store(ids, docs, metas, persist_dir="chroma_db"):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=persist_dir)

    # Free, local, CPU-friendly embedding model — downloaded once from
    # Hugging Face Hub on first run, then cached. No API key, no cost.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recreate collection each run so re-ingesting a fresh CSV doesn't
    # leave stale chunks behind.
    try:
        client.delete_collection("dld_aggregates")
    except Exception:
        pass
    collection = client.create_collection("dld_aggregates", embedding_function=embed_fn)

    # Chroma has a batch-size ceiling; chunk the inserts to be safe.
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])

    return collection


# ---------------------------------------------------------------------------
# 5. Overall market summary (kept for macro-trend questions)
# ---------------------------------------------------------------------------

def build_market_summary(sales: pd.DataFrame) -> str:
    clean = sales[~sales["is_outlier"]]
    monthly = (
        clean.set_index("INSTANCE_DATE")
        .resample("MS")["price_per_sqft"]
        .median()
        .dropna()
    )
    latest = monthly.iloc[-1]
    prev = monthly.iloc[-2] if len(monthly) > 1 else latest
    growth = ((latest - prev) / prev * 100) if prev else 0.0

    return (
        f"DUBAI MARKET REPORT (auto-generated from DLD transactions)\n"
        f"Data window: {sales['INSTANCE_DATE'].min().date()} to {sales['INSTANCE_DATE'].max().date()}\n"
        f"Total genuine sale transactions analyzed: {len(clean)}\n"
        f"Latest month median price/sqft: AED {latest:.0f}\n"
        f"Month-over-month change: {growth:+.2f}%\n"
        f"Areas covered: {clean['AREA_EN'].nunique()}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str):
    print(f"Loading {csv_path} ...")
    sales = load_and_clean(csv_path)
    print(f"Cleaned sales rows: {len(sales)}")

    sales.to_csv("cleaned_sales.csv", index=False)

    agg = build_aggregates(sales)
    agg.to_csv("area_aggregates.csv", index=False)
    print(f"Aggregate groups: {len(agg)}")

    ids, docs, metas = build_chunks(agg)
    print(f"Chunks eligible for embedding (count>=3): {len(ids)}")

    summary = build_market_summary(sales)
    with open("market_intelligence.txt", "w") as f:
        f.write(summary)
    print(summary)

    print("Building ChromaDB store (requires internet for model download)...")
    build_chroma_store(ids, docs, metas)
    print("Done. chroma_db/, cleaned_sales.csv, area_aggregates.csv, market_intelligence.txt are ready.")
    print("Commit all of these to your HF Space repo alongside app.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py /path/to/transactions.csv")
        sys.exit(1)
    main(sys.argv[1])