"""
ingest_laws.py — Dubai Property Intelligence: legal grounding pipeline

Ingests the DLD real-estate legislation compilation (book.pdf) into a
ChromaDB collection separate from the transaction-price data, chunked by
legal Article so answers can cite a precise law + article number rather
than a vague paraphrase.

WHY THIS ISN'T A SIMPLE pdftotext JOB:
The book is laid out in two text columns per page. Naive linear text
extraction (pdftotext -layout) interleaves the two columns mid-line,
producing garbled text (verified: one line mixed a "Hotel Project"
definition from the left column with an unrelated "Owner" definition from
the right column). This script extracts each column separately by
cropping on the page's horizontal midpoint before reading text.

WHY ARTICLE HEADINGS AREN'T DETECTED BY A SIMPLE SUBSTRING SEARCH:
The law text frequently cross-references itself mid-sentence, e.g.
"...pursuant to paragraph (a) of this Article..." or "...provisions of
Article (37) of Law No...". A naive `"Article (" in line` search treats
every one of these as a new heading, which is wrong. Verified: a real
Article heading appears as its OWN isolated line containing nothing but
"Article (N)" - cross-references appear embedded within a sentence.
Detection here requires the full line to match the heading pattern
exactly, which correctly separates true headings from citations (checked
against the whole 127-page document: 449 real headings found this way,
vs. 499 raw substring matches - the ~50 difference is exactly the
cross-reference noise this approach filters out).

WHY LAW BOUNDARIES ARE DETECTED VIA ARTICLE-NUMBER RESETS, NOT THE INDEX:
The book's own table of contents is itself printed in two columns and
resists reliable regex parsing. Each law restarts its own Article
numbering at (1), so a drop in article number (e.g. ...52, then 1...) is
a reliable, verifiable signal of a new law starting. Cross-checked: 23
such resets found, consistent with the ~24 distinct laws visible in the
table of contents.

Run:
    pip install pdfplumber chromadb sentence-transformers
    python ingest_laws.py /path/to/book.pdf
"""

import sys
import re
import pdfplumber

LAW_TITLE_PATTERN = re.compile(
    r"^((?:Law No\.|Decree No\.|Regulation No\.|Executive Council Resolution No\.|"
    r"Resolution No\.|By law No\.)\s*\(\d+\)\s*of\s*\d{4})",
)
ARTICLE_HEADING_PATTERN = re.compile(r"^Article\s*\((\d+)\)$")


def extract_columns(pdf_path: str):
    """Yields (page_number, column_label, text) for each column of each page."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            mid = page.width / 2
            left = page.crop((0, 0, mid, page.height)).extract_text() or ""
            right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
            yield i + 1, "L", left
            yield i + 1, "R", right


def find_law_title_in_buffer(buffer_lines):
    """Search a buffer of lines (everything since the previous Article
    heading) for the FIRST 'Law No. (N) of YYYY' style match.

    Must take the FIRST match, not just any match in a short trailing
    window: each law's preamble cites several OTHER laws by number
    ("After perusal of: Law No. (3) of 2003 Establishing..."), and those
    citations appear AFTER the real title banner. A naive short rolling
    window can scroll past the real title and land on a citation instead
    (verified: this produced wrong law names like "Law No. (16) of 2007"
    for a page whose real title was "Law No. (4) of 2019" - the citation
    text was still in the window, the real title had already scrolled out
    of it). The real title reliably appears first in the buffer.
    """
    for line in buffer_lines:
        m = LAW_TITLE_PATTERN.search(line)
        if m:
            return m.group(1)
    return None


def parse_articles(pdf_path: str):
    """Returns a list of dicts: {law, article_num, text, page}."""
    current_law = "Unknown Law"
    prev_article_num = 0
    is_first_heading = True
    articles = []
    current = None  # {"law":..., "article_num":..., "text": [...], "page": ...}

    # Buffer ALL lines since the previous heading, so the real law title
    # (which always appears first, before any preamble citations of other
    # laws) can be found reliably rather than scrolling out of a short window.
    buffer = []

    for page_num, col, col_text in extract_columns(pdf_path):
        if not col_text:
            continue
        lines = col_text.split("\n")
        for line in lines:
            stripped = line.strip()

            heading_match = ARTICLE_HEADING_PATTERN.match(stripped)
            if heading_match:
                num = int(heading_match.group(1))

                # A drop in article number (allowing a little OCR/columnar
                # noise tolerance) means a new law has started. The very
                # first heading in the whole document is a special case:
                # there is no prior number to drop FROM, so it would never
                # trigger this check on its own, yet it still needs its
                # title looked up (this was the source of the "Unknown
                # Law" articles found in testing - the book's opening law
                # never got a title assigned without this).
                is_new_law = (
                    is_first_heading
                    or num < prev_article_num - 2
                    or (num == 1 and prev_article_num > 1)
                )
                if is_new_law:
                    title = find_law_title_in_buffer(buffer)
                    if title:
                        current_law = title
                is_first_heading = False

                if current is not None:
                    articles.append(current)
                current = {
                    "law": current_law,
                    "article_num": num,
                    "text": [],
                    "page": page_num,
                }
                prev_article_num = num
                buffer = []
                continue

            buffer.append(stripped)
            if current is not None and stripped:
                current["text"].append(stripped)

    if current is not None:
        articles.append(current)

    for a in articles:
        a["text"] = " ".join(a["text"])

    return articles


def build_chunks(articles):
    """One chunk per Article. Skips near-empty articles (heading with no
    body captured - can happen at column-boundary edge cases) since an
    empty chunk would be embedded and retrieved with zero useful content."""
    ids, docs, metas = [], [], []
    for i, a in enumerate(articles):
        if len(a["text"]) < 20:
            continue
        doc = f"{a['law']}, Article ({a['article_num']}): {a['text']}"
        ids.append(f"law-{i}")
        docs.append(doc)
        metas.append({
            "law": a["law"],
            "article_num": a["article_num"],
            "page": a["page"],
        })
    return ids, docs, metas


def build_chroma_store(ids, docs, metas, persist_dir="chroma_db"):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection("dld_laws")
    except Exception:
        pass
    collection = client.create_collection("dld_laws", embedding_function=embed_fn)

    BATCH = 300
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])

    return collection


def main(pdf_path: str):
    print(f"Extracting articles from {pdf_path} ...")
    articles = parse_articles(pdf_path)
    print(f"Articles found: {len(articles)}")

    laws_seen = sorted(set(a["law"] for a in articles))
    print(f"Distinct laws detected: {len(laws_seen)}")
    for law in laws_seen:
        print(f"  - {law}")

    ids, docs, metas = build_chunks(articles)
    print(f"Chunks ready for embedding: {len(ids)} (dropped {len(articles) - len(ids)} near-empty)")

    print("Building ChromaDB legal store (requires internet for model download)...")
    build_chroma_store(ids, docs, metas)
    print("Done. chroma_db/ now also contains the 'dld_laws' collection.")
    print("Commit chroma_db/ to your repo alongside the price-data collection.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_laws.py /path/to/book.pdf")
        sys.exit(1)
    main(sys.argv[1])
