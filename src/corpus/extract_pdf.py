"""
Extract 3 targeted page ranges from the full PostgreSQL 17 PDF.
Run: PYTHONPATH=. python src/corpus/extract_pdf.py
"""
import random
from pathlib import Path

import fitz  # pymupdf

PDF_PATH = "/Users/pranav/AI/Projects/prostgres17/postgresql-17-A4.pdf"
OUT_DIR = Path("corpus/raw")

# Page ranges verified from TOC (0-based PDF page index = doc page - 1).
# Non-contiguous ranges are merged into one PDF via multiple insert_pdf calls.
EXTRACTIONS = [
    {
        "name": "postgres_data_definition.pdf",
        "desc": "Ch 5: Data Definition (Tables, Constraints, Partitioning, Privileges)",
        # doc p97–149 → index 96–148
        "ranges": [(96, 148)],
    },
    {
        "name": "postgres_queries_indexes.pdf",
        "desc": "Ch 7: Queries + Ch 11: Indexes + Ch 13: Concurrency Control",
        # Ch 7: doc p154–184 → index 153–183
        # Ch 11: doc p485–500 → index 484–499
        # Ch 13: doc p543–558 → index 542–557
        "ranges": [(153, 183), (484, 499), (542, 557)],
    },
    {
        "name": "postgres_administration.pdf",
        "desc": "Ch 21: Database Roles + Ch 24: Routine Maintenance + Ch 25: Backup",
        # Ch 21: doc p769–776 → index 768–775
        # Ch 24: doc p808–819 → index 807–818
        # Ch 25: doc p820–835 → index 819–834
        "ranges": [(768, 775), (807, 834)],
    },
]


def print_toc(doc: fitz.Document) -> None:
    print(f"\nTotal pages in PDF: {len(doc)}")
    print("\nTable of Contents (levels 1-2):")
    toc = doc.get_toc()
    for level, title, page in toc:
        if level <= 2:
            print(f"  p{page:5d}  {'  ' * (level - 1)}{title}")


def extract_ranges(src_path: str, out_path: Path, ranges: list[tuple[int, int]]) -> None:
    doc = fitz.open(src_path)
    max_page = len(doc) - 1
    out = fitz.open()
    total = 0
    for start, end in ranges:
        end = min(end, max_page)
        out.insert_pdf(doc, from_page=start, to_page=end)
        total += end - start + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path))
    print(f"Saved {total} pages → {out_path}")


def spot_check(pdf_path: Path, n_pages: int = 3) -> None:
    doc = fitz.open(str(pdf_path))
    pages = random.sample(range(len(doc)), min(n_pages, len(doc)))
    print(f"\n=== Spot-check: {pdf_path.name} ({len(doc)} pages) ===")
    for p in sorted(pages):
        text = doc[p].get_text()[:500]
        print(f"\n--- Page {p} sample ---")
        print(text)
        print("---")


def main() -> None:
    doc = fitz.open(PDF_PATH)
    print_toc(doc)

    print("\n" + "=" * 60)
    print("Extracting PDFs...")
    print("=" * 60)

    for entry in EXTRACTIONS:
        out_path = OUT_DIR / entry["name"]
        if out_path.exists():
            print(f"\n[SKIP] {entry['name']} already exists — delete to re-extract")
            spot_check(out_path)
            continue
        print(f"\n[EXTRACT] {entry['desc']}")
        extract_ranges(PDF_PATH, out_path, entry["ranges"])
        spot_check(out_path)

    print("\n" + "=" * 60)
    print("Extraction complete. Review spot-checks above.")
    print("If any PDF shows garbled tables, adjust page ranges and re-run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
