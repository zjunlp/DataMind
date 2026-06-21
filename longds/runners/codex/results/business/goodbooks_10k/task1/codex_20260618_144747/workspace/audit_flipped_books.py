from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent


def author_count(authors: str) -> int:
    return len([part for part in str(authors).split(",") if part.strip()])


def tag_evidence(rows: pd.DataFrame, limit: int = 3) -> str:
    ordered = rows.sort_values(["count", "tag_name"], ascending=[False, True]).head(limit)
    return "; ".join(f"{row.tag_name}:{int(row['count'])}" for _, row in ordered.iterrows())


def main() -> None:
    clean = pd.read_csv(WORKSPACE / "cleaned_books_analysis.csv")
    tags = pd.read_csv(WORKSPACE / "recognized_genre_tag_rows.csv")
    summary = json.loads((WORKSPACE / "exposure_aware_quality_summary.json").read_text())

    movement_by_key = {
        **{int(book_key): "entered" for book_key in summary["entered_book_keys"]},
        **{int(book_key): "left" for book_key in summary["left_book_keys"]},
    }
    flipped_keys = summary["entered_book_keys"] + summary["left_book_keys"]

    strict_duplicate = clean["strict_title_key"].duplicated(keep=False)
    clean = clean.copy()
    clean["duplicate_cleaned_title"] = strict_duplicate

    records: list[dict[str, object]] = []
    for book_key in flipped_keys:
        book = clean.loc[clean["book_key"].eq(book_key)].iloc[0]
        tag_rows = tags.loc[tags["book_key"].eq(book_key)].copy()
        listed_author_count = author_count(book["authors"])
        restored_tag_rows = int(len(tag_rows))

        if bool(book["duplicate_cleaned_title"]):
            risk = "duplicate_title_handling"
        elif listed_author_count > 2:
            risk = "author_ambiguity"
        elif restored_tag_rows < 3:
            risk = "compressed_genre_evidence"
        else:
            risk = "low_source_risk"

        records.append(
            {
                "movement": movement_by_key[int(book_key)],
                "book_key": int(book_key),
                "display_title": book["original_title"],
                "full_authors": book["authors"],
                "cover_link": book["small_image_url"],
                "main_genre": book["main_genre"],
                "tag_evidence": tag_evidence(tag_rows),
                "restored_recognized_tag_row_count": restored_tag_rows,
                "duplicate_cleaned_title": bool(book["duplicate_cleaned_title"]),
                "listed_author_count": listed_author_count,
                "main_source_risk": risk,
            }
        )

    pd.DataFrame(records).to_csv(WORKSPACE / "flipped_books_source_audit.csv", index=False)
    with (WORKSPACE / "flipped_books_source_audit.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)
    print(json.dumps(records, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
