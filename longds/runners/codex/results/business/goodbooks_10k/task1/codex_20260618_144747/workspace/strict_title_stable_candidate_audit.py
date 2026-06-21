from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent


def listed_author_count(authors: str) -> int:
    return len([part for part in str(authors).split(",") if part.strip()])


def round_record(record: dict[str, object], decimal_columns: set[str]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in record.items():
        if key in decimal_columns and pd.notna(value):
            rounded[key] = round(float(value), 3)
        elif isinstance(value, float) and pd.isna(value):
            rounded[key] = None
        else:
            rounded[key] = value
    return rounded


def collision_examples(clean: pd.DataFrame, key: str, book_key: int) -> str:
    peers = clean.loc[
        clean["strict_title_key"].eq(key) & clean["book_key"].ne(book_key),
        ["book_key", "original_title"],
    ].sort_values(["book_key"])
    return "; ".join(
        f"{int(row.book_key)}:{row.original_title}" for _, row in peers.iterrows()
    )


def main() -> None:
    clean = pd.read_csv(WORKSPACE / "cleaned_books_analysis.csv")
    initial_pool = pd.read_csv(WORKSPACE / "first_review_pool.csv")
    relative_summary = json.loads((WORKSPACE / "relative_demand_summary.json").read_text())
    exposure_summary = json.loads((WORKSPACE / "exposure_aware_quality_summary.json").read_text())

    monitoring_sets = {
        "initial_volume": initial_pool.head(10)["book_key"].astype(int).tolist(),
        "relative_demand": [int(book_key) for book_key in relative_summary["relative_top10_book_keys"]],
        "exposure_aware_quality": [
            int(book_key) for book_key in exposure_summary["exposure_aware_top10_book_keys"]
        ],
    }
    appearances = Counter(book_key for keys in monitoring_sets.values() for book_key in keys)
    stable_keys = [book_key for book_key, count in appearances.items() if count >= 2]

    clean = clean.copy()
    clean["early_title_key_size"] = clean.groupby("early_title_key")["book_key"].transform("size")
    clean["strict_title_key_size"] = clean.groupby("strict_title_key")["book_key"].transform("size")
    clean["early_title_collision_count"] = clean["early_title_key_size"] - 1
    clean["strict_title_collision_count"] = clean["strict_title_key_size"] - 1
    clean["strict_collision_increase"] = (
        clean["strict_title_collision_count"] > clean["early_title_collision_count"]
    )
    clean["original_title_length"] = clean["original_title"].astype(str).str.len()
    median_title_length = clean["original_title_length"].median()
    clean["short_title"] = clean["original_title_length"].le(median_title_length)
    clean["listed_author_count"] = clean["authors"].map(listed_author_count)
    clean["author_ambiguity"] = clean["listed_author_count"].gt(2)

    clean["title_merging_points"] = clean["strict_collision_increase"].astype(int) * 25
    clean["short_title_points"] = clean["short_title"].astype(int) * 20
    clean["author_ambiguity_points"] = clean["author_ambiguity"].astype(int) * 10
    clean["title_risk_score"] = (
        clean["title_merging_points"]
        + clean["short_title_points"]
        + clean["author_ambiguity_points"]
    )

    stable = clean.loc[clean["book_key"].isin(stable_keys)].copy()
    stable["monitoring_set_appearances"] = stable["book_key"].map(appearances)
    for set_name, keys in monitoring_sets.items():
        stable[f"in_{set_name}"] = stable["book_key"].isin(keys)
    stable["strict_collision_examples"] = [
        collision_examples(clean, row.strict_title_key, int(row.book_key))
        for row in stable.itertuples(index=False)
    ]

    ranked = stable.sort_values(["title_risk_score", "book_key"], ascending=[False, True])
    columns = [
        "book_key",
        "original_title",
        "authors",
        "strict_title_key",
        "early_title_collision_count",
        "strict_title_collision_count",
        "strict_collision_increase",
        "original_title_length",
        "short_title",
        "listed_author_count",
        "author_ambiguity",
        "title_merging_points",
        "short_title_points",
        "author_ambiguity_points",
        "title_risk_score",
        "monitoring_set_appearances",
        "in_initial_volume",
        "in_relative_demand",
        "in_exposure_aware_quality",
        "strict_collision_examples",
    ]
    top_five = ranked.loc[:, columns].head(5)

    summary = {
        "stable_candidate_count": int(len(stable)),
        "appeared_in_all_three_count": int(stable["monitoring_set_appearances"].eq(3).sum()),
        "appeared_in_exactly_two_count": int(stable["monitoring_set_appearances"].eq(2).sum()),
        "strict_collision_increase_count": int(stable["strict_collision_increase"].sum()),
        "short_title_count": int(stable["short_title"].sum()),
        "author_ambiguity_count": int(stable["author_ambiguity"].sum()),
        "nonzero_title_risk_count": int(stable["title_risk_score"].gt(0).sum()),
        "median_original_title_length": float(median_title_length),
        "monitoring_sets": monitoring_sets,
        "stable_book_keys": sorted(int(book_key) for book_key in stable["book_key"]),
        "top_five_title_risk": [
            round_record(record, {"median_original_title_length"})
            for record in top_five.to_dict(orient="records")
        ],
    }

    ranked.loc[:, columns].to_csv(WORKSPACE / "strict_title_stable_candidate_audit.csv", index=False)
    with (WORKSPACE / "strict_title_stable_candidate_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
