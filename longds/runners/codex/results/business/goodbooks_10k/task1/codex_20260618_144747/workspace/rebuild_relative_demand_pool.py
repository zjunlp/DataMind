from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    value = series if ascending else -series
    return value.rank(method="average", pct=True) * 100.0


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


def movement_rows(rows: pd.DataFrame, early_high_intent_keys: set[int]) -> list[dict[str, object]]:
    columns = [
        "book_key",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity_count",
        "intent_per_cleaned_rating",
        "relative_demand_score",
        "relative_review_score",
        "initial_pool_rank",
        "relative_pool_rank",
        "initial_monitor_rank",
        "relative_monitor_rank",
        "in_early_high_intent_group",
    ]
    output = rows.loc[:, columns].copy()
    output["in_early_high_intent_group"] = output["book_key"].isin(early_high_intent_keys)
    decimal_columns = {
        "intent_per_cleaned_rating",
        "relative_demand_score",
        "relative_review_score",
    }
    return [
        round_record(record, decimal_columns)
        for record in output.to_dict(orient="records")
    ]


def main() -> None:
    clean = pd.read_csv(WORKSPACE / "cleaned_books_analysis.csv")
    initial_pool = pd.read_csv(WORKSPACE / "first_review_pool.csv")

    clean = clean.copy()
    clean["intent_per_cleaned_rating"] = (
        clean["reading_intent_count"] / clean["cleaned_rating_activity_count"]
    ).where(clean["cleaned_rating_activity_count"].ne(0), 0.0)
    clean["relative_demand_score"] = percentile_score(clean["intent_per_cleaned_rating"], ascending=True)
    clean["relative_reader_intent_contribution"] = 0.34 * clean["relative_demand_score"]
    clean["relative_review_score"] = (
        clean["relative_reader_intent_contribution"]
        + 0.24 * clean["exposure_gap_score"]
        + 0.22 * clean["genre_scarcity_score"]
        + 0.20 * clean["quality_score"]
    )

    active = clean.loc[
        clean["language_known"] & clean["publication_year_known"] & clean["main_genre"].notna()
    ].copy()
    relative_demand_threshold = active["relative_demand_score"].quantile(0.80)
    rating_count_cap = active["ratings_count"].quantile(0.85)
    relative_pool = active.loc[
        active["relative_demand_score"].ge(relative_demand_threshold)
        & active["ratings_count"].le(rating_count_cap)
    ].copy()
    relative_pool = relative_pool.sort_values(["relative_review_score", "book_key"], ascending=[False, True])

    initial_top10_keys = initial_pool.head(10)["book_key"].astype(int).tolist()
    relative_top10_keys = relative_pool.head(10)["book_key"].astype(int).tolist()
    early_high_intent_book_keys = initial_pool.head(25)["book_key"].astype(int).tolist()
    early_high_intent_keys = set(early_high_intent_book_keys)
    initial_pool_rank = {
        book_key: rank for rank, book_key in enumerate(initial_pool["book_key"].astype(int), start=1)
    }
    relative_pool_rank = {
        book_key: rank for rank, book_key in enumerate(relative_pool["book_key"].astype(int), start=1)
    }
    initial_rank = {book_key: rank for rank, book_key in enumerate(initial_top10_keys, start=1)}
    relative_rank = {book_key: rank for rank, book_key in enumerate(relative_top10_keys, start=1)}

    comparison = clean.copy()
    comparison["initial_pool_rank"] = comparison["book_key"].map(initial_pool_rank)
    comparison["relative_pool_rank"] = comparison["book_key"].map(relative_pool_rank)
    comparison["initial_monitor_rank"] = comparison["book_key"].map(initial_rank)
    comparison["relative_monitor_rank"] = comparison["book_key"].map(relative_rank)
    comparison["in_early_high_intent_group"] = comparison["book_key"].isin(early_high_intent_keys)

    entered_keys = [book_key for book_key in relative_top10_keys if book_key not in initial_rank]
    left_keys = [book_key for book_key in initial_top10_keys if book_key not in relative_rank]
    entered = comparison.loc[comparison["book_key"].isin(entered_keys)].copy()
    left = comparison.loc[comparison["book_key"].isin(left_keys)].copy()
    entered["_order"] = entered["book_key"].map({book_key: idx for idx, book_key in enumerate(entered_keys)})
    left["_order"] = left["book_key"].map({book_key: idx for idx, book_key in enumerate(left_keys)})
    entered = entered.sort_values("_order")
    left = left.sort_values("_order")

    top_columns = [
        "book_key",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity_count",
        "intent_per_cleaned_rating",
        "relative_demand_score",
        "relative_reader_intent_contribution",
        "exposure_gap_score",
        "exposure_gap_contribution",
        "genre_scarcity_score",
        "genre_scarcity_contribution",
        "quality_score",
        "quality_contribution",
        "relative_review_score",
    ]
    top_five = relative_pool.loc[:, top_columns].head(5).copy()
    decimal_columns = {
        "intent_per_cleaned_rating",
        "relative_demand_score",
        "relative_reader_intent_contribution",
        "exposure_gap_score",
        "exposure_gap_contribution",
        "genre_scarcity_score",
        "genre_scarcity_contribution",
        "quality_score",
        "quality_contribution",
        "relative_review_score",
    }
    top_five_records = [
        round_record(record, decimal_columns)
        for record in top_five.to_dict(orient="records")
    ]

    entered_records = movement_rows(entered, early_high_intent_keys)
    left_records = movement_rows(left, early_high_intent_keys)
    movement_from_early = sum(record["in_early_high_intent_group"] for record in entered_records)
    movement_total = len(entered_records)
    mainly_from_early = movement_from_early > movement_total / 2 if movement_total else False

    thresholds = {
        "active_book_count": int(len(active)),
        "relative_pool_book_count": int(len(relative_pool)),
        "active_relative_demand_score_80th_percentile": float(relative_demand_threshold),
        "active_ratings_count_85th_percentile": float(rating_count_cap),
    }
    summary = {
        "thresholds": thresholds,
        "initial_top10_book_keys": initial_top10_keys,
        "relative_top10_book_keys": relative_top10_keys,
        "early_high_intent_book_keys": early_high_intent_book_keys,
        "top_five": top_five_records,
        "entered_monitoring_set": entered_records,
        "left_monitoring_set": left_records,
        "movement_assessment": {
            "entered_count": movement_total,
            "entered_from_early_high_intent_count": int(movement_from_early),
            "mainly_from_early_high_intent_group": bool(mainly_from_early),
        },
    }

    clean.to_csv(WORKSPACE / "cleaned_books_relative_demand.csv", index=False)
    clean.to_pickle(WORKSPACE / "cleaned_books_relative_demand.pkl")
    relative_pool.to_csv(WORKSPACE / "relative_demand_review_pool.csv", index=False)
    pd.DataFrame(top_five_records).to_csv(WORKSPACE / "relative_demand_top5.csv", index=False)
    with (WORKSPACE / "relative_demand_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
