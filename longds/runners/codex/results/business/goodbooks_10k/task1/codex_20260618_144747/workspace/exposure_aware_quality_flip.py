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


def main() -> None:
    clean = pd.read_csv(WORKSPACE / "cleaned_books_analysis.csv")
    initial_pool = pd.read_csv(WORKSPACE / "first_review_pool.csv")

    exposure_floor = clean["ratings_count"].quantile(0.60)
    overall_weighted_mean = (
        clean["average_rating"].mul(clean["ratings_count"]).sum() / clean["ratings_count"].sum()
    )

    clean = clean.copy()
    clean["exposure_aware_quality_value"] = (
        clean["average_rating"].mul(clean["ratings_count"])
        + overall_weighted_mean * exposure_floor
    ) / (clean["ratings_count"] + exposure_floor)
    clean["exposure_aware_quality_score"] = percentile_score(clean["exposure_aware_quality_value"])
    clean["exposure_aware_quality_contribution"] = 0.20 * clean["exposure_aware_quality_score"]
    clean["exposure_aware_review_score"] = (
        0.34 * clean["demand_absolute_score"]
        + 0.24 * clean["exposure_gap_score"]
        + 0.22 * clean["genre_scarcity_score"]
        + clean["exposure_aware_quality_contribution"]
    )

    clean["demand_percentile_gap"] = 0.0
    clean["quality_percentile_gap"] = clean["exposure_aware_quality_score"] - clean["quality_score"]
    clean["flip_strength"] = (
        clean["demand_percentile_gap"] + clean["quality_percentile_gap"]
    ).abs()

    active = clean.loc[
        clean["language_known"] & clean["publication_year_known"] & clean["main_genre"].notna()
    ].copy()
    demand_threshold = active["demand_absolute_score"].quantile(0.80)
    rating_count_cap = active["ratings_count"].quantile(0.85)
    exposure_aware_pool = active.loc[
        active["demand_absolute_score"].ge(demand_threshold)
        & active["ratings_count"].le(rating_count_cap)
    ].copy()
    exposure_aware_pool = exposure_aware_pool.sort_values(
        ["exposure_aware_review_score", "book_key"],
        ascending=[False, True],
    )

    initial_top10_keys = initial_pool.head(10)["book_key"].astype(int).tolist()
    exposure_aware_top10_keys = exposure_aware_pool.head(10)["book_key"].astype(int).tolist()
    initial_top10_set = set(initial_top10_keys)
    exposure_aware_top10_set = set(exposure_aware_top10_keys)
    entered_keys = [book_key for book_key in exposure_aware_top10_keys if book_key not in initial_top10_set]
    left_keys = [book_key for book_key in initial_top10_keys if book_key not in exposure_aware_top10_set]
    changed_keys = set(entered_keys) | set(left_keys)

    initial_pool_rank = {
        book_key: rank for rank, book_key in enumerate(initial_pool["book_key"].astype(int), start=1)
    }
    exposure_aware_pool_rank = {
        book_key: rank
        for rank, book_key in enumerate(exposure_aware_pool["book_key"].astype(int), start=1)
    }
    initial_monitor_rank = {book_key: rank for rank, book_key in enumerate(initial_top10_keys, start=1)}
    exposure_aware_monitor_rank = {
        book_key: rank for rank, book_key in enumerate(exposure_aware_top10_keys, start=1)
    }

    changed = clean.loc[clean["book_key"].isin(changed_keys)].copy()
    changed["movement"] = changed["book_key"].map(
        {**{book_key: "entered" for book_key in entered_keys}, **{book_key: "left" for book_key in left_keys}}
    )
    changed["initial_pool_rank"] = changed["book_key"].map(initial_pool_rank)
    changed["exposure_aware_pool_rank"] = changed["book_key"].map(exposure_aware_pool_rank)
    changed["initial_monitor_rank"] = changed["book_key"].map(initial_monitor_rank)
    changed["exposure_aware_monitor_rank"] = changed["book_key"].map(exposure_aware_monitor_rank)
    changed["raw_quality_contribution"] = 0.20 * changed["quality_score"]
    changed["score_change"] = changed["exposure_aware_review_score"] - changed["first_review_score"]

    movement_columns = [
        "movement",
        "book_key",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "exposure_aware_quality_value",
        "demand_absolute_score",
        "demand_percentile_gap",
        "quality_score",
        "exposure_aware_quality_score",
        "quality_percentile_gap",
        "flip_strength",
        "reader_intent_contribution",
        "exposure_gap_contribution",
        "genre_scarcity_contribution",
        "raw_quality_contribution",
        "exposure_aware_quality_contribution",
        "first_review_score",
        "exposure_aware_review_score",
        "score_change",
        "initial_pool_rank",
        "exposure_aware_pool_rank",
        "initial_monitor_rank",
        "exposure_aware_monitor_rank",
    ]
    strongest = changed.sort_values(["flip_strength", "book_key"], ascending=[False, True]).head(5)

    decimal_columns = {
        "average_rating",
        "exposure_aware_quality_value",
        "demand_absolute_score",
        "demand_percentile_gap",
        "quality_score",
        "exposure_aware_quality_score",
        "quality_percentile_gap",
        "flip_strength",
        "reader_intent_contribution",
        "exposure_gap_contribution",
        "genre_scarcity_contribution",
        "raw_quality_contribution",
        "exposure_aware_quality_contribution",
        "first_review_score",
        "exposure_aware_review_score",
        "score_change",
    }
    strongest_records = [
        round_record(record, decimal_columns)
        for record in strongest.loc[:, movement_columns].to_dict(orient="records")
    ]

    top_columns = [
        "book_key",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "exposure_aware_quality_value",
        "demand_absolute_score",
        "exposure_gap_score",
        "genre_scarcity_score",
        "exposure_aware_quality_score",
        "exposure_aware_review_score",
    ]
    top10 = exposure_aware_pool.head(10).loc[:, top_columns].copy()
    top10["exposure_aware_rank"] = range(1, len(top10) + 1)
    top10_records = [
        round_record(record, decimal_columns | {"exposure_gap_score", "genre_scarcity_score"})
        for record in top10.to_dict(orient="records")
    ]

    summary = {
        "exposure_floor_60th_percentile_ratings_count": float(exposure_floor),
        "overall_weighted_average_rating": float(overall_weighted_mean),
        "active_book_count": int(len(active)),
        "exposure_aware_pool_book_count": int(len(exposure_aware_pool)),
        "demand_absolute_score_80th_percentile": float(demand_threshold),
        "ratings_count_85th_percentile": float(rating_count_cap),
        "initial_top10_book_keys": initial_top10_keys,
        "exposure_aware_top10_book_keys": exposure_aware_top10_keys,
        "entered_book_keys": entered_keys,
        "left_book_keys": left_keys,
        "strongest_membership_flips": strongest_records,
        "exposure_aware_top10": top10_records,
    }

    clean.to_csv(WORKSPACE / "cleaned_books_exposure_aware_quality.csv", index=False)
    clean.to_pickle(WORKSPACE / "cleaned_books_exposure_aware_quality.pkl")
    exposure_aware_pool.to_csv(WORKSPACE / "exposure_aware_quality_review_pool.csv", index=False)
    pd.DataFrame(strongest_records).to_csv(WORKSPACE / "exposure_aware_quality_flips.csv", index=False)
    with (WORKSPACE / "exposure_aware_quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
