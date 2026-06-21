from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent


def percentile(series: pd.Series, *, higher_is_stronger: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=higher_is_stronger) * 100


def records_for_display(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    return json.loads(frame.loc[:, columns].to_json(orient="records"))


def main() -> None:
    cleaned_books = pd.read_pickle(WORKSPACE_DIR / "cleaned_books.pkl").copy()
    cleaned_ratings = pd.read_pickle(WORKSPACE_DIR / "cleaned_ratings.pkl")
    initial_pool = pd.read_pickle(WORKSPACE_DIR / "first_review_pool.pkl").copy()

    cleaned_rating_activity = cleaned_ratings.groupby("book_key").size()
    cleaned_books["cleaned_rating_activity"] = (
        cleaned_books["book_key"].map(cleaned_rating_activity).fillna(0).astype("int64")
    )
    cleaned_books["intent_per_cleaned_rating"] = 0.0
    has_activity = cleaned_books["cleaned_rating_activity"].gt(0)
    cleaned_books.loc[has_activity, "intent_per_cleaned_rating"] = (
        cleaned_books.loc[has_activity, "reading_intent_count"]
        / cleaned_books.loc[has_activity, "cleaned_rating_activity"]
    )

    cleaned_books["demand_relative_score"] = percentile(
        cleaned_books["intent_per_cleaned_rating"], higher_is_stronger=True
    )
    cleaned_books["relative_reader_intent_contribution"] = (
        0.34 * cleaned_books["demand_relative_score"]
    )
    cleaned_books["relative_review_score"] = (
        cleaned_books["relative_reader_intent_contribution"]
        + cleaned_books["exposure_gap_contribution"]
        + cleaned_books["genre_scarcity_contribution"]
        + cleaned_books["quality_contribution"]
    )

    active_mask = (
        cleaned_books["known_language"]
        & cleaned_books["known_publication_year"]
        & cleaned_books["main_genre"].notna()
    )
    active_books = cleaned_books.loc[active_mask].copy()
    relative_demand_80th = active_books["demand_relative_score"].quantile(0.80)
    ratings_count_85th = active_books["ratings_count"].quantile(0.85)

    relative_pool = active_books.loc[
        (active_books["demand_relative_score"] >= relative_demand_80th)
        & (active_books["ratings_count"] <= ratings_count_85th)
    ].copy()
    relative_pool = relative_pool.sort_values(
        ["relative_review_score", "book_key"], ascending=[False, True]
    )
    relative_pool["relative_rank"] = range(1, len(relative_pool) + 1)

    initial_monitor = initial_pool.head(10).copy()
    initial_monitor["initial_rank"] = range(1, len(initial_monitor) + 1)
    relative_metric_columns = [
        "book_key",
        "cleaned_rating_activity",
        "intent_per_cleaned_rating",
        "relative_review_score",
    ]
    initial_monitor = initial_monitor.merge(
        cleaned_books.loc[:, relative_metric_columns],
        on="book_key",
        how="left",
    )
    relative_monitor = relative_pool.head(10).copy()

    early_high_intent_keys = set(initial_pool.head(25)["book_key"])
    initial_monitor_keys = set(initial_monitor["book_key"])
    relative_monitor_keys = set(relative_monitor["book_key"])

    entering = relative_monitor.loc[
        ~relative_monitor["book_key"].isin(initial_monitor_keys)
    ].copy()
    entering["was_in_early_high_intent_group"] = entering["book_key"].isin(early_high_intent_keys)

    leaving = initial_monitor.loc[
        ~initial_monitor["book_key"].isin(relative_monitor_keys)
    ].copy()
    leaving["was_in_early_high_intent_group"] = leaving["book_key"].isin(early_high_intent_keys)
    new_rank_lookup = relative_pool.set_index("book_key")["relative_rank"]
    leaving["relative_rank"] = leaving["book_key"].map(new_rank_lookup)

    top5_columns = [
        "relative_rank",
        "book_key",
        "original_title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity",
        "intent_per_cleaned_rating",
        "ratings_count",
        "average_rating",
        "demand_relative_score",
        "relative_reader_intent_contribution",
        "exposure_gap_contribution",
        "genre_scarcity_contribution",
        "quality_contribution",
        "relative_review_score",
    ]
    movement_columns = [
        "book_key",
        "original_title",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity",
        "intent_per_cleaned_rating",
        "relative_review_score",
        "relative_rank",
        "was_in_early_high_intent_group",
    ]
    leaving_columns = [
        "book_key",
        "original_title",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity",
        "intent_per_cleaned_rating",
        "first_review_score",
        "initial_rank",
        "relative_rank",
        "was_in_early_high_intent_group",
    ]

    summary = {
        "active_book_count": int(len(active_books)),
        "relative_demand_80th_percentile": float(relative_demand_80th),
        "active_ratings_count_85th_percentile": float(ratings_count_85th),
        "relative_review_pool_count": int(len(relative_pool)),
        "initial_monitor_count": int(len(initial_monitor)),
        "relative_monitor_count": int(len(relative_monitor)),
        "entering_count": int(len(entering)),
        "leaving_count": int(len(leaving)),
        "entering_from_early_high_intent_count": int(
            entering["was_in_early_high_intent_group"].sum()
        ),
        "leaving_from_early_high_intent_count": int(
            leaving["was_in_early_high_intent_group"].sum()
        ),
        "top5": records_for_display(relative_pool.head(5), top5_columns),
        "entering_monitoring_set": records_for_display(entering, movement_columns),
        "leaving_monitoring_set": records_for_display(leaving, leaving_columns),
    }

    cleaned_books.to_pickle(WORKSPACE_DIR / "cleaned_books_with_relative_demand.pkl")
    relative_pool.to_pickle(WORKSPACE_DIR / "relative_review_pool.pkl")
    relative_pool.to_csv(WORKSPACE_DIR / "relative_review_pool.csv", index=False)
    pd.DataFrame(summary["top5"]).to_csv(
        WORKSPACE_DIR / "relative_review_pool_top5.csv", index=False
    )
    (WORKSPACE_DIR / "relative_review_movement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
