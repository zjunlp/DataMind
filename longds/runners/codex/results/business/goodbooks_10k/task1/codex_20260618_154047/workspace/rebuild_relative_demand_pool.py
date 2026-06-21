import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path("results/business/goodbooks_10k/task1/codex_20260618_154047/workspace")


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series if ascending else -series
    return ranked.rank(method="average", pct=True) * 100


def main() -> None:
    books = pd.read_csv(WORKSPACE / "cleaned_books_with_scores.csv")
    initial_pool = pd.read_csv(WORKSPACE / "first_review_pool.csv")

    denominator = books["cleaned_rating_activity_count"]
    books["intent_rate"] = (books["reading_intent_count"] / denominator).where(
        denominator.gt(0),
        0.0,
    )
    books["demand_relative_score"] = percentile_rank(books["intent_rate"])
    books["relative_intent_contribution"] = 0.34 * books["demand_relative_score"]
    books["relative_first_review_score"] = (
        books["relative_intent_contribution"]
        + books["exposure_contribution"]
        + books["genre_contribution"]
        + books["quality_contribution"]
    )

    active = books.loc[
        books["known_language"] & books["known_publication_year"] & books["main_genre"].notna()
    ].copy()
    relative_demand_80 = active["demand_relative_score"].quantile(0.80)
    ratings_count_85 = active["ratings_count"].quantile(0.85)
    relative_pool = active.loc[
        (active["demand_relative_score"] >= relative_demand_80)
        & (active["ratings_count"] <= ratings_count_85)
    ].copy()
    relative_pool = relative_pool.sort_values(
        ["relative_first_review_score", "book_key"], ascending=[False, True]
    )

    old_top10 = initial_pool.sort_values(
        ["first_review_score", "book_key"], ascending=[False, True]
    ).head(10)
    new_top10 = relative_pool.head(10)
    early_high_intent = set(
        initial_pool.sort_values(["first_review_score", "book_key"], ascending=[False, True])
        .head(25)["book_key"]
        .astype(int)
    )

    old_top10_keys = set(old_top10["book_key"].astype(int))
    new_top10_keys = set(new_top10["book_key"].astype(int))
    entering_keys = new_top10_keys - old_top10_keys
    leaving_keys = old_top10_keys - new_top10_keys

    movement_keys = entering_keys | leaving_keys
    rank_map_old = {
        int(row.book_key): rank
        for rank, row in enumerate(old_top10.itertuples(index=False), start=1)
    }
    rank_map_new = {
        int(row.book_key): rank
        for rank, row in enumerate(new_top10.itertuples(index=False), start=1)
    }
    movement_base = books.loc[
        books["book_key"].astype(int).isin(movement_keys),
    ].copy()
    movement_base["old_rank"] = movement_base["book_key"].astype(int).map(rank_map_old)
    movement_base["new_rank"] = movement_base["book_key"].astype(int).map(rank_map_new)
    entering = movement_base.loc[
        movement_base["book_key"].astype(int).isin(entering_keys)
    ].sort_values(["new_rank", "book_key"])
    leaving = movement_base.loc[
        movement_base["book_key"].astype(int).isin(leaving_keys)
    ].sort_values(["old_rank", "book_key"])
    entering["in_early_high_intent_group"] = entering["book_key"].astype(int).isin(
        early_high_intent
    )
    leaving["in_early_high_intent_group"] = leaving["book_key"].astype(int).isin(
        early_high_intent
    )

    top5_columns = [
        "book_key",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "cleaned_rating_activity_count",
        "intent_rate",
        "ratings_count",
        "average_rating",
        "demand_relative_score",
        "exposure_gap_score",
        "genre_rarity_score",
        "quality_score",
        "relative_intent_contribution",
        "exposure_contribution",
        "genre_contribution",
        "quality_contribution",
        "relative_first_review_score",
    ]
    movement_columns = [
        "book_key",
        "title",
        "main_genre",
        "old_rank",
        "new_rank",
        "reading_intent_count",
        "cleaned_rating_activity_count",
        "intent_rate",
        "relative_first_review_score",
        "in_early_high_intent_group",
    ]

    books.to_csv(WORKSPACE / "cleaned_books_with_relative_scores.csv", index=False)
    relative_pool.to_csv(WORKSPACE / "relative_review_pool.csv", index=False)
    relative_pool.loc[:, top5_columns].head(5).to_csv(
        WORKSPACE / "relative_review_pool_top5.csv", index=False
    )
    entering.loc[:, movement_columns].to_csv(
        WORKSPACE / "relative_monitoring_entries.csv", index=False
    )
    leaving.loc[:, movement_columns].to_csv(
        WORKSPACE / "relative_monitoring_leavers.csv", index=False
    )

    movement_in_early = sum(int(key in early_high_intent) for key in movement_keys)
    summary = {
        "cleaned_books": int(len(books)),
        "active_review_books": int(len(active)),
        "relative_demand_score_80th_percentile": float(relative_demand_80),
        "active_ratings_count_85th_percentile": float(ratings_count_85),
        "relative_review_pool_books": int(len(relative_pool)),
        "initial_monitoring_set_size": int(len(old_top10)),
        "relative_monitoring_set_size": int(len(new_top10)),
        "entering_count": int(len(entering_keys)),
        "leaving_count": int(len(leaving_keys)),
        "movement_unique_books": int(len(movement_keys)),
        "movement_unique_books_in_early_high_intent_group": int(movement_in_early),
        "movement_mainly_from_early_high_intent_group": bool(
            movement_in_early > len(movement_keys) / 2
        ),
        "entering_in_early_high_intent_group": int(
            entering["in_early_high_intent_group"].sum()
        ),
        "leaving_in_early_high_intent_group": int(
            leaving["in_early_high_intent_group"].sum()
        ),
    }
    (WORKSPACE / "relative_review_pool_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("TOP5")
    print(relative_pool.loc[:, top5_columns].head(5).to_json(orient="records", indent=2))
    print("ENTERING")
    print(entering.loc[:, movement_columns].to_json(orient="records", indent=2))
    print("LEAVING")
    print(leaving.loc[:, movement_columns].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
