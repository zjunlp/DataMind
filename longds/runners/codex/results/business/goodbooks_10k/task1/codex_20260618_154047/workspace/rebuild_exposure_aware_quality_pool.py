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

    exposure_floor = books["ratings_count"].quantile(0.60)
    overall_weighted_rating = (
        books["average_rating"].mul(books["ratings_count"]).sum()
        / books["ratings_count"].sum()
    )
    own_weight = books["ratings_count"] / (books["ratings_count"] + exposure_floor)
    books["exposure_aware_quality_value"] = (
        books["average_rating"].mul(own_weight)
        + overall_weighted_rating * (1.0 - own_weight)
    )
    books["exposure_aware_quality_score"] = percentile_rank(
        books["exposure_aware_quality_value"]
    )
    books["exposure_aware_quality_contribution"] = (
        0.20 * books["exposure_aware_quality_score"]
    )
    books["exposure_aware_first_review_score"] = (
        books["intent_contribution"]
        + books["exposure_contribution"]
        + books["genre_contribution"]
        + books["exposure_aware_quality_contribution"]
    )

    active = books.loc[
        books["known_language"] & books["known_publication_year"] & books["main_genre"].notna()
    ].copy()
    demand_80 = active["demand_absolute_score"].quantile(0.80)
    ratings_count_85 = active["ratings_count"].quantile(0.85)
    exposure_aware_pool = active.loc[
        (active["demand_absolute_score"] >= demand_80)
        & (active["ratings_count"] <= ratings_count_85)
    ].copy()
    exposure_aware_pool = exposure_aware_pool.sort_values(
        ["exposure_aware_first_review_score", "book_key"], ascending=[False, True]
    )

    old_top10 = initial_pool.sort_values(
        ["first_review_score", "book_key"], ascending=[False, True]
    ).head(10)
    old_full_rank = {
        int(row.book_key): rank
        for rank, row in enumerate(
            initial_pool.sort_values(
                ["first_review_score", "book_key"], ascending=[False, True]
            ).itertuples(index=False),
            start=1,
        )
    }
    new_top10 = exposure_aware_pool.head(10)
    new_full_rank = {
        int(row.book_key): rank
        for rank, row in enumerate(exposure_aware_pool.itertuples(index=False), start=1)
    }

    old_rank = {
        int(row.book_key): rank
        for rank, row in enumerate(old_top10.itertuples(index=False), start=1)
    }
    new_rank = {
        int(row.book_key): rank
        for rank, row in enumerate(new_top10.itertuples(index=False), start=1)
    }
    old_keys = set(old_rank)
    new_keys = set(new_rank)
    changed_keys = old_keys ^ new_keys

    movement = books.loc[books["book_key"].astype(int).isin(changed_keys)].copy()
    movement["movement"] = movement["book_key"].astype(int).map(
        lambda key: "entering" if key in new_keys else "leaving"
    )
    movement["old_rank"] = movement["book_key"].astype(int).map(old_rank)
    movement["new_rank"] = movement["book_key"].astype(int).map(new_rank)
    movement["old_full_pool_rank"] = movement["book_key"].astype(int).map(old_full_rank)
    movement["new_full_pool_rank"] = movement["book_key"].astype(int).map(new_full_rank)
    movement["demand_percentile_gap"] = 0.0
    movement["quality_percentile_gap"] = (
        movement["exposure_aware_quality_score"] - movement["quality_score"]
    )
    movement["flip_strength"] = (
        movement["demand_percentile_gap"] + movement["quality_percentile_gap"]
    ).abs()
    movement["score_delta"] = (
        movement["exposure_aware_first_review_score"] - movement["first_review_score"]
    )

    movement = movement.sort_values(
        ["flip_strength", "book_key"], ascending=[False, True]
    )

    top5_columns = [
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
        "genre_rarity_score",
        "quality_score",
        "exposure_aware_quality_score",
        "intent_contribution",
        "exposure_contribution",
        "genre_contribution",
        "quality_contribution",
        "exposure_aware_quality_contribution",
        "first_review_score",
        "exposure_aware_first_review_score",
    ]
    movement_columns = [
        "movement",
        "book_key",
        "title",
        "main_genre",
        "old_rank",
        "new_rank",
        "old_full_pool_rank",
        "new_full_pool_rank",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "exposure_aware_quality_value",
        "demand_absolute_score",
        "quality_score",
        "exposure_aware_quality_score",
        "demand_percentile_gap",
        "quality_percentile_gap",
        "flip_strength",
        "intent_contribution",
        "exposure_contribution",
        "genre_contribution",
        "quality_contribution",
        "exposure_aware_quality_contribution",
        "first_review_score",
        "exposure_aware_first_review_score",
        "score_delta",
    ]

    books.to_csv(WORKSPACE / "cleaned_books_with_exposure_aware_quality.csv", index=False)
    exposure_aware_pool.to_csv(WORKSPACE / "exposure_aware_quality_review_pool.csv", index=False)
    exposure_aware_pool.loc[:, top5_columns].head(5).to_csv(
        WORKSPACE / "exposure_aware_quality_review_pool_top5.csv", index=False
    )
    movement.loc[:, movement_columns].to_csv(
        WORKSPACE / "exposure_aware_quality_monitoring_flips.csv", index=False
    )

    summary = {
        "cleaned_books": int(len(books)),
        "active_review_books": int(len(active)),
        "exposure_floor_ratings_count_p60": float(exposure_floor),
        "overall_weighted_average_rating": float(overall_weighted_rating),
        "active_demand_absolute_score_80th_percentile": float(demand_80),
        "active_ratings_count_85th_percentile": float(ratings_count_85),
        "exposure_aware_review_pool_books": int(len(exposure_aware_pool)),
        "old_monitoring_set_size": int(len(old_top10)),
        "new_monitoring_set_size": int(len(new_top10)),
        "changed_membership_books": int(len(changed_keys)),
        "entering_count": int(sum(key in new_keys for key in changed_keys)),
        "leaving_count": int(sum(key in old_keys for key in changed_keys)),
    }
    (WORKSPACE / "exposure_aware_quality_review_pool_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("TOP5_POOL")
    print(exposure_aware_pool.loc[:, top5_columns].head(5).to_json(orient="records", indent=2))
    print("FLIPS")
    print(movement.loc[:, movement_columns].head(5).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
