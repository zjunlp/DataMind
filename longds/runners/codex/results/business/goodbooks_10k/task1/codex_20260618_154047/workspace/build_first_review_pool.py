import json
import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/goodbooks-10k")
WORKSPACE = Path("workspace")

BOOKS_PATH = DATA_DIR / "books.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TO_READ_PATH = DATA_DIR / "to_read.csv"
BOOK_TAGS_PATH = DATA_DIR / "book_tags.csv"
TAGS_PATH = DATA_DIR / "tags.csv"

STAR_COLUMNS = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
REQUIRED_BOOK_COLUMNS = [
    "original_title",
    "authors",
    "average_rating",
    "ratings_count",
    *STAR_COLUMNS,
]

GENRE_CANONICAL = {
    "art": "art",
    "biography": "biography",
    "business": "business",
    "chick-lit": "chick_lit",
    "children": "children",
    "childrens": "children",
    "christian": "christian",
    "classics": "classics",
    "comics": "comics",
    "contemporary": "contemporary",
    "cookbooks": "cookbooks",
    "crime": "crime",
    "ebooks": "ebooks",
    "fantasy": "fantasy",
    "fiction": "fiction",
    "historical-fiction": "historical_fiction",
    "history": "history",
    "horror": "horror",
    "humor": "humor",
    "manga": "manga",
    "memoir": "memoir",
    "music": "music",
    "mystery": "mystery",
    "nonfiction": "nonfiction",
    "paranormal": "paranormal",
    "philosophy": "philosophy",
    "poetry": "poetry",
    "romance": "romance",
    "science": "science",
    "science-fiction": "science_fiction",
    "self-help": "self_help",
    "sports": "sports",
    "suspense": "suspense",
    "thriller": "thriller",
    "travel": "travel",
    "young-adult": "young_adult",
}


def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def early_title_key(value: str) -> str:
    return collapse_spaces(str(value).lower())


def strict_title_key(value: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(value))
    text = re.sub(r"\b(?:a|an|the)\b", " ", text, flags=re.IGNORECASE)
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return collapse_spaces(text.lower())


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series if ascending else -series
    return ranked.rank(method="average", pct=True) * 100


def main() -> None:
    WORKSPACE.mkdir(exist_ok=True)

    books_raw = pd.read_csv(BOOKS_PATH)
    books_raw["book_key"] = books_raw["id"]

    missing_required = books_raw[REQUIRED_BOOK_COLUMNS].isna().any(axis=1)
    duplicated_raw_title = books_raw["original_title"].duplicated(keep=False)
    books = books_raw.loc[~missing_required & ~duplicated_raw_title].copy()

    books["early_title_key"] = books["original_title"].map(early_title_key)
    books["strict_title_key"] = books["original_title"].map(strict_title_key)
    books["lead_author"] = books["authors"].astype(str).str.split(",", n=1).str[0].str.strip()
    books["known_language"] = books["language_code"].notna() & (
        books["language_code"].astype(str).str.strip() != ""
    )
    books["known_publication_year"] = books["original_publication_year"].notna()
    books["star_total"] = books[STAR_COLUMNS].sum(axis=1)
    books["star_weighted_sum"] = sum((idx + 1) * books[col] for idx, col in enumerate(STAR_COLUMNS))
    books["star_average_from_counts"] = books["star_weighted_sum"] / books["star_total"]
    books["star_average_gap"] = books["star_average_from_counts"] - books["average_rating"]

    ratings_raw = pd.read_csv(RATINGS_PATH)
    repeated_rating_pair = ratings_raw.duplicated(["user_id", "book_id"], keep=False)
    ratings_dedup = ratings_raw.loc[~repeated_rating_pair].copy()
    cleaned_book_keys = set(books["book_key"])
    cleaned_ratings = ratings_dedup.loc[ratings_dedup["book_id"].isin(cleaned_book_keys)].copy()
    rating_activity = cleaned_ratings.groupby("book_id").agg(
        cleaned_rating_activity_count=("rating", "size"),
        cleaned_rating_activity_mean=("rating", "mean"),
    )
    books = books.merge(
        rating_activity,
        how="left",
        left_on="book_key",
        right_index=True,
    )
    books["cleaned_rating_activity_count"] = (
        books["cleaned_rating_activity_count"].fillna(0).astype("int64")
    )

    to_read = pd.read_csv(TO_READ_PATH)
    reading_intent = to_read.groupby("book_id").size().rename("reading_intent_count")
    books = books.merge(reading_intent, how="left", left_on="book_key", right_index=True)
    books["reading_intent_count"] = books["reading_intent_count"].fillna(0).astype("int64")

    tags = pd.read_csv(TAGS_PATH)
    tags["normalized_tag_name"] = (
        tags["tag_name"].astype(str).str.lower().str.replace("_", "-", regex=False)
    )
    tags["canonical_genre"] = tags["normalized_tag_name"].map(GENRE_CANONICAL)
    genre_tags = tags.loc[tags["canonical_genre"].notna(), ["tag_id", "canonical_genre"]]

    book_map = books[["book_key", "book_id"]].copy()
    book_tags = pd.read_csv(BOOK_TAGS_PATH)
    genre_evidence = (
        book_tags.merge(genre_tags, how="inner", on="tag_id")
        .merge(book_map, how="inner", left_on="goodreads_book_id", right_on="book_id")
    )
    genre_sums = (
        genre_evidence.groupby(["book_key", "canonical_genre"], as_index=False)["count"].sum()
    )
    main_genres = (
        genre_sums.sort_values(
            ["book_key", "count", "canonical_genre"],
            ascending=[True, False, True],
        )
        .drop_duplicates("book_key", keep="first")
        .rename(
            columns={
                "canonical_genre": "main_genre",
                "count": "main_genre_tag_count",
            }
        )
    )
    books = books.merge(
        main_genres[["book_key", "main_genre", "main_genre_tag_count"]],
        how="left",
        on="book_key",
    )

    genre_book_counts = books["main_genre"].value_counts(dropna=True).sort_index()
    books["main_genre_cleaned_book_count"] = books["main_genre"].map(genre_book_counts)

    books["demand_absolute_score"] = percentile_rank(books["reading_intent_count"])
    books["exposure_gap_score"] = percentile_rank(books["ratings_count"], ascending=False)
    books["genre_rarity_score"] = percentile_rank(
        books["main_genre_cleaned_book_count"], ascending=False
    )
    books["quality_score"] = percentile_rank(books["average_rating"])

    books["intent_contribution"] = 0.34 * books["demand_absolute_score"]
    books["exposure_contribution"] = 0.24 * books["exposure_gap_score"]
    books["genre_contribution"] = 0.22 * books["genre_rarity_score"]
    books["quality_contribution"] = 0.20 * books["quality_score"]
    books["first_review_score"] = (
        books["intent_contribution"]
        + books["exposure_contribution"]
        + books["genre_contribution"]
        + books["quality_contribution"]
    )

    active = books.loc[
        books["known_language"] & books["known_publication_year"] & books["main_genre"].notna()
    ].copy()
    demand_80 = active["demand_absolute_score"].quantile(0.80)
    ratings_count_85 = active["ratings_count"].quantile(0.85)
    first_review_pool = active.loc[
        (active["demand_absolute_score"] >= demand_80)
        & (active["ratings_count"] <= ratings_count_85)
    ].copy()
    first_review_pool = first_review_pool.sort_values(
        ["first_review_score", "book_key"], ascending=[False, True]
    )

    top5_columns = [
        "book_key",
        "title",
        "authors",
        "original_publication_year",
        "language_code",
        "main_genre",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "demand_absolute_score",
        "exposure_gap_score",
        "genre_rarity_score",
        "quality_score",
        "intent_contribution",
        "exposure_contribution",
        "genre_contribution",
        "quality_contribution",
        "first_review_score",
    ]
    top5 = first_review_pool.loc[:, top5_columns].head(5).copy()

    # Persist the cleaned analysis state and focused outputs for later turns.
    books.to_csv(WORKSPACE / "cleaned_books_with_scores.csv", index=False)
    cleaned_ratings.to_csv(WORKSPACE / "cleaned_ratings.csv", index=False)
    genre_sums.to_csv(WORKSPACE / "recognized_genre_sums.csv", index=False)
    genre_book_counts.rename("cleaned_book_count").to_csv(WORKSPACE / "genre_book_counts.csv")
    first_review_pool.to_csv(WORKSPACE / "first_review_pool.csv", index=False)
    top5.to_csv(WORKSPACE / "first_review_pool_top5.csv", index=False)

    summary = {
        "raw_books": int(len(books_raw)),
        "books_missing_required": int(missing_required.sum()),
        "books_removed_for_duplicate_raw_original_title": int(
            ((~missing_required) & duplicated_raw_title).sum()
        ),
        "cleaned_books": int(len(books)),
        "raw_rating_rows": int(len(ratings_raw)),
        "rating_rows_removed_for_repeated_user_book_pairs": int(repeated_rating_pair.sum()),
        "cleaned_rating_rows_joined_to_clean_books": int(len(cleaned_ratings)),
        "active_review_books": int(len(active)),
        "active_demand_absolute_score_80th_percentile": float(demand_80),
        "active_ratings_count_85th_percentile": float(ratings_count_85),
        "first_review_pool_books": int(len(first_review_pool)),
    }
    (WORKSPACE / "cleaned_state_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(top5.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
