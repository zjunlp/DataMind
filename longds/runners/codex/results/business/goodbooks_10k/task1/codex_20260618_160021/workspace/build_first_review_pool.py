from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "goodbooks-10k"
WORKSPACE_DIR = BASE_DIR / "workspace"

STAR_COLUMNS = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
BOOK_REQUIRED_COLUMNS = [
    "original_title",
    "authors",
    "average_rating",
    "ratings_count",
    *STAR_COLUMNS,
]

GENRE_TAG_TO_CANONICAL = {
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
    return " ".join(value.split())


def make_early_title_key(value: str) -> str:
    return collapse_spaces(value.lower())


def make_strict_title_key(value: str) -> str:
    without_parenthetical = re.sub(r"\([^)]*\)", " ", value)
    without_articles = re.sub(r"\b(?:a|an|the)\b", " ", without_parenthetical, flags=re.IGNORECASE)
    alnum_spaced = re.sub(r"[^0-9A-Za-z]+", " ", without_articles)
    return collapse_spaces(alnum_spaced.lower())


def first_author(authors: str) -> str:
    return authors.split(",", maxsplit=1)[0].strip()


def percentile(series: pd.Series, *, higher_is_stronger: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=higher_is_stronger) * 100


def main() -> None:
    books = pd.read_csv(DATA_DIR / "books.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    to_read = pd.read_csv(DATA_DIR / "to_read.csv")
    book_tags = pd.read_csv(DATA_DIR / "book_tags.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    raw_original_title_counts = books.loc[books["original_title"].notna(), "original_title"].value_counts()
    duplicate_original_titles = set(raw_original_title_counts[raw_original_title_counts > 1].index)

    required_mask = books[BOOK_REQUIRED_COLUMNS].notna().all(axis=1)
    unique_raw_title_mask = ~books["original_title"].isin(duplicate_original_titles)
    cleaned_books = books.loc[required_mask & unique_raw_title_mask].copy()
    cleaned_books = cleaned_books.rename(columns={"id": "book_key"})

    cleaned_books["early_title_key"] = cleaned_books["original_title"].map(make_early_title_key)
    cleaned_books["strict_title_key"] = cleaned_books["original_title"].map(make_strict_title_key)
    cleaned_books["lead_author"] = cleaned_books["authors"].map(first_author)
    cleaned_books["known_language"] = cleaned_books["language_code"].notna() & (
        cleaned_books["language_code"].astype(str).str.strip() != ""
    )
    cleaned_books["known_publication_year"] = cleaned_books["original_publication_year"].notna()

    pair_row_counts = ratings.groupby(["user_id", "book_id"])["rating"].transform("size")
    ratings_no_repeated_pairs = ratings.loc[pair_row_counts.eq(1)].copy()
    cleaned_book_keys = set(cleaned_books["book_key"])
    cleaned_ratings = ratings_no_repeated_pairs.loc[
        ratings_no_repeated_pairs["book_id"].isin(cleaned_book_keys)
    ].copy()
    cleaned_ratings = cleaned_ratings.rename(columns={"book_id": "book_key"})

    reading_intent_counts = to_read.groupby("book_id").size()
    cleaned_books["reading_intent_count"] = (
        cleaned_books["book_key"].map(reading_intent_counts).fillna(0).astype("int64")
    )

    tags = tags.copy()
    tags["normalized_tag_name"] = tags["tag_name"].astype(str).str.lower().str.replace("_", "-", regex=False)
    recognized_tags = tags.loc[
        tags["normalized_tag_name"].isin(GENRE_TAG_TO_CANONICAL),
        ["tag_id", "tag_name", "normalized_tag_name"],
    ].copy()
    recognized_tags["canonical_genre"] = recognized_tags["normalized_tag_name"].map(GENRE_TAG_TO_CANONICAL)

    book_key_lookup = cleaned_books[["book_key", "book_id"]].rename(
        columns={"book_id": "goodreads_book_id"}
    )
    recognized_book_tags = (
        book_tags.merge(recognized_tags[["tag_id", "canonical_genre"]], on="tag_id", how="inner")
        .merge(book_key_lookup, on="goodreads_book_id", how="inner")
    )

    genre_counts = (
        recognized_book_tags.groupby(["book_key", "canonical_genre"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "main_genre_tag_count"})
    )
    main_genres = (
        genre_counts.sort_values(
            ["book_key", "main_genre_tag_count", "canonical_genre"],
            ascending=[True, False, True],
        )
        .drop_duplicates("book_key", keep="first")
        .rename(columns={"canonical_genre": "main_genre"})
    )
    cleaned_books = cleaned_books.merge(
        main_genres[["book_key", "main_genre", "main_genre_tag_count"]],
        on="book_key",
        how="left",
    )
    genre_book_counts = cleaned_books["main_genre"].value_counts(dropna=True)
    cleaned_books["main_genre_cleaned_book_count"] = cleaned_books["main_genre"].map(genre_book_counts)

    star_weights = pd.Series({f"ratings_{i}": i for i in range(1, 6)})
    cleaned_books["star_rating_total"] = cleaned_books[STAR_COLUMNS].sum(axis=1)
    cleaned_books["star_rating_weighted_mean"] = (
        cleaned_books[STAR_COLUMNS].mul(star_weights).sum(axis=1) / cleaned_books["star_rating_total"]
    )
    cleaned_books["star_average_gap"] = (
        cleaned_books["average_rating"] - cleaned_books["star_rating_weighted_mean"]
    )

    cleaned_books["demand_absolute_score"] = percentile(
        cleaned_books["reading_intent_count"], higher_is_stronger=True
    )
    cleaned_books["exposure_gap_score"] = percentile(
        cleaned_books["ratings_count"], higher_is_stronger=False
    )
    cleaned_books["quality_score"] = percentile(
        cleaned_books["average_rating"], higher_is_stronger=True
    )
    genre_known = cleaned_books["main_genre_cleaned_book_count"].notna()
    cleaned_books.loc[genre_known, "genre_rarity_score"] = percentile(
        cleaned_books.loc[genre_known, "main_genre_cleaned_book_count"],
        higher_is_stronger=False,
    )

    cleaned_books["reader_intent_contribution"] = 0.34 * cleaned_books["demand_absolute_score"]
    cleaned_books["exposure_gap_contribution"] = 0.24 * cleaned_books["exposure_gap_score"]
    cleaned_books["genre_scarcity_contribution"] = 0.22 * cleaned_books["genre_rarity_score"]
    cleaned_books["quality_contribution"] = 0.20 * cleaned_books["quality_score"]
    cleaned_books["first_review_score"] = (
        cleaned_books["reader_intent_contribution"]
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
    demand_80th = active_books["demand_absolute_score"].quantile(0.80)
    ratings_count_85th = active_books["ratings_count"].quantile(0.85)

    first_review_pool = active_books.loc[
        (active_books["demand_absolute_score"] >= demand_80th)
        & (active_books["ratings_count"] <= ratings_count_85th)
    ].copy()
    first_review_pool = first_review_pool.sort_values(
        ["first_review_score", "book_key"], ascending=[False, True]
    )

    top5_columns = [
        "book_key",
        "original_title",
        "title",
        "authors",
        "main_genre",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "demand_absolute_score",
        "reader_intent_contribution",
        "exposure_gap_score",
        "exposure_gap_contribution",
        "genre_rarity_score",
        "genre_scarcity_contribution",
        "quality_score",
        "quality_contribution",
        "first_review_score",
    ]
    top5 = first_review_pool.head(5).loc[:, top5_columns].copy()

    output_paths = {
        "cleaned_books_pickle": WORKSPACE_DIR / "cleaned_books.pkl",
        "cleaned_ratings_pickle": WORKSPACE_DIR / "cleaned_ratings.pkl",
        "genre_counts_csv": WORKSPACE_DIR / "genre_counts_by_book.csv",
        "first_review_pool_pickle": WORKSPACE_DIR / "first_review_pool.pkl",
        "first_review_pool_csv": WORKSPACE_DIR / "first_review_pool.csv",
        "top5_csv": WORKSPACE_DIR / "first_review_pool_top5.csv",
        "summary_json": WORKSPACE_DIR / "first_review_summary.json",
    }

    cleaned_books.to_pickle(output_paths["cleaned_books_pickle"])
    cleaned_ratings.to_pickle(output_paths["cleaned_ratings_pickle"])
    genre_counts.to_csv(output_paths["genre_counts_csv"], index=False)
    first_review_pool.to_pickle(output_paths["first_review_pool_pickle"])
    first_review_pool.to_csv(output_paths["first_review_pool_csv"], index=False)
    top5.to_csv(output_paths["top5_csv"], index=False)

    summary = {
        "raw_book_count": int(len(books)),
        "books_after_required_fields_and_duplicate_title_removal": int(len(cleaned_books)),
        "raw_duplicate_original_title_values": int(len(duplicate_original_titles)),
        "raw_ratings_count": int(len(ratings)),
        "rating_rows_removed_for_repeated_user_book_pairs": int(len(ratings) - len(ratings_no_repeated_pairs)),
        "cleaned_ratings_count": int(len(cleaned_ratings)),
        "recognized_tag_names": int(len(recognized_tags)),
        "cleaned_books_with_recognized_main_genre": int(cleaned_books["main_genre"].notna().sum()),
        "active_book_count": int(len(active_books)),
        "active_demand_absolute_score_80th_percentile": float(demand_80th),
        "active_ratings_count_85th_percentile": float(ratings_count_85th),
        "first_review_pool_count": int(len(first_review_pool)),
        "top5": json.loads(top5.to_json(orient="records")),
    }
    output_paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
