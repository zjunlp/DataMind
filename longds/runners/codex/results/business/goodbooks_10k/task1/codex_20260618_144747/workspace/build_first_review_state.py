from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "goodbooks-10k"
WORKSPACE = ROOT / "workspace"

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

GENRE_MAP = {
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


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    value = series if ascending else -series
    return value.rank(method="average", pct=True) * 100.0


def early_title_key(value: object) -> str:
    text = str(value).lower()
    return re.sub(r"\s+", " ", text).strip()


def strict_title_key(value: object) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(value))
    text = text.lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def known_text(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(BOOKS_PATH),
        pd.read_csv(RATINGS_PATH),
        pd.read_csv(TO_READ_PATH),
        pd.read_csv(BOOK_TAGS_PATH),
        pd.read_csv(TAGS_PATH),
    )


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    required_mask = books[REQUIRED_BOOK_COLUMNS].notna().all(axis=1)
    clean = books.loc[required_mask].copy()
    duplicate_raw_title = clean["original_title"].duplicated(keep=False)
    clean = clean.loc[~duplicate_raw_title].copy()

    clean["book_key"] = clean["id"].astype(int)
    clean["lead_author"] = clean["authors"].astype(str).str.split(",", n=1).str[0].str.strip()
    clean["language_known"] = known_text(clean["language_code"])
    clean["publication_year_known"] = clean["original_publication_year"].notna()
    clean["early_title_key"] = clean["original_title"].map(early_title_key)
    clean["strict_title_key"] = clean["original_title"].map(strict_title_key)

    star_total = clean[STAR_COLUMNS].sum(axis=1)
    star_weighted_sum = sum((idx + 1) * clean[column] for idx, column in enumerate(STAR_COLUMNS))
    clean["star_ratings_total"] = star_total
    clean["star_weighted_average"] = star_weighted_sum / star_total
    clean["average_rating_star_gap"] = clean["average_rating"] - clean["star_weighted_average"]

    return clean


def clean_ratings(ratings: pd.DataFrame, cleaned_book_keys: set[int]) -> pd.DataFrame:
    pair_size = ratings.groupby(["user_id", "book_id"], sort=False)["rating"].transform("size")
    ratings_clean = ratings.loc[pair_size.eq(1)].copy()
    ratings_clean = ratings_clean.loc[ratings_clean["book_id"].isin(cleaned_book_keys)].copy()
    return ratings_clean


def add_rating_activity(clean: pd.DataFrame, ratings_clean: pd.DataFrame) -> pd.DataFrame:
    activity = (
        ratings_clean.groupby("book_id", as_index=False)
        .agg(
            cleaned_rating_activity_count=("rating", "size"),
            cleaned_rating_activity_mean=("rating", "mean"),
        )
        .rename(columns={"book_id": "book_key"})
    )
    clean = clean.merge(activity, on="book_key", how="left")
    clean["cleaned_rating_activity_count"] = clean["cleaned_rating_activity_count"].fillna(0).astype(int)
    return clean


def add_reading_intent(clean: pd.DataFrame, to_read: pd.DataFrame) -> pd.DataFrame:
    intent = to_read.groupby("book_id", as_index=False).size().rename(
        columns={"book_id": "book_key", "size": "reading_intent_count"}
    )
    clean = clean.merge(intent, on="book_key", how="left")
    clean["reading_intent_count"] = clean["reading_intent_count"].fillna(0).astype(int)
    return clean


def build_genres(
    clean: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tag_lookup = tags.copy()
    tag_lookup["normalized_tag_name"] = (
        tag_lookup["tag_name"].astype("string").str.lower().str.replace("_", "-", regex=False)
    )
    tag_lookup["canonical_genre"] = tag_lookup["normalized_tag_name"].map(GENRE_MAP)
    recognized_tags = tag_lookup.loc[tag_lookup["canonical_genre"].notna()].copy()

    key_map = clean[["book_key", "book_id"]].rename(columns={"book_id": "goodreads_book_id"})
    recognized_rows = (
        book_tags.merge(
            recognized_tags[["tag_id", "tag_name", "normalized_tag_name", "canonical_genre"]],
            on="tag_id",
            how="inner",
        )
        .merge(key_map, on="goodreads_book_id", how="inner")
        .loc[
            :,
            [
                "book_key",
                "goodreads_book_id",
                "tag_id",
                "tag_name",
                "normalized_tag_name",
                "canonical_genre",
                "count",
            ],
        ]
    )

    genre_sums = (
        recognized_rows.groupby(["book_key", "canonical_genre"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "genre_tag_count"})
    )
    main_genres = genre_sums.sort_values(
        ["book_key", "genre_tag_count", "canonical_genre"],
        ascending=[True, False, True],
    )
    main_genres = main_genres.drop_duplicates("book_key", keep="first").rename(
        columns={
            "canonical_genre": "main_genre",
            "genre_tag_count": "main_genre_tag_count",
        }
    )

    clean = clean.merge(main_genres, on="book_key", how="left")
    main_genre_counts = clean["main_genre"].value_counts(dropna=True).rename("main_genre_book_count")
    clean["main_genre_book_count"] = clean["main_genre"].map(main_genre_counts)
    return clean, genre_sums, recognized_rows


def add_component_scores(clean: pd.DataFrame) -> pd.DataFrame:
    clean["demand_absolute_score"] = percentile_score(clean["reading_intent_count"], ascending=True)
    clean["exposure_gap_score"] = percentile_score(clean["ratings_count"], ascending=False)
    clean["genre_scarcity_score"] = percentile_score(clean["main_genre_book_count"], ascending=False)
    clean["quality_score"] = percentile_score(clean["average_rating"], ascending=True)
    clean["first_review_score"] = (
        0.34 * clean["demand_absolute_score"]
        + 0.24 * clean["exposure_gap_score"]
        + 0.22 * clean["genre_scarcity_score"]
        + 0.20 * clean["quality_score"]
    )
    clean["reader_intent_contribution"] = 0.34 * clean["demand_absolute_score"]
    clean["exposure_gap_contribution"] = 0.24 * clean["exposure_gap_score"]
    clean["genre_scarcity_contribution"] = 0.22 * clean["genre_scarcity_score"]
    clean["quality_contribution"] = 0.20 * clean["quality_score"]
    return clean


def build_review_pool(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    active = clean.loc[
        clean["language_known"] & clean["publication_year_known"] & clean["main_genre"].notna()
    ].copy()
    demand_threshold = active["demand_absolute_score"].quantile(0.80)
    ratings_count_cap = active["ratings_count"].quantile(0.85)
    pool = active.loc[
        active["demand_absolute_score"].ge(demand_threshold)
        & active["ratings_count"].le(ratings_count_cap)
    ].copy()
    pool = pool.sort_values(["first_review_score", "book_key"], ascending=[False, True])

    thresholds = {
        "active_book_count": int(len(active)),
        "pool_book_count": int(len(pool)),
        "active_demand_absolute_score_80th_percentile": float(demand_threshold),
        "active_ratings_count_85th_percentile": float(ratings_count_cap),
    }
    return pool, thresholds


def display_top_five(pool: pd.DataFrame) -> list[dict[str, object]]:
    columns = [
        "book_key",
        "title",
        "original_title",
        "authors",
        "lead_author",
        "language_code",
        "original_publication_year",
        "main_genre",
        "reading_intent_count",
        "ratings_count",
        "average_rating",
        "demand_absolute_score",
        "reader_intent_contribution",
        "exposure_gap_score",
        "exposure_gap_contribution",
        "genre_scarcity_score",
        "genre_scarcity_contribution",
        "quality_score",
        "quality_contribution",
        "first_review_score",
    ]
    top = pool.loc[:, columns].head(5).copy()
    decimal_columns = [
        "original_publication_year",
        "average_rating",
        "demand_absolute_score",
        "reader_intent_contribution",
        "exposure_gap_score",
        "exposure_gap_contribution",
        "genre_scarcity_score",
        "genre_scarcity_contribution",
        "quality_score",
        "quality_contribution",
        "first_review_score",
    ]
    for column in decimal_columns:
        top[column] = top[column].round(3)
    return top.to_dict(orient="records")


def write_outputs(
    clean: pd.DataFrame,
    ratings_clean: pd.DataFrame,
    genre_sums: pd.DataFrame,
    recognized_rows: pd.DataFrame,
    pool: pd.DataFrame,
    thresholds: dict[str, float | int],
    top_five: list[dict[str, object]],
    source_counts: dict[str, int],
) -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    clean.to_csv(WORKSPACE / "cleaned_books_analysis.csv", index=False)
    clean.to_pickle(WORKSPACE / "cleaned_books_analysis.pkl")
    ratings_clean.to_csv(WORKSPACE / "cleaned_ratings.csv", index=False)
    ratings_clean.to_pickle(WORKSPACE / "cleaned_ratings.pkl")
    genre_sums.to_csv(WORKSPACE / "genre_sums_by_book.csv", index=False)
    recognized_rows.to_csv(WORKSPACE / "recognized_genre_tag_rows.csv", index=False)
    pool.to_csv(WORKSPACE / "first_review_pool.csv", index=False)
    pd.DataFrame(top_five).to_csv(WORKSPACE / "first_review_top5.csv", index=False)

    manifest = {
        "source_counts": source_counts,
        "cleaned_book_count": int(len(clean)),
        "cleaned_rating_count": int(len(ratings_clean)),
        "recognized_genre_book_count": int(clean["main_genre"].notna().sum()),
        "thresholds": thresholds,
        "top_five": top_five,
        "score_formula": {
            "first_review_score": (
                "0.34*demand_absolute_score + 0.24*exposure_gap_score "
                "+ 0.22*genre_scarcity_score + 0.20*quality_score"
            ),
            "demand_absolute_score": "rank(pct=True)*100 of reading_intent_count on cleaned books",
            "exposure_gap_score": "rank(pct=True)*100 of negative ratings_count on cleaned books",
            "genre_scarcity_score": "rank(pct=True)*100 of negative main_genre_book_count on cleaned books with recognized main genres",
            "quality_score": "rank(pct=True)*100 of average_rating on cleaned books",
        },
        "paths": {
            "cleaned_books": "workspace/cleaned_books_analysis.csv",
            "cleaned_ratings": "workspace/cleaned_ratings.csv",
            "genre_sums": "workspace/genre_sums_by_book.csv",
            "recognized_genre_tag_rows": "workspace/recognized_genre_tag_rows.csv",
            "first_review_pool": "workspace/first_review_pool.csv",
            "first_review_top5": "workspace/first_review_top5.csv",
        },
    }
    with (WORKSPACE / "state_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def main() -> None:
    books, ratings, to_read, book_tags, tags = load_sources()
    source_counts = {
        "books": int(len(books)),
        "ratings": int(len(ratings)),
        "to_read": int(len(to_read)),
        "book_tags": int(len(book_tags)),
        "tags": int(len(tags)),
    }

    clean = clean_books(books)
    cleaned_book_keys = set(clean["book_key"])
    ratings_clean = clean_ratings(ratings, cleaned_book_keys)
    clean = add_rating_activity(clean, ratings_clean)
    clean = add_reading_intent(clean, to_read)
    clean, genre_sums, recognized_rows = build_genres(clean, book_tags, tags)
    clean = add_component_scores(clean)
    pool, thresholds = build_review_pool(clean)
    top_five = display_top_five(pool)
    write_outputs(clean, ratings_clean, genre_sums, recognized_rows, pool, thresholds, top_five, source_counts)

    print(json.dumps({"thresholds": thresholds, "top_five": top_five}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
