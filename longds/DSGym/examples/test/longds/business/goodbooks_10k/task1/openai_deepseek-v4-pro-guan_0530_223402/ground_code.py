###### Task 1:
# Context: Clean the Goodreads sources into a single persistent working set for all later tasks. Use books.id as the analysis book_key. ratings.book_id and to_read.book_id join to books.id; book_tags.goodreads_book_id joins to books.book_id. Before computing rating activity, if a (user_id, book_id) pair appears more than once in ratings, remove all rating rows for that repeated pair rather than keeping one row. Remove all books sharing an exact raw books.original_title value; do not use normalized title keys for this duplicate-title removal. The cleaned table excludes records missing original_title, authors, average_rating, ratings_count, or star-rating columns. After component percentile ranks are built on the cleaned table, the active review range excludes books without known language, known publication year, or a recognized main genre. Use two title treatments based on original_title. The early key lowercases and collapses spaces without folding accents, while the stricter key later removes parenthetical text, removes standalone articles, replaces non-alphanumeric characters with spaces, lowercases, and collapses spaces. Connect reading-intent counts and genre evidence to the correct book records. Recognize genre evidence only from exact normalized tag names in these families: art, biography, business, chick-lit, children and childrens, christian, classics, comics, contemporary, cookbooks, crime, ebooks, fantasy, fiction, historical-fiction, history, horror, humor, manga, memoir, music, mystery, nonfiction, paranormal, philosophy, poetry, romance, science, science-fiction, self-help, sports, suspense, thriller, travel, and young-adult. Normalize tag names by lowercasing and replacing underscores with hyphens. Map recognized tags to canonical genre families, including children and childrens -> children, historical-fiction -> historical_fiction, science-fiction -> science_fiction, self-help -> self_help, young-adult -> young_adult, and chick-lit -> chick_lit. For each book_key, sum tag counts within each canonical genre, choose the main genre by largest summed count, and break ties alphabetically by canonical genre. Keep analysis-ready fields for title matching, lead author, language, star-rating diagnostics, reading intent, and genre rarity. Build component percentile ranks on the cleaned table before applying the active review filter; percentile ranks use average tie handling and pandas style rank(pct=True) * 100, i.e. average_rank / nonmissing_count * 100. Do not rescale ranks with (rank - 1) / (n - 1). In the early genre-scarcity treatment, rarer recognized main genres receive higher scores. The active review range excludes books without known language, known publication year, or a recognized main genre. The first review score blends 34% reader-intent volume, 24% exposure gap where lower Goodreads rating count is stronger, 22% genre rarity, and 20% average-rating quality. Define demand_absolute_score as the percentile rank of reading-intent count. Keep active-range books whose demand_absolute_score alone reaches the active-range 80th percentile; do not include exposure gap in this filtering threshold. Also require raw Goodreads rating count to be no higher than the active-range 85th percentile. Sort ranked outputs by the named score in the natural direction implied by the question; tied ranked rows and top-N truncation use the smaller book_key first. Restore display titles, full authors, covers, publication years, raw star columns, ISBN fields, and raw tag rows only when an audit needs source evidence. Throughout this analysis, compute derived quantities including sums, gaps, ratios, means, similarities, and correlations using unrounded values, and report decimal-valued final results rounded to 3 decimal places. Counts use the cleaned denominator established in the same step unless the question explicitly asks for a comparison across two states. Absent neighbor evidence contributes zero. This step establishes the persistent cleaned analysis state for all later tasks; later tasks must reuse these book keys, cleaned ratings, genre assignments, title keys, and percentile columns unless explicitly instructed to recompute one component.
# Question: Build the first review pool by combining reader intent, exposure gap, genre scarcity, and quality. Show the five leading books and the component contributions behind their scores.

# Establishes the cleaned analysis state used by later tasks.
DATA_DIR = Path("../../../../../data/longds/business/goodbooks_10k/task1/data")

books_raw = pd.read_csv(DATA_DIR / "goodbooks-10k/books.csv", on_bad_lines="skip")
ratings_raw = pd.read_csv(DATA_DIR / "goodbooks-10k/ratings.csv")
to_read_raw = pd.read_csv(DATA_DIR / "goodbooks-10k/to_read.csv")
book_tags_raw = pd.read_csv(DATA_DIR / "goodbooks-10k/book_tags.csv")
tags_raw = pd.read_csv(DATA_DIR / "goodbooks-10k/tags.csv")
netflix_raw = pd.read_csv(DATA_DIR / "netflix-shows/netflix_titles.csv")

def norm_spaces(x):
    return re.sub(r"\s+", " ", str(x).strip())

def title_basic(x):
    return norm_spaces(x).lower()

def title_compact(x):
    s = title_basic(x)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return norm_spaces(s)

def key_author(x):
    return re.sub(r"[^a-z0-9]+", " ", str(x).split(",")[0].lower()).strip()

def pct_rank(s, ascending=True):
    s = pd.Series(s).astype(float)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    if float(s.max() - s.min()) == 0:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return s.rank(pct=True, ascending=ascending, method="average") * 100

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 3)
    if pd.isna(obj):
        return None
    return obj

def emit(obj):
    print(json.dumps(clean_json(obj), ensure_ascii=False, indent=2))

def top_records(df, cols, n=5):
    out = df.loc[:, cols].head(n).copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = out[c].round(3)
    return out.to_dict(orient="records")

def stable_sort(df, by, ascending):
    by_list = by if isinstance(by, list) else [by]
    asc_list = ascending if isinstance(ascending, list) else [ascending] * len(by_list)
    for tie_col in ["book_key", "seed_key", "neighbor_key"]:
        if tie_col in df.columns and tie_col not in by_list:
            by_list = by_list + [tie_col]
            asc_list = asc_list + [True]
            break
    return df.sort_values(by_list, ascending=asc_list, kind="mergesort")

def stable_abs_sort(df, col, ascending=False):
    temp = df.assign(_abs_sort=df[col].abs())
    return stable_sort(temp, "_abs_sort", ascending).drop(columns="_abs_sort")

ratings_clean = (ratings_raw.rename(columns={"book_id": "book_key"})
                 .sort_values(["user_id", "book_key", "rating"])
                 .drop_duplicates(["user_id", "book_key"], keep=False)
                 .dropna(subset=["user_id", "book_key", "rating"]))
ratings_clean[["user_id", "book_key", "rating"]] = ratings_clean[["user_id", "book_key", "rating"]].astype(int)

books_clean = books_raw.drop_duplicates("original_title", keep=False).copy()
books_clean = books_clean.dropna(subset=[
    "id", "book_id", "original_title", "authors", "average_rating", "ratings_count",
    "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"
])
books_clean["book_key"] = books_clean["id"].astype(int)
books_clean["goodreads_book_id"] = books_clean["book_id"].astype(int)

raw_book_evidence = books_clean[[
    "book_key", "goodreads_book_id", "original_title", "title", "authors",
    "image_url", "small_image_url", "original_publication_year",
    "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"
]].copy()
raw_book_evidence["title_key_basic"] = raw_book_evidence["original_title"].map(title_basic)
raw_book_evidence["title_key_compact"] = raw_book_evidence["original_title"].map(title_compact)
raw_book_evidence["lead_author_key"] = raw_book_evidence["authors"].map(key_author)

star_cols = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
star_total = books_clean[star_cols].sum(axis=1).replace(0, np.nan)
star_probs = books_clean[star_cols].div(star_total, axis=0).fillna(0)
rating_entropy = -(star_probs.replace(0, np.nan) * np.log(star_probs.replace(0, np.nan))).sum(axis=1).fillna(0)

to_read_counts = (to_read_raw.rename(columns={"book_id": "book_key"})
                  .groupby("book_key").size().rename("intent_count"))
rating_activity = ratings_clean.groupby("book_key").size().rename("activity_count")

genre_terms = {
    "art": "art", "biography": "biography", "business": "business",
    "chick-lit": "chick_lit", "children": "children", "childrens": "children",
    "christian": "christian", "classics": "classics", "comics": "comics",
    "contemporary": "contemporary", "cookbooks": "cookbooks", "crime": "crime",
    "ebooks": "ebooks", "fantasy": "fantasy", "fiction": "fiction",
    "historical-fiction": "historical_fiction", "history": "history",
    "horror": "horror", "humor": "humor", "manga": "manga", "memoir": "memoir",
    "music": "music", "mystery": "mystery", "nonfiction": "nonfiction",
    "paranormal": "paranormal", "philosophy": "philosophy", "poetry": "poetry",
    "romance": "romance", "science": "science", "science-fiction": "science_fiction",
    "self-help": "self_help", "sports": "sports", "suspense": "suspense",
    "thriller": "thriller", "travel": "travel", "young-adult": "young_adult"
}
tag_lookup = tags_raw.copy()
tag_lookup["tag_norm"] = tag_lookup["tag_name"].astype(str).str.lower().str.replace("_", "-", regex=False)
tag_lookup["genre"] = tag_lookup["tag_norm"].map(genre_terms)
genre_rows = (book_tags_raw.merge(tag_lookup[["tag_id", "tag_name", "genre"]], on="tag_id", how="left")
              .dropna(subset=["genre"]))
genre_rows = genre_rows.merge(
    books_clean[["book_key", "goodreads_book_id"]],
    left_on="goodreads_book_id",
    right_on="goodreads_book_id",
    how="inner"
)
genre_strength = (genre_rows.groupby(["book_key", "genre"], as_index=False)["count"].sum()
                  .sort_values(["book_key", "count", "genre"], ascending=[True, False, True], kind="mergesort"))
main_genre = genre_strength.drop_duplicates("book_key").set_index("book_key")[["genre", "count"]]
main_genre.columns = ["main_genre", "main_genre_tag_count"]
raw_tag_evidence = (genre_rows.sort_values(["book_key", "count", "tag_name"], ascending=[True, False, True], kind="mergesort")
                   .groupby("book_key")
                   .head(8)
                   .groupby("book_key")
                   .apply(lambda x: x[["tag_name", "count"]].to_dict("records"))
                   .rename("tag_evidence"))

books_audit = pd.DataFrame({
    "book_key": books_clean["book_key"],
    "goodreads_book_id": books_clean["goodreads_book_id"],
    "title_key_basic": books_clean["original_title"].map(title_basic),
    "title_key_compact": books_clean["original_title"].map(title_compact),
    "lead_author_key": books_clean["authors"].map(key_author),
    "language_group": books_clean["language_code"].fillna("unknown").astype(str).str.lower(),
    "average_rating": books_clean["average_rating"].astype(float),
    "ratings_count": books_clean["ratings_count"].astype(float),
    "books_count": books_clean["books_count"].fillna(0).astype(float),
    "review_count": books_clean["work_text_reviews_count"].fillna(0).astype(float),
    "publication_year_known": books_clean["original_publication_year"].notna(),
    "title_length": books_clean["original_title"].astype(str).str.len(),
    "five_star_share": (books_clean["ratings_5"] / star_total).fillna(0),
    "low_star_share": ((books_clean["ratings_1"] + books_clean["ratings_2"]) / star_total).fillna(0),
    "star_balance": ((books_clean["ratings_5"] - books_clean["ratings_1"] - books_clean["ratings_2"]) / star_total).fillna(0),
    "rating_entropy": rating_entropy,
}).reset_index(drop=True)
books_audit = books_audit.merge(main_genre, left_on="book_key", right_index=True, how="left")
books_audit["main_genre"] = books_audit["main_genre"].fillna("unknown")
books_audit["main_genre_tag_count"] = books_audit["main_genre_tag_count"].fillna(0)
books_audit = books_audit.merge(to_read_counts, left_on="book_key", right_index=True, how="left")
books_audit = books_audit.merge(rating_activity, left_on="book_key", right_index=True, how="left")
books_audit[["intent_count", "activity_count"]] = books_audit[["intent_count", "activity_count"]].fillna(0)
books_audit["intent_share"] = books_audit["intent_count"] / books_audit["activity_count"].replace(0, np.nan)
books_audit["intent_share"] = books_audit["intent_share"].fillna(0)

genre_sizes = books_audit.groupby("main_genre").size()
books_audit["genre_supply"] = books_audit["main_genre"].map(genre_sizes).astype(float)
books_audit["genre_rarity_initial"] = pct_rank(-books_audit["genre_supply"], ascending=True)
books_audit["genre_rarity_score"] = books_audit["genre_rarity_initial"]
books_audit["demand_absolute_score"] = pct_rank(books_audit["intent_count"], ascending=True)
books_audit["demand_relative_score"] = pct_rank(books_audit["intent_share"], ascending=True)
global_share = float(books_audit["intent_count"].sum() / max(books_audit["activity_count"].sum(), 1))
alpha = 100.0
books_audit["intent_share_smoothed"] = (books_audit["intent_count"] + alpha * global_share) / (books_audit["activity_count"] + alpha)
books_audit["demand_smoothed_score"] = pct_rank(books_audit["intent_share_smoothed"], ascending=True)
books_audit["exposure_gap_score"] = pct_rank(-books_audit["ratings_count"], ascending=True)
books_audit["quality_simple_score"] = pct_rank(books_audit["average_rating"], ascending=True)
global_rating = float(np.average(books_audit["average_rating"], weights=np.maximum(books_audit["ratings_count"], 1)))
exposure_floor = float(books_audit["ratings_count"].quantile(0.60))
books_audit["quality_exposure_score_raw"] = (
    (books_audit["ratings_count"] / (books_audit["ratings_count"] + exposure_floor)) * books_audit["average_rating"]
    + (exposure_floor / (books_audit["ratings_count"] + exposure_floor)) * global_rating
)
books_audit["quality_exposure_score"] = pct_rank(books_audit["quality_exposure_score_raw"], ascending=True)
books_audit["quality_skew_score_raw"] = (
    0.70 * books_audit["quality_exposure_score"]
    + 20.0 * books_audit["star_balance"]
    - 5.0 * books_audit["rating_entropy"]
)
books_audit["quality_skew_score"] = pct_rank(books_audit["quality_skew_score_raw"], ascending=True)
books_audit["short_title_flag"] = books_audit["title_length"] <= books_audit["title_length"].median()

active_book_keys = set(books_audit.loc[
    books_audit["language_group"].ne("unknown")
    & books_audit["publication_year_known"]
    & books_audit["main_genre"].ne("unknown"),
    "book_key"
])
current_demand_col = "demand_absolute_score"
current_quality_col = "quality_simple_score"
current_title_col = "title_key_basic"
branch_tables = {}
branch_rankings = {}
audit_sets = {}

def score_review_pool(demand_col, quality_col):
    df = books_audit[books_audit["book_key"].isin(active_book_keys)].copy()
    df["review_score"] = (
        0.34 * df[demand_col]
        + 0.24 * df["exposure_gap_score"]
        + 0.22 * df["genre_rarity_score"]
        + 0.20 * df[quality_col]
    )
    demand_cut = df[demand_col].quantile(0.80)
    df = df[(df[demand_col] >= demand_cut) & (df["ratings_count"] <= df["ratings_count"].quantile(0.85))]
    return stable_sort(df, "review_score", False)

review_pool_initial = score_review_pool(current_demand_col, current_quality_col)
audit_sets["review_pool_initial"] = review_pool_initial["book_key"].head(12).tolist()

emit({
    "review_pool_top5": top_records(
        review_pool_initial,
        ["book_key", "title_key_basic", "review_score", "demand_absolute_score",
         "exposure_gap_score", "genre_rarity_score", "quality_simple_score"],
        5
    )
})

###### Task 2:
# Context: Keep the cleaned book range and first-pass scoring setup, but measure demand by the percentile rank of reading-intent count divided by cleaned rating activity. Books with no cleaned rating activity have an intent rate of zero. The early high-intent group is the first 25 rows from the initial review pool.
# Question: Rebuild the review pool with relative demand. Show five leading books, the books entering and leaving the ten-book monitoring set, and whether the movement mainly comes from books already in the early high-intent group.

# Depends on Task 1 (definition update): reuse cleaned books and initial review pool; update only demand.
previous_pool = audit_sets["review_pool_initial"][:10]
current_demand_col = "demand_relative_score"
review_pool_relative = score_review_pool(current_demand_col, current_quality_col)
audit_sets["review_pool_relative"] = review_pool_relative["book_key"].head(12).tolist()

new_pool = audit_sets["review_pool_relative"][:10]
entered = [k for k in new_pool if k not in previous_pool]
exited = [k for k in previous_pool if k not in new_pool]
kept = [k for k in new_pool if k in previous_pool]
early_high_intent = set(review_pool_initial.head(25)["book_key"])

change_df = books_audit[books_audit["book_key"].isin(entered + exited)].copy()
change_df["change"] = np.where(change_df["book_key"].isin(entered), "entered", "exited")
change_df["was_early_high_intent"] = change_df["book_key"].isin(early_high_intent)

emit({
    "relative_pool_top5": top_records(
        review_pool_relative,
        ["book_key", "title_key_basic", "review_score", "demand_relative_score",
         "exposure_gap_score", "genre_rarity_score", "quality_simple_score"],
        5
    ),
    "entered": top_records(change_df[change_df["change"].eq("entered")], ["book_key", "title_key_basic", "demand_relative_score", "was_early_high_intent"], 5),
    "exited": top_records(change_df[change_df["change"].eq("exited")], ["book_key", "title_key_basic", "demand_absolute_score", "was_early_high_intent"], 5),
    "kept_count": len(kept)
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 3:
# Context: Shrink lightly exposed titles toward the overall rating level. Use the 60th percentile of Goodreads rating count as the exposure floor, compute the overall mean weighted by Goodreads rating count, weight each book's own average by its Goodreads rating count relative to that count plus the exposure floor, weight the overall mean by the remaining floor share, and percentile-rank the blended value. Use the earlier volume-based demand lens here. Flip strength is the absolute value of the unweighted sum of the demand-percentile gap and quality-percentile gap between the compared treatments.
# Question: Find books whose membership in the ten-book monitoring set changes when volume demand is paired with exposure-aware quality. Show up to five strongest flips and the score components behind each movement.

# Depends on Task 2 (explicit rollback): demand is temporarily viewed through the first demand lens.
# Depends on Task 1 (definition update): quality is updated to exposure-aware scoring and kept for later tasks.
current_quality_col = "quality_exposure_score"
rollback_quality_pool = score_review_pool("demand_absolute_score", current_quality_col)
audit_sets["review_pool_absolute_exposure_quality"] = rollback_quality_pool["book_key"].head(12).tolist()

relative_set = set(audit_sets["review_pool_relative"][:10])
rollback_set = set(audit_sets["review_pool_absolute_exposure_quality"][:10])
flip_keys = list((relative_set ^ rollback_set))
audit_sets["flipped_review_books"] = flip_keys

flip_df = books_audit[books_audit["book_key"].isin(flip_keys)].copy()
flip_df["flip_direction"] = np.where(flip_df["book_key"].isin(rollback_set), "entered_under_rollback", "left_under_rollback")
flip_df["component_gap"] = (
    flip_df["demand_absolute_score"] - flip_df["demand_relative_score"]
    + flip_df["quality_exposure_score"] - flip_df["quality_simple_score"]
)
flip_df["flip_strength"] = flip_df["component_gap"].abs()
flip_df = stable_sort(flip_df, "flip_strength", False)

emit({
    "flipped_count": int(len(flip_df)),
    "strongest_flips": top_records(
        flip_df,
        ["book_key", "title_key_basic", "flip_direction", "demand_absolute_score",
         "demand_relative_score", "quality_simple_score", "quality_exposure_score",
         "component_gap", "flip_strength"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 4:
# Context: Restore original_title, authors, small_image_url, and raw recognized tag rows for the flipped books while keeping the same cleaned range and duplicate-title treatment. Classify source risk by duplicate cleaned title, then more than two listed authors, then fewer than three restored recognized genre-tag rows; otherwise mark the risk as low_source_risk.
# Question: Audit the flipped books by restoring original_title as the display title, authors as the full author string, small_image_url as the cover link, and the three strongest restored recognized raw tag rows as evidence. Format tag evidence as tag_name:count entries separated by semicolons, ordered by descending count with tag_name as the tie-breaker. For each book, identify whether the main source risk is duplicate-title handling, author ambiguity, compressed genre evidence, or low source risk.

# Depends on Task 3 (definition inheritance): audit the flipped set from the rollback comparison.
# Depends on Task 1 (long-span data-state dependency): restore source evidence without changing the cleaned book range.
evidence = (books_audit[books_audit["book_key"].isin(audit_sets["flipped_review_books"])]
            .merge(raw_book_evidence, on=["book_key", "goodreads_book_id"], how="left", suffixes=("", "_raw"))
            .merge(raw_tag_evidence, left_on="book_key", right_index=True, how="left"))
raw_title_counts = books_raw.assign(title_key_basic=books_raw["original_title"].map(title_basic)).groupby("title_key_basic").size()
evidence["raw_title_count"] = evidence["title_key_basic"].map(raw_title_counts).fillna(1)
evidence["author_count"] = evidence["authors"].fillna("").str.count(",") + 1
evidence["tag_evidence"] = evidence["tag_evidence"].apply(lambda x: x if isinstance(x, list) else [])
evidence["genre_evidence_count"] = evidence["tag_evidence"].apply(len)
evidence["strongest_tag_evidence"] = evidence["tag_evidence"].apply(
    lambda rows: "; ".join(f"{r.get('tag_name')}:{int(r.get('count', 0))}" for r in rows[:3])
)
evidence["risk_driver"] = np.select(
    [
        evidence["raw_title_count"] > 1,
        evidence["author_count"] > 2,
        evidence["genre_evidence_count"] < 3
    ],
    ["duplicate_title_handling", "author_ambiguity", "thin_genre_evidence"],
    default="low_source_risk"
)
evidence = stable_sort(evidence, ["risk_driver", "quality_exposure_score"], [True, False])

emit({
    "evidence_review": top_records(
        evidence,
        ["book_key", "original_title", "authors", "small_image_url", "main_genre",
         "strongest_tag_evidence", "risk_driver", "raw_title_count", "author_count", "genre_evidence_count"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 5:
# Context: Some titles become unstable under stricter title alignment because series suffixes, leading articles, and punctuation can collapse separate display names. Move title alignment to the stricter treatment while keeping the current demand, quality, and scarcity treatments. This changes matching keys, feature text, and collision risk, without merging or deleting book records. Focus on candidates that appeared in at least two of the three ten-book monitoring sets. Title-risk score adds 25 points when the stricter title key has more collisions than the earlier key, 20 points when original-title length is no greater than the cleaned-table median, and 10 points when the restored author string lists more than two authors.
# Question: Re-audit the stable candidates after stricter title alignment. Show five candidates most affected by title merging, short-title status, or author ambiguity, and summarize how the stable candidate set changes.

# Depends on Tasks 1-3 (multi-hop composition): stable candidates come from repeated review-pool appearances.
# Depends on Task 4 (definition update): title alignment changes while demand and quality definitions are inherited.
current_title_col = "title_key_compact"
pool_sources = [
    audit_sets["review_pool_initial"][:10],
    audit_sets["review_pool_relative"][:10],
    audit_sets["review_pool_absolute_exposure_quality"][:10],
]
appearances = Counter(k for src in pool_sources for k in src)
stable_candidate_keys = [k for k, v in appearances.items() if v >= 2]
audit_sets["stable_review_candidates"] = stable_candidate_keys

compact_counts = books_audit.groupby("title_key_compact").size()
basic_counts = books_audit.groupby("title_key_basic").size()
title_risk_all = books_audit[["book_key", "title_key_basic", "title_key_compact", "short_title_flag"]].copy()
title_risk_all["compact_title_count"] = title_risk_all["title_key_compact"].map(compact_counts).fillna(1)
title_risk_all["basic_title_count"] = title_risk_all["title_key_basic"].map(basic_counts).fillna(1)
title_risk_all = title_risk_all.merge(raw_book_evidence[["book_key", "authors"]], on="book_key", how="left")
title_risk_all["author_count"] = title_risk_all["authors"].fillna("").str.count(",") + 1
title_risk_all["title_risk_score"] = (
    25 * (title_risk_all["compact_title_count"] > title_risk_all["basic_title_count"]).astype(int)
    + 20 * title_risk_all["short_title_flag"].astype(int)
    + 10 * (title_risk_all["author_count"] > 2).astype(int)
)
stable_title_audit = title_risk_all[title_risk_all["book_key"].isin(stable_candidate_keys)].copy()
stable_title_audit = stable_sort(stable_title_audit, "title_risk_score", False)
audit_sets["title_risky_stable"] = stable_title_audit.loc[stable_title_audit["title_risk_score"] > 0, "book_key"].tolist()

emit({
    "affected_stable_candidates": top_records(
        stable_title_audit,
        ["book_key", "title_key_basic", "title_key_compact", "title_risk_score",
         "compact_title_count", "short_title_flag", "author_count"],
        5
    ),
    "stable_candidate_count": int(len(stable_candidate_keys)),
    "candidates_with_title_risk": int((stable_title_audit["title_risk_score"] > 0).sum())
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 6:
# Context: Use books that survived earlier review steps as seeds. For each seed, build count-vector content features with English stop-word removal from the current title key, lead-author key, and a five-bin average-rating token, then take eight nearest non-self cosine neighbors. Summarize each seed with average similarity, neighbor quality, neighbor demand, genre agreement, and title-risk counts; genre agreement is the share of returned neighbors with the seed's main genre. Seed evidence blends 30% average similarity, 25% neighbor quality, 25% neighbor demand, and 20% genre agreement, with a three-point penalty for each title-risk neighbor.
# Question: Build content-based neighbors for the stable candidate seeds. Show five seeds with the weakest recommendation evidence, including neighbor quality, demand, genre agreement, and title-risk measures.

# Depends on Task 5 (definition inheritance): seed books and title treatment come from the stable-candidate audit.
# Depends on Task 3 (definition inheritance): recommendation quality uses the exposure-aware quality treatment.
def build_content_neighbors(seed_keys, use_quality=True, use_genre_language=False, title_col=None, k=8):
    title_col = title_col or current_title_col
    work = books_audit[books_audit["book_key"].isin(active_book_keys)].copy().reset_index(drop=True)
    quality_bucket = pd.qcut(work["average_rating"].rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)])
    soup = work[title_col].fillna("") + " " + work["lead_author_key"].fillna("")
    if use_quality:
        soup = soup + " quality_" + quality_bucket.astype(str)
    if use_genre_language:
        soup = soup + " genre_" + work["main_genre"].fillna("unknown") + " lang_" + work["language_group"].fillna("unknown")
    vec = CountVectorizer(stop_words="english")
    mat = vec.fit_transform(soup)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(k + 1, len(work)))
    model.fit(mat)
    pos = pd.Series(work.index, index=work["book_key"]).to_dict()
    rows = []
    for seed in seed_keys:
        if seed not in pos:
            continue
        distances, indices = model.kneighbors(mat[pos[seed]], n_neighbors=min(k + 1, len(work)))
        seed_genre = work.loc[pos[seed], "main_genre"]
        for dist, idx in zip(distances[0], indices[0]):
            neighbor = int(work.loc[idx, "book_key"])
            if neighbor == seed:
                continue
            rows.append({
                "seed_key": int(seed),
                "neighbor_key": neighbor,
                "similarity_pct": (1 - float(dist)) * 100,
                "same_genre": bool(work.loc[idx, "main_genre"] == seed_genre),
                "neighbor_quality": float(work.loc[idx, current_quality_col]),
                "neighbor_demand": float(work.loc[idx, current_demand_col]),
                "neighbor_title_risk": bool(neighbor in audit_sets.get("title_risky_stable", []))
            })
    return pd.DataFrame(rows)

content_neighbors_current = build_content_neighbors(stable_candidate_keys, use_quality=True, use_genre_language=False, title_col=current_title_col)
content_seed_summary = (content_neighbors_current.groupby("seed_key")
    .agg(avg_similarity_pct=("similarity_pct", "mean"),
         avg_neighbor_quality=("neighbor_quality", "mean"),
         avg_neighbor_demand=("neighbor_demand", "mean"),
         genre_agreement_pct=("same_genre", lambda s: 100 * s.mean()),
         title_risk_neighbors=("neighbor_title_risk", "sum"))
    .reset_index())
content_seed_summary["evidence_score"] = (
    0.30 * content_seed_summary["avg_similarity_pct"]
    + 0.25 * content_seed_summary["avg_neighbor_quality"]
    + 0.25 * content_seed_summary["avg_neighbor_demand"]
    + 0.20 * content_seed_summary["genre_agreement_pct"]
    - 3.0 * content_seed_summary["title_risk_neighbors"]
)
content_seed_summary = content_seed_summary.merge(books_audit[["book_key", "title_key_basic"]], left_on="seed_key", right_on="book_key", how="left")
content_seed_summary = stable_sort(content_seed_summary, "evidence_score", True)

emit({
    "weakest_recommendation_evidence": top_records(
        content_seed_summary,
        ["seed_key", "title_key_basic", "evidence_score", "avg_similarity_pct",
         "avg_neighbor_quality", "avg_neighbor_demand", "genre_agreement_pct", "title_risk_neighbors"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 7:
# Context: Check whether the content evidence is driven by the quality token or by stricter title alignment rather than meaningful similarity. Run one pass without quality in the content features and another pass with the earlier title alignment. Measure neighbor drift with Jaccard overlap, using the union of the two eight-neighbor sets for each seed as the denominator. After the checks, keep the stricter-title, quality-aware recommendation evidence.
# Question: Compare current content-neighbor evidence with the no-quality and earlier-title checks. Show five seeds with the highest combined drift and indicate which check drives more of each drift.

# Depends on Task 6 (counterfactual): compare two temporary recommendation branches to current content neighbors.
# Depends on Task 5 (explicit rollback): one branch returns only the title alignment to the earlier treatment.
def neighbor_sets(df):
    return df.groupby("seed_key")["neighbor_key"].apply(lambda x: set(map(int, x))).to_dict()

current_sets = neighbor_sets(content_neighbors_current)
content_no_quality = build_content_neighbors(stable_candidate_keys, use_quality=False, use_genre_language=False, title_col=current_title_col)
content_old_title = build_content_neighbors(stable_candidate_keys, use_quality=True, use_genre_language=False, title_col="title_key_basic")
no_quality_sets = neighbor_sets(content_no_quality)
old_title_sets = neighbor_sets(content_old_title)

drift_rows = []
for seed in stable_candidate_keys:
    base = current_sets.get(seed, set())
    nq = no_quality_sets.get(seed, set())
    ot = old_title_sets.get(seed, set())
    def jacc(a, b):
        return 100 * len(a & b) / len(a | b) if len(a | b) else 100.0
    drift_rows.append({
        "seed_key": int(seed),
        "quality_removed_jaccard_pct": jacc(base, nq),
        "old_title_jaccard_pct": jacc(base, ot),
        "larger_drift_source": "quality_feature" if jacc(base, nq) < jacc(base, ot) else "title_alignment"
    })
content_drift_audit = pd.DataFrame(drift_rows).merge(books_audit[["book_key", "title_key_basic"]], left_on="seed_key", right_on="book_key", how="left")
content_drift_audit["worst_jaccard_pct"] = content_drift_audit[["quality_removed_jaccard_pct", "old_title_jaccard_pct"]].min(axis=1)
content_drift_audit = stable_sort(content_drift_audit, "worst_jaccard_pct", True)

emit({
    "largest_neighbor_drifts": top_records(
        content_drift_audit,
        ["seed_key", "title_key_basic", "quality_removed_jaccard_pct", "old_title_jaccard_pct", "larger_drift_source"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 8:
# Context: Extend the content features with genre and language while keeping the same seeds. Use the union of previous and new neighbor sets for drift, and use the new neighbors as the denominator for genre and language agreement. Earlier title and quality checks remain diagnostic context.
# Question: Add genre and language to the content-neighbor evidence for the same seeds. Show five seeds that drift most, and decompose the drift into genre agreement, language agreement, title-risk, and quality changes.

# Depends on Task 6 (definition update): the content recommender gains genre/language features.
# Depends on Task 7 (definition inheritance): temporary branches do not replace the main recommendation state.
content_neighbors_before_genre = content_neighbors_current.copy()
content_neighbors_current = build_content_neighbors(stable_candidate_keys, use_quality=True, use_genre_language=True, title_col=current_title_col)

before_sets = neighbor_sets(content_neighbors_before_genre)
after_sets = neighbor_sets(content_neighbors_current)
seed_lang = books_audit.set_index("book_key")["language_group"].to_dict()

drift_detail = []
for seed in stable_candidate_keys:
    before = before_sets.get(seed, set())
    after = after_sets.get(seed, set())
    union = before | after
    jaccard_pct = 100 * len(before & after) / len(union) if union else 100.0
    before_df = content_neighbors_before_genre[content_neighbors_before_genre["seed_key"].eq(seed)].copy()
    after_df = content_neighbors_current[content_neighbors_current["seed_key"].eq(seed)].copy()
    seed_language = seed_lang.get(seed, "unknown")
    after_df = after_df.merge(books_audit[["book_key", "language_group"]], left_on="neighbor_key", right_on="book_key", how="left")
    before_quality = before_df["neighbor_quality"].mean() if len(before_df) else 0
    after_quality = after_df["neighbor_quality"].mean() if len(after_df) else 0
    drift_detail.append({
        "seed_key": int(seed),
        "jaccard_pct": jaccard_pct,
        "genre_agreement_pct": 100 * after_df["same_genre"].mean() if len(after_df) else 0,
        "language_agreement_pct": 100 * (after_df["language_group"] == seed_language).mean() if len(after_df) else 0,
        "avg_neighbor_quality": after_quality,
        "quality_change": after_quality - before_quality,
        "title_risk_neighbors": int(after_df["neighbor_title_risk"].sum()) if len(after_df) else 0
    })
content_genre_drift = pd.DataFrame(drift_detail).merge(books_audit[["book_key", "title_key_basic"]], left_on="seed_key", right_on="book_key", how="left")
content_genre_drift = stable_sort(content_genre_drift, "jaccard_pct", True)

emit({
    "genre_language_drift": top_records(
        content_genre_drift,
        ["seed_key", "title_key_basic", "jaccard_pct", "genre_agreement_pct",
         "language_agreement_pct", "avg_neighbor_quality", "quality_change", "title_risk_neighbors"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 9:
# Context: Some candidates may gain recommendation support mainly from the genre and language features. Compare current content evidence with the immediately previous recommendation treatment while leaving title alignment and demand unchanged. Feature-dependence score blends 40% added recommendation-hit percentile, 30% current demand, and 30% current quality.
# Question: Find candidates whose recommendation support mainly appears after adding genre and language. Show five strongest cases with their current support, earlier support, and demand-quality context.

# Depends on Task 8 (explicit rollback): compare current recommendation features to the immediately previous treatment.
# Depends on Task 2 and Task 3 (definition inheritance): demand and quality remain as currently established.
current_neighbor_counts = content_neighbors_current.groupby("neighbor_key").size().rename("current_content_hits")
previous_neighbor_counts = content_neighbors_before_genre.groupby("neighbor_key").size().rename("previous_content_hits")
feature_lift = books_audit[books_audit["book_key"].isin(active_book_keys)].copy()
feature_lift = feature_lift.merge(current_neighbor_counts, left_on="book_key", right_index=True, how="left")
feature_lift = feature_lift.merge(previous_neighbor_counts, left_on="book_key", right_index=True, how="left")
feature_lift[["current_content_hits", "previous_content_hits"]] = feature_lift[["current_content_hits", "previous_content_hits"]].fillna(0)
feature_lift["feature_added_hits"] = feature_lift["current_content_hits"] - feature_lift["previous_content_hits"]
feature_lift["feature_dependence_score"] = (
    40 * pct_rank(feature_lift["feature_added_hits"], ascending=True)
    + 30 * feature_lift[current_demand_col]
    + 30 * feature_lift[current_quality_col]
) / 100
feature_lift = stable_sort(feature_lift[feature_lift["feature_added_hits"] > 0], "feature_dependence_score", False)
audit_sets["feature_dependent_candidates"] = feature_lift.head(15)["book_key"].tolist()

emit({
    "feature_dependent_candidates": top_records(
        feature_lift,
        ["book_key", "title_key_basic", "feature_dependence_score", "current_content_hits",
         "previous_content_hits", "feature_added_hits", current_demand_col, current_quality_col],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 10:
# Context: Add a collaborative-neighbor check for the same candidate history and demand-quality treatments, leaving content evidence unchanged. Build cosine nearest neighbors from the cleaned book-user rating matrix; the initial matrix keeps books with at least 60 cleaned ratings and users with at least 50 cleaned ratings, and takes eight non-self neighbors per seed. Content/collaborative overlap uses the union of the two neighbor sets for the seed as denominator. Cross-model support combines 40% content/collaborative neighbor overlap, 30% collaborative-neighbor quality, and 30% collaborative-neighbor demand.
# Question: Build collaborative neighbors for the review seeds and compare them with current content neighbors. Show five seeds with the weakest cross-model support, including overlap, collaborative-neighbor quality, and collaborative-neighbor demand.

# Depends on Task 8 (multi-hop composition): cross-check collaborative neighbors against current content neighbors.
# Depends on Tasks 2, 3, and 5 (definition inheritance): use current demand, quality, title, and seed groups.
def build_cf_neighbors(seed_keys, book_min, user_min, k=8):
    candidate_books = set(active_book_keys)
    r = ratings_clean[ratings_clean["book_key"].isin(candidate_books)].copy()
    book_counts = r.groupby("book_key").size()
    keep_books = set(book_counts[book_counts >= book_min].index)
    r = r[r["book_key"].isin(keep_books)]
    user_counts = r.groupby("user_id").size()
    keep_users = set(user_counts[user_counts >= user_min].index)
    r = r[r["user_id"].isin(keep_users)]
    if r.empty:
        return pd.DataFrame(), {"books": 0, "users": 0, "ratings": 0, "book_min": book_min, "user_min": user_min}
    book_ids = sorted(r["book_key"].unique())
    user_ids = sorted(r["user_id"].unique())
    bpos = {b: i for i, b in enumerate(book_ids)}
    upos = {u: i for i, u in enumerate(user_ids)}
    mat = csr_matrix((
        r["rating"].astype(float),
        ([bpos[b] for b in r["book_key"]], [upos[u] for u in r["user_id"]])
    ), shape=(len(book_ids), len(user_ids)))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(k + 1, len(book_ids)))
    model.fit(mat)
    rows = []
    for seed in seed_keys:
        if seed not in bpos:
            continue
        distances, indices = model.kneighbors(mat[bpos[seed]], n_neighbors=min(k + 1, len(book_ids)))
        for dist, idx in zip(distances[0], indices[0]):
            neighbor = int(book_ids[idx])
            if neighbor == seed:
                continue
            rows.append({"seed_key": int(seed), "neighbor_key": neighbor, "cf_similarity_pct": (1 - float(dist)) * 100})
    return pd.DataFrame(rows), {"books": len(book_ids), "users": len(user_ids), "ratings": int(len(r)), "book_min": int(book_min), "user_min": int(user_min)}

book_activity_counts = ratings_clean[ratings_clean["book_key"].isin(active_book_keys)].groupby("book_key").size()
user_activity_counts = ratings_clean[ratings_clean["book_key"].isin(active_book_keys)].groupby("user_id").size()
cf_neighbors_current, cf_state_current = build_cf_neighbors(stable_candidate_keys, book_min=60, user_min=50)
cf_neighbors_current = cf_neighbors_current.merge(books_audit[["book_key", current_quality_col, current_demand_col]], left_on="neighbor_key", right_on="book_key", how="left")
cf_sets = neighbor_sets(cf_neighbors_current) if len(cf_neighbors_current) else {}
content_sets = neighbor_sets(content_neighbors_current)

cross_rows = []
for seed in stable_candidate_keys:
    cset = content_sets.get(seed, set())
    fset = cf_sets.get(seed, set())
    union = cset | fset
    overlap_pct = 100 * len(cset & fset) / len(union) if union else 0
    sub = cf_neighbors_current[cf_neighbors_current["seed_key"].eq(seed)]
    cross_rows.append({
        "seed_key": int(seed),
        "cross_model_overlap_pct": overlap_pct,
        "cf_avg_quality": sub[current_quality_col].mean() if len(sub) else 0,
        "cf_avg_demand": sub[current_demand_col].mean() if len(sub) else 0,
        "cf_neighbor_count": int(len(sub))
    })
cross_model_support = pd.DataFrame(cross_rows).merge(books_audit[["book_key", "title_key_basic"]], left_on="seed_key", right_on="book_key", how="left")
cross_model_support["cross_support_score"] = (
    0.40 * cross_model_support["cross_model_overlap_pct"]
    + 0.30 * cross_model_support["cf_avg_quality"]
    + 0.30 * cross_model_support["cf_avg_demand"]
)
cross_model_support = stable_sort(cross_model_support, "cross_support_score", True)

emit({
    "weakest_cross_model_support": top_records(
        cross_model_support,
        ["seed_key", "title_key_basic", "cross_support_score", "cross_model_overlap_pct",
         "cf_avg_quality", "cf_avg_demand", "cf_neighbor_count"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 11:
# Context: Move the collaborative check to distribution-based thresholds and also run a user-relaxed check. The data-driven matrix keeps books at or above the median cleaned book-activity count and users at or above the upper-quartile cleaned user-activity count; the relaxed check keeps the same book cutoff but lowers the user cutoff to the median. Neighbor drift uses Jaccard overlap with the union of compared neighbor sets for each seed as denominator. Changed-neighbor demand and quality are averaged over the symmetric difference between the previous and data-driven neighbor sets, and values above the midpoint indicate stronger evidence. Compare the previous, data-driven, and user-relaxed matrices by cleaned book count, cleaned user count, retained rating count, and the two cutoff values. Use the data-driven matrix as collaborative evidence after this comparison.
# Question: Update the collaborative-neighbor threshold with distribution cutoffs and compare it with the user-relaxed check. Show the matrix counts, five main neighbor drifts, and whether the affected candidates are weaker or stronger under the current demand-quality treatment.

# Depends on Task 10 (definition update): collaborative thresholds move to distribution-based cutoffs.
# Depends on Task 10 (counterfactual): user-only relaxation is temporary and does not replace the updated state.
cf_state_previous = dict(cf_state_current)
book_min_quantile = int(max(1, book_activity_counts.quantile(0.50)))
user_min_quantile = int(max(1, user_activity_counts.quantile(0.75)))
cf_neighbors_quantile, cf_state_quantile = build_cf_neighbors(stable_candidate_keys, book_min=book_min_quantile, user_min=user_min_quantile)
cf_neighbors_relaxed_user, cf_state_relaxed_user = build_cf_neighbors(stable_candidate_keys, book_min=book_min_quantile, user_min=int(max(1, user_activity_counts.quantile(0.50))))

old_cf_sets = cf_sets
quantile_sets = neighbor_sets(cf_neighbors_quantile) if len(cf_neighbors_quantile) else {}
relaxed_sets = neighbor_sets(cf_neighbors_relaxed_user) if len(cf_neighbors_relaxed_user) else {}

cf_drift_rows = []
for seed in stable_candidate_keys:
    old = old_cf_sets.get(seed, set())
    new = quantile_sets.get(seed, set())
    relaxed = relaxed_sets.get(seed, set())
    def jaccard_pct(a, b):
        return 100 * len(a & b) / len(a | b) if len(a | b) else 0
    changed = new ^ old
    changed_quality = books_audit.loc[books_audit["book_key"].isin(changed), current_quality_col].mean() if changed else 0
    changed_demand = books_audit.loc[books_audit["book_key"].isin(changed), current_demand_col].mean() if changed else 0
    changed_signal = "stronger" if changed and (changed_quality + changed_demand) / 2 >= 50 else ("weaker" if changed else "no_changed_neighbors")
    cf_drift_rows.append({
        "seed_key": int(seed),
        "old_to_distribution_jaccard_pct": jaccard_pct(old, new),
        "distribution_to_user_relaxed_jaccard_pct": jaccard_pct(new, relaxed),
        "changed_neighbor_quality": changed_quality,
        "changed_neighbor_demand": changed_demand,
        "changed_neighbor_signal": changed_signal
    })
cf_threshold_drift = pd.DataFrame(cf_drift_rows).merge(books_audit[["book_key", "title_key_basic"]], left_on="seed_key", right_on="book_key", how="left")
cf_threshold_drift = stable_sort(cf_threshold_drift, "old_to_distribution_jaccard_pct", True)
cf_neighbors_current = cf_neighbors_quantile.merge(books_audit[["book_key", current_quality_col, current_demand_col]], left_on="neighbor_key", right_on="book_key", how="left") if len(cf_neighbors_quantile) else cf_neighbors_quantile
cf_state_current = cf_state_quantile

emit({
    "matrix_comparison": {
        "previous": cf_state_previous,
        "distribution_based": cf_state_quantile,
        "user_relaxed": cf_state_relaxed_user
    },
    "largest_cf_threshold_drifts": top_records(
        cf_threshold_drift,
        ["seed_key", "title_key_basic", "old_to_distribution_jaccard_pct",
         "distribution_to_user_relaxed_jaccard_pct", "changed_neighbor_quality",
         "changed_neighbor_demand", "changed_neighbor_signal"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 12:
# Context: Combine demand, quality, genre scarcity, content-neighbor support, and collaborative-neighbor support into an adaptation-opportunity score across the active review range. Content and collaborative support are percentile ranks, over the cleaned book range, of how often a book appears as a neighbor across the current seed audits; books never appearing as a neighbor receive zero support. The first opportunity blend is 25% demand, 25% quality, 15% genre scarcity, 20% content-neighbor support, and 15% collaborative-neighbor support.
# Question: Build the first adaptation-opportunity ranking. Show five leading books with demand, quality, genre, content, and collaborative contributions.

# Depends on Tasks 2, 3, 8, and 11 (multi-hop composition): combine current demand, quality, content, and CF evidence.
def support_component(neighbor_df, name):
    if neighbor_df is None or len(neighbor_df) == 0:
        return pd.Series(0.0, index=books_audit["book_key"], name=name)
    counts = neighbor_df.groupby("neighbor_key").size()
    raw = pd.Series(
        books_audit["book_key"].map(counts).fillna(0).to_numpy(),
        index=books_audit["book_key"],
        name=name
    )
    if raw.max() == 0:
        return pd.Series(0.0, index=books_audit["book_key"], name=name)
    ranked = pct_rank(raw, ascending=True).rename(name)
    return ranked.where(raw > 0, 0.0)

def compute_opportunity(demand_col=None, quality_col=None, genre_col="genre_rarity_score",
                        content_df=None, cf_df=None, coverage_df=None, lag_df=None,
                        weights=None, label="branch"):
    demand_col = demand_col or current_demand_col
    quality_col = quality_col or current_quality_col
    content_df = content_neighbors_current if content_df is None else content_df
    cf_df = cf_neighbors_current if cf_df is None else cf_df
    weights = weights or {"demand": 0.25, "quality": 0.25, "genre": 0.15, "content": 0.20, "cf": 0.15}
    df = books_audit[books_audit["book_key"].isin(active_book_keys)].copy()
    df["demand_component"] = df[demand_col]
    df["quality_component"] = df[quality_col]
    df["genre_component"] = df[genre_col] if genre_col else 0.0
    content_scores = support_component(content_df, "content_component")
    cf_scores = support_component(cf_df, "cf_component")
    df = df.merge(content_scores, left_on="book_key", right_index=True, how="left")
    df = df.merge(cf_scores, left_on="book_key", right_index=True, how="left")
    df[["content_component", "cf_component"]] = df[["content_component", "cf_component"]].fillna(0)
    if coverage_df is not None:
        df = df.merge(coverage_df[["book_key", "covered_author_safe", "coverage_path"]], on="book_key", how="left")
        df["covered_author_safe"] = df["covered_author_safe"].fillna(False)
        df["coverage_gap_component"] = (~df["covered_author_safe"]).astype(float) * 100
    else:
        df["covered_author_safe"] = False
        df["coverage_path"] = "not_checked"
        df["coverage_gap_component"] = 0.0
    if lag_df is not None:
        df = df.merge(lag_df[["book_key", "lag_penalty_component"]], on="book_key", how="left")
        df["lag_penalty_component"] = df["lag_penalty_component"].fillna(0.0)
    else:
        df["lag_penalty_component"] = 0.0
    df["opportunity_score"] = (
        weights.get("demand", 0) * df["demand_component"]
        + weights.get("quality", 0) * df["quality_component"]
        + weights.get("genre", 0) * df["genre_component"]
        + weights.get("content", 0) * df["content_component"]
        + weights.get("cf", 0) * df["cf_component"]
        + weights.get("coverage_gap", 0) * df["coverage_gap_component"]
        - weights.get("lag_penalty", 0) * df["lag_penalty_component"]
    )
    df = stable_sort(df, "opportunity_score", False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    branch_tables[label] = df.copy()
    branch_rankings[label] = dict(zip(df["book_key"], df["rank"]))
    return df

opportunity_current = compute_opportunity(label="first_opportunity")
audit_sets["opportunity_first_top"] = opportunity_current.head(20)["book_key"].tolist()

emit({
    "opportunity_top5": top_records(
        opportunity_current,
        ["book_key", "title_key_basic", "rank", "opportunity_score",
         "demand_component", "quality_component", "genre_component", "content_component", "cf_component"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 13:
# Context: Reduce sensitivity to small denominators in the reading-intent ratio. Move demand to a smoothed intensity treatment: compute the catalog-wide intent share over the cleaned table, add 100 pseudo-rating activities at that share, and then percentile-rank the resulting intent rate. Compare this ranking with both earlier demand lenses, evaluating sensitivity among books that appear in the first 25 rows of at least one demand ranking.
# Question: Re-rank adaptation opportunities with smoothed relative demand. Show five leading candidates, the top-ten overlap with the previous opportunity ranking, and five candidates most sensitive to the demand treatments.

# Depends on Task 12 (definition update): opportunity ranking is recomputed under smoothed demand.
# Depends on Tasks 1 and 2 (explicit rollback): compare against absolute and intermediate relative demand treatments.
previous_opportunity = opportunity_current.copy()
current_demand_col = "demand_smoothed_score"
opportunity_smoothed = compute_opportunity(demand_col=current_demand_col, quality_col=current_quality_col, label="smoothed_demand")
opportunity_absolute_branch = compute_opportunity(demand_col="demand_absolute_score", quality_col=current_quality_col, label="absolute_demand_branch")
opportunity_relative_branch = compute_opportunity(demand_col="demand_relative_score", quality_col=current_quality_col, label="relative_demand_branch")
opportunity_current = opportunity_smoothed

top_prev = set(previous_opportunity.head(10)["book_key"])
top_smooth = set(opportunity_smoothed.head(10)["book_key"])
union_keys = list(set(opportunity_smoothed.head(25)["book_key"]) | set(opportunity_absolute_branch.head(25)["book_key"]) | set(opportunity_relative_branch.head(25)["book_key"]))
sensitivity = books_audit[books_audit["book_key"].isin(union_keys)][["book_key", "title_key_basic"]].copy()
for label in ["smoothed_demand", "absolute_demand_branch", "relative_demand_branch"]:
    sensitivity[label + "_rank"] = sensitivity["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
sensitivity["demand_rank_range"] = sensitivity[["smoothed_demand_rank", "absolute_demand_branch_rank", "relative_demand_branch_rank"]].max(axis=1) - sensitivity[["smoothed_demand_rank", "absolute_demand_branch_rank", "relative_demand_branch_rank"]].min(axis=1)
sensitivity = stable_sort(sensitivity, "demand_rank_range", False)

emit({
    "smoothed_top5": top_records(
        opportunity_smoothed,
        ["book_key", "title_key_basic", "rank", "opportunity_score", "demand_component", "quality_component"],
        5
    ),
    "top10_overlap_with_previous": int(len(top_prev & top_smooth)),
    "most_demand_sensitive": top_records(
        sensitivity,
        ["book_key", "title_key_basic", "smoothed_demand_rank", "absolute_demand_branch_rank",
         "relative_demand_branch_rank", "demand_rank_range"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 14:
# Context: Average rating can hide polarized star distributions. Restore folded star evidence for the current opportunity candidates and use it to adjust quality. Five-star and low-star percentages use each book's total restored star counts as the denominator. The five-star-minus-low-star balance is the 0-1 share difference, and rating entropy is natural-log entropy over the five restored star shares. The star-aware quality raw score is 70% exposure-aware quality percentile plus 20 times the balance minus 5 times entropy; percentile-rank that raw score before using it in the opportunity blend. Interpret rank movements over the union of the previous and star-aware top thirty as star-distribution effects.
# Question: Use star-skew adjusted quality to re-rank adaptation opportunities, and find five books whose movement shows the strongest disagreement between average rating and star structure.

# Depends on Task 13 (definition inheritance): keep smoothed demand and current recommendation evidence.
# Depends on Task 1 (long-span source restoration): use folded star evidence from the cleaned raw field path.
previous_opportunity = opportunity_current.copy()
current_quality_col = "quality_skew_score"
opportunity_star_quality = compute_opportunity(demand_col=current_demand_col, quality_col=current_quality_col, label="star_skew_quality")
opportunity_current = opportunity_star_quality

compare_keys = list(set(previous_opportunity.head(30)["book_key"]) | set(opportunity_star_quality.head(30)["book_key"]))
star_compare = books_audit[books_audit["book_key"].isin(compare_keys)][[
    "book_key", "title_key_basic", "average_rating", "five_star_share", "low_star_share",
    "star_balance", "quality_exposure_score", "quality_skew_score"
]].copy()
star_compare["previous_rank"] = star_compare["book_key"].map(dict(zip(previous_opportunity["book_key"], previous_opportunity["rank"]))).fillna(len(books_audit) + 1)
star_compare["star_quality_rank"] = star_compare["book_key"].map(branch_rankings["star_skew_quality"]).fillna(len(books_audit) + 1)
star_compare["rank_change"] = star_compare["previous_rank"] - star_compare["star_quality_rank"]
star_compare["five_star_pct"] = 100 * star_compare["five_star_share"]
star_compare["low_star_pct"] = 100 * star_compare["low_star_share"]
star_compare = stable_abs_sort(star_compare, "rank_change", False)

emit({
    "star_quality_top5": top_records(
        opportunity_star_quality,
        ["book_key", "title_key_basic", "rank", "opportunity_score", "quality_component",
         "demand_component", "content_component", "cf_component"],
        5
    ),
    "largest_star_structure_movements": top_records(
        star_compare,
        ["book_key", "title_key_basic", "previous_rank", "star_quality_rank", "rank_change",
         "average_rating", "five_star_pct", "low_star_pct", "star_balance"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 15:
# Context: Add cross-media coverage evidence to the current opportunity list. Compare exact restored-original-title alignment, stricter normalized alignment, and normalized alignment with book-side author disambiguation. Treat a normalized match as author-safe when the normalized book title has no book-side collision, or when the exact title already matched despite the collision. Coverage-path counts are simple counts of the current top thirty by final coverage path. Path sensitivity is the normalized-versus-exact coverage difference plus the normalized-versus-author-safe coverage difference; use that sensitivity with current opportunity score to choose audit rows, with unchanged rows included only when needed to complete the five-row list.
# Question: Audit Netflix coverage for the current top thirty opportunity candidates across the three matching paths. Show coverage-path counts and five path-sensitivity audit rows.

# Depends on Tasks 5 and 14 (multi-hop composition): use current title alignment and current opportunity candidates.
# Depends on Task 4 (definition inheritance): author disambiguation uses restored author evidence.
netflix_titles = netflix_raw.copy()
netflix_titles["title_basic"] = netflix_titles["title"].map(title_basic)
netflix_titles["title_compact"] = netflix_titles["title"].map(title_compact)
netflix_titles["date_added_year"] = pd.to_datetime(netflix_titles["date_added"], errors="coerce").dt.year

book_titles = raw_book_evidence[["book_key", "original_title", "authors", "title_key_basic", "title_key_compact", "lead_author_key"]].copy()
exact_keys = set(netflix_titles["title_basic"].dropna())
compact_keys = set(netflix_titles["title_compact"].dropna())
compact_counts_books = book_titles.groupby("title_key_compact").size()

coverage_rows = []
for _, row in book_titles.iterrows():
    exact = row["title_key_basic"] in exact_keys
    normalized = row["title_key_compact"] in compact_keys
    book_collision = compact_counts_books.get(row["title_key_compact"], 0) > 1
    author_safe = bool(normalized and (not book_collision or exact))
    if author_safe:
        path = "author_safe_normalized" if not exact else "exact"
    elif normalized:
        path = "normalized_ambiguous"
    else:
        path = "uncovered"
    nsub = netflix_titles[netflix_titles["title_compact"].eq(row["title_key_compact"])]
    coverage_rows.append({
        "book_key": int(row["book_key"]),
        "covered_exact": bool(exact),
        "covered_normalized": bool(normalized),
        "covered_author_safe": bool(author_safe),
        "coverage_path": path,
        "netflix_types": ",".join(sorted(nsub["type"].dropna().unique())) if len(nsub) else "",
        "netflix_release_year": float(nsub["release_year"].min()) if len(nsub) else np.nan,
        "netflix_date_added_year": float(nsub["date_added_year"].min()) if len(nsub) else np.nan
    })
netflix_coverage = pd.DataFrame(coverage_rows)
current_top = opportunity_current.head(30)[["book_key", "title_key_basic", "opportunity_score", "rank"]].merge(netflix_coverage, on="book_key", how="left")
current_top["path_sensitivity"] = (
    current_top["covered_normalized"].astype(int) - current_top["covered_exact"].astype(int)
    + current_top["covered_normalized"].astype(int) - current_top["covered_author_safe"].astype(int)
)
sensitive_coverage = stable_sort(current_top, ["path_sensitivity", "opportunity_score"], [False, False])

emit({
    "coverage_path_counts_top30": current_top["coverage_path"].value_counts().to_dict(),
    "path_sensitive_candidates": top_records(
        sensitive_coverage,
        ["book_key", "title_key_basic", "rank", "opportunity_score",
         "covered_exact", "covered_normalized", "covered_author_safe", "coverage_path"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 16:
# Context: Check coverage by media type and by the title rule used before author disambiguation. The Movie-only check uses normalized titles among Netflix Movie rows without book-side author disambiguation; the earlier-title check uses the earlier basic title key against all Netflix titles. Compare those checks with current coverage evidence within the best fifty current opportunity rows, and evaluate whether affected books concentrate in earlier high-intent, short-title, or scarce-genre groups. Use coverage-change count and opportunity score as the audit signals, with unchanged rows included only when needed to complete the five-row list.
# Question: Run the Movie-only and earlier-title coverage checks. Show five coverage-change audit rows and whether changed rows concentrate in the earlier demand, short-title, or genre-scarcity signals.

# Depends on Task 15 (counterfactual): Movie-only and title-rollback coverage are temporary checks.
# Depends on Tasks 1, 2, and 5 (long-span anchors): concentration is checked against early demand, short-title, and genre scarcity signals.
movie_titles = netflix_titles[netflix_titles["type"].eq("Movie")].copy()
movie_compact_keys = set(movie_titles["title_compact"].dropna())
basic_keys = set(netflix_titles["title_basic"].dropna())

coverage_branch = netflix_coverage.copy()
coverage_branch["covered_movie_only"] = coverage_branch["book_key"].map(
    raw_book_evidence.set_index("book_key")["title_key_compact"].isin(movie_compact_keys)
).fillna(False).astype(bool)
coverage_branch["covered_title_rollback"] = coverage_branch["book_key"].map(
    raw_book_evidence.set_index("book_key")["title_key_basic"].isin(basic_keys)
).fillna(False).astype(bool)

early_high_demand = set(review_pool_initial.head(25)["book_key"])
early_rare_genre_cut = books_audit["genre_rarity_initial"].quantile(0.80)
coverage_audit = opportunity_current.head(50)[["book_key", "title_key_basic", "rank", "opportunity_score"]].merge(coverage_branch, on="book_key", how="left")
coverage_audit = coverage_audit.merge(books_audit[["book_key", "short_title_flag", "genre_rarity_initial"]], on="book_key", how="left")
coverage_audit["high_intent_early"] = coverage_audit["book_key"].isin(early_high_demand)
coverage_audit["scarce_genre_early"] = coverage_audit["genre_rarity_initial"] >= early_rare_genre_cut
coverage_audit["coverage_change_count"] = (
    coverage_audit["covered_author_safe"].astype(int).sub(coverage_audit["covered_movie_only"].astype(int)).abs()
    + coverage_audit["covered_author_safe"].astype(int).sub(coverage_audit["covered_title_rollback"].astype(int)).abs()
)
coverage_audit = stable_sort(coverage_audit, ["coverage_change_count", "opportunity_score"], [False, False])
changed_coverage = coverage_audit[coverage_audit["coverage_change_count"] > 0]
coverage_signal_counts = {
    "early_high_intent": int(changed_coverage["high_intent_early"].sum()),
    "short_title": int(changed_coverage["short_title_flag"].sum()),
    "scarce_genre": int(changed_coverage["scarce_genre_early"].sum()),
    "changed_count": int(len(changed_coverage)),
}

emit({
    "coverage_changes": top_records(
        coverage_audit,
        ["book_key", "title_key_basic", "rank", "covered_author_safe",
         "covered_movie_only", "covered_title_rollback", "high_intent_early",
         "short_title_flag", "scarce_genre_early", "coverage_change_count"],
        5
    ),
    "coverage_change_signal_counts": coverage_signal_counts
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 17:
# Context: Author-side title disambiguation may affect the opportunity ranking. Recompute only that title treatment, leaving demand, quality, recommendation evidence, and coverage weights unchanged. For this isolated comparison, use a light coverage blend of 24% demand, 24% quality, 14% genre scarcity, 19% content support, 14% collaborative support, and 5% uncovered-gap credit, and compare the first ten rows as the top group. Treat movement as coverage-led when current uncovered-gap contribution is at least as large as both recommendation contributions; otherwise treat it as recommendation-led.
# Question: Recompute opportunity candidates with the pre-disambiguation title treatment. State whether the top group changes, then show five main rank shifts and whether they are mainly due to coverage labels or recommendation support.

# Depends on Task 15 (explicit rollback): title matching returns to the pre-disambiguation treatment only.
# Depends on Tasks 13, 14, and 11 (partial inheritance): demand, quality, and recommender evidence stay current.
pre_author_coverage = netflix_coverage.copy()
pre_author_coverage["covered_author_safe"] = pre_author_coverage["covered_normalized"]
pre_author_coverage["coverage_path"] = np.where(pre_author_coverage["covered_normalized"], "normalized_without_author_check", "uncovered")
weights_with_light_coverage = {"demand": 0.24, "quality": 0.24, "genre": 0.14, "content": 0.19, "cf": 0.14, "coverage_gap": 0.05}
opportunity_pre_author_title = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=pre_author_coverage,
    weights=weights_with_light_coverage,
    label="pre_author_title_branch"
)
opportunity_current_light_coverage = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=netflix_coverage,
    weights=weights_with_light_coverage,
    label="author_safe_light_coverage"
)
top_current = set(opportunity_current_light_coverage.head(10)["book_key"])
top_rollback = set(opportunity_pre_author_title.head(10)["book_key"])
change_keys = list(top_current | top_rollback)
title_branch_changes = books_audit[books_audit["book_key"].isin(change_keys)][["book_key", "title_key_basic"]].copy()
title_branch_changes["current_rank"] = title_branch_changes["book_key"].map(branch_rankings["author_safe_light_coverage"]).fillna(len(books_audit) + 1)
title_branch_changes["rollback_rank"] = title_branch_changes["book_key"].map(branch_rankings["pre_author_title_branch"]).fillna(len(books_audit) + 1)
title_branch_changes["rank_shift"] = title_branch_changes["current_rank"] - title_branch_changes["rollback_rank"]
title_branch_changes = title_branch_changes.merge(netflix_coverage[["book_key", "coverage_path"]], on="book_key", how="left")
title_branch_changes = title_branch_changes.merge(
    opportunity_current_light_coverage[["book_key", "coverage_gap_component", "content_component", "cf_component"]],
    on="book_key",
    how="left"
)
title_branch_changes = title_branch_changes.rename(columns={
    "coverage_gap_component": "current_coverage_gap_component",
    "content_component": "current_content_component",
    "cf_component": "current_cf_component"
})
title_branch_changes["movement_driver"] = np.where(
    title_branch_changes["current_coverage_gap_component"] >=
    title_branch_changes[["current_content_component", "current_cf_component"]].max(axis=1),
    "coverage_label",
    "recommendation_support"
)
title_branch_changes["change_type"] = np.select(
    [
        title_branch_changes["book_key"].isin(top_current & top_rollback),
        title_branch_changes["book_key"].isin(top_rollback - top_current),
        title_branch_changes["book_key"].isin(top_current - top_rollback),
    ],
    ["retained", "entered_under_pre_author_title", "left_under_pre_author_title"],
    default="outside_top_group"
)
title_branch_changes = stable_abs_sort(title_branch_changes, "rank_shift", False)

emit({
    "top_group_changed": bool(top_current != top_rollback),
    "change_counts": title_branch_changes["change_type"].value_counts().to_dict(),
    "largest_title_treatment_movements": top_records(
        title_branch_changes,
        ["book_key", "title_key_basic", "change_type", "current_rank", "rollback_rank",
         "rank_shift", "movement_driver", "coverage_path", "current_content_component", "current_cf_component"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 18:
# Context: Make coverage part of the opportunity score: keep smoothed demand as its own component and give books without author-safe coverage a fixed uncovered-gap credit, while covered books receive zero for that component. Before scoring, update genre scarcity for cross-media supply; within each main genre, fewer author-safe Netflix-covered books means stronger scarcity, then percentile-rank that scarcity. The coverage-aware blend is 21% demand, 22% quality, 12% cross-media genre scarcity, 17% content support, 13% collaborative support, and 15% uncovered-gap credit, where the uncovered-gap component is 100 for books without author-safe coverage and 0 otherwise. For rank-change explanations among books appearing in either top thirty, use the largest current contribution among coverage gap, demand, quality, content support, and collaborative support. Leave recommendation and quality treatments as already established.
# Question: Add the fixed uncovered-gap credit and coverage discount to the opportunity score. Show five leading books and five main rank changes, explaining whether movement is driven by coverage, demand, quality, content support, or collaborative support.

# Depends on Task 15 (definition update): Netflix coverage now affects opportunity scoring.
# Depends on Tasks 13, 14, and 11 (multi-hop composition): keep smoothed demand, skew quality, content, and CF evidence.
coverage_weights = {"demand": 0.21, "quality": 0.22, "genre": 0.12, "content": 0.17, "cf": 0.13, "coverage_gap": 0.15}
previous_no_coverage = opportunity_current.copy()
genre_coverage_supply = (books_audit[["book_key", "main_genre"]]
                         .merge(netflix_coverage[["book_key", "covered_author_safe"]], on="book_key", how="left"))
genre_coverage_supply["covered_author_safe"] = genre_coverage_supply["covered_author_safe"].fillna(False)
covered_by_genre = genre_coverage_supply.groupby("main_genre")["covered_author_safe"].sum()
books_audit["genre_netflix_supply"] = books_audit["main_genre"].map(covered_by_genre).fillna(0).astype(float)
books_audit["genre_rarity_crossmedia"] = pct_rank(-books_audit["genre_netflix_supply"], ascending=True)
books_audit["genre_rarity_score"] = books_audit["genre_rarity_crossmedia"]
opportunity_coverage = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=netflix_coverage,
    weights=coverage_weights,
    label="coverage_weighted"
)
opportunity_coverage["unmet_demand_component"] = opportunity_coverage["coverage_gap_component"] * opportunity_coverage["demand_component"] / 100
opportunity_current = opportunity_coverage

movement_keys = list(set(previous_no_coverage.head(30)["book_key"]) | set(opportunity_coverage.head(30)["book_key"]))
movement = opportunity_coverage[opportunity_coverage["book_key"].isin(movement_keys)][[
    "book_key", "title_key_basic", "rank", "opportunity_score", "coverage_gap_component",
    "demand_component", "quality_component", "content_component", "cf_component"
]].copy()
old_rank_map = dict(zip(previous_no_coverage["book_key"], previous_no_coverage["rank"]))
movement["previous_rank"] = movement["book_key"].map(old_rank_map).fillna(len(books_audit) + 1)
movement["rank_change"] = movement["previous_rank"] - movement["rank"]
movement["movement_driver"] = movement[["coverage_gap_component", "demand_component", "quality_component", "content_component", "cf_component"]].idxmax(axis=1).str.replace("_component", "")
movement = stable_abs_sort(movement, "rank_change", False)

emit({
    "coverage_weighted_top5": top_records(
        opportunity_coverage,
        ["book_key", "title_key_basic", "rank", "opportunity_score",
         "demand_component", "quality_component", "coverage_gap_component", "content_component", "cf_component"],
        5
    ),
    "largest_rank_changes": top_records(
        movement,
        ["book_key", "title_key_basic", "previous_rank", "rank", "rank_change", "movement_driver"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 19:
# Context: For books with a matched Netflix title, including ambiguous normalized matches, publication-to-screen timing changes how much remaining opportunity gets credited. Restore exact publication years only for the current cleaned book range, compute the lag to the earliest available matched Netflix release or date-added year, convert valid nonnegative lags up to 300 years into a percentile penalty, reduce uncovered-gap credit to 13%, and subtract 8% of that penalty in the lag-aware score. Unmatched or invalid lags receive no lag penalty. Compare the lag-aware score with no-lag, Movie-only, and earlier-title checks, focusing on penalties among the best forty lag-aware rows; zero-penalty rows may complete the five-row list when needed.
# Question: Add a lag penalty to the coverage-aware opportunity score. Show five best-forty candidates with lag-penalty context and compare their ranks across the no-lag, Movie-only, and earlier-title checks.

# Depends on Task 18 (definition update): lag penalty is added to coverage-weighted opportunity.
# Depends on Tasks 1, 16, and 17 (long-span + counterfactual): restore publication year and compare Movie/title branches.
publication_years = raw_book_evidence[["book_key", "original_publication_year"]].copy()
lag_df = netflix_coverage.merge(publication_years, on="book_key", how="left")
lag_df["screen_year"] = lag_df[["netflix_release_year", "netflix_date_added_year"]].min(axis=1)
lag_df["raw_lag_years"] = lag_df["screen_year"] - lag_df["original_publication_year"]
valid_lag = lag_df["raw_lag_years"].where(lag_df["raw_lag_years"].between(0, 300))
lag_df["lag_penalty_component"] = 0.0
lag_df.loc[valid_lag.notna(), "lag_penalty_component"] = pct_rank(valid_lag.dropna(), ascending=True).to_numpy()

lag_weights = dict(coverage_weights)
lag_weights["lag_penalty"] = 0.08
lag_weights["coverage_gap"] = 0.13
opportunity_lag = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=netflix_coverage,
    lag_df=lag_df,
    weights=lag_weights,
    label="lag_penalized"
)
opportunity_no_lag_branch = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=netflix_coverage,
    weights=coverage_weights,
    label="no_lag_branch"
)
movie_coverage = netflix_coverage.copy()
movie_coverage["covered_author_safe"] = coverage_branch["covered_movie_only"].values
movie_coverage["coverage_path"] = np.where(movie_coverage["covered_author_safe"], "movie_only", "uncovered")
opportunity_movie_branch = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=movie_coverage,
    lag_df=lag_df,
    weights=lag_weights,
    label="movie_only_branch"
)
opportunity_current = opportunity_lag

lag_movement = opportunity_lag.head(40)[["book_key", "title_key_basic", "rank", "opportunity_score", "lag_penalty_component"]].copy()
for label in ["no_lag_branch", "movie_only_branch", "pre_author_title_branch"]:
    lag_movement[label + "_rank"] = lag_movement["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
lag_movement["no_lag_rank_shift"] = lag_movement["no_lag_branch_rank"] - lag_movement["rank"]
lag_movement = stable_sort(lag_movement, "lag_penalty_component", False)

emit({
    "largest_lag_penalties": top_records(
        lag_movement,
        ["book_key", "title_key_basic", "rank", "no_lag_branch_rank",
         "movie_only_branch_rank", "pre_author_title_branch_rank", "lag_penalty_component"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 20:
# Context: Genre scarcity has influenced earlier rankings. Remove that contribution without renormalizing, then compare the result with the earlier demand, earlier title, Movie-only, and no-lag checks. Use the first fifteen rows of each check as the top-candidate range for this stability view.
# Question: Assess top-candidate stability across the main score and the demand, title, Movie-only, no-lag, and no-genre checks. Show five stable books and five candidates whose inclusion depends most on genre scarcity.

# Depends on Tasks 13, 17, 19 (multi-branch composition): compare demand/title/movie/no-lag checks.
# Depends on Task 1 (counterfactual): genre scarcity is removed only for this branch.
no_genre_weights = dict(lag_weights)
no_genre_weights["genre"] = 0.0
opportunity_no_genre = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    genre_col=None,
    coverage_df=netflix_coverage,
    lag_df=lag_df,
    weights=no_genre_weights,
    label="no_genre_branch"
)

comparison_labels = ["lag_penalized", "relative_demand_branch", "pre_author_title_branch", "movie_only_branch", "no_lag_branch", "no_genre_branch"]
union_top = set()
for label in comparison_labels:
    union_top |= set(branch_tables[label].head(15)["book_key"])
stability = books_audit[books_audit["book_key"].isin(union_top)][["book_key", "title_key_basic"]].copy()
for label in comparison_labels:
    stability[label + "_rank"] = stability["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
stability["top15_branch_hits"] = sum((stability[label + "_rank"] <= 15).astype(int) for label in comparison_labels)
stability["rank_range"] = stability[[label + "_rank" for label in comparison_labels]].max(axis=1) - stability[[label + "_rank" for label in comparison_labels]].min(axis=1)
stability = stable_sort(stability, ["top15_branch_hits", "rank_range"], [False, True])
genre_dependency = stability.copy()
genre_dependency["genre_rank_loss"] = genre_dependency["no_genre_branch_rank"] - genre_dependency["lag_penalized_rank"]
genre_dependency = stable_sort(genre_dependency, "genre_rank_loss", False)

emit({
    "most_stable_candidates": top_records(
        stability,
        ["book_key", "title_key_basic", "top15_branch_hits", "rank_range",
         "lag_penalized_rank", "relative_demand_branch_rank", "pre_author_title_branch_rank",
         "movie_only_branch_rank", "no_lag_branch_rank", "no_genre_branch_rank"],
        5
    ),
    "most_genre_dependent": top_records(
        genre_dependency,
        ["book_key", "title_key_basic", "lag_penalized_rank", "no_genre_branch_rank", "genre_rank_loss"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 21:
# Context: Collaborative support is another possible weak point. Within the forty best current rows, re-run only the collaborative evidence with its earliest threshold, the 60-cleaned-ratings book cutoff and 50-cleaned-ratings user cutoff, while leaving the current demand, quality, title, coverage, lag, and content treatments intact. Focus the comparison on current-rank loss; zero-loss rows may complete the five-row audit table when needed.
# Question: Find five high-ranking rows that lose most under the earlier collaborative threshold. Show their current rank, comparison rank, and the component evidence behind the fragility.

# Depends on Task 10 (explicit rollback): collaborative evidence returns to earliest threshold only.
# Depends on Tasks 18 and 19 (partial inheritance): coverage and lag scoring stay current.
cf_early_again, _cf_early_state = build_cf_neighbors(stable_candidate_keys, book_min=60, user_min=50)
opportunity_cf_rollback = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    coverage_df=netflix_coverage,
    lag_df=lag_df,
    cf_df=cf_early_again,
    weights=lag_weights,
    label="cf_rollback_branch"
)
cf_fragility = opportunity_current.head(40)[[
    "book_key", "title_key_basic", "rank", "opportunity_score",
    "cf_component", "content_component", "demand_component", "quality_component"
]].copy()
cf_fragility["cf_rollback_rank"] = cf_fragility["book_key"].map(branch_rankings["cf_rollback_branch"]).fillna(len(books_audit) + 1)
cf_fragility["cf_rank_loss"] = cf_fragility["cf_rollback_rank"] - cf_fragility["rank"]
cf_fragility = stable_sort(cf_fragility, "cf_rank_loss", False)
audit_sets["cf_fragile_candidates"] = cf_fragility.head(12)["book_key"].tolist()

emit({
    "cf_fragile_candidates": top_records(
        cf_fragility,
        ["book_key", "title_key_basic", "rank", "cf_rollback_rank", "cf_rank_loss",
         "cf_component", "content_component", "demand_component", "quality_component"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 22:
# Context: Add source-risk diagnosis for the collaborative-fragile candidates. Restore full author strings, original titles, and star evidence for the same cleaned records, then decide whether each risk is mostly author ambiguity, title merging, rating skew, or collaborative instability. Prioritize the diagnosis by upper-quartile collaborative rank loss, then title merging, then more than two listed authors, then above-median absolute star-quality movement.
# Question: Create a compact risk attribution table for the five collaborative-fragile candidates with the largest collaborative rank loss, combining source-field restoration, title/author risk, star skew, and collaborative rank loss.

# Depends on Task 21 (definition inheritance): diagnose the fragile candidate set.
# Depends on Tasks 1, 4, and 14 (long-span source restoration): restore source fields and star evidence within the cleaned range.
fragile_evidence = books_audit[books_audit["book_key"].isin(audit_sets["cf_fragile_candidates"])].merge(
    raw_book_evidence, on=["book_key", "goodreads_book_id"], how="left", suffixes=("", "_raw")
)
fragile_evidence = fragile_evidence.merge(cf_fragility[["book_key", "rank", "cf_rollback_rank", "cf_rank_loss"]], on="book_key", how="left")
fragile_evidence["raw_title_count"] = fragile_evidence["title_key_basic"].map(raw_title_counts).fillna(1)
fragile_evidence = fragile_evidence.merge(title_risk_all[["book_key", "compact_title_count", "basic_title_count"]], on="book_key", how="left")
fragile_evidence["author_count"] = fragile_evidence["authors"].fillna("").str.count(",") + 1
fragile_evidence["star_skew_gap"] = fragile_evidence["quality_skew_score"] - fragile_evidence["quality_exposure_score"]
fragile_evidence["primary_risk"] = np.select(
    [
        fragile_evidence["cf_rank_loss"] >= fragile_evidence["cf_rank_loss"].quantile(0.75),
        fragile_evidence["compact_title_count"] > fragile_evidence["basic_title_count"],
        fragile_evidence["author_count"] > 2,
        fragile_evidence["star_skew_gap"].abs() > fragile_evidence["star_skew_gap"].abs().median()
    ],
    ["collaborative_instability", "title_merging", "author_ambiguity", "rating_skew"],
    default="mixed_low_risk"
)
fragile_evidence = stable_sort(fragile_evidence, "cf_rank_loss", False)

emit({
    "fragile_risk_attribution": top_records(
        fragile_evidence,
        ["book_key", "original_title", "authors", "rank", "cf_rollback_rank",
         "cf_rank_loss", "raw_title_count", "author_count", "star_skew_gap", "primary_risk"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 23:
# Context: The score checks now support a stability core. Use the main score together with the earlier demand, earlier title, earlier collaborative, Movie-only, no-genre, and no-lag checks. A book belongs to the core when it appears in the top 20 of at least five of those seven checks; assess core strength with hit count, average rank, and rank range.
# Question: Find the stable core across the seven scoring checks. Show five strongest core books with hit count, average rank, and rank range.

# Depends on Tasks 13, 17, 19, 20, and 21 (multi-branch long-span composition).
core_labels = ["lag_penalized", "relative_demand_branch", "pre_author_title_branch", "cf_rollback_branch", "movie_only_branch", "no_genre_branch", "no_lag_branch"]
core_union = set()
for label in core_labels:
    core_union |= set(branch_tables[label].head(20)["book_key"])
core = books_audit[books_audit["book_key"].isin(core_union)][["book_key", "title_key_basic"]].copy()
for label in core_labels:
    core[label + "_rank"] = core["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
rank_cols = [label + "_rank" for label in core_labels]
core["top20_hits"] = sum((core[c] <= 20).astype(int) for c in rank_cols)
core["avg_branch_rank"] = core[rank_cols].mean(axis=1)
core["rank_range"] = core[rank_cols].max(axis=1) - core[rank_cols].min(axis=1)
stable_core = stable_sort(core[core["top20_hits"] >= 5], ["top20_hits", "avg_branch_rank", "rank_range"], [False, True, True])
audit_sets["stable_core"] = stable_core["book_key"].head(20).tolist()

emit({
    "stable_core_top5": top_records(
        stable_core,
        ["book_key", "title_key_basic", "top20_hits", "avg_branch_rank", "rank_range",
         "lag_penalized_rank", "relative_demand_branch_rank", "pre_author_title_branch_rank",
         "cf_rollback_branch_rank", "movie_only_branch_rank", "no_genre_branch_rank", "no_lag_branch_rank"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 24:
# Context: Some high-ranking books do not enter the stability core. Explain those failures through components and score-check behavior. Attribute failure first to top-quartile rank volatility, then title risk, then already-covered status, then weak recommendation consistency, weak demand, or weak quality according to the weakest visible component.
# Question: Among high-scoring books outside the stability core, show five strongest non-core books and the main reason each one failed to stabilize: demand, quality, recommendation consistency, coverage, title risk, or rank volatility.

# Depends on Task 23 (definition inheritance): explain exclusions from the stable core.
# Depends on Tasks 18, 20, and 22 (multi-hop attribution): use components, branch stability, and source-risk diagnostics.
stable_set = set(audit_sets["stable_core"])
outside = opportunity_current.head(60)[~opportunity_current.head(60)["book_key"].isin(stable_set)].copy()
outside = outside.merge(core[["book_key", "top20_hits", "rank_range"]], on="book_key", how="left")
outside[["top20_hits", "rank_range"]] = outside[["top20_hits", "rank_range"]].fillna(0)
outside = outside.merge(title_risk_all[["book_key", "title_risk_score"]], on="book_key", how="left")
outside["title_risk_score"] = outside["title_risk_score"].fillna(0)
component_cols = ["demand_component", "quality_component", "content_component", "cf_component", "coverage_gap_component"]
outside["weakest_component"] = outside[component_cols].idxmin(axis=1).str.replace("_component", "")
outside["failure_reason"] = np.select(
    [
        outside["rank_range"] > outside["rank_range"].quantile(0.75),
        outside["title_risk_score"] > 0,
        outside["weakest_component"].eq("coverage_gap"),
        outside["weakest_component"].eq("cf") | outside["weakest_component"].eq("content"),
        outside["weakest_component"].eq("demand"),
        outside["weakest_component"].eq("quality")
    ],
    ["branch_volatility", "title_risk", "already_covered", "weak_recommendation_consistency", "weak_demand", "weak_quality"],
    default="mixed_component_gap"
)
outside = stable_sort(outside, ["opportunity_score", "top20_hits"], [False, True])

emit({
    "high_score_exclusions": top_records(
        outside,
        ["book_key", "title_key_basic", "rank", "opportunity_score", "top20_hits",
         "rank_range", "weakest_component", "title_risk_score", "failure_reason"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 25:
# Context: Combine two earlier business choices: relative demand and the title rule from before author disambiguation. Keep the current quality, collaborative evidence, Netflix coverage framework, and lag treatment. Compare movements across books appearing in the first thirty rows of either ranking to isolate sensitivity to those choices together.
# Question: Recompute final candidates with relative demand and the pre-disambiguation title treatment while keeping the other current states. Show five leading candidates and five main movements versus the main stable score.

# Depends on Task 13 (explicit rollback): demand returns to the middle relative treatment.
# Depends on Task 17 (explicit rollback): title/coverage returns to the pre-author-disambiguation treatment.
# Depends on Tasks 14, 19, and 21 (partial inheritance): quality, lag, and CF treatment stay current.
opportunity_middle_joint = compute_opportunity(
    demand_col="demand_relative_score",
    quality_col=current_quality_col,
    coverage_df=pre_author_coverage,
    lag_df=lag_df,
    weights=lag_weights,
    label="middle_demand_middle_title"
)
movement_keys = list(set(opportunity_current.head(30)["book_key"]) | set(opportunity_middle_joint.head(30)["book_key"]))
middle_move = books_audit[books_audit["book_key"].isin(movement_keys)][["book_key", "title_key_basic"]].copy()
middle_move["main_rank"] = middle_move["book_key"].map(branch_rankings["lag_penalized"]).fillna(len(books_audit) + 1)
middle_move["middle_joint_rank"] = middle_move["book_key"].map(branch_rankings["middle_demand_middle_title"]).fillna(len(books_audit) + 1)
middle_move["rank_shift"] = middle_move["main_rank"] - middle_move["middle_joint_rank"]
middle_move = stable_abs_sort(middle_move, "rank_shift", False)

emit({
    "middle_joint_top5": top_records(
        opportunity_middle_joint,
        ["book_key", "title_key_basic", "rank", "opportunity_score",
         "demand_component", "quality_component", "coverage_gap_component", "lag_penalty_component"],
        5
    ),
    "largest_joint_movements": top_records(
        middle_move,
        ["book_key", "title_key_basic", "main_rank", "middle_joint_rank", "rank_shift"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 26:
# Context: Compare the stable core from the main ranking with the first twenty rows under the combined earlier choices, using both membership and rank. Use component changes and earlier signal flags to explain which books enter, exit, or remain in that combined-choice top group.
# Question: Compare the main stable core with the combined-choice top group. Show five entering, exiting, or retained examples with component changes and whether they hit the early high-intent, short-title, or scarce-genre signals.

# Depends on Task 23 (definition inheritance): main stable core.
# Depends on Task 25 (definition update): middle-choice branch.
# Depends on Tasks 1, 2, and 5 (long-span anchors): early high-intent, short-title, and scarce-genre signals.
middle_core_candidates = opportunity_middle_joint.head(20)["book_key"].tolist()
main_core_set = set(audit_sets["stable_core"][:20])
middle_core_set = set(middle_core_candidates)
change_set = main_core_set | middle_core_set
core_change = books_audit[books_audit["book_key"].isin(change_set)][["book_key", "title_key_basic", "short_title_flag", "genre_rarity_initial"]].copy()
core_change["core_change"] = np.select(
    [core_change["book_key"].isin(main_core_set & middle_core_set),
     core_change["book_key"].isin(middle_core_set - main_core_set),
     core_change["book_key"].isin(main_core_set - middle_core_set)],
    ["retained", "entered_middle_core", "exited_middle_core"],
    default="outside"
)
core_change["main_rank"] = core_change["book_key"].map(branch_rankings["lag_penalized"]).fillna(len(books_audit) + 1)
core_change["middle_rank"] = core_change["book_key"].map(branch_rankings["middle_demand_middle_title"]).fillna(len(books_audit) + 1)
core_change["rank_shift"] = core_change["main_rank"] - core_change["middle_rank"]
main_components = opportunity_current.set_index("book_key")[["opportunity_score", "demand_component", "coverage_gap_component"]]
middle_components = opportunity_middle_joint.set_index("book_key")[["opportunity_score", "demand_component", "coverage_gap_component"]]
core_change["main_score"] = core_change["book_key"].map(main_components["opportunity_score"]).fillna(0)
core_change["middle_score"] = core_change["book_key"].map(middle_components["opportunity_score"]).fillna(0)
core_change["score_change"] = core_change["middle_score"] - core_change["main_score"]
core_change["demand_component_change"] = (
    core_change["book_key"].map(middle_components["demand_component"]).fillna(0)
    - core_change["book_key"].map(main_components["demand_component"]).fillna(0)
)
core_change["coverage_component_change"] = (
    core_change["book_key"].map(middle_components["coverage_gap_component"]).fillna(0)
    - core_change["book_key"].map(main_components["coverage_gap_component"]).fillna(0)
)
core_change["early_high_intent"] = core_change["book_key"].isin(early_high_demand)
core_change["early_scarce_genre"] = core_change["genre_rarity_initial"] >= early_rare_genre_cut
core_change = stable_sort(core_change, ["core_change", "rank_shift"], [True, False])

emit({
    "core_change_examples": top_records(
        core_change,
        ["book_key", "title_key_basic", "core_change", "main_rank", "middle_rank",
         "rank_shift", "score_change", "demand_component_change", "coverage_component_change",
         "early_high_intent", "short_title_flag", "early_scarce_genre"],
        5
    ),
    "change_counts": core_change["core_change"].value_counts().to_dict()
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 27:
# Context: Prepare a compact final review table with current rank, behavior across important checks, and remaining source-level risks. Use the first dozen books from the stable core as the review set, and report the strongest five rows.
# Question: Create the final audit output for five strongest candidates: current rank, important check ranks, early-signal flags, title and author risk, star-skew evidence, and Netflix coverage path.

# Depends on Task 23 (definition inheritance): strongest candidates come from the stable core.
# Depends on Tasks 15, 22, and 26 (multi-hop composition): include coverage path, source risk, and early-signal flags.
final_candidate_keys = audit_sets["stable_core"][:12]
final_audit = opportunity_current[opportunity_current["book_key"].isin(final_candidate_keys)].copy()
for label in ["relative_demand_branch", "pre_author_title_branch", "cf_rollback_branch", "movie_only_branch", "no_genre_branch", "no_lag_branch", "middle_demand_middle_title"]:
    final_audit[label + "_rank"] = final_audit["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
if "coverage_path" not in final_audit.columns:
    final_audit = final_audit.merge(netflix_coverage[["book_key", "coverage_path"]], on="book_key", how="left")
else:
    final_audit["coverage_path"] = final_audit["coverage_path"].fillna("uncovered")
final_audit = final_audit.merge(title_risk_all[["book_key", "title_risk_score"]], on="book_key", how="left")
final_audit["title_risk_score"] = final_audit["title_risk_score"].fillna(0)
final_audit = final_audit.merge(raw_book_evidence[["book_key", "authors"]], on="book_key", how="left")
final_audit["author_count"] = final_audit["authors"].fillna("").str.count(",") + 1
final_audit["early_high_intent"] = final_audit["book_key"].isin(early_high_demand)
final_audit["early_scarce_genre"] = final_audit["genre_component"] >= final_audit["genre_component"].quantile(0.75)
final_audit["star_skew_gap"] = final_audit["book_key"].map(books_audit.set_index("book_key")["quality_skew_score"] - books_audit.set_index("book_key")["quality_exposure_score"])
final_audit = stable_sort(final_audit, "rank", True)

emit({
    "final_candidate_audit": top_records(
        final_audit,
        ["book_key", "title_key_basic", "rank", "opportunity_score",
         "relative_demand_branch_rank", "pre_author_title_branch_rank", "cf_rollback_branch_rank",
         "movie_only_branch_rank", "no_genre_branch_rank", "no_lag_branch_rank",
         "early_high_intent", "early_scarce_genre", "title_risk_score", "author_count",
         "star_skew_gap", "coverage_path"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 28:
# Context: Stress-test the first-dozen stable-core review set by setting each major component contribution to zero in turn without renormalizing the remaining weights, then check how much each candidate's rank deteriorates. Mark a candidate for lower manual-review priority when the largest rank loss exceeds 20 positions.
# Question: Run the single-component removal stress test for the first-dozen final review set. Show five candidates most dependent on one component, and flag any manual-review priority downgrades.

# Depends on Task 27 (definition inheritance): stress-test the final candidate set.
# Depends on Task 18 (counterfactual composition): remove one scoring component at a time without changing the main state.
component_weights = dict(lag_weights)
component_labels = ["demand", "quality", "genre", "content", "cf", "coverage_gap", "lag_penalty"]
pressure_rows = []
for comp in component_labels:
    w = dict(component_weights)
    w[comp] = 0.0
    label = f"zero_{comp}"
    table = compute_opportunity(
        demand_col=current_demand_col,
        quality_col=current_quality_col,
        coverage_df=netflix_coverage,
        lag_df=lag_df,
        weights=w,
        label=label
    )
for key in final_candidate_keys:
    base_rank = branch_rankings["lag_penalized"].get(key, len(books_audit) + 1)
    losses = {comp: branch_rankings[f"zero_{comp}"].get(key, len(books_audit) + 1) - base_rank for comp in component_labels}
    worst_comp = max(losses, key=losses.get)
    pressure_rows.append({
        "book_key": int(key),
        "title_key_basic": books_audit.set_index("book_key").loc[key, "title_key_basic"],
        "base_rank": base_rank,
        "largest_rank_loss": losses[worst_comp],
        "most_dependent_component": worst_comp,
        "downgrade_for_manual_review": bool(losses[worst_comp] > 20)
    })
component_pressure = stable_sort(pd.DataFrame(pressure_rows), "largest_rank_loss", False)

emit({
    "component_pressure": top_records(
        component_pressure,
        ["book_key", "title_key_basic", "base_rank", "largest_rank_loss",
         "most_dependent_component", "downgrade_for_manual_review"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 29:
# Context: Some books remain high opportunity despite source risk or component dependence. Draw the human-review queue from candidates already flagged for component-driven rank loss, candidates surfaced by the earlier collaborative-threshold fragility audit, and stable candidates carrying title-alignment risk. Prioritize them only when the remaining unmet value is large enough to justify the risk. The manual-priority score is 30% current opportunity, 20% uncovered-gap credit, 15% genre-scarcity value, 20% stability-hit strength, minus 10% of the component stress-test rank loss and 5% title-risk score.
# Question: Build the manual-review priority list for high-risk but high-opportunity candidates. Show five strongest priorities after balancing stability, component dependence, title/author risk, coverage gap, and scarce-genre value.

# Depends on Task 28 (definition inheritance): use component-dependence results.
# Depends on Tasks 22, 23, and 27 (multi-hop composition): combine source risk, stability, and coverage gap.
high_risk_keys = set(component_pressure[component_pressure["downgrade_for_manual_review"]]["book_key"]) | set(audit_sets["cf_fragile_candidates"]) | set(audit_sets["title_risky_stable"])
manual = opportunity_current[opportunity_current["book_key"].isin(high_risk_keys)].copy()
manual = manual.merge(component_pressure[["book_key", "largest_rank_loss", "most_dependent_component"]], on="book_key", how="left")
manual = manual.merge(title_risk_all[["book_key", "title_risk_score"]], on="book_key", how="left")
manual[["largest_rank_loss", "title_risk_score"]] = manual[["largest_rank_loss", "title_risk_score"]].fillna(0)
manual["most_dependent_component"] = manual["most_dependent_component"].fillna("not_component_pressure_top")
manual["branch_hit_score"] = manual["book_key"].map(core.set_index("book_key")["top20_hits"]).fillna(0) * (100 / len(core_labels))
manual["manual_priority_score"] = (
    0.30 * manual["opportunity_score"]
    + 0.20 * manual["coverage_gap_component"]
    + 0.15 * manual["genre_component"]
    + 0.20 * manual["branch_hit_score"]
    - 0.10 * manual["largest_rank_loss"]
    - 0.05 * manual["title_risk_score"]
)
manual = stable_sort(manual, "manual_priority_score", False)
audit_sets["manual_priority_candidates"] = manual.head(20)["book_key"].tolist()

emit({
    "manual_review_priority": top_records(
        manual,
        ["book_key", "title_key_basic", "rank", "manual_priority_score", "opportunity_score",
         "coverage_gap_component", "genre_component", "branch_hit_score",
         "largest_rank_loss", "most_dependent_component", "title_risk_score"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 30:
# Context: Check whether the high-priority review list over-concentrates in a few authors or genres. Use the top 20 current priority rows, and report raw counts for the three most frequent authors and three most frequent genres. Then recalculate scarcity with the earliest genre-supply treatment within the same manual-priority candidate set while leaving the rest of the scoring state intact. Treat the concentration conclusion as changed only if the leading genre changes.
# Question: Assess author and genre concentration in the manual-review priorities, then test whether the leading-genre concentration remains under the early genre-scarcity treatment. Show the three main author and genre concentration risks and whether the leading-genre conclusion changes.

# Depends on Task 29 (definition inheritance): inspect manual-priority candidates.
# Depends on Task 1 (explicit rollback): genre scarcity returns to the earliest supply treatment only.
priority = manual.head(20).merge(raw_book_evidence[["book_key", "authors"]], on="book_key", how="left")
priority["lead_author_display"] = priority["authors"].fillna("").str.split(",").str[0]
author_concentration = (priority.groupby("lead_author_display").size().reset_index(name="count")
                        .sort_values(["count", "lead_author_display"], ascending=[False, True], kind="mergesort")
                        .set_index("lead_author_display")["count"])
genre_concentration = (priority.groupby("main_genre").size().reset_index(name="count")
                       .sort_values(["count", "main_genre"], ascending=[False, True], kind="mergesort")
                       .set_index("main_genre")["count"])

books_audit["genre_rarity_early_recheck"] = books_audit["genre_rarity_initial"]
opportunity_early_genre = compute_opportunity(
    demand_col=current_demand_col,
    quality_col=current_quality_col,
    genre_col="genre_rarity_early_recheck",
    coverage_df=netflix_coverage,
    lag_df=lag_df,
    weights=lag_weights,
    label="early_genre_rollback"
)
early_priority = opportunity_early_genre[opportunity_early_genre["book_key"].isin(audit_sets["manual_priority_candidates"])].head(20)
genre_before_top = genre_concentration.index[0] if len(genre_concentration) else None
genre_after_counts = (early_priority.groupby("main_genre").size().reset_index(name="count")
                      .sort_values(["count", "main_genre"], ascending=[False, True], kind="mergesort"))
genre_after_top = genre_after_counts.iloc[0]["main_genre"] if len(genre_after_counts) else None

emit({
    "top_author_concentration": author_concentration.head(3).to_dict(),
    "top_genre_concentration": genre_concentration.head(3).to_dict(),
    "early_genre_rollback_top_genre": genre_after_top,
    "concentration_conclusion_changed": bool(genre_before_top != genre_after_top)
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 31:
# Context: Some covered books remain high opportunity. Within the covered subset, take the fifty best-ranked books under the current score and determine whether each score reflects genuine remaining demand after coverage, recommendation support, or scoring residue. Use a residue check equal to the current score minus half of demand, three tenths of content support, and two tenths of collaborative support; classify first by upper-quartile demand within those fifty covered rows, then upper-quartile combined recommendation support, then above-median residue.
# Question: Among the best-ranked covered books, identify whether the score is driven by real remaining demand, recommendation support, or scoring residue. Show five strongest reverse-audit cases.

# Depends on Task 18 (definition inheritance): coverage-aware opportunity score.
# Depends on Tasks 13, 20, and 23 (multi-hop attribution): use demand smoothing, branch stability, and score components.
covered_high = opportunity_current[opportunity_current["covered_author_safe"]].head(50).copy()
covered_high["branch_hits"] = covered_high["book_key"].map(core.set_index("book_key")["top20_hits"]).fillna(0)
covered_high["residue_score"] = (
    covered_high["opportunity_score"]
    - 0.5 * covered_high["demand_component"]
    - 0.3 * covered_high["content_component"]
    - 0.2 * covered_high["cf_component"]
)
covered_high["reverse_audit_reason"] = np.select(
    [
        covered_high["demand_component"] >= covered_high["demand_component"].quantile(0.75),
        (covered_high["content_component"] + covered_high["cf_component"]) >= (covered_high["content_component"] + covered_high["cf_component"]).quantile(0.75),
        covered_high["residue_score"] > covered_high["residue_score"].median()
    ],
    ["real_remaining_demand", "strong_recommendation_support", "scoring_residue"],
    default="mixed"
)
covered_high = stable_sort(covered_high, ["opportunity_score", "branch_hits"], [False, False])

emit({
    "covered_high_opportunity_cases": top_records(
        covered_high,
        ["book_key", "title_key_basic", "rank", "opportunity_score", "demand_component",
         "content_component", "cf_component", "branch_hits", "residue_score", "reverse_audit_reason"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 32:
# Context: Separate uncovered stable candidates into immediate adaptation prospects and cases needing title or source review first. Use the lag-aware score, smoothed demand, recommendation consistency, title risk, and star-skew evidence together. Tier score is 30% demand, 25% average recommendation consistency, 20% current quality, 15% genre scarcity, minus 5% title risk and 5% absolute star-quality movement; split tiers at the median and 80th percentile of that score.
# Question: Segment uncovered stable candidates into adaptation-potential tiers. Show tier counts and five strongest tier examples with the factors that determine their tier.

# Depends on Tasks 23 and 31 (definition inheritance): focus on stable candidates that are not author-safe covered.
# Depends on Tasks 14, 15, and 19 (multi-hop composition): use star skew, coverage, and lag-aware opportunity.
uncovered_stable = opportunity_current[
    opportunity_current["book_key"].isin(audit_sets["stable_core"])
    & (~opportunity_current["covered_author_safe"])
].copy()
uncovered_stable = uncovered_stable.merge(title_risk_all[["book_key", "title_risk_score"]], on="book_key", how="left")
uncovered_stable["title_risk_score"] = uncovered_stable["title_risk_score"].fillna(0)
uncovered_stable["star_skew_gap"] = uncovered_stable["book_key"].map(books_audit.set_index("book_key")["quality_skew_score"] - books_audit.set_index("book_key")["quality_exposure_score"])
uncovered_stable["recommendation_consistency"] = (uncovered_stable["content_component"] + uncovered_stable["cf_component"]) / 2
uncovered_stable["tier_score"] = (
    0.30 * uncovered_stable["demand_component"]
    + 0.25 * uncovered_stable["recommendation_consistency"]
    + 0.20 * uncovered_stable["quality_component"]
    + 0.15 * uncovered_stable["genre_component"]
    - 0.05 * uncovered_stable["title_risk_score"]
    - 0.05 * uncovered_stable["star_skew_gap"].abs()
)
uncovered_stable["tier"] = pd.cut(
    uncovered_stable["tier_score"],
    bins=[-np.inf, uncovered_stable["tier_score"].quantile(0.50), uncovered_stable["tier_score"].quantile(0.80), np.inf],
    labels=["review_first", "strong", "immediate"]
)
uncovered_stable = stable_sort(uncovered_stable, ["tier", "tier_score"], [False, False])

emit({
    "tier_counts": uncovered_stable["tier"].astype(str).value_counts().to_dict(),
    "tier_examples": top_records(
        uncovered_stable,
        ["book_key", "title_key_basic", "tier", "tier_score", "demand_component",
         "recommendation_consistency", "quality_component", "genre_component", "title_risk_score", "star_skew_gap"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 33:
# Context: Quantify rank stability across the major historical and sensitivity checks: main lag-aware score, absolute-demand and relative-demand checks, earlier-title check, earlier-collaborative check, Movie-only, no-genre, no-lag, combined relative-demand/title check, and early-genre check. The matrix includes every book appearing in the first thirty rows of at least one check. Attribute rank volatility to the largest current component among demand, quality, genre, content, collaborative support, coverage gap, and lag penalty.
# Question: Build the ranking-stability matrix across the major checks. Show five stable and five volatile candidates, along with the component most associated with each volatile movement.

# Depends on Tasks 13, 17, 19, 20, 21, 25, and 30 (long-span multi-branch composition).
matrix_labels = ["lag_penalized", "relative_demand_branch", "absolute_demand_branch", "pre_author_title_branch",
                 "cf_rollback_branch", "movie_only_branch", "no_genre_branch", "no_lag_branch",
                 "middle_demand_middle_title", "early_genre_rollback"]
matrix_union = set()
for label in matrix_labels:
    matrix_union |= set(branch_tables[label].head(30)["book_key"])
rank_matrix = books_audit[books_audit["book_key"].isin(matrix_union)][["book_key", "title_key_basic"]].copy()
for label in matrix_labels:
    rank_matrix[label + "_rank"] = rank_matrix["book_key"].map(branch_rankings[label]).fillna(len(books_audit) + 1)
rank_cols = [label + "_rank" for label in matrix_labels]
rank_matrix["rank_mean"] = rank_matrix[rank_cols].mean(axis=1)
rank_matrix["rank_std"] = rank_matrix[rank_cols].std(axis=1, ddof=0)
rank_matrix["rank_range"] = rank_matrix[rank_cols].max(axis=1) - rank_matrix[rank_cols].min(axis=1)
component_snapshot = opportunity_current.set_index("book_key")[["demand_component", "quality_component", "genre_component", "content_component", "cf_component", "coverage_gap_component", "lag_penalty_component"]]
rank_matrix = rank_matrix.merge(component_snapshot, left_on="book_key", right_index=True, how="left")
rank_matrix["volatile_component"] = rank_matrix[["demand_component", "quality_component", "genre_component", "content_component", "cf_component", "coverage_gap_component", "lag_penalty_component"]].idxmax(axis=1).str.replace("_component", "")
most_stable = stable_sort(rank_matrix, ["rank_std", "rank_mean"], [True, True])
most_volatile = stable_sort(rank_matrix, "rank_std", False)

emit({
    "most_stable": top_records(
        most_stable,
        ["book_key", "title_key_basic", "rank_mean", "rank_std", "rank_range"],
        5
    ),
    "most_volatile": top_records(
        most_volatile,
        ["book_key", "title_key_basic", "rank_mean", "rank_std", "rank_range", "volatile_component"],
        5
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 34:
# Context: Before finalizing, test whether the conclusion changes when candidates with the highest source-risk flags are excluded one risk type at a time. Within the current lag-aware ranking, remove the top decile of title risk, author ambiguity, and absolute star-quality movement separately. Compare top-ten sets with Jaccard overlap using the union of original and filtered top-ten sets as denominator, and treat the conclusion as sensitive when fewer than eight original top-ten candidates remain after an exclusion.
# Question: Exclude high title-risk, high author-ambiguity, and high star-skew candidates one group at a time. Show how the top recommendation set changes and whether the final conclusion is sensitive to any exclusion.

# Depends on Tasks 22, 27, and 33 (counterfactual exclusion audit): remove source-risk groups one at a time.
source_risk = opportunity_current.copy()
source_risk = source_risk.merge(title_risk_all[["book_key", "title_risk_score", "author_count"]], on="book_key", how="left")
source_risk["star_skew_gap"] = source_risk["book_key"].map(books_audit.set_index("book_key")["quality_skew_score"] - books_audit.set_index("book_key")["quality_exposure_score"])
source_risk[["title_risk_score", "author_count", "star_skew_gap"]] = source_risk[["title_risk_score", "author_count", "star_skew_gap"]].fillna(0)
base_top = set(source_risk.head(10)["book_key"])
exclusion_rows = []
remove_n = max(1, int(math.ceil(0.10 * len(source_risk))))
risk_scores = {
    "high_title_risk": source_risk["title_risk_score"],
    "high_author_ambiguity": source_risk["author_count"],
    "high_star_skew": source_risk["star_skew_gap"].abs(),
}
rules = {name: source_risk.index.isin(score.nlargest(remove_n, keep="first").index) for name, score in risk_scores.items()}
for rule, mask in rules.items():
    filtered = stable_sort(source_risk[~mask], "opportunity_score", False)
    new_top = set(filtered.head(10)["book_key"])
    exclusion_rows.append({
        "exclusion": rule,
        "removed_count": int(mask.sum()),
        "top10_jaccard_pct": 100 * len(base_top & new_top) / len(base_top | new_top),
        "new_top_book": int(filtered.iloc[0]["book_key"]) if len(filtered) else None,
        "new_top_title": filtered.iloc[0]["title_key_basic"] if len(filtered) else None,
        "conclusion_sensitive": bool(len(base_top & new_top) < 8)
    })
exclusion_sensitivity = pd.DataFrame(exclusion_rows).sort_values(["top10_jaccard_pct", "exclusion"], ascending=[True, True], kind="mergesort")

emit({
    "exclusion_sensitivity": top_records(
        exclusion_sensitivity,
        ["exclusion", "removed_count", "top10_jaccard_pct", "new_top_book", "new_top_title", "conclusion_sensitive"],
        3
    )
})

# Structured JSON output: emit(...) prints with json.dumps(...).

###### Task 35:
# Context: Synthesize the main ranking, score stability, component evidence, exclusion sensitivity, and reasons why high-scoring alternatives were not selected. Retain books by the first applicable reason among lowest-quartile rank volatility, clear uncovered gap, strong cross-model support, strong unmet demand, or otherwise balanced evidence. Keep the output compact.
# Question: Give the final five robust adaptation candidates. For each, report current rank, average check rank, component contributions, coverage path, and the main reason it is retained. Also list the strongest excluded high-score candidate and why it was excluded, plus how many source-risk exclusion checks changed the conclusion.

# Depends on Tasks 23, 24, 27, 33, and 34 (terminal multi-hop synthesis): combine stability, attribution, and sensitivity.
final = opportunity_current[opportunity_current["book_key"].isin(audit_sets["stable_core"])].copy()
final = final.merge(rank_matrix[["book_key", "rank_mean", "rank_std", "rank_range"]], on="book_key", how="left")
if "coverage_path" not in final.columns:
    final = final.merge(netflix_coverage[["book_key", "coverage_path"]], on="book_key", how="left")
else:
    final["coverage_path"] = final["coverage_path"].fillna("uncovered")
final["retention_reason"] = np.select(
    [
        final["rank_std"] <= final["rank_std"].quantile(0.25),
        final["coverage_gap_component"] >= 90,
        (final["content_component"] + final["cf_component"]) / 2 >= 75,
        final["demand_component"] >= 90
    ],
    ["stable_across_branches", "clear_uncovered_gap", "cross_model_support", "strong_unmet_demand"],
    default="balanced_evidence"
)
final = stable_sort(final, ["rank_mean", "rank_std", "opportunity_score"], [True, True, False])
excluded_best = stable_sort(outside, "opportunity_score", False).head(1)

emit({
    "robust_candidates": top_records(
        final,
        ["book_key", "title_key_basic", "rank", "rank_mean", "rank_std",
         "opportunity_score", "demand_component", "quality_component", "genre_component",
         "content_component", "cf_component", "coverage_path", "retention_reason"],
        5
    ),
    "strongest_excluded_high_score": top_records(
        excluded_best,
        ["book_key", "title_key_basic", "rank", "opportunity_score", "top20_hits",
         "failure_reason", "weakest_component"],
        1
    ),
    "sensitive_exclusion_checks": int(exclusion_sensitivity["conclusion_sensitive"].sum())
})

# Structured JSON output: emit(...) prints with json.dumps(...).

