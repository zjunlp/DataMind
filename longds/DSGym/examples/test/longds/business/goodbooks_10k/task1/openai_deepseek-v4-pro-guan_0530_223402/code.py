############## turn 1
import os

data_dir = '/data/longds/business/goodbooks_10k/task1/data'
print("Files in data directory:")
for f in os.listdir(data_dir):
    fpath = os.path.join(data_dir, f)
    size = os.path.getsize(fpath)
    print(f"{f} ({size} bytes)")

import os

data_dir = '/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k'
for root, dirs, files in os.walk(data_dir):
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        print(f"{os.path.relpath(fpath, data_dir)} ({size} bytes)")

import pandas as pd

books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
print("books.csv columns:", books.columns.tolist())
print("Shape:", books.shape)
print(books.head(2).to_string())
print("\nDtypes:")
print(books.dtypes)

ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
print("ratings.csv columns:", ratings.columns.tolist())
print("Shape:", ratings.shape)
print(ratings.head(5).to_string())
print(ratings.dtypes)

to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
print("to_read.csv columns:", to_read.columns.tolist())
print("Shape:", to_read.shape)
print(to_read.head(5).to_string())

book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
print("book_tags.csv columns:", book_tags.columns.tolist())
print("Shape:", book_tags.shape)
print(book_tags.head(5).to_string())

tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')
print("tags.csv columns:", tags.columns.tolist())
print("Shape:", tags.shape)
print(tags.head(10).to_string())
print(tags.dtypes)

# Check nulls in critical books columns
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
print("Null counts in critical books columns:")
for c in critical_cols:
    print(f"  {c}: {books[c].isna().sum()} / {len(books)}")

# Check for duplicate original_titles
print(f"\nTotal books: {len(books)}")
print(f"Unique original_title: {books['original_title'].nunique()}")
print(f"Duplicate original_title count: {books['original_title'].duplicated(keep=False).sum()}")

# Check ratings duplicates
print(f"\nRatings shape: {ratings.shape}")
dupes = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
print(f"Ratings rows with duplicate (user_id, book_id): {dupes.sum()}")

import pandas as pd
import numpy as np

# Load all data
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# Step 1: Remove duplicate (user_id, book_id) pairs in ratings
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
print(f"Ratings rows to remove (duplicate pairs): {dup_mask.sum()}")
ratings_clean = ratings[~dup_mask].copy()
print(f"Ratings rows remaining: {len(ratings_clean)}")

# Step 2: Remove books with missing critical columns
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
initial_books = len(books)
books_clean = books.dropna(subset=critical_cols).copy()
print(f"Books after dropping missing critical columns: {len(books_clean)} (dropped {initial_books - len(books_clean)})")

# Step 3: Remove all books sharing duplicate original_title
# Identify duplicate original_titles (keep=False marks all occurrences)
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
print(f"Books after removing duplicate original_title: {len(books_clean)} (dropped {dupe_titles.sum()})")

import pandas as pd
import numpy as np

# Re-load fresh
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

import pandas as pd
import numpy as np
import re

# Load all data
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# --- Clean ratings: remove duplicate (user_id, book_id) pairs entirely ---
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
ratings_clean = ratings[~dup_mask].copy()

# --- Clean books ---
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
books_clean = books.dropna(subset=critical_cols).copy()
# Remove books with duplicate original_title (keep none)
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
books_clean['book_key'] = books_clean['id']

# --- Title keys ---
# Early key: lowercase and collapse spaces (no accent folding)
def early_key(title):
    if not isinstance(title, str):
        return None
    return re.sub(r'\s+', ' ', title.lower().strip())

# Stricter key: remove parenthetical text, remove standalone articles,
# replace non-alphanumeric with spaces, lowercase, collapse spaces
def strict_key(title):
    if not isinstance(title, str):
        return None
    t = title.lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)  # remove parenthetical text
    # remove standalone articles (the, a, an) as whole words
    t = re.sub(r'\b(the|a|an)\b', '', t, flags=re.IGNORECASE)
    # replace non-alphanumeric with spaces
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

books_clean['title_key_early'] = books_clean['original_title'].apply(early_key)
books_clean['title_key_strict'] = books_clean['original_title'].apply(strict_key)

# --- Reading intent count ---
intent_counts = to_read['book_id'].value_counts().reset_index()
intent_counts.columns = ['id', 'reading_intent_count']  # to_read.book_id = books.id

# --- Genre processing ---
# Normalize tag names: lowercase, underscores to hyphens
tags['tag_norm'] = tags['tag_name'].str.lower().str.replace('_', '-')

# Define canonical families mapping
genre_families = {
    'art': 'art',
    'biography': 'biography',
    'business': 'business',
    'chick-lit': 'chick_lit',
    'children': 'children',
    'childrens': 'children',
    'christian': 'christian',
    'classics': 'classics',
    'comics': 'comics',
    'contemporary': 'contemporary',
    'cookbooks': 'cookbooks',
    'crime': 'crime',
    'ebooks': 'ebooks',
    'fantasy': 'fantasy',
    'fiction': 'fiction',
    'historical-fiction': 'historical_fiction',
    'history': 'history',
    'horror': 'horror',
    'humor': 'humor',
    'manga': 'manga',
    'memoir': 'memoir',
    'music': 'music',
    'mystery': 'mystery',
    'nonfiction': 'nonfiction',
    'paranormal': 'paranormal',
    'philosophy': 'philosophy',
    'poetry': 'poetry',
    'romance': 'romance',
    'science': 'science',
    'science-fiction': 'science_fiction',
    'self-help': 'self_help',
    'sports': 'sports',
    'suspense': 'suspense',
    'thriller': 'thriller',
    'travel': 'travel',
    'young-adult': 'young_adult'
}

tags['canonical_genre'] = tags['tag_norm'].map(genre_families)
canonical_tags = tags.dropna(subset=['canonical_genre'])

# Merge book_tags with canonical tags
book_tags_genre = book_tags.merge(canonical_tags[['tag_id', 'canonical_genre']], on='tag_id', how='inner')

# Sum counts per goodreads_book_id and canonical genre
genre_sums = book_tags_genre.groupby(['goodreads_book_id', 'canonical_genre'])['count'].sum().reset_index()

# Choose main genre: max sum, break ties alphabetically by canonical genre
# Sort by count descending then genre ascending, pick first
genre_sums = genre_sums.sort_values(['goodreads_book_id', 'count', 'canonical_genre'], 
                                     ascending=[True, False, True])
main_genres = genre_sums.groupby('goodreads_book_id').first().reset_index()
main_genres = main_genres[['goodreads_book_id', 'canonical_genre']].rename(
    columns={'canonical_genre': 'main_genre'})

# Merge genres to books: book_tags.goodreads_book_id joins to books.book_id
books_clean = books_clean.merge(main_genres, left_on='book_id', right_on='goodreads_book_id', how='left')
books_clean.drop(columns='goodreads_book_id', inplace=True)

# Merge reading intent counts
books_clean = books_clean.merge(intent_counts, left_on='id', right_on='id', how='left')
books_clean['reading_intent_count'] = books_clean['reading_intent_count'].fillna(0)

# --- Percentile ranks on cleaned table ---
# demand_absolute_score = percentile rank of reading_intent_count
books_clean['demand_absolute_score'] = books_clean['reading_intent_count'].rank(pct=True, method='average') * 100

# exposure_gap_score = percentile rank of -ratings_count (lower count stronger)
books_clean['exposure_gap_score'] = (-books_clean['ratings_count']).rank(pct=True, method='average') * 100

# avg_rating_score = percentile rank of average_rating
books_clean['avg_rating_score'] = books_clean['average_rating'].rank(pct=True, method='average') * 100

# genre_rarity: compute frequency of main genres among cleaned table
genre_freq = books_clean['main_genre'].value_counts(normalize=False)
books_clean['genre_freq'] = books_clean['main_genre'].map(genre_freq)
# rarity_value = -freq (higher for rarer)
books_clean['rarity_value'] = -books_clean['genre_freq']
books_clean['genre_rarity_score'] = books_clean['rarity_value'].rank(pct=True, method='average') * 100

# --- Active review range: exclude books without language, pub year, or main_genre ---
active = books_clean[(books_clean['language_code'].notna()) & 
                     (books_clean['original_publication_year'].notna()) &
                     (books_clean['main_genre'].notna())].copy()

print(f"Cleaned books: {len(books_clean)}")
print(f"Active range books: {len(active)}")

# --- Thresholds ---
demand_80 = active['demand_absolute_score'].quantile(0.80)
ratings_85 = active['ratings_count'].quantile(0.85)
print(f"Demand 80th percentile: {demand_80:.3f}")
print(f"Ratings count 85th percentile: {ratings_85:.3f}")

# Filter active pool
pool = active[(active['demand_absolute_score'] >= demand_80) &
              (active['ratings_count'] <= ratings_85)].copy()
print(f"First review pool size: {len(pool)}")

# Compute final score
pool['review_score'] = (0.34 * pool['demand_absolute_score'] +
                        0.24 * pool['exposure_gap_score'] +
                        0.22 * pool['genre_rarity_score'] +
                        0.20 * pool['avg_rating_score'])

# Sort by score descending, then book_key ascending for ties
pool_sorted = pool.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top5 = pool_sorted.head(5)

# Show leading five with component contributions
cols_show = ['book_key', 'original_title', 'main_genre', 'review_score',
             'demand_absolute_score', 'exposure_gap_score', 'genre_rarity_score', 'avg_rating_score']
print(top5[cols_show].to_string())

import pandas as pd
import numpy as np
import re

# Re-run entire pipeline to ensure correctness
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# Clean ratings
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
ratings_clean = ratings[~dup_mask].copy()

# Clean books
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
books_clean = books.dropna(subset=critical_cols).copy()
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
books_clean['book_key'] = books_clean['id']

# Title keys
def early_key(title):
    if not isinstance(title, str):
        return None
    return re.sub(r'\s+', ' ', title.lower().strip())

def strict_key(title):
    if not isinstance(title, str):
        return None
    t = title.lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r'\b(the|a|an)\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

books_clean['title_key_early'] = books_clean['original_title'].apply(early_key)
books_clean['title_key_strict'] = books_clean['original_title'].apply(strict_key)

# Reading intent
intent_counts = to_read['book_id'].value_counts().reset_index()
intent_counts.columns = ['id', 'reading_intent_count']
books_clean = books_clean.merge(intent_counts, on='id', how='left')
books_clean['reading_intent_count'] = books_clean['reading_intent_count'].fillna(0)

# Genre processing
tags['tag_norm'] = tags['tag_name'].str.lower().str.replace('_', '-')
genre_families = {
    'art': 'art', 'biography': 'biography', 'business': 'business',
    'chick-lit': 'chick_lit', 'children': 'children', 'childrens': 'children',
    'christian': 'christian', 'classics': 'classics', 'comics': 'comics',
    'contemporary': 'contemporary', 'cookbooks': 'cookbooks', 'crime': 'crime',
    'ebooks': 'ebooks', 'fantasy': 'fantasy', 'fiction': 'fiction',
    'historical-fiction': 'historical_fiction', 'history': 'history',
    'horror': 'horror', 'humor': 'humor', 'manga': 'manga',
    'memoir': 'memoir', 'music': 'music', 'mystery': 'mystery',
    'nonfiction': 'nonfiction', 'paranormal': 'paranormal',
    'philosophy': 'philosophy', 'poetry': 'poetry', 'romance': 'romance',
    'science': 'science', 'science-fiction': 'science_fiction',
    'self-help': 'self_help', 'sports': 'sports', 'suspense': 'suspense',
    'thriller': 'thriller', 'travel': 'travel', 'young-adult': 'young_adult'
}
tags['canonical_genre'] = tags['tag_norm'].map(genre_families)
canonical_tags = tags.dropna(subset=['canonical_genre'])
book_tags_genre = book_tags.merge(canonical_tags[['tag_id', 'canonical_genre']], on='tag_id', how='inner')
genre_sums = book_tags_genre.groupby(['goodreads_book_id', 'canonical_genre'])['count'].sum().reset_index()
genre_sums = genre_sums.sort_values(['goodreads_book_id', 'count', 'canonical_genre'], 
                                     ascending=[True, False, True])
main_genres = genre_sums.groupby('goodreads_book_id').first().reset_index()
main_genres = main_genres[['goodreads_book_id', 'canonical_genre']].rename(
    columns={'canonical_genre': 'main_genre'})
books_clean = books_clean.merge(main_genres, left_on='book_id', right_on='goodreads_book_id', how='left')
books_clean.drop(columns='goodreads_book_id', inplace=True)

# Percentile ranks on cleaned table
books_clean['demand_absolute_score'] = books_clean['reading_intent_count'].rank(pct=True, method='average') * 100
books_clean['exposure_gap_score'] = (-books_clean['ratings_count']).rank(pct=True, method='average') * 100
books_clean['avg_rating_score'] = books_clean['average_rating'].rank(pct=True, method='average') * 100

genre_freq = books_clean['main_genre'].value_counts(normalize=False)
books_clean['genre_freq'] = books_clean['main_genre'].map(genre_freq)
books_clean['rarity_value'] = -books_clean['genre_freq']
books_clean['genre_rarity_score'] = books_clean['rarity_value'].rank(pct=True, method='average') * 100

# Active review range
active = books_clean[(books_clean['language_code'].notna()) & 
                     (books_clean['original_publication_year'].notna()) &
                     (books_clean['main_genre'].notna())].copy()

# Thresholds
demand_80 = active['demand_absolute_score'].quantile(0.80)
ratings_85 = active['ratings_count'].quantile(0.85)

# Filter pool
pool = active[(active['demand_absolute_score'] >= demand_80) &
              (active['ratings_count'] <= ratings_85)].copy()

# Final score
pool['review_score'] = (0.34 * pool['demand_absolute_score'] +
                        0.24 * pool['exposure_gap_score'] +
                        0.22 * pool['genre_rarity_score'] +
                        0.20 * pool['avg_rating_score'])

# Sort
pool_sorted = pool.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top5 = pool_sorted.head(5)

# Verify counts
print(f"books raw: {len(books)}")
print(f"ratings clean: {len(ratings_clean)}")
print(f"books_clean: {len(books_clean)}")
print(f"active: {len(active)}")
print(f"pool: {len(pool)}")
print(f"demand_80: {demand_80:.3f}")
print(f"ratings_85: {ratings_85:.3f}")
print()

# Display top5 with rounded values
for i, (idx, row) in enumerate(top5.iterrows()):
    print(f"Rank {i+1}: book_key={row['book_key']}")
    print(f"  Title: {row['original_title']}")
    print(f"  Main Genre: {row['main_genre']}")
    print(f"  Review Score: {row['review_score']:.3f}")
    print(f"  Demand Absolute Score (34%): {row['demand_absolute_score']:.3f}")
    print(f"  Exposure Gap Score (24%): {row['exposure_gap_score']:.3f}")
    print(f"  Genre Rarity Score (22%): {row['genre_rarity_score']:.3f}")
    print(f"  Average Rating Score (20%): {row['avg_rating_score']:.3f}")
    print()

# Generate exact values with full precision and re-confirm
print("Exact component contributions for top 5:")
for i, (idx, row) in enumerate(top5.iterrows()):
    s = row['review_score']
    d = row['demand_absolute_score']
    e = row['exposure_gap_score']
    g = row['genre_rarity_score']
    a = row['avg_rating_score']
    weighted = 0.34*d + 0.24*e + 0.22*g + 0.20*a
    print(f"\nRank {i+1}: book_key={int(row['book_key'])}, title='{row['original_title']}'")
    print(f"  review_score: {s:.6f} (recomputed: {weighted:.6f})")
    print(f"  demand_absolute_score: {d:.6f}  (weight 0.34 -> {0.34*d:.6f})")
    print(f"  exposure_gap_score:    {e:.6f}  (weight 0.24 -> {0.24*e:.6f})")
    print(f"  genre_rarity_score:    {g:.6f}  (weight 0.22 -> {0.22*g:.6f})")
    print(f"  avg_rating_score:      {a:.6f}  (weight 0.20 -> {0.20*a:.6f})")
    print(f"  Sum: {0.34*d + 0.24*e + 0.22*g + 0.20*a:.6f}  vs stored: {s:.6f}")

############## turn 2
import pandas as pd
import numpy as np
import re

# Load all data
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# Clean ratings: remove duplicate (user_id, book_id) pairs
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
ratings_clean = ratings[~dup_mask].copy()

# Compute cleaned rating activity per book (ratings.book_id = books.id)
rating_activity = ratings_clean.groupby('book_id').size().reset_index(name='cleaned_rating_activity')
# Note: ratings.book_id maps to books.id

# Clean books
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
books_clean = books.dropna(subset=critical_cols).copy()
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
books_clean['book_key'] = books_clean['id']

# Title keys (for state, not essential now)
def early_key(title):
    if not isinstance(title, str): return None
    return re.sub(r'\s+', ' ', title.lower().strip())
def strict_key(title):
    if not isinstance(title, str): return None
    t = title.lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r'\b(the|a|an)\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t
books_clean['title_key_early'] = books_clean['original_title'].apply(early_key)
books_clean['title_key_strict'] = books_clean['original_title'].apply(strict_key)

# Reading intent counts
intent_counts = to_read['book_id'].value_counts().reset_index()
intent_counts.columns = ['book_id', 'reading_intent_count']  # to_read.book_id = books.id
books_clean = books_clean.merge(intent_counts, left_on='id', right_on='book_id', how='left')
books_clean.drop(columns='book_id', inplace=True)
books_clean['reading_intent_count'] = books_clean['reading_intent_count'].fillna(0)

# Merge cleaned rating activity
books_clean = books_clean.merge(rating_activity, left_on='id', right_on='book_id', how='left')
books_clean.drop(columns='book_id', inplace=True)
books_clean['cleaned_rating_activity'] = books_clean['cleaned_rating_activity'].fillna(0)

# Compute relative demand: reading_intent_count / cleaned_rating_activity
# Books with no cleaned rating activity: rate = 0
books_clean['relative_demand'] = np.where(
    books_clean['cleaned_rating_activity'] > 0,
    books_clean['reading_intent_count'] / books_clean['cleaned_rating_activity'],
    0.0
)

# Genre processing
tags['tag_norm'] = tags['tag_name'].str.lower().str.replace('_', '-')
genre_families = {
    'art': 'art', 'biography': 'biography', 'business': 'business',
    'chick-lit': 'chick_lit', 'children': 'children', 'childrens': 'children',
    'christian': 'christian', 'classics': 'classics', 'comics': 'comics',
    'contemporary': 'contemporary', 'cookbooks': 'cookbooks', 'crime': 'crime',
    'ebooks': 'ebooks', 'fantasy': 'fantasy', 'fiction': 'fiction',
    'historical-fiction': 'historical_fiction', 'history': 'history',
    'horror': 'horror', 'humor': 'humor', 'manga': 'manga',
    'memoir': 'memoir', 'music': 'music', 'mystery': 'mystery',
    'nonfiction': 'nonfiction', 'paranormal': 'paranormal',
    'philosophy': 'philosophy', 'poetry': 'poetry', 'romance': 'romance',
    'science': 'science', 'science-fiction': 'science_fiction',
    'self-help': 'self_help', 'sports': 'sports', 'suspense': 'suspense',
    'thriller': 'thriller', 'travel': 'travel', 'young-adult': 'young_adult'
}
tags['canonical_genre'] = tags['tag_norm'].map(genre_families)
canonical_tags = tags.dropna(subset=['canonical_genre'])
book_tags_genre = book_tags.merge(canonical_tags[['tag_id', 'canonical_genre']], on='tag_id', how='inner')
genre_sums = book_tags_genre.groupby(['goodreads_book_id', 'canonical_genre'])['count'].sum().reset_index()
genre_sums = genre_sums.sort_values(['goodreads_book_id', 'count', 'canonical_genre'], 
                                     ascending=[True, False, True])
main_genres = genre_sums.groupby('goodreads_book_id').first().reset_index()
main_genres = main_genres[['goodreads_book_id', 'canonical_genre']].rename(
    columns={'canonical_genre': 'main_genre'})
books_clean = books_clean.merge(main_genres, left_on='book_id', right_on='goodreads_book_id', how='left')
books_clean.drop(columns='goodreads_book_id', inplace=True)

# Percentile ranks on cleaned table
books_clean['demand_relative_score'] = books_clean['relative_demand'].rank(pct=True, method='average') * 100
books_clean['exposure_gap_score'] = (-books_clean['ratings_count']).rank(pct=True, method='average') * 100
books_clean['avg_rating_score'] = books_clean['average_rating'].rank(pct=True, method='average') * 100

genre_freq = books_clean['main_genre'].value_counts(normalize=False)
books_clean['genre_freq'] = books_clean['main_genre'].map(genre_freq)
books_clean['rarity_value'] = -books_clean['genre_freq']
books_clean['genre_rarity_score'] = books_clean['rarity_value'].rank(pct=True, method='average') * 100

# Active review range
active = books_clean[(books_clean['language_code'].notna()) & 
                     (books_clean['original_publication_year'].notna()) &
                     (books_clean['main_genre'].notna())].copy()

print(f"Books clean: {len(books_clean)}, Active: {len(active)}")

# New thresholds based on relative demand
demand_80_active = active['demand_relative_score'].quantile(0.80)
ratings_85_active = active['ratings_count'].quantile(0.85)
print(f"Demand 80th pct: {demand_80_active:.3f}, Ratings 85th pct: {ratings_85_active:.3f}")

# Filter pool
pool_new = active[(active['demand_relative_score'] >= demand_80_active) &
                  (active['ratings_count'] <= ratings_85_active)].copy()

print(f"New pool size: {len(pool_new)}")

# Final score
pool_new['review_score'] = (0.34 * pool_new['demand_relative_score'] +
                             0.24 * pool_new['exposure_gap_score'] +
                             0.22 * pool_new['genre_rarity_score'] +
                             0.20 * pool_new['avg_rating_score'])

# Sort
pool_new = pool_new.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top5_new = pool_new.head(5)
top10_new = pool_new.head(10)

print("\nTop 5 with new relative demand:")
for i, (idx, row) in enumerate(top5_new.iterrows()):
    print(f"Rank {i+1}: book_key={int(row['book_key'])}, {row['original_title']} ({row['main_genre']})")
    print(f"  Score: {row['review_score']:.3f}, Demand: {row['demand_relative_score']:.3f}, Exposure: {row['exposure_gap_score']:.3f}, Rarity: {row['genre_rarity_score']:.3f}, Rating: {row['avg_rating_score']:.3f}")

# Store new top 10 keys
top10_new_keys = set(top10_new['book_key'].values)
print("\nTop 10 new book_keys:", sorted(top10_new_keys))

# Rebuild properly with correct column naming
import pandas as pd
import numpy as np
import re

books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# Clean ratings
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
ratings_clean = ratings[~dup_mask].copy()
rating_activity = ratings_clean.groupby('book_id').size().reset_index(name='cleaned_rating_activity')

# Clean books
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
books_clean = books.dropna(subset=critical_cols).copy()
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
books_clean['book_key'] = books_clean['id']

def early_key(title):
    if not isinstance(title, str): return None
    return re.sub(r'\s+', ' ', title.lower().strip())
def strict_key(title):
    if not isinstance(title, str): return None
    t = title.lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r'\b(the|a|an)\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t
books_clean['title_key_early'] = books_clean['original_title'].apply(early_key)
books_clean['title_key_strict'] = books_clean['original_title'].apply(strict_key)

# Reading intent: aggregate separately and join
intent_counts = to_read['book_id'].value_counts().reset_index()
intent_counts.columns = ['book_id_for_intent', 'reading_intent_count']
books_clean = books_clean.merge(intent_counts, left_on='id', right_on='book_id_for_intent', how='left')
books_clean.drop(columns='book_id_for_intent', inplace=True)
books_clean['reading_intent_count'] = books_clean['reading_intent_count'].fillna(0)

# Merge rating activity
books_clean = books_clean.merge(rating_activity, left_on='id', right_on='book_id', how='left')
books_clean['cleaned_rating_activity'] = books_clean['cleaned_rating_activity'].fillna(0)
# drop the duplicate 'book_id' column from rating_activity (which matches ratings.book_id = books.id)
# The merge adds 'book_id' from rating_activity; books already has 'book_id' from original.
# Check columns for duplication
print("Columns after activity merge:", [c for c in books_clean.columns if 'book_id' in c])

# Fix columns: drop book_id_y, rename book_id_x back to book_id
books_clean.drop(columns='book_id_y', inplace=True)
books_clean.rename(columns={'book_id_x': 'book_id'}, inplace=True)

print("Fixed columns check:", [c for c in books_clean.columns if 'book_id' in c])
print("book_id sample:", books_clean['book_id'].head(3).tolist())
print("id sample:", books_clean['id'].head(3).tolist())

# Compute relative demand
books_clean['relative_demand'] = np.where(
    books_clean['cleaned_rating_activity'] > 0,
    books_clean['reading_intent_count'] / books_clean['cleaned_rating_activity'],
    0.0
)

# Genre processing
tags['tag_norm'] = tags['tag_name'].str.lower().str.replace('_', '-')
genre_families = {
    'art': 'art', 'biography': 'biography', 'business': 'business',
    'chick-lit': 'chick_lit', 'children': 'children', 'childrens': 'children',
    'christian': 'christian', 'classics': 'classics', 'comics': 'comics',
    'contemporary': 'contemporary', 'cookbooks': 'cookbooks', 'crime': 'crime',
    'ebooks': 'ebooks', 'fantasy': 'fantasy', 'fiction': 'fiction',
    'historical-fiction': 'historical_fiction', 'history': 'history',
    'horror': 'horror', 'humor': 'humor', 'manga': 'manga',
    'memoir': 'memoir', 'music': 'music', 'mystery': 'mystery',
    'nonfiction': 'nonfiction', 'paranormal': 'paranormal',
    'philosophy': 'philosophy', 'poetry': 'poetry', 'romance': 'romance',
    'science': 'science', 'science-fiction': 'science_fiction',
    'self-help': 'self_help', 'sports': 'sports', 'suspense': 'suspense',
    'thriller': 'thriller', 'travel': 'travel', 'young-adult': 'young_adult'
}
tags['canonical_genre'] = tags['tag_norm'].map(genre_families)
canonical_tags = tags.dropna(subset=['canonical_genre'])
book_tags_genre = book_tags.merge(canonical_tags[['tag_id', 'canonical_genre']], on='tag_id', how='inner')
genre_sums = book_tags_genre.groupby(['goodreads_book_id', 'canonical_genre'])['count'].sum().reset_index()
genre_sums = genre_sums.sort_values(['goodreads_book_id', 'count', 'canonical_genre'], 
                                     ascending=[True, False, True])
main_genres = genre_sums.groupby('goodreads_book_id').first().reset_index()
main_genres = main_genres[['goodreads_book_id', 'canonical_genre']].rename(
    columns={'canonical_genre': 'main_genre'})
books_clean = books_clean.merge(main_genres, left_on='book_id', right_on='goodreads_book_id', how='left')
books_clean.drop(columns='goodreads_book_id', inplace=True)

print("After genre merge, shape:", books_clean.shape)
print("Main genre nulls:", books_clean['main_genre'].isna().sum())

# Percentile ranks on cleaned table
books_clean['demand_relative_score'] = books_clean['relative_demand'].rank(pct=True, method='average') * 100
books_clean['exposure_gap_score'] = (-books_clean['ratings_count']).rank(pct=True, method='average') * 100
books_clean['avg_rating_score'] = books_clean['average_rating'].rank(pct=True, method='average') * 100

genre_freq = books_clean['main_genre'].value_counts(normalize=False)
books_clean['genre_freq'] = books_clean['main_genre'].map(genre_freq)
books_clean['rarity_value'] = -books_clean['genre_freq']
books_clean['genre_rarity_score'] = books_clean['rarity_value'].rank(pct=True, method='average') * 100

# Active review range
active = books_clean[(books_clean['language_code'].notna()) & 
                     (books_clean['original_publication_year'].notna()) &
                     (books_clean['main_genre'].notna())].copy()

print(f"Active range: {len(active)}")

# New thresholds
demand_80 = active['demand_relative_score'].quantile(0.80)
ratings_85 = active['ratings_count'].quantile(0.85)
print(f"Demand 80th pct: {demand_80:.3f}")
print(f"Ratings 85th pct: {ratings_85:.3f}")

# Filter pool
pool_new = active[(active['demand_relative_score'] >= demand_80) &
                  (active['ratings_count'] <= ratings_85)].copy()
print(f"New pool size: {len(pool_new)}")

# Final score
pool_new['review_score'] = (0.34 * pool_new['demand_relative_score'] +
                             0.24 * pool_new['exposure_gap_score'] +
                             0.22 * pool_new['genre_rarity_score'] +
                             0.20 * pool_new['avg_rating_score'])

# Sort
pool_new = pool_new.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top5_new = pool_new.head(5)
top10_new = pool_new.head(10)

print("\nTop 5 with relative demand:")
for i, (idx, row) in enumerate(top5_new.iterrows()):
    print(f"Rank {i+1}: book_key={int(row['book_key'])}, '{row['original_title']}' ({row['main_genre']})")
    print(f"  Score: {row['review_score']:.3f}, Demand: {row['demand_relative_score']:.3f}, Exposure: {row['exposure_gap_score']:.3f}, Rarity: {row['genre_rarity_score']:.3f}, Rating: {row['avg_rating_score']:.3f}")

top10_new_keys = set(top10_new['book_key'].values)
print("\nTop 10 new book_keys:", sorted(top10_new_keys))

# Build previous pool using absolute demand (from the same cleaned table)
# We already have books_clean with all percentile ranks; we need demand_absolute_score
books_clean['demand_absolute_score'] = books_clean['reading_intent_count'].rank(pct=True, method='average') * 100

active_abs = books_clean[(books_clean['language_code'].notna()) & 
                         (books_clean['original_publication_year'].notna()) &
                         (books_clean['main_genre'].notna())].copy()

demand_80_abs = active_abs['demand_absolute_score'].quantile(0.80)
ratings_85_abs = active_abs['ratings_count'].quantile(0.85)

pool_prev = active_abs[(active_abs['demand_absolute_score'] >= demand_80_abs) &
                        (active_abs['ratings_count'] <= ratings_85_abs)].copy()

pool_prev['review_score'] = (0.34 * pool_prev['demand_absolute_score'] +
                              0.24 * pool_prev['exposure_gap_score'] +
                              0.22 * pool_prev['genre_rarity_score'] +
                              0.20 * pool_prev['avg_rating_score'])

pool_prev = pool_prev.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top10_prev = pool_prev.head(10)
early_high_intent = pool_prev.head(25)

print(f"Previous pool size: {len(pool_prev)}")
print("Previous top 10 book_keys:", sorted(top10_prev['book_key'].values))
print("Early high-intent group (first 25) book_keys:", sorted(early_high_intent['book_key'].values))

# Convert to plain Python sets of ints for comparison
prev_top10_set = set(int(x) for x in top10_prev['book_key'].values)
new_top10_set = set(int(x) for x in top10_new['book_key'].values)
early_high_set = set(int(x) for x in early_high_intent['book_key'].values)

entering = new_top10_set - prev_top10_set
leaving = prev_top10_set - new_top10_set
staying = new_top10_set & prev_top10_set

print("Previous top 10:", sorted(prev_top10_set))
print("New top 10:", sorted(new_top10_set))
print("Entering:", sorted(entering))
print("Leaving:", sorted(leaving))
print("Staying:", sorted(staying))
print()

# Are entering books in the early high-intent group?
for bk in sorted(entering):
    in_early = bk in early_high_set
    print(f"Entering book_key={bk}: in_early_high_intent={in_early}")

# Are leaving books in the early high-intent group?
for bk in sorted(leaving):
    in_early = bk in early_high_set
    print(f"Leaving book_key={bk}: in_early_high_intent={in_early}")

print("\nEarly high-intent group size:", len(early_high_set))
print("Number of entering books in early group:", len(entering & early_high_set))
print("Number of leaving books in early group:", len(leaving & early_high_set))

############## turn 3
import pandas as pd
import numpy as np
import re

# Reload and rebuild the full cleaned state as before
books = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/books.csv')
ratings = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/ratings.csv')
to_read = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/to_read.csv')
book_tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/book_tags.csv')
tags = pd.read_csv('/data/longds/business/goodbooks_10k/task1/data/goodbooks-10k/tags.csv')

# Clean ratings
dup_mask = ratings.duplicated(subset=['user_id', 'book_id'], keep=False)
ratings_clean = ratings[~dup_mask].copy()

# Clean books
critical_cols = ['original_title', 'authors', 'average_rating', 'ratings_count',
                 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
books_clean = books.dropna(subset=critical_cols).copy()
dupe_titles = books_clean['original_title'].duplicated(keep=False)
books_clean = books_clean[~dupe_titles].copy()
books_clean['book_key'] = books_clean['id']

# Title keys
def early_key(title):
    if not isinstance(title, str): return None
    return re.sub(r'\s+', ' ', title.lower().strip())
def strict_key(title):
    if not isinstance(title, str): return None
    t = title.lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r'\b(the|a|an)\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[^a-z0-9]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t
books_clean['title_key_early'] = books_clean['original_title'].apply(early_key)
books_clean['title_key_strict'] = books_clean['original_title'].apply(strict_key)

# Reading intent
intent_counts = to_read['book_id'].value_counts().reset_index()
intent_counts.columns = ['book_id_for_intent', 'reading_intent_count']
books_clean = books_clean.merge(intent_counts, left_on='id', right_on='book_id_for_intent', how='left')
books_clean.drop(columns='book_id_for_intent', inplace=True)
books_clean['reading_intent_count'] = books_clean['reading_intent_count'].fillna(0)

# Genre
tags['tag_norm'] = tags['tag_name'].str.lower().str.replace('_', '-')
genre_families = {
    'art': 'art', 'biography': 'biography', 'business': 'business',
    'chick-lit': 'chick_lit', 'children': 'children', 'childrens': 'children',
    'christian': 'christian', 'classics': 'classics', 'comics': 'comics',
    'contemporary': 'contemporary', 'cookbooks': 'cookbooks', 'crime': 'crime',
    'ebooks': 'ebooks', 'fantasy': 'fantasy', 'fiction': 'fiction',
    'historical-fiction': 'historical_fiction', 'history': 'history',
    'horror': 'horror', 'humor': 'humor', 'manga': 'manga',
    'memoir': 'memoir', 'music': 'music', 'mystery': 'mystery',
    'nonfiction': 'nonfiction', 'paranormal': 'paranormal',
    'philosophy': 'philosophy', 'poetry': 'poetry', 'romance': 'romance',
    'science': 'science', 'science-fiction': 'science_fiction',
    'self-help': 'self_help', 'sports': 'sports', 'suspense': 'suspense',
    'thriller': 'thriller', 'travel': 'travel', 'young-adult': 'young_adult'
}
tags['canonical_genre'] = tags['tag_norm'].map(genre_families)
canonical_tags = tags.dropna(subset=['canonical_genre'])
book_tags_genre = book_tags.merge(canonical_tags[['tag_id', 'canonical_genre']], on='tag_id', how='inner')
genre_sums = book_tags_genre.groupby(['goodreads_book_id', 'canonical_genre'])['count'].sum().reset_index()
genre_sums = genre_sums.sort_values(['goodreads_book_id', 'count', 'canonical_genre'], 
                                     ascending=[True, False, True])
main_genres = genre_sums.groupby('goodreads_book_id').first().reset_index()
main_genres = main_genres[['goodreads_book_id', 'canonical_genre']].rename(columns={'canonical_genre': 'main_genre'})
books_clean = books_clean.merge(main_genres, left_on='book_id', right_on='goodreads_book_id', how='left')
books_clean.drop(columns='goodreads_book_id', inplace=True)

print("Books clean shape:", books_clean.shape)

# Compute exposure floor K = 60th percentile of ratings_count on cleaned table
K = books_clean['ratings_count'].quantile(0.60)
print(f"Exposure floor K (60th pct of ratings_count): {K}")

# Overall weighted mean mu = sum(ratings_count * avg_rating) / sum(ratings_count)
total_weight = books_clean['ratings_count'].sum()
weighted_sum = (books_clean['ratings_count'] * books_clean['average_rating']).sum()
mu = weighted_sum / total_weight
print(f"Overall weighted mean mu: {mu:.6f}")

# Compute shrunk rating for each book
books_clean['blended_rating'] = (books_clean['ratings_count'] * books_clean['average_rating'] + K * mu) / (books_clean['ratings_count'] + K)

# Percentile rank blended_rating -> quality_score_exposure
books_clean['quality_score_exposure'] = books_clean['blended_rating'].rank(pct=True, method='average') * 100

print("Blended rating sample:", books_clean[['book_key', 'average_rating', 'ratings_count', 'blended_rating', 'quality_score_exposure']].head())

# Percentile ranks on cleaned table
books_clean['demand_absolute_score'] = books_clean['reading_intent_count'].rank(pct=True, method='average') * 100
books_clean['exposure_gap_score'] = (-books_clean['ratings_count']).rank(pct=True, method='average') * 100
books_clean['avg_rating_score'] = books_clean['average_rating'].rank(pct=True, method='average') * 100

genre_freq = books_clean['main_genre'].value_counts(normalize=False)
books_clean['genre_freq'] = books_clean['main_genre'].map(genre_freq)
books_clean['rarity_value'] = -books_clean['genre_freq']
books_clean['genre_rarity_score'] = books_clean['rarity_value'].rank(pct=True, method='average') * 100

# Active range
active = books_clean[(books_clean['language_code'].notna()) & 
                     (books_clean['original_publication_year'].notna()) &
                     (books_clean['main_genre'].notna())].copy()

# Thresholds
demand_80 = active['demand_absolute_score'].quantile(0.80)
ratings_85 = active['ratings_count'].quantile(0.85)
print(f"Demand 80th: {demand_80:.3f}, Ratings 85th: {ratings_85:.3f}")

# Pool with original quality (avg_rating_score)
pool_orig = active[(active['demand_absolute_score'] >= demand_80) &
                   (active['ratings_count'] <= ratings_85)].copy()
pool_orig['review_score'] = (0.34 * pool_orig['demand_absolute_score'] +
                             0.24 * pool_orig['exposure_gap_score'] +
                             0.22 * pool_orig['genre_rarity_score'] +
                             0.20 * pool_orig['avg_rating_score'])
pool_orig = pool_orig.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top10_orig = pool_orig.head(10)

# Pool with exposure-aware quality
pool_new = active[(active['demand_absolute_score'] >= demand_80) &
                  (active['ratings_count'] <= ratings_85)].copy()
pool_new['review_score'] = (0.34 * pool_new['demand_absolute_score'] +
                            0.24 * pool_new['exposure_gap_score'] +
                            0.22 * pool_new['genre_rarity_score'] +
                            0.20 * pool_new['quality_score_exposure'])
pool_new = pool_new.sort_values(by=['review_score', 'book_key'], ascending=[False, True])
top10_new = pool_new.head(10)

print(f"\nOriginal top 10: {sorted(top10_orig['book_key'].values)}")
print(f"New top 10: {sorted(top10_new['book_key'].values)}")

# Identify flipping books
orig_set = set(int(x) for x in top10_orig['book_key'].values)
new_set = set(int(x) for x in top10_new['book_key'].values)

entering = new_set - orig_set
leaving = orig_set - new_set
flip_keys = entering | leaving
print("Entering:", entering)
print("Leaving:", leaving)
print("All flips:", flip_keys)

# Get the pool rows for these books from both pools
# pool_new and pool_orig are already computed
flip_new = pool_new[pool_new['book_key'].isin(flip_keys)]
flip_orig = pool_orig[pool_orig['book_key'].isin(flip_keys)]

# Merge their scores side by side
flip_new_scores = flip_new[['book_key', 'review_score', 'demand_absolute_score', 
                              'exposure_gap_score', 'genre_rarity_score', 'quality_score_exposure']].copy()
flip_orig_scores = flip_orig[['book_key', 'review_score', 'avg_rating_score']].copy()

flip_combined = flip_new_scores.merge(flip_orig_scores, on='book_key', suffixes=('_new', '_orig'))
flip_combined['quality_gap'] = abs(flip_combined['quality_score_exposure'] - flip_combined['avg_rating_score'])
flip_combined['flip_strength'] = flip_combined['quality_gap']  # demand gap is 0

# Sort by flip strength descending
flip_combined = flip_combined.sort_values('flip_strength', ascending=False)

print("\nFlipping books details:")
for _, row in flip_combined.iterrows():
    print(f"\nbook_key={int(row['book_key'])}, {books_clean[books_clean['book_key']==row['book_key']]['original_title'].values[0]}")
    print(f"  Treatment OLD (avg_rating_score): review_score={row['review_score_orig']:.3f}, demand={row['demand_absolute_score']:.3f}, exposure={row['exposure_gap_score']:.3f}, rarity={row['genre_rarity_score']:.3f}, quality={row['avg_rating_score']:.3f}")
    print(f"  Treatment NEW (exposure quality): review_score={row['review_score_new']:.3f}, demand={row['demand_absolute_score']:.3f}, exposure={row['exposure_gap_score']:.3f}, rarity={row['genre_rarity_score']:.3f}, quality={row['quality_score_exposure']:.3f}")
    print(f"  Flip strength (|quality_gap|): {row['flip_strength']:.3f}")
