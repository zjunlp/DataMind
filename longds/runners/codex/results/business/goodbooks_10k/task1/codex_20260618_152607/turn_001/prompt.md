You are running LongDS-Bench directly as Codex.

Use only the local benchmark data directory below and the current prompt. Do not inspect parent
directories, repository files, LongDS task files such as task.json, task.py, task.ipynb,
metadata.json, or any ground-truth answer/code files.

Local data directory:
/mnt/40t/xkw/LongMemDA/DataMind/longds/scripts/results/business/goodbooks_10k/task1/codex_20260618_152607/data

Codex execution directory:
/mnt/40t/xkw/LongMemDA/DataMind/longds/scripts/results/business/goodbooks_10k/task1/codex_20260618_152607

Scratch workspace directory for analysis scripts and intermediate files:
/mnt/40t/xkw/LongMemDA/DataMind/longds/scripts/results/business/goodbooks_10k/task1/codex_20260618_152607/workspace

Python executable to use for analysis:
/mnt/20t/xkw/anaconda3/envs/longds/bin/python

You may run Python or shell commands, create helper scripts, and save intermediate artifacts in
the scratch workspace directory. Keep useful state for later turns in this Codex session. Later turns will
resume this same Codex session, so reuse earlier definitions and assumptions when applicable.

Environment constraint:
- All Python analysis commands MUST use the exact executable above.
- Do NOT use bare `python`, `python3`, `pip`, `ipython`, or another interpreter for analysis.
- If you need to install or inspect packages, use `/mnt/20t/xkw/anaconda3/envs/longds/bin/python -m pip ...` or
  `/mnt/20t/xkw/anaconda3/envs/longds/bin/python -c ...`.
- Treat the Codex execution directory as the filesystem boundary for this benchmark task.
- Put temporary code, notebooks, caches, and intermediate outputs under the scratch workspace
  directory.
- Do not search outside the Codex execution directory, except for using the exact Python executable
  listed above.

For this benchmark:
- Solve only the current turn. Do not ask for or infer future turns.
- Use exact calculations from data, not mental arithmetic, when data files are involved.
- Round decimal-valued final results only when the task asks for rounding.
- Preserve requested ordering and tie-breaking rules.
- Return the final response as JSON matching the provided schema. Put the direct user-facing
  result in the "answer" string.

LongDS task turn 1 of 3.

Context:
Clean the Goodreads sources into a single persistent working set for all later tasks. Use books.id as the analysis book_key.
ratings.book_id and to_read.book_id join to books.id; book_tags.goodreads_book_id joins to books.book_id. Before computing rating activity, if a (user_id, book_id) pair appears more than once in ratings, remove all rating rows for that repeated pair rather than keeping one row. Remove all books sharing an exact raw books.original_title value; do not use normalized title keys for this duplicate-title removal. The cleaned table excludes records missing original_title, authors, average_rating, ratings_count, or star-rating columns. After component percentile ranks are built on the cleaned table, the active review range excludes books without known language, known publication year, or a recognized main genre.
Use two title treatments based on original_title. The early key lowercases and collapses spaces without folding accents, while the stricter key later removes parenthetical text, removes standalone articles, replaces non-alphanumeric characters with spaces, lowercases,
and collapses spaces.
Connect reading-intent counts and genre evidence to the correct book records. Recognize genre evidence only from exact normalized tag names in these families: art, biography, business, chick-lit, children and childrens, christian, classics, comics, contemporary, cookbooks, crime, ebooks, fantasy, fiction, historical-fiction, history, horror, humor, manga, memoir, music, mystery, nonfiction, paranormal, philosophy, poetry, romance, science, science-fiction, self-help, sports, suspense, thriller, travel, and young-adult. Normalize tag names by lowercasing and replacing underscores with hyphens. Map recognized tags to canonical genre families, including children and childrens -> children, historical-fiction -> historical_fiction, science-fiction -> science_fiction, self-help -> self_help, young-adult -> young_adult, and chick-lit -> chick_lit. For each book_key, sum tag counts within each canonical genre, choose the main genre by largest summed count, and break ties alphabetically by canonical genre.
Keep analysis-ready fields for title matching, lead author, language, star-rating diagnostics, reading intent, and genre rarity. Build component percentile ranks on the cleaned table before applying the active review filter; percentile ranks use average tie handling and pandas style rank(pct=True) * 100, i.e. average_rank / nonmissing_count * 100. Do not rescale ranks with (rank - 1) / (n - 1).
In the early genre-scarcity treatment, rarer recognized main genres receive higher scores. The active review range excludes books without known language, known publication year, or a recognized main genre.
The first review score blends 34% reader-intent volume, 24% exposure gap where lower Goodreads rating count is stronger, 22% genre rarity, and 20% average-rating quality. Define demand_absolute_score as the percentile rank of reading-intent count. Keep active-range books whose demand_absolute_score alone reaches the active-range 80th percentile; do not include exposure gap in this filtering threshold. Also require raw Goodreads rating count to be no higher than the active-range 85th percentile. Sort ranked outputs by the named score in the natural direction implied by the question; tied ranked rows and top-N truncation use the smaller book_key first.
Restore display titles, full authors, covers, publication years, raw star columns, ISBN fields, and raw tag rows only when an audit needs source evidence. Throughout this analysis, compute derived quantities including sums, gaps, ratios, means, similarities, and correlations using unrounded values, and report decimal-valued final results rounded to 3 decimal places. Counts use the cleaned denominator established in the same step unless the question explicitly asks for a comparison across two states. Absent neighbor evidence contributes zero.
This step establishes the persistent cleaned analysis state for all later tasks; later tasks must reuse these book keys, cleaned ratings, genre assignments, title keys, and percentile columns unless explicitly instructed to recompute one component.

Question:
Build the first review pool by combining reader intent, exposure gap, genre scarcity, and quality. Show the five leading books and the component contributions behind their scores.