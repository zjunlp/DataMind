LongDS task turn 4 of 5.

Context:
Restore original_title, authors, small_image_url, and raw recognized tag rows for the flipped books while keeping the same cleaned range and duplicate-title treatment. Classify source risk by duplicate cleaned title, then more than two listed authors, then fewer than three restored recognized genre-tag rows; otherwise mark the risk as low_source_risk.

Question:
Audit the flipped books by restoring original_title as the display title, authors as the full author string, small_image_url as the cover link, and the three strongest restored recognized raw tag rows as evidence. Format tag evidence as tag_name:count entries separated by semicolons, ordered by descending count with tag_name as the tie-breaker. For each book, identify whether the main source risk is duplicate-title handling, author ambiguity, compressed genre evidence, or low source risk.