---
name: Average_Transaction_Value_Stats
description: Skill for computing average transaction value statistics from payment transaction data in the dabstep dataset. Use this skill when a question asks about average transaction amount/value grouped by a categorical field (e.g., shopper_interaction, issuing_country, acquirer_country, aci), possibly filtered by merchant, card scheme, and/or date range.
---

# Average Transaction Value Stats

This skill covers computing mean `eur_amount` from `payments.csv`, with optional filters on merchant, card scheme, and month range, grouped by a categorical column.

## Dataset Overview

**`payments.csv`** — core transaction table. Key columns:
- `merchant`: merchant name (e.g., `Crossfit_Hanna`, `Rafa_AI`, `Golfclub_Baron_Friso`, `Belles_cookbook_store`, `Martinis_Fine_Steakhouse`)
- `card_scheme`: payment network (`NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`)
- `year`: always 2023 in this dataset
- `day_of_year`: 1–365 (no month column — compute month ranges manually)
- `eur_amount`: transaction amount in euros
- `shopper_interaction`: `Ecommerce` or `POS`
- `issuing_country`, `acquirer_country`, `ip_country`: country codes (`SE`, `NL`, `LU`, `IT`, `BE`, `FR`, `GR`, `ES`)
- `aci`: Authorization Characteristics Indicator (`A`–`G`)
- `email_address`: hashed email (may have NaN values)

## Month → day_of_year Mapping (2023, non-leap year)

| Month | day_of_year range |
|-------|-------------------|
| January | 1–31 |
| February | 32–59 |
| March | 60–90 |
| April | 91–120 |
| May | 121–151 |
| June | 152–181 |
| July | 182–212 |
| August | 213–243 |
| September | 244–273 |
| October | 274–304 |
| November | 305–334 |
| December | 335–365 |

Use `(df['day_of_year'] >= start) & (df['day_of_year'] <= end)` to filter date ranges.

## Critical Filtering Rules

### Card scheme is a filter column, not part of the merchant name
When a question says **"Merchant_X's TransactPlus transactions"**, it means:
```python
df[(df['merchant'] == 'Merchant_X') & (df['card_scheme'] == 'TransactPlus')]
```
Never omit the `card_scheme` filter when a card scheme is mentioned in the question. This is the most common source of wrong answers.

### Multi-month ranges
"Between January and April" means months January through April inclusive (day_of_year 1–120). Include both endpoint months.

## Standard Solution Pattern

```python
import pandas as pd

df = pd.read_csv('/path/to/payments.csv')

# 1. Filter (apply all conditions that the question specifies)
mask = pd.Series([True] * len(df), index=df.index)

# Optional: filter by merchant
mask &= (df['merchant'] == 'Merchant_Name')

# Optional: filter by card scheme
mask &= (df['card_scheme'] == 'CardSchemeName')

# Optional: filter by date range (use day_of_year ranges from the table above)
mask &= (df['day_of_year'] >= DAY_START) & (df['day_of_year'] <= DAY_END)

filtered = df[mask]

# 2. Group by the specified column and compute mean eur_amount
result = filtered.groupby('grouping_column')['eur_amount'].mean()

# 3. Round (default: 2 decimal places unless question specifies otherwise)
result = result.round(2)

# 4. Sort ascending by value
result = result.sort_values(ascending=True)

# 5. Format as list of strings
answer = [f"{idx}: {val}" for idx, val in result.items()]
print(answer)
```

## Output Format

- **List of grouped averages**: `['GroupA: 71.18', 'GroupB: 86.79', ...]`
  - Elements sorted in **ascending order** by amount
  - Amounts rounded to **2 decimal places** (unless otherwise specified)
  - Use the grouping key exactly as it appears in the data (e.g., country codes like `FR`, `SE`)
- **Single scalar average** (e.g., "average transaction amount per unique email"): return as a number (e.g., `90.696`)

## Interpreting "Average per Unique X"

When the question asks for "average transaction amount per unique email" (or similar per-unique-entity phrasing):
- Compute the average transaction amount **for each entity**, then take the **mean of those per-entity averages**:
  ```python
  result = df.groupby('email_address')['eur_amount'].mean().mean()
  ```
- This is **not** `total_amount / count_of_unique_entities` (which gives a different, incorrect result).

## Common Mistakes to Avoid

1. **Missing card_scheme filter**: "NexPay transactions" = `card_scheme == 'NexPay'`, not just merchant filter.
2. **Wrong date range**: Check the month table. September–October is days 244–304, not 244–273.
3. **Sorting direction**: Always sort **ascending** by `eur_amount` unless the question says descending.
4. **Rounding**: Use `.round(2)` for 2 decimal places. For display ensure `f"{val:.2f}"` to show trailing zeros.
5. **NaN in email_address**: When grouping by `email_address`, pandas `groupby` automatically excludes NaN keys — no need to drop them explicitly.
