---
name: Average_Transaction_Value_Stats
description: Solve dabstep questions that ask for average transaction value (eur_amount) grouped by a categorical dimension (e.g., issuing_country, acquirer_country, aci, shopper_interaction, email_address), with optional filters for merchant, card_scheme, and date range (months). Use this skill whenever a question asks for average/mean transaction amounts broken down by a grouping column in the dabstep payments dataset.
---

# Average Transaction Value Stats

Questions in this category ask: "What is the average transaction value grouped by X for merchant Y's card_scheme Z transactions between months A and B?"

The answer is always a list of strings sorted ascending by average amount, formatted with exactly 2 decimal places.

## Data Source

Load `payments.csv`. Key columns:
- `merchant` — exact merchant name (e.g., `Crossfit_Hanna`, `Rafa_AI`, `Golfclub_Baron_Friso`, `Belles_cookbook_store`, `Martinis_Fine_Steakhouse`)
- `card_scheme` — exact scheme name (`NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`)
- `day_of_year` — integer 1–365; use this for month filtering (no separate month column)
- `eur_amount` — transaction amount in euros (the value to average)
- Grouping columns: `issuing_country`, `acquirer_country`, `aci`, `shopper_interaction`, `email_address`, `ip_country`, `device_type`, etc.

## Month → day_of_year Mapping (2023, non-leap year)

| Month | Start | End |
|-------|-------|-----|
| January | 1 | 31 |
| February | 32 | 59 |
| March | 60 | 90 |
| April | 91 | 120 |
| May | 121 | 151 |
| June | 152 | 181 |
| July | 182 | 212 |
| August | 213 | 243 |
| September | 244 | 273 |
| October | 274 | 304 |
| November | 305 | 334 |
| December | 335 | 365 |

For a range like "September and October", use start=244 and end=304.

## Solution Pattern

```python
import pandas as pd

df = pd.read_csv('/path/to/payments.csv')

# Apply filters
mask = (
    (df['merchant'] == 'MerchantName') &
    (df['card_scheme'] == 'SchemeName') &
    (df['day_of_year'] >= START_DAY) &
    (df['day_of_year'] <= END_DAY)
)
filtered = df[mask]

# Group and compute mean
avg = filtered.groupby('grouping_col')['eur_amount'].mean().round(2)
avg_sorted = avg.sort_values(ascending=True)

# Format — always use :.2f to ensure trailing zeros (e.g., 75.70 not 75.7)
result = [f"{k}: {v:.2f}" for k, v in avg_sorted.items()]
```

## Output Format

The answer is a Python list of strings: `['KEY1: XX.XX', 'KEY2: YY.YY', ...]`

- Sorted ascending by amount
- Each element: `"CATEGORY: VALUE"` — string format, **not** tuples like `('KEY', value)`
- All amounts use exactly 2 decimal places via `:.2f`

**Correct:** `['C: 75.70', 'A: 87.91', 'B: 131.71']`  
**Wrong:** `[('C', 75.7), ('A', 87.91), ('B', 131.71)]`

## Special Case: Grouping by Email Address

When asked for "average transaction value per unique email", compute the mean of per-email means (not total_amount / unique_email_count):

```python
# Correct: average of per-email averages
per_email_avg = df.groupby('email_address')['eur_amount'].mean()
result = per_email_avg.mean().round(3)

# Wrong: total / count
# result = df['eur_amount'].sum() / df['email_address'].nunique()
```

The two methods give different results. Use the mean-of-means approach unless the question explicitly says "total divided by unique count".

## Common Mistakes to Avoid

1. **Tuple output instead of strings** — the most frequent error. Always format as `f"{k}: {v:.2f}"`.
2. **Wrong date boundary** — for multi-month ranges, use the start of the first month and the end of the last month. "September and October" = days 244–304, not 244–273.
3. **Missing :.2f formatting** — pandas `.round(2)` alone may drop trailing zeros (e.g., `75.7`). Always apply `:.2f` in the format string.
4. **Wrong averaging for email grouping** — see Special Case above.

## Verification

After computing, sanity-check:
- Count of filtered rows is reasonable (hundreds to thousands for a 2-month merchant+scheme filter)
- All grouping keys present in results match the unique values in the filtered data
- Values are sorted ascending
