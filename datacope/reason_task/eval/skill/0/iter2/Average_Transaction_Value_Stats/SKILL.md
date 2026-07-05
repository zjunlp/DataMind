---
name: Average_Transaction_Value_Stats
description: Skill for computing average transaction value statistics from payment transaction data in the dabstep dataset. Use this skill when a question asks about average transaction amount/value grouped by a categorical field (e.g., shopper_interaction, issuing_country, acquirer_country, aci, device_type, ip_country, is_credit, card_scheme, card_bin), possibly filtered by merchant, card scheme, and/or date range.
---

# Average Transaction Value Stats

This skill covers computing mean `eur_amount` from `payments.csv`, with optional filters on merchant, card scheme, and month range, grouped by a categorical column.

## Dataset Overview

**`payments.csv`** — core transaction table. Key columns:
- `merchant`: `Crossfit_Hanna`, `Rafa_AI`, `Golfclub_Baron_Friso`, `Belles_cookbook_store`, `Martinis_Fine_Steakhouse`
- `card_scheme`: `NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`
- `year`: always 2023 in this dataset
- `day_of_year`: 1–365 (no month column — compute month ranges using the table below)
- `eur_amount`: transaction amount in euros
- `shopper_interaction`: `Ecommerce` or `POS`
- `issuing_country`: `BE`, `ES`, `FR`, `GR`, `IT`, `LU`, `NL`, `SE`
- `acquirer_country`: `FR`, `GB`, `IT`, `NL`, `US` (NOT the same as issuing_country values)
- `ip_country`: `BE`, `ES`, `FR`, `GR`, `IT`, `LU`, `NL`, `SE`
- `aci`: `A`, `B`, `C`, `D`, `E`, `F`, `G`
- `device_type`: `Windows`, `Linux`, `MacOS`, `iOS`, `Android`, `Other`
- `is_credit`: `True` or `False`
- `card_bin`: 13 possible values (e.g., 4017, 4133, 4236, ...)
- `email_address`: hashed email (has NaN values — `groupby` excludes NaN automatically)
- `card_number`, `ip_address`: hashed IDs (no NaN)

**Merchant → acquirer_country mapping** (each merchant has a fixed acquirer):
- `Belles_cookbook_store` → `US`
- `Crossfit_Hanna` → `NL` (majority) and `GB`
- `Golfclub_Baron_Friso` → `IT`
- `Martinis_Fine_Steakhouse` → `FR`
- `Rafa_AI` → `NL`

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
Never omit the `card_scheme` filter when a card scheme is mentioned in the question.

### Multi-month ranges
"Between January and April" means months January through April inclusive (day_of_year 1–120). Include both endpoint months.

## Standard Solution Pattern

```python
import pandas as pd

df = pd.read_csv('/path/to/payments.csv')

# 1. Apply all filters the question specifies
mask = pd.Series([True] * len(df), index=df.index)

# Optional: filter by merchant
mask &= (df['merchant'] == 'Merchant_Name')

# Optional: filter by card scheme
mask &= (df['card_scheme'] == 'CardSchemeName')

# Optional: filter by date range
mask &= (df['day_of_year'] >= DAY_START) & (df['day_of_year'] <= DAY_END)

filtered = df[mask]

# 2. Group by the specified column and compute mean eur_amount
result = filtered.groupby('grouping_column')['eur_amount'].mean()

# 3. Round to 2 decimal places (unless question specifies otherwise)
result = result.round(2)

# 4. Sort ascending by value
result = result.sort_values(ascending=True)

# 5. Format as list of strings
answer = [f"{idx}: {val}" for idx, val in result.items()]
print(answer)
```

## Output Format

- **List of grouped averages**: `['GroupA: 71.18', 'GroupB: 86.79', ...]`
  - Sorted in **ascending order** by amount
  - Amounts rounded to **2 decimal places**
  - Use the grouping key exactly as it appears in the data (e.g., `FR`, `SE`, `Ecommerce`)
- **Single scalar average**: return as a number (e.g., `90.696`)

## Interpreting "Average per Unique X"

When the question asks for "average transaction amount per unique email/card/customer":
- Compute the mean per entity, then take the mean of those per-entity averages:
  ```python
  result = df.groupby('email_address')['eur_amount'].mean().mean()
  ```
- This differs from `total_amount / count_of_unique_entities` (which gives a wrong result).

## Common Mistakes to Avoid

1. **Wrong acquirer_country values**: acquirer_country is `FR, GB, IT, NL, US` — not the same as issuing_country (`BE, ES, FR, GR, IT, LU, NL, SE`).
2. **Missing card_scheme filter**: "NexPay transactions" = `card_scheme == 'NexPay'`, not just merchant filter.
3. **Wrong date range**: Verify month boundaries from the table. September–October is days 244–304.
4. **Sorting direction**: Always sort **ascending** by `eur_amount` unless the question says descending.
5. **Rounding display**: Use `f"{val:.2f}"` to show trailing zeros when formatting strings.
6. **NaN in email_address**: `groupby` automatically excludes NaN keys — no need to drop them explicitly.
