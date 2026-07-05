---
name: Average_Transaction_Value_Stats
description: Solve dabstep Average_Transaction_Value_Stats questions that ask for average transaction value (eur_amount) grouped by a dimension (shopper_interaction, issuing_country, acquirer_country, aci) for a specific merchant and card scheme over a date range. Use this skill whenever a question asks for average transaction value/amount broken down by a grouping variable with merchant and/or card scheme filters on the payments dataset.
---

# Average Transaction Value Stats — Solver Guide

## Task Pattern

Questions in this category ask:
> "What is the average transaction value grouped by **[dimension]** for **[merchant]**'s **[card_scheme]** transactions between **[month_start]** and **[month_end]** [year]?"

Dimensions include: `shopper_interaction`, `issuing_country`, `acquirer_country`, `aci`, or `email_address`.

## Dataset

Use only `payments.csv`. Key columns:
- `merchant` — exact merchant name (e.g., `Crossfit_Hanna`, `Golfclub_Baron_Friso`, `Belles_cookbook_store`, `Rafa_AI`, `Martinis_Fine_Steakhouse`)
- `card_scheme` — one of: `NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`
- `year` — all data is 2023
- `day_of_year` — integer 1–365 (no actual date column exists; convert months to day ranges)
- `eur_amount` — transaction amount in euros
- `shopper_interaction` — `Ecommerce` or `POS`
- `issuing_country`, `acquirer_country` — country codes (SE, NL, LU, IT, BE, FR, GR, ES, US, ...)
- `aci` — Authorization Characteristics Indicator: A, B, C, D, E, F, G
- `email_address` — hashed email (may contain NaN)

## 2023 Month → day_of_year Mapping (Non-Leap Year)

| Month     | day_of_year range |
|-----------|-------------------|
| January   | 1–31              |
| February  | 32–59             |
| March     | 60–90             |
| April     | 91–120            |
| May       | 121–151           |
| June      | 152–181           |
| July      | 182–212           |
| August    | 213–243           |
| September | 244–273           |
| October   | 274–304           |
| November  | 305–334           |
| December  | 335–365           |

For a multi-month range (e.g., "between May and June"), use the start of the first month through the end of the last month.

## Standard Solution Steps

```python
import pandas as pd

df = pd.read_csv('<data_path>/payments.csv')

# Step 1: Apply ALL filters — both merchant AND card_scheme are required
filtered = df[
    (df['merchant'] == 'Merchant_Name') &
    (df['card_scheme'] == 'CardSchemeName') &
    (df['day_of_year'] >= START_DAY) &
    (df['day_of_year'] <= END_DAY)
]
# year filter not needed if all data is 2023, but add (df['year'] == 2023) if uncertain

# Step 2: Group and compute mean
avg = filtered.groupby('dimension_column')['eur_amount'].mean()

# Step 3: Sort ascending by value, round to 2 decimal places
avg_sorted = avg.sort_values(ascending=True).round(2)

# Step 4: Format as list of "KEY: VALUE" strings with exactly 2 decimal places
result = [f"{k}: {v:.2f}" for k, v in avg_sorted.items()]
print(result)
```

## Critical Rules

### Always filter by card_scheme
The question says "Merchant's **CardScheme** transactions" — both filters are mandatory. Omitting card_scheme filter yields wrong results (different transaction population).

### Use exact 2 decimal place formatting
Use `f"{value:.2f}"` not `round(value, 2)` in format strings. Python drops trailing zeros (75.7 ≠ 75.70 as a string), but answers require `75.70`.

### Output format
Produce a list of strings: `['BE: 86.39', 'SE: 91.89', ...]` — not tuples. The question specifies `[grouping_i: amount_i, ]` format.

### Sort ascending by amount
Always sort by the computed average value, not alphabetically by the group key.

## Special Case: "Average transaction amount per unique email"

When the question asks for "average transaction amount per unique email":
- **Correct interpretation**: For each unique email, compute the average of its transactions; then take the mean of those per-email averages.
- **Wrong interpretation**: total_eur_amount / count_of_unique_emails

```python
# Correct
per_email_avg = df.groupby('email_address')['eur_amount'].mean()
result = per_email_avg.mean().round(3)
```

NaN email addresses are automatically excluded by groupby.

## Common Errors to Avoid

1. **Missing card_scheme filter** — The most frequent source of wrong answers. Always apply it.
2. **Wrong date boundaries** — Double-check month boundaries using the table above. "Between January and April" = days 1–120.
3. **Tuple format instead of string format** — Output `['FR: 71.18', ...]` not `[('FR', 71.18), ...]`.
4. **Floating-point display** — Use `:.2f` format; `round(x, 2)` alone won't guarantee two decimal places in string output.

## Example

**Query**: "Average transaction value grouped by `issuing_country` for Golfclub_Baron_Friso's NexPay transactions between May and June 2023?"

```python
filtered = df[
    (df['merchant'] == 'Golfclub_Baron_Friso') &
    (df['card_scheme'] == 'NexPay') &
    (df['day_of_year'] >= 121) &   # May 1
    (df['day_of_year'] <= 181)     # June 30
]
avg = filtered.groupby('issuing_country')['eur_amount'].mean().sort_values()
result = [f"{k}: {v:.2f}" for k, v in avg.items()]
# → ['FR: 71.18', 'GR: 77.56', 'NL: 86.79', 'IT: 90.72', 'BE: 95.83', 'SE: 105.05', 'ES: 109.18', 'LU: 143.98']
```
