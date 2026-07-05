---
name: Average_Transaction_Value_Stats
description: Solve average transaction value statistics questions on the dabstep payments dataset. Use this skill for questions that ask for average transaction value/amount grouped by a dimension (e.g., shopper_interaction, issuing_country, acquirer_country, aci), optionally filtered by merchant, card scheme, and/or date range. Triggers on questions like "What is the average transaction value grouped by X for merchant Y's card_scheme Z transactions between months A and B?"
---

# Average Transaction Value Stats — dabstep Dataset

## Dataset Overview

**payments.csv** is the primary dataset. Key columns:
- `merchant`: Merchant name (e.g., `Crossfit_Hanna`, `Belles_cookbook_store`, `Golfclub_Baron_Friso`, `Martinis_Fine_Steakhouse`, `Rafa_AI`)
- `card_scheme`: Payment network (e.g., `NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`)
- `year`: Transaction year (all data is 2023)
- `day_of_year`: Day 1–365 (no date column — months must be derived from this)
- `eur_amount`: Transaction amount in euros
- `shopper_interaction`: `Ecommerce` or `POS`
- `issuing_country`: Card-issuing country code (SE, NL, LU, IT, BE, FR, GR, ES)
- `acquirer_country`: Acquiring bank country code
- `aci`: Authorization Characteristics Indicator (A–G)
- `email_address`: Hashed shopper email (can be NaN)

## Critical: Apply ALL Filters Specified in the Question

When a question says **"merchant X's card_scheme Y transactions"**, filter by **BOTH** `merchant` AND `card_scheme`. Forgetting one filter is the most common error.

```python
df_filtered = df[
    (df['merchant'] == 'Merchant_Name') &
    (df['card_scheme'] == 'SchemeName') &
    (df['day_of_year'] >= start_day) &
    (df['day_of_year'] <= end_day)
]
```

## Month → day_of_year Conversion (2023, non-leap year)

| Month     | day_of_year range |
|-----------|-------------------|
| January   | 1 – 31            |
| February  | 32 – 59           |
| March     | 60 – 90           |
| April     | 91 – 120          |
| May       | 121 – 151         |
| June      | 152 – 181         |
| July      | 182 – 212         |
| August    | 213 – 243         |
| September | 244 – 273         |
| October   | 274 – 304         |
| November  | 305 – 334         |
| December  | 335 – 365         |

Use the **start of the first month** and **end of the last month** as inclusive bounds.

## Standard Analysis Pattern

```python
import pandas as pd

df = pd.read_csv('/path/to/payments.csv')

# Step 1: Apply all filters
df_filtered = df[
    (df['merchant'] == 'Merchant_Name') &       # if specified
    (df['card_scheme'] == 'SchemeName') &         # if specified
    (df['day_of_year'] >= START_DAY) &            # if date range given
    (df['day_of_year'] <= END_DAY)
]

# Step 2: Group and compute mean
result = (
    df_filtered
    .groupby('grouping_column')['eur_amount']
    .mean()
    .round(2)
    .sort_values(ascending=True)  # always ascending unless stated otherwise
)

# Step 3: Format output
answer = [f"{k}: {v:.2f}" for k, v in result.items()]
```

## Output Format

The required format `[grouping_i: amount_i, ]` means a **Python list of strings**:
```python
['BE: 86.39', 'SE: 91.89', 'ES: 92.83', ...]
```
- Each element: `"GROUPING_VALUE: AMOUNT"` (colon-space separator)
- When grouping by country: use the country code
- Amounts rounded to **2 decimal places** (e.g., `101.00` not `101.0`)
- Sorted in **ascending order** by amount

**Do NOT** return tuples like `[('BE', 86.39), ...]`.

## Special Case: "Average per Unique X"

When the question says **"average transaction amount per unique email"** (or per unique card, etc.):

```python
# Correct: mean of per-group means
per_group_mean = df.groupby('email_address')['eur_amount'].mean()
answer = round(per_group_mean.mean(), 3)
```

This is NOT `total_amount / count_unique_emails`. The correct interpretation is:
1. For each unique email, compute their average transaction amount
2. Take the average of those per-email averages

## Validation Checklist

Before submitting your answer:
1. Did you filter by **all** criteria mentioned (merchant, card_scheme, date range)?
2. Did you use the correct `day_of_year` bounds for the requested months?
3. Is the output sorted **ascending** by amount?
4. Are amounts rounded to **2 decimal places** and formatted as strings with `:`?
5. For "per unique X" questions: did you use mean-of-means, not total/count?
