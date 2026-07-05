---
name: Fraud_and_General_Macro_Analysis
description: Skill for analyzing payment transaction fraud patterns and general macro-level statistics in the dabstep dataset. Use when questions involve: fraud rates by card scheme/merchant/year, percentage of fraudulent transactions, correlation between transaction attributes and fraud, credit vs debit fraud likelihood, transaction counts by country/merchant/card scheme, email address coverage, shopper interaction types, or any aggregate analysis over the payments.csv dataset. Always invoke this skill for dabstep Fraud_and_General_Macro_Analysis category questions.
---

# Fraud and General Macro Analysis — dabstep Dataset

## Dataset Overview

**Primary file**: `payments.csv` — 138,236 payment transactions (year=2023 only).

**Key columns**:
- `merchant`: Merchant name
- `card_scheme`: Card scheme (actual values: `NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`)
- `year`: Payment year (only 2023 in dataset)
- `hour_of_day`: Hour of payment (0–23)
- `eur_amount`: Transaction amount in euros
- `ip_country`: Shopper's IP-based country
- `issuing_country`: Card-issuing country
- `is_credit`: True=credit card, False=debit card
- `email_address`: Hashed email (NaN = missing; 13,824 missing out of 138,236)
- `shopper_interaction`: `Ecommerce` or `POS`
- `has_fraudulent_dispute`: Boolean — True if issuing bank flagged as fraud
- `acquirer_country`: Country of acquiring bank

**Other reference files** (load only when needed):
- `manual.md` — domain definitions, especially fraud rate formula
- `payments-readme.md` — column descriptions
- `merchant_data.json` — merchant metadata
- `fees.json` — fee rules

---

## Critical: Fraud Rate Definition

Per `manual.md` Section 7:
> **Fraud is defined as the ratio of fraudulent volume over total volume.**

This means fraud rate = **sum of eur_amount where has_fraudulent_dispute=True** divided by **sum of all eur_amount** — NOT a count-based ratio.

```python
fraud_rate = df[df['has_fraudulent_dispute']]['eur_amount'].sum() / df['eur_amount'].sum()
```

**Common mistake**: Using `has_fraudulent_dispute.mean()` (count-based) instead of volume-based calculation. These give different rankings.

---

## Standard Workflow

### Step 1: Read documentation first
Always read `manual.md` and `payments-readme.md` before analyzing. Domain definitions (especially fraud rate) are essential for correct answers.

### Step 2: Load and inspect data
```python
import pandas as pd
payments = pd.read_csv('/path/to/payments.csv')
print(payments.shape, payments.dtypes)
```

### Step 3: Apply relevant filters
- If question specifies a year, filter first: `df = payments[payments['year'] == 2023]`
- For missing-value questions, use `.isnull()` / `.notna()`

### Step 4: Compute the metric
Follow the patterns below for each question type.

---

## Analysis Patterns

### Fraud rate by group (card_scheme, merchant, etc.)
```python
# Volume-based fraud rate (CORRECT per manual)
fraud_stats = df.groupby('card_scheme').agg(
    total_volume=('eur_amount', 'sum'),
    fraud_volume=('eur_amount', lambda x: x[df.loc[x.index, 'has_fraudulent_dispute']].sum())
)
fraud_stats['fraud_rate'] = fraud_stats['fraud_volume'] / fraud_stats['total_volume']
highest = fraud_stats['fraud_rate'].idxmax()
```

### Percentage of fraudulent transactions (count-based for "what % are fraudulent")
When asked "what percentage of transactions are fraudulent" (not fraud rate by volume):
```python
pct = payments_2023['has_fraudulent_dispute'].mean() * 100
# Round to 6 decimal places if required
round(pct, 6)
```

### Percentage with non-null field (e.g., email address)
```python
pct = payments['email_address'].notna().mean() * 100  # Returns percentage (0–100)
round(pct, 6)
```
**Critical**: Multiply by 100 to get percentage (not proportion 0–1).

### Most common value in a column
```python
most_common = payments['card_scheme'].value_counts().idxmax()
# OR
most_common = payments['shopper_interaction'].mode()[0]
```

### Highest count by group
```python
top_country = payments['ip_country'].value_counts().idxmax()
top_merchant = payments['merchant'].value_counts().idxmax()
```

### Highest average transaction amount
```python
top_merchant = payments.groupby('merchant')['eur_amount'].mean().idxmax()
```

### Correlation between numeric and boolean columns
```python
from scipy import stats
# Point-biserial correlation = Pearson when one variable is binary
corr, pval = stats.pointbiserialr(payments['hour_of_day'], payments['has_fraudulent_dispute'].astype(int))
strong = abs(corr) > 0.50
answer = 'yes' if strong else 'no'
```

### Credit vs debit fraud likelihood
```python
credit_rate = payments[payments['is_credit']]['has_fraudulent_dispute'].mean()
debit_rate = payments[~payments['is_credit']]['has_fraudulent_dispute'].mean()
answer = 'yes' if credit_rate > debit_rate else 'no'
```

### Filtering by missing values
```python
missing_email = payments[payments['email_address'].isnull()]
top_scheme = missing_email['card_scheme'].value_counts().idxmax()
```

---

## Output Format Rules

- **Numerical percentages**: Round to 6 decimal places — `round(value, 6)`
- **Country/merchant/scheme names**: Return exact string as-is from data
- **yes/no questions**: Return lowercase `yes` or `no`
- **Not applicable**: If the question has no relevant answer, return `Not Applicable`
- **Single value answers**: Return just the value, no extra text

---

## Key Data Facts (verify these don't change)

| Fact | Value |
|------|-------|
| Total transactions | 138,236 |
| Years in dataset | 2023 only |
| Card schemes | NexPay, GlobalCard, SwiftCharge, TransactPlus |
| Fraudulent transactions | 10,765 (all are credit card: is_credit=True) |
| Shopper interactions | Ecommerce (125,839), POS (12,397) |
| Missing email_address | 13,824 rows |
| Top IP country | NL (29,760) |
| Top issuing country | NL (29,622) |
| Top merchant by count | Crossfit_Hanna (55,139) |

---

## Common Pitfalls

1. **Volume vs count fraud rate**: The manual defines fraud rate as volume-based. When asked which card scheme has the *highest fraud rate*, use `eur_amount` sums — this may give a different answer than count-based calculation.

2. **Percentage vs proportion**: "What percentage..." expects a number like `89.999711` (multiply by 100). Don't return `0.899997`.

3. **Year filter**: Always check if the question specifies a year. Filter before computing.

4. **card_scheme values**: The readme mentions MasterCard/Visa/Amex/Other, but the actual data uses NexPay/GlobalCard/SwiftCharge/TransactPlus. Trust the data, not the readme examples.

5. **Credit-only fraud**: In this dataset, all fraudulent disputes (has_fraudulent_dispute=True) are on credit transactions (is_credit=True). Debit transactions have 0 fraudulent disputes.
