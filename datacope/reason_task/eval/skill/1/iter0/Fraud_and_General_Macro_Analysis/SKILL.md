---
name: dabstep-fraud-general-macro-analysis
description: Solve Fraud_and_General_Macro_Analysis questions in the dabstep payment processing dataset. Use this skill for macro-level transaction analysis questions: fraud rates, fraud comparisons by segment, percentage calculations, correlation analysis, most-frequent category lookups, and population-level statistics across merchants, countries, card schemes, and payment methods. Trigger whenever questions ask about fraud rates, fraudulent disputes, transaction counts by category, most/highest/lowest statistics, percentage of transactions, or correlation between payment attributes.
---

# Dabstep: Fraud & General Macro Analysis

## Dataset Overview

Primary file: `payments.csv` — 138,236 payment transactions with 21 columns.
Supporting files: `manual.md` (domain definitions), `payments-readme.md` (column schema).

### Key Columns
| Column | Type | Notes |
|--------|------|-------|
| `merchant` | categorical | Crossfit_Hanna, Belles_cookbook_store, Golfclub_Baron_Friso, Martinis_Fine_Steakhouse, Rafa_AI |
| `card_scheme` | categorical | NexPay, GlobalCard, SwiftCharge, TransactPlus |
| `year` | int | Currently only 2023 in data |
| `hour_of_day` | int | 0–23 |
| `eur_amount` | float | Transaction amount in euros |
| `ip_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `issuing_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `acquirer_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `email_address` | str/NaN | Hashed; NaN means missing (13,824 missing) |
| `is_credit` | bool | True=credit, False=debit |
| `has_fraudulent_dispute` | bool | True=fraudulent |
| `shopper_interaction` | categorical | Ecommerce, POS |
| `aci` | categorical | A–G (Authorization Characteristics Indicator) |

## Step 1: Always Read Documentation First

Before any analysis:
1. Read `manual.md` — authoritative fraud definition and domain rules
2. Read `payments-readme.md` — column-by-column schema

**Critical definition from manual.md:**
> "Fraud is defined as the ratio of fraudulent volume over total volume."

"Volume" = monetary volume (sum of `eur_amount`), NOT transaction count. This distinction changes results.

## Analysis Patterns

### Counting / Most-Frequent Queries
**"Which X has the highest number of transactions?" / "Most common Y?"**

```python
import pandas as pd
df = pd.read_csv('payments.csv')
result = df['column_name'].value_counts()
print(result)
answer = result.idxmax()
```

### Percentage Calculations
**"What percentage of transactions have/are X?"**

```python
total = len(df)
matching = df['column'].condition.sum()  # or .notna().sum() for non-null
percentage = (matching / total) * 100
answer = round(percentage, 6)
```

**Critical:** Return the value multiplied by 100 (e.g., `89.999711`), not the raw proportion (e.g., `0.899997`). The question asks for a percentage.

Missing values check: `df['email_address'].notna().sum()` counts rows with email.

### Fraud Rate by Segment
**"Which X has the highest fraud rate?" / "What is the fraud rate for Y?"**

Fraud rate = **fraudulent volume / total volume** (monetary, not count):

```python
df_filtered = df[df['year'] == 2023]  # apply any filters first

fraud_by_segment = df_filtered.groupby('card_scheme').apply(
    lambda g: g.loc[g['has_fraudulent_dispute'], 'eur_amount'].sum() / g['eur_amount'].sum()
).reset_index(name='fraud_rate')

print(fraud_by_segment.sort_values('fraud_rate', ascending=False))
```

**Do NOT use transaction count** (`has_fraudulent_dispute.mean()`) for fraud rate — that gives count-based ratio, contradicting the manual definition and producing wrong segment rankings.

### Comparing Groups
**"Are credit payments more likely to result in fraud?" / "Is X more likely than Y?"**

```python
credit_fraud = df[df['is_credit']]['has_fraudulent_dispute'].mean()
debit_fraud = df[~df['is_credit']]['has_fraudulent_dispute'].mean()
answer = 'yes' if credit_fraud > debit_fraud else 'no'
```

Note: The dataset shows only credit transactions (`is_credit=True`) carry fraudulent disputes — debit fraud rate is 0%.

### Correlation Analysis
**"Is there a strong correlation (>threshold) between X and Y?"**

For continuous vs binary variables, use point-biserial correlation (equivalent to Pearson):

```python
from scipy.stats import pointbiserialr
corr, pvalue = pointbiserialr(df['hour_of_day'], df['has_fraudulent_dispute'])
answer = 'yes' if abs(corr) > 0.50 else 'no'
```

Use `abs(corr)` to handle negative correlations.

### Average / Aggregate Statistics
**"Which merchant has the highest average transaction amount?"**

```python
avg_by_merchant = df.groupby('merchant')['eur_amount'].mean()
answer = avg_by_merchant.idxmax()
```

## Critical Pitfalls

**Fraud rate definition**: "Fraudulent volume over total volume" = sum(eur_amount where fraud) / sum(eur_amount). Using count ratio yields different segment rankings (SwiftCharge ≠ TransactPlus).

**Percentage vs proportion**: When asked for "percentage", return `value * 100`. Never report 0.899997 when the correct answer is 89.999711.

**Missing emails**: Use `df['email_address'].isna()` or `.notna()`. Empty string `''` and NaN are different; the data uses NaN for missing.

**Boolean columns**: `is_credit` and `has_fraudulent_dispute` are Python bools. Sum them directly or use `.mean()` for proportions.

**Year filter**: Although only 2023 data exists, always apply year filters when the question specifies a year — do not assume.

**Fraud and `is_refused_by_adyen`**: These are separate indicators. `has_fraudulent_dispute` = fraud from issuing bank; `is_refused_by_adyen` = payment refusal. Do not conflate them.

## Answer Format Rules

- **Categorical answers**: Use exact names from data (e.g., `Crossfit_Hanna`, `NL`, `GlobalCard`, `Ecommerce`)
- **Numerical answers**: Round to 6 decimal places unless specified otherwise
- **Yes/No questions**: `yes` or `no` (lowercase)
- **Not found**: `Not Applicable`
