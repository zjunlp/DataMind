---
name: dabstep-fraud-general-macro-analysis
description: Solve Fraud_and_General_Macro_Analysis questions in the dabstep payment processing dataset. Use this skill for macro-level transaction analysis questions: fraud rates, fraud comparisons by segment, percentage calculations, correlation analysis, most-frequent category lookups, and population-level statistics across merchants, countries, card schemes, and payment methods. Trigger whenever questions ask about fraud rates, fraudulent disputes, transaction counts by category, most/highest/lowest statistics, percentage of transactions, or correlation between payment attributes.
---

# Dabstep: Fraud & General Macro Analysis

## Dataset Overview

Primary file: `payments.csv` — 138,236 payment transactions with 21 columns (all year 2023).
Supporting files: `manual.md` (domain definitions), `payments-readme.md` (column schema).

### Key Columns
| Column | Type | Notes |
|--------|------|-------|
| `merchant` | categorical | Crossfit_Hanna, Belles_cookbook_store, Golfclub_Baron_Friso, Martinis_Fine_Steakhouse, Rafa_AI |
| `card_scheme` | categorical | NexPay, GlobalCard, SwiftCharge, TransactPlus |
| `year` | int | Only 2023 in the data |
| `hour_of_day` | int | 0–23 |
| `eur_amount` | float | Transaction amount in euros |
| `ip_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `issuing_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `acquirer_country` | categorical | SE, NL, LU, IT, BE, FR, GR, ES |
| `email_address` | str/NaN | Hashed; NaN means missing (13,824 missing) |
| `is_credit` | bool | True=credit, False=debit |
| `has_fraudulent_dispute` | bool | True=fraudulent transaction |
| `shopper_interaction` | categorical | Ecommerce, POS |
| `aci` | categorical | A–G (Authorization Characteristics Indicator) |
| `device_type` | categorical | Windows, Linux, MacOS, iOS, Android, Other |
| `is_refused_by_adyen` | bool | Payment refusal (separate from fraud) |

## CRITICAL: Fraud Metric Decision Rule

**Read the question wording carefully first — this is the #1 source of error:**

| Question wording | Formula | Result |
|------------------|---------|--------|
| "What **percentage of transactions** are fraudulent?" | `has_fraudulent_dispute.mean() * 100` | **7.787407** |
| "What is the **fraud rate**?" (fee/segment context) | `fraud_eur / total_eur * 100` | **9.154306** |
| "Which X has the highest **fraud rate**?" | volume-based per segment | varies |
| "Are credit payments **more likely** to result in fraud?" | count-based `.mean()` comparison | yes |

**"Percentage of transactions"** → ALWAYS count-based: `has_fraudulent_dispute.mean() * 100`

**"Fraud rate"** in a fee calculation or segment comparison context → volume-based: `sum(fraud eur_amount) / sum(total eur_amount) * 100`

The manual (Section 7) defines: "Fraud is defined as the ratio of fraudulent volume over total volume." This definition is specifically for the **fraud rate metric** used in fee calculations and segment comparisons — it does NOT override questions that explicitly ask about "percentage of transactions."

## Step 1: Always Read Documentation First

Before any analysis, read `manual.md` for domain rules, especially Section 7 (fraud definition) and Section 5 (fee structure). Apply the decision rule above to choose the right formula.

## Analysis Patterns

### Counting / Most-Frequent Queries
**"Which X has the highest number of transactions?" / "Most common Y?"**

```python
import pandas as pd
df = pd.read_csv('payments.csv')
result = df['column_name'].value_counts()
answer = result.idxmax()
```

### Percentage of Transactions
**"What percentage of transactions have/are X?"**

```python
total = len(df)
matching = df['column'].condition.sum()  # or .notna().sum() for non-null check
percentage = (matching / total) * 100
answer = round(percentage, 6)
```

**Return the value multiplied by 100** (e.g., `7.787407`), not the raw proportion (e.g., `0.077874`).

Fraudulent transaction percentage: `df['has_fraudulent_dispute'].mean() * 100`
Transactions with email: `df['email_address'].notna().mean() * 100`

### Fraud Rate by Segment (Volume-Based)
**"Which X has the highest fraud rate?" / "What is the fraud rate for Y?"**

Use volume-based per manual definition:

```python
df_filtered = df[df['year'] == 2023]  # apply any filters first

fraud_by_segment = df_filtered.groupby('card_scheme').apply(
    lambda g: g.loc[g['has_fraudulent_dispute'], 'eur_amount'].sum() / g['eur_amount'].sum(),
    include_groups=False
).reset_index(name='fraud_rate')

print(fraud_by_segment.sort_values('fraud_rate', ascending=False))
```

Volume-based ranking for card_scheme: TransactPlus > SwiftCharge > NexPay > GlobalCard.
Count-based ranking is different: SwiftCharge > NexPay > TransactPlus > GlobalCard.

### Comparing Groups
**"Are credit payments more likely to result in fraud?" / "Is X more likely than Y?"**

```python
credit_fraud = df[df['is_credit']]['has_fraudulent_dispute'].mean()
debit_fraud = df[~df['is_credit']]['has_fraudulent_dispute'].mean()
answer = 'yes' if credit_fraud > debit_fraud else 'no'
```

Note: Only credit transactions (`is_credit=True`) carry fraudulent disputes — debit fraud count is 0.

### Correlation Analysis
**"Is there a strong correlation (>threshold) between X and Y?"**

```python
from scipy.stats import pointbiserialr
corr, pvalue = pointbiserialr(df['hour_of_day'], df['has_fraudulent_dispute'])
answer = 'yes' if abs(corr) > 0.50 else 'no'
```

Use `abs(corr)` to handle negative correlations. `hour_of_day` vs `has_fraudulent_dispute` correlation is weak (~-0.028).

### Average / Aggregate Statistics
**"Which merchant has the highest average transaction amount?"**

```python
avg_by_merchant = df.groupby('merchant')['eur_amount'].mean()
answer = avg_by_merchant.idxmax()
```

## Critical Pitfalls

**Fraud metric confusion**: "Percentage of transactions" uses `has_fraudulent_dispute.mean() * 100` (= 7.787407%). "Fraud rate" per manual uses `sum(eur_amount where fraud) / sum(eur_amount) * 100` (= 9.154306%). These give different numbers — picking the wrong one is the #1 error. The manual's volume-based definition applies only when the question is explicitly about fraud *rate* in fee/segment context.

**Percentage vs proportion**: Return `value * 100` for percentage answers. Never report 0.077874 when the correct answer is 7.787407.

**Missing emails**: Use `df['email_address'].isna()` or `.notna()`. NaN means missing; do not use empty string check.

**Year filter**: Apply `df[df['year'] == 2023]` when the question specifies a year — only 2023 data exists but filter anyway per the question.

**Fraud vs refusal**: `has_fraudulent_dispute` = fraudulent dispute from issuing bank. `is_refused_by_adyen` = payment refusal (~6.38%). These are separate indicators.

**GroupBy deprecation**: Use `include_groups=False` in `.apply()` for newer pandas versions to avoid FutureWarning.

## Answer Format Rules

- **Categorical answers**: Use exact names from data (e.g., `Crossfit_Hanna`, `NL`, `GlobalCard`, `Ecommerce`)
- **Numerical answers**: Round to 6 decimal places unless specified otherwise
- **Yes/No questions**: `yes` or `no` (lowercase)
- **Not found**: `Not Applicable`
