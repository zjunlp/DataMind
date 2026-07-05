---
name: Fraud_and_General_Macro_Analysis
description: Solve fraud analysis and general macro transaction analysis questions on the dabstep payment dataset. Use this skill for questions about fraud rates, transaction distributions, merchant rankings, card scheme analysis, country breakdowns, correlation studies, and yes/no comparative questions on payment data.
---

# Fraud and General Macro Analysis

## Dataset Overview

Primary file: `payments.csv` — 138,236 payment transactions.

Key columns:
- `merchant`: merchant name (e.g., Crossfit_Hanna, Golfclub_Baron_Friso)
- `card_scheme`: NexPay, GlobalCard, SwiftCharge, TransactPlus
- `year`: payment year
- `hour_of_day`: 0–23
- `eur_amount`: transaction amount in euros
- `ip_country`, `issuing_country`, `acquirer_country`: country codes (SE, NL, LU, IT, BE, FR, GR, ES)
- `device_type`: Windows, Linux, MacOS, iOS, Android, Other
- `shopper_interaction`: Ecommerce or POS
- `has_fraudulent_dispute`: **Boolean — the fraud indicator** (True = fraudulent)
- `is_refused_by_adyen`: Boolean — refusal indicator (NOT the same as fraud)
- `is_credit`: Boolean — True = credit card, False = debit card
- `email_address`: hashed email, NaN when missing
- `aci`: Authorization Characteristics Indicator (A–G)
- `card_bin`: Bank Identification Number

Other files: `merchant_data.json`, `fees.json`, `acquirer_countries.csv`, `merchant_category_codes.csv`, `manual.md`, `payments-readme.md`.

## Critical Domain Rules

### Fraud Rate Definition
**Fraud = ratio of fraudulent volume over total volume** (from manual.md, Section 7).

```python
# CORRECT: volume-based fraud rate
fraud_rate = df[df['has_fraudulent_dispute']]['eur_amount'].sum() / df['eur_amount'].sum()

# WRONG: count-based rate gives different results and wrong answers
# fraud_rate = df['has_fraudulent_dispute'].mean()  # DO NOT USE for fraud rate
```

This matters: volume-based and count-based calculations yield different rankings across card schemes.

### Percentage vs Proportion
When a question asks for a **percentage**, return a value in the 0–100 range (multiply proportion by 100).

```python
# "What percentage of transactions have email addresses?"
pct = (df['email_address'].notna().sum() / len(df)) * 100  # → 89.999711
# NOT: df['email_address'].notna().mean()  # → 0.899997 (wrong for percentage questions)
```

### Fraud Indicator
Always use `has_fraudulent_dispute` for fraud. Never use `is_refused_by_adyen` as a proxy for fraud.

## Analysis Patterns

### Step 1: Load and inspect
```python
import pandas as pd
df = pd.read_csv('payments.csv')
print(df.shape, df.dtypes)
```

### Fraud rate by group (volume-based)
```python
def fraud_rate_by_group(df, group_col):
    g = df.groupby(group_col).apply(
        lambda x: x.loc[x['has_fraudulent_dispute'], 'eur_amount'].sum() / x['eur_amount'].sum()
    )
    return g.sort_values(ascending=False)
```

### Fraud percentage vs fraud rate — key disambiguation

These two look similar but require different calculations:

- **"What percentage of transactions are fraudulent?"** → count-based:
  ```python
  period_df = df[df['year'] == 2023]
  fraud_pct = (period_df['has_fraudulent_dispute'].sum() / len(period_df)) * 100
  ```
- **"What is the fraud rate / avg fraud rate?"** → volume-based (per manual definition):
  ```python
  fraud_rate = (period_df.loc[period_df['has_fraudulent_dispute'], 'eur_amount'].sum()
                / period_df['eur_amount'].sum())
  ```

Volume-based and count-based rates yield different group rankings, so misapplying them produces wrong answers.

### Top/bottom rankings
```python
# Most common / highest count
df['merchant'].value_counts().idxmax()

# Highest average transaction amount
df.groupby('merchant')['eur_amount'].mean().idxmax()

# Most frequent card scheme where email is missing
df[df['email_address'].isna()]['card_scheme'].value_counts().idxmax()
```

### Correlation analysis (continuous vs binary)
```python
from scipy import stats
corr, pval = stats.pointbiserialr(df['hour_of_day'], df['has_fraudulent_dispute'].astype(int))
# Pearson on numeric+bool is equivalent; threshold 0.5 for "strong"
```

### Comparing groups (yes/no questions)
```python
# "Are credit payments more likely to be fraudulent?"
credit_rate = df[df['is_credit']]['has_fraudulent_dispute'].mean()
debit_rate = df[~df['is_credit']]['has_fraudulent_dispute'].mean()
answer = "yes" if credit_rate > debit_rate else "no"
```

## Common Question Types and Approach

| Question type | Key columns | Method |
|---|---|---|
| Highest/lowest fraud rate by group | `has_fraudulent_dispute`, `eur_amount` | Volume-based groupby |
| % transactions with property | relevant col | count notna or condition / total × 100 |
| Top merchant by count | `merchant` | `value_counts().idxmax()` |
| Top merchant by avg amount | `merchant`, `eur_amount` | `groupby.mean().idxmax()` |
| Country with most transactions | `ip_country` or `issuing_country` | `value_counts().idxmax()` |
| Most common category | any categorical col | `value_counts().idxmax()` |
| Correlation strength | continuous + binary cols | point-biserial, check abs > threshold |
| Credit vs debit fraud comparison | `is_credit`, `has_fraudulent_dispute` | Compare fraud rates |
| Missing data patterns | `email_address`, `ip_address` | `.isna()` filter then analyze |

## Output Format Rules

- **Numerical answers**: round to the decimal places specified in the question (often 6)
- **Categorical answers**: use the exact string from the data (e.g., "Crossfit_Hanna", not "crossfit_hanna")
- **Yes/no questions**: return exactly "yes" or "no" (lowercase)
- **Country codes**: return the code (e.g., "NL"), not the full country name
- If no applicable answer exists after exhausting all approaches, return "Not Applicable"

## Workflow

1. Read `manual.md` first — it defines fraud, ACI codes, account types, and fee rules
2. Read `payments-readme.md` — understand exact column names and value ranges
3. Load `payments.csv` with pandas
4. Apply any time/segment filters (e.g., year == 2023) before computing metrics
5. Choose volume-based vs count-based method based on the question's semantics
6. Round to specified precision and return in the exact required format
