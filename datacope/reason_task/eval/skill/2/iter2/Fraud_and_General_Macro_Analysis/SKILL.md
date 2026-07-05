---
name: Fraud_and_General_Macro_Analysis
description: Solve fraud detection and macro-level transaction analysis questions on the dabstep payment dataset. Use this skill for any question involving the dabstep payments.csv data, including: fraud rate calculation, identifying highest/lowest transaction counts by category, missing data analysis, correlation between transaction attributes and fraud, comparisons between payment types (credit vs. debit), refused transaction analysis, and general aggregation/statistics over transaction data. Trigger whenever the question involves dabstep transaction data, fraud rates, card schemes, merchant statistics, or payment behavioral patterns.
---

# Fraud and General Macro Analysis — dabstep Dataset

This skill guides analysis of fraud patterns and macro-level transaction statistics over the dabstep synthetic payment dataset.

## Dataset Overview

The dabstep dataset contains **138,236 payment transactions** (all from year 2023) across these files:

| File | Purpose |
|------|---------|
| `payments.csv` | Main transactions table (primary data source, 21 columns) |
| `payments-readme.md` | Column definitions |
| `manual.md` | Domain definitions — **read this for fraud rate formula and terminology** |
| `fees.json` | Fee rules per card scheme and merchant type |
| `merchant_data.json` | Merchant metadata |
| `merchant_category_codes.csv` | MCC codes and descriptions |
| `acquirer_countries.csv` | Acquirer country reference |

Always read `manual.md` and `payments-readme.md` before analysis to confirm terminology, especially when the question uses domain terms like "fraud rate" or "volume."

## Key Columns in payments.csv

| Column | Type | Notes |
|--------|------|-------|
| `psp_reference` | ID | Unique transaction ID |
| `merchant` | Categorical | Merchant name (e.g., Crossfit_Hanna, Rafa_AI) |
| `card_scheme` | Categorical | NexPay, GlobalCard, SwiftCharge, TransactPlus |
| `year` | Numeric | All 2023 in this dataset |
| `hour_of_day` | Numeric | 0–23 |
| `minute_of_hour` | Numeric | 0–59 |
| `day_of_year` | Numeric | 1–365 |
| `eur_amount` | Numeric | Transaction amount in EUR |
| `ip_country` | Categorical | Country of shopper at transaction time (from IP) |
| `issuing_country` | Categorical | Country that issued the card |
| `shopper_interaction` | Categorical | Ecommerce or POS |
| `email_address` | ID / nullable | Hashed email — **null for ~10% of transactions** |
| `ip_address` | ID | Hashed shopper IP address |
| `card_number` | ID | Hashed card number |
| `card_bin` | ID | Bank Identification Number |
| `has_fraudulent_dispute` | Boolean | Fraud indicator (primary fraud signal) |
| `is_refused_by_adyen` | Boolean | Adyen refusal indicator |
| `is_credit` | Boolean | True = credit card, False = debit card |
| `device_type` | Categorical | Windows, Linux, MacOS, iOS, Android, Other |
| `aci` | Categorical | Authorization Characteristics Indicator (A–G) |
| `acquirer_country` | Categorical | Country of acquiring bank |

**Important**: `ip_country` and `issuing_country` are different columns. Questions about "country of the shopper" use `ip_country`; questions about "card issuing country" use `issuing_country`.

**Card scheme note**: The readme example mentions MasterCard/Visa, but actual data values are NexPay, GlobalCard, SwiftCharge, TransactPlus. Always use observed data values, not readme examples.

## Critical Rule: Answer Scale for Percentages

When a question asks "what percentage," the answer is always on the **0–100 scale** (e.g., `89.999711`), never the decimal fraction (e.g., `0.899997`).

- "What percentage of transactions have email?" → compute `(with_email / total) * 100` → `89.999711`
- "What percentage are fraudulent?" → compute `(fraud_count / total) * 100` → `7.787407`

> **Warning**: Instructions may show a format example like `<answer>0.23</answer>`. This is only a formatting hint — the `0.23` is a placeholder, not a hint about the expected magnitude or scale of your answer. Always compute from the data.

## Analysis Workflow

### Step 1: Load data and read documentation

```python
import pandas as pd

payments = pd.read_csv('<data_dir>/payments.csv')
# Also read manual.md for domain definitions when question uses terminology like "fraud rate"
```

### Step 2: Identify the question type

| Question Pattern | Approach |
|-----------------|---------|
| "Which X has the highest/most transactions?" | `value_counts()` or `groupby().count()` |
| "Which merchant has highest average amount?" | `groupby('merchant')['eur_amount'].mean().idxmax()` |
| "What percentage of transactions have X?" | `condition.sum() / len(df) * 100` (0–100 scale) |
| "Which card scheme has highest fraud rate?" | Volume-based fraud rate (see below) |
| "Is there a strong correlation between X and fraud?" | Pearson correlation, threshold > 0.50 |
| "Are credit cards more likely to be fraudulent?" | Compare fraud rates for credit vs. debit |
| "What percentage were refused?" | `is_refused_by_adyen.sum() / len(df) * 100` |

### Step 3: Apply the correct calculation

#### Fraud Rate (critical: use volume-based, not count-based)

Per `manual.md`: *"Fraud is defined as the ratio of fraudulent volume over total volume."*
**Volume = EUR amount**, not transaction count.

```python
# Correct: volume-based fraud rate
fraud_by_scheme = payments.groupby('card_scheme').agg(
    total_volume=('eur_amount', 'sum'),
    fraud_volume=('eur_amount', lambda x: x[payments.loc[x.index, 'has_fraudulent_dispute']].sum())
)
fraud_by_scheme['fraud_rate'] = fraud_by_scheme['fraud_volume'] / fraud_by_scheme['total_volume']
highest_fraud_scheme = fraud_by_scheme['fraud_rate'].idxmax()

# Wrong (count-based — produces different rankings):
# fraud_rate = fraudulent_count / total_count
```

Using transaction count instead of EUR volume is the most common error in this category and leads to wrong card scheme rankings.

#### Overall Fraud Percentage

```python
fraud_pct = payments['has_fraudulent_dispute'].sum() / len(payments) * 100
# Result: ~7.787407
```

#### Email Address Analysis

```python
# Percentage WITH email (the common question)
with_email = payments['email_address'].notnull().sum()
pct_with_email = with_email / len(payments) * 100
# Result: 89.999711 (124,412 transactions)

# Percentage WITHOUT email
missing_email = payments['email_address'].isnull().sum()
pct_without_email = missing_email / len(payments) * 100
# Result: 10.000289 (13,824 transactions)

# Most common card scheme among missing-email transactions:
most_common = payments[payments['email_address'].isnull()]['card_scheme'].value_counts().idxmax()
# Result: GlobalCard
```

#### Refused Transactions Analysis

```python
refused_pct = payments['is_refused_by_adyen'].sum() / len(payments) * 100
# Result: ~6.384010 (8,825 refused transactions)
```

#### Correlation with Fraud

```python
# Convert boolean to int for correlation
payments['fraud_int'] = payments['has_fraudulent_dispute'].astype(int)
corr = payments['hour_of_day'].corr(payments['fraud_int'])
is_strong = abs(corr) > 0.50
# hour_of_day vs fraud: corr ≈ -0.028 (not strong)
```

For binary vs. continuous correlation, Pearson on the 0/1 encoded variable is equivalent to point-biserial correlation. The threshold for "strong" correlation in this dataset is > 0.50 in absolute value.

#### Credit vs. Debit Fraud Comparison

```python
credit = payments[payments['is_credit'] == True]
debit = payments[payments['is_credit'] == False]

credit_fraud_rate = credit['has_fraudulent_dispute'].sum() / len(credit)
debit_fraud_rate = debit['has_fraudulent_dispute'].sum() / len(debit)
# credit: ~10.65%, debit: ~0%
answer = "yes" if credit_fraud_rate > debit_fraud_rate else "no"
```

## Answer Format Guidelines

Return answers as clean, minimal values — no explanations or units unless the question asks for them:

| Answer Type | Format | Example |
|-------------|--------|---------|
| Country code | Uppercase 2-letter code | `NL` |
| Merchant name | Exact name from data | `Crossfit_Hanna` |
| Card scheme | Exact name from data | `TransactPlus` |
| Interaction type | Exact name | `Ecommerce` |
| Percentage (0–100 scale) | Numeric, 6 decimal places | `89.999711` |
| Yes/No question | Lowercase | `yes` or `no` |
| Correlation conclusion | Based on abs(corr) > threshold | `yes` or `no` |

## Known Data Facts (use to verify results)

- Total transactions: 138,236
- All transactions are year 2023
- **Transactions WITH email: 124,412 (89.999711%)**
- Missing email addresses: 13,824 (10.000289%)
- Overall fraud rate: 7.787407% (10,765 fraudulent transactions)
- Refused transactions (is_refused_by_adyen): 8,825 (6.384010%)
- Most frequent country (ip_country): NL (29,760)
- Most frequent country (issuing_country): NL (29,622)
- Most common shopper interaction: Ecommerce (125,839 vs. 12,397 POS)
- Merchant with most transactions: Crossfit_Hanna (55,139)
- Merchant with highest average transaction amount: Crossfit_Hanna (~€92.07)
- Card scheme distribution: GlobalCard (48,150) > NexPay (41,679) > TransactPlus (34,674) > SwiftCharge (13,733)
- Card scheme with most missing-email transactions: GlobalCard (4,752)
- Card scheme with highest fraud rate (volume-based): TransactPlus (9.677%)
- Card scheme volume-based fraud rates: TransactPlus 9.677% > SwiftCharge 9.608% > NexPay 8.934% > GlobalCard 8.835%
- Card scheme with highest fraud rate (count-based): SwiftCharge (8.024%) — use volume-based for domain-correct answer
- Credit card fraud rate: ~10.65%; debit card fraud rate: ~0%
- hour_of_day vs. fraud correlation: ~-0.028 (not strong)
- Most common ACI: D (49,642), followed by F (29,266)
- Most common device type: Other (31,018), followed by Android (22,599)

## Common Pitfalls

1. **Confusing ip_country with issuing_country** — they are different columns and answer different questions about "where."

2. **Using transaction count for fraud rate** — the manual defines fraud rate as volume (EUR amount) ratio. Count-based ranking gives SwiftCharge as highest; volume-based gives TransactPlus. The domain-correct answer uses volume.

3. **Returning decimal instead of percentage** — if asked "what percentage," multiply by 100. `89.999711` is correct; `0.899997` is wrong format. This applies to all percentage questions: email presence, fraud rate, refusal rate, etc.

4. **Anchoring on instruction format examples** — the instructions sometimes show `<answer>0.23</answer>` as a formatting example. The value `0.23` is a placeholder — ignore it entirely. Always compute the answer from the data.

5. **Confusing "with email" vs. "without email"** — "what percentage have an associated email" asks for 89.999711% (those WITH email), not 10.000289% (those WITHOUT). Read the question carefully.

6. **Assuming card scheme values match the readme example** — the readme mentions MasterCard/Visa as examples, but actual data uses NexPay, GlobalCard, SwiftCharge, TransactPlus. Always check the data.

7. **Conflating statistical significance with practical correlation** — with 138,236 rows, even tiny correlations (0.028) are statistically significant. Evaluate effect size (magnitude) vs. the question's threshold, not p-value alone.
