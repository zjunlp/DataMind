---
name: dabstep-dataset-metadata-business-rules
description: Solve Dataset_Metadata_and_Business_Rules questions in the dabstep payment processing dataset. Use this skill for questions about dataset column meanings, fee factor directionality, business rules encoded in fee structures, "Not Applicable" detection, and metadata interpretation for the dabstep payment processing domain. Trigger whenever questions reference payment fees, fraud indicators, boolean fee factors, volume tiers, ACI codes, or any field from the dabstep dataset.
---

# Dabstep: Dataset Metadata & Business Rules

## Dataset Overview

Seven files describe a payment processing system:

| File | Contents |
|------|----------|
| `payments.csv` | Transaction records (fraud flag, amounts, card types, countries) |
| `payments-readme.md` | Column-by-column documentation for payments.csv |
| `fees.json` | 1000 fee rules with conditions and rates |
| `manual.md` | Domain definitions, fee factor descriptions, business rules |
| `merchant_data.json` | Per-merchant config (account type, capture delay, acquirers) |
| `merchant_category_codes.csv` | MCC codes and descriptions |
| `acquirer_countries.csv` | Country codes for acquirers |

## Step 1: Always Read Documentation First

Before any analysis, read both:
1. `manual.md` — authoritative definitions of all fee factors and their cost direction
2. `payments-readme.md` — column-by-column schema for payments.csv

These files contain the ground truth for interpreting field semantics. Do not rely on raw data averages alone for factor directionality questions.

## Key Domain Knowledge (from manual.md)

### Fee Formula
```
fee = fixed_amount + rate * transaction_value / 10000
```
`null` values in fee rules mean "applies to all values of that field."

### Fee Factor Directionality (authoritative — from manual.md)

| Factor | Direction | Cheaper when... |
|--------|-----------|-----------------|
| `is_credit` | True → more expensive | `is_credit = False` (debit) |
| `intracountry` | True → domestic → cheaper | `intracountry = True` |
| `monthly_fraud_level` | Higher % → more expensive | Lower fraud level |
| `monthly_volume` | Higher volume → cheaper | Higher monthly volume |
| `capture_delay` | Faster → more expensive | Slower (manual/delayed) |

### Boolean Factors Summary
- `is_credit`: Credit cards cost more → cheaper when **False**
- `intracountry`: Domestic (True) is cheaper; international (False) is more expensive → cheaper when **True**

### Volume Tiers (lowest to highest)
`<100k` → `100k-1m` → `1m-5m` → `>5m`

### Fraud Level Tiers (lowest to highest)
`<7.2%` → `7.2%-7.7%` → `7.7%-8.3%` → `>8.3%`

### payments.csv Key Columns
- `has_fraudulent_dispute`: Boolean — fraud indicator from issuing bank
- `is_credit`: Boolean — credit vs debit card
- `intracountry`: Derived from `issuing_country == acquirer_country`
- `aci`: Authorization Characteristics Indicator (A–G)

### ACI Codes
| Code | Meaning |
|------|---------|
| A | Card present - Non-authenticated |
| B | Card Present - Authenticated |
| C | Tokenized card with mobile device |
| D | Card Not Present - Card On File |
| E | Card Not Present - Recurring Bill Payment |
| F | Card Not Present - 3-D Secure |
| G | Card Not Present - Non-3-D Secure |

## Question-Type Strategies

### 1. Metadata / Column Identification
**"What column indicates X?" / "What does field Y mean?"**

Check `payments-readme.md` first (columns), then `manual.md` (definitions).

Example: fraud column → `has_fraudulent_dispute` (from payments-readme.md: "Indicator of fraudulent dispute from issuing bank (Boolean)")

### 2. Boolean Fee Factor Questions
**"Which boolean factor is cheaper when True/False?"**

Rely on manual.md explicit statements, not raw data averages. Raw averages are confounded by other correlated variables (card scheme, MCC, etc.) and can be misleading.

Boolean factors in fees.json: `is_credit`, `intracountry`

- Cheaper when **True**: `intracountry` (domestic transactions are cheaper)
- Cheaper when **False**: `is_credit` (non-credit/debit is cheaper)

### 3. General Factor Directionality
**"Which factors lead to cheaper fees when decreased/increased?"**

Map each factor using manual.md definitions:
- Decreasing → cheaper: `monthly_fraud_level`, `is_credit` (True→False)
- Increasing → cheaper: `monthly_volume`, `intracountry` (False→True)
- Decreasing → more expensive: `capture_delay` (faster = more expensive), `monthly_volume`, `intracountry`

### 4. Volume/Threshold Analysis
**"At what volume do fees stop decreasing?" / "Highest volume where fees are not cheaper?"**

Analyze actual fee data from fees.json:
```python
import json, pandas as pd
with open('fees.json') as f:
    fees = json.load(f)
df = pd.DataFrame(fees)

for vol in ['<100k', '100k-1m', '1m-5m', '>5m']:
    s = df[df['monthly_volume'] == vol]
    print(vol, s['rate'].mean(), s['fixed_amount'].mean())
```

The actual data shows `>5m` has the highest average rate — fees do **not** become cheaper at `>5m`. The `>5m` tier is the highest volume at which fees are not cheaper.

### 5. "Not Applicable" Detection
**"Does X fee exist?" / "What is the value of Y?"**

A question should be answered "Not Applicable" when the concept genuinely does not exist in any dataset file. Verify by:
1. Searching manual.md for the concept
2. Searching all keys in fees.json
3. Checking payments.csv columns
4. Searching payments-readme.md

Example: "excessive retry fee" — manual mentions excessive retrying causes downgrades but defines no specific retry fee amount → answer is "Not Applicable"

## Common Pitfalls

**Don't trust raw averages for boolean factors.** The raw average rate for `is_credit=True` can appear lower than `is_credit=False` due to confounding (different MCCs and card schemes have both different base rates and different credit ratios). The manual's explicit statement is authoritative.

**intracountry in fees.json uses 0.0/1.0 floats**, not Python booleans. `1.0 = True (domestic)`, `0.0 = False (international)`.

**null in fees.json means "applies to all values"** — do not treat null rules as lower-priority; they apply when no more-specific rule matches.

**Read the full manual.md** — it has 10+ sections. The output may be truncated; if so, use `grep` or targeted searches to find specific definitions.

## Answer Format Rules

- Column names: exact string as in payments-readme.md (e.g., `has_fraudulent_dispute`)
- Multiple factors: comma-separated, alphabetical or as listed (e.g., `monthly_fraud_level, is_credit`)  
- Volume tiers: exact strings from data (e.g., `>5m`, `1m-5m`)
- Fraud levels: exact strings from data (e.g., `<7.2%`, `>8.3%`)
- Not found: `Not Applicable`
