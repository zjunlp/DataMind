---
name: dabstep-dataset-metadata-business-rules
description: Solve questions about the dabstep payment dataset's metadata, schema, column definitions, and business rules. Use this skill when answering questions about column names/meanings, fee structures, how factors affect fees (is_credit, intracountry, monthly_fraud_level, monthly_volume, capture_delay), fee thresholds, volume tiers, or any business rule from the manual. Trigger on any dabstep question involving "column", "field", "fee", "factor", "cheaper", "threshold", "volume tier", "boolean", "rule", or asking about dataset structure/documentation.
---

# Dataset Metadata and Business Rules — DABSTEP

## Dataset Files

| File | Purpose |
|------|---------|
| `payments.csv` | Transaction records (21 columns) |
| `fees.json` | Fee rules (~1000 rules, JSON array) |
| `manual.md` | Business rules and field definitions |
| `payments-readme.md` | Column descriptions for payments.csv |
| `merchant_data.json` | Merchant-level information |
| `merchant_category_codes.csv` | MCC code lookup |
| `acquirer_countries.csv` | Acquirer country codes |

## Core Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

`fees.json` rules contain these fields:

| Field | Type | Description |
|-------|------|-------------|
| `fixed_amount` | float | Flat fee per transaction (euros) |
| `rate` | int | Variable rate × transaction_value / 10000 |
| `is_credit` | bool | True = credit card (typically more expensive) |
| `intracountry` | bool | True = domestic (same issuer/acquirer country; cheaper) |
| `monthly_fraud_level` | str | Fraud ratio tier: `<7.2%`, `7.2%-7.7%`, `7.7%-8.3%`, `>8.3%` |
| `monthly_volume` | str | Volume tier: `<100k`, `100k-1m`, `1m-5m`, `>5m` (euros) |
| `capture_delay` | str | `immediate`, `<3`, `3-5`, `>5`, `manual` |
| `card_scheme` | str | MasterCard, Visa, Amex, Other |
| `account_type` | list | R, D, H, F, S, O |
| `merchant_category_code` | list | MCC integers |
| `aci` | list | Authorization Characteristics Indicator values |
| `ID` | int | Rule identifier |

**Null values** in any field mean the rule applies to ALL values of that field.

## payments.csv Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `psp_reference` | ID | Unique payment identifier |
| `has_fraudulent_dispute` | Boolean | Fraudulent dispute flagged by issuing bank |
| `is_refused_by_adyen` | Boolean | Adyen refusal indicator |
| `is_credit` | Categorical | Credit or debit card |
| `eur_amount` | Numeric | Payment amount in euros |
| `card_scheme` | Categorical | MasterCard, Visa, Amex, Other |
| `aci` | Categorical | Authorization Characteristics Indicator |
| `acquirer_country` | Categorical | Acquiring bank country |
| `issuing_country` | Categorical | Card-issuing country |
| `merchant` | Categorical | Merchant name |

## Factor Directionality (from manual.md)

These are the definitive business rule statements for how each factor affects fees:

| Factor | Manual Statement | Cheaper When |
|--------|-----------------|--------------|
| `is_credit` | "credit transactions are more expensive" | `is_credit = False` (debit) |
| `intracountry` | "international transactions are typically more expensive" | `intracountry = True` (domestic) |
| `monthly_fraud_level` | "more expensive as fraud rate increases" | Decrease fraud level |
| `monthly_volume` | "higher volume → cheaper fees" | Increase volume |
| `capture_delay` | "faster capture is more expensive" | Increase delay (slower) |

**Key rule**: When asked "which boolean factor, when set to True, leads to cheaper fees?" — only `intracountry=True` is cheaper (domestic). `is_credit=True` is more expensive.

**Key rule**: When asked "which factors, when DECREASED, lead to cheaper fees?" — `monthly_fraud_level` (lower fraud = cheaper) and `is_credit` (False < True in terms of cost, so decreasing/switching to False is cheaper).

## Strategy by Question Type

### Type 1: Column/Field Identification
*"What column indicates X?" / "What field stores Y?"*

1. Read `payments-readme.md` for payments.csv columns
2. Read `manual.md` Section 5 for fees.json field definitions
3. Match the described concept to the exact column name

```python
# Load and inspect
import pandas as pd, json
df = pd.read_csv('payments.csv', nrows=3)
print(df.columns.tolist())

with open('fees.json') as f:
    fees = json.load(f)
print(list(fees[0].keys()))
```

### Type 2: Fee/Rule Existence Questions
*"Does fee X exist?" / "How much is fee Y?"*

Search strategy:
1. Search `manual.md` for the concept (e.g., grep for "retry", "refund", "chargeback")
2. Inspect `fees.json` field names — the schema defines what fee types are modeled
3. If the manual mentions a behavior causally (e.g., "excessive retrying causes downgrades") but no numeric fee is defined → answer **"Not Applicable"**
4. If the concept is absent entirely → answer **"Not Applicable"**

```python
import json
with open('fees.json') as f:
    fees = json.load(f)
# Check available fields
print(list(fees[0].keys()))
# Fields present: ID, card_scheme, account_type, capture_delay,
# monthly_fraud_level, monthly_volume, merchant_category_code,
# is_credit, aci, fixed_amount, rate, intracountry
# No "retry_fee", "refund_fee", etc.
```

### Type 3: Boolean Factor Analysis
*"Which boolean factors contribute to cheaper fees when True/False?"*

**Always trust manual.md over empirical data averages.**

Empirical mean/median comparisons across all rules are unreliable because rules for different boolean values serve different contexts. Use the manual's explicit directional statements.

```python
import json, pandas as pd
with open('fees.json') as f:
    fees = json.load(f)
df = pd.DataFrame(fees)

# For a proper comparison, find rule PAIRS differing only in the boolean
# Use a representative transaction amount for fee calculation
amount = 100  # euros
df['sample_fee'] = df['fixed_amount'] + df['rate'] * amount / 10000

# Compare within matched groups
non_bool_cols = ['card_scheme', 'account_type', 'capture_delay',
                 'monthly_fraud_level', 'monthly_volume',
                 'merchant_category_code', 'aci']
# Look at mean by boolean value as a sanity check, but defer to manual
for col in ['is_credit', 'intracountry']:
    print(f"\n{col}:")
    print(df.groupby(col)['sample_fee'].mean())
```

**Expected results per manual:**
- `is_credit=True` → higher fees; `is_credit=False` → cheaper
- `intracountry=True` → cheaper (domestic); `intracountry=False` → expensive (international)

### Type 4: Volume/Threshold Analysis
*"At what volume do fees become cheaper?" / "Highest volume where fees do NOT become cheaper?"*

Volume tier order (ascending): `<100k` → `100k-1m` → `1m-5m` → `>5m`

Manual rule: higher volume → cheaper fees (economies of scale). The pricing curve **flattens out** at higher volumes — meaning at the highest tier (`>5m`), fees cannot decrease further.

**Answering volume-boundary questions:**

- "At what volume do fees become cheaper?" — the tier(s) where lower rates apply; in practice `>5m` is the cheapest tier.
- "What is the highest volume at which fees do NOT become cheaper?" — Answer: **`>5m`**
  - Reasoning: `>5m` is the maximum volume tier. There is no higher tier to transition to, so fees cannot become cheaper at this volume level. The pricing curve has flattened at this ceiling.
  - Do NOT answer `1m-5m` — that is the tier below the cheapest, not the highest tier.

```python
import json
with open('fees.json') as f:
    fees = json.load(f)

# Enumerate all distinct monthly_volume tier values in logical order
tier_order = {'<100k': 0, '100k-1m': 1, '1m-5m': 2, '>5m': 3}
volumes = sorted({r['monthly_volume'] for r in fees if r.get('monthly_volume')},
                 key=lambda x: tier_order[x])
print(volumes)  # ['<100k', '100k-1m', '1m-5m', '>5m']
# Highest tier = volumes[-1] = '>5m'
```

### Type 5: Multi-Factor Direction Analysis
*"Which factors lead to cheaper fees when decreased?"*

For each candidate factor, determine if its value range has a monotonic direction:

```python
import json, pandas as pd
with open('fees.json') as f:
    fees = json.load(f)
df = pd.DataFrame(fees)

amount = 100
df['sample_fee'] = df['fixed_amount'] + df['rate'] * amount / 10000

# Define ordered tiers for categorical factors
vol_order = {'<100k': 0, '100k-1m': 1, '1m-5m': 2, '>5m': 3}
fraud_order = {'<7.2%': 0, '7.2%-7.7%': 1, '7.7%-8.3%': 2, '>8.3%': 3}
delay_order = {'immediate': 0, '<3': 1, '3-5': 2, '>5': 3, 'manual': 4}

for factor, order_map in [('monthly_volume', vol_order),
                           ('monthly_fraud_level', fraud_order),
                           ('capture_delay', delay_order)]:
    subset = df[df[factor].notna()].copy()
    subset['rank'] = subset[factor].map(order_map)
    print(f"\n{factor}:")
    print(subset.groupby('rank')['sample_fee'].mean().sort_index())
```

## Reading manual.md Efficiently

The manual is organized as:
1. Account types (Section 2)
2. MCC codes (Section 3)
3. ACI values (Section 4)
4. **Fee fields and formulas (Section 5)** ← most relevant
5. PIN limits, fraud management, reporting (Sections 6-8)
6. Glossary (Appendix)

For any question about a fee factor's meaning or directionality, go directly to **Section 5** which contains the authoritative definitions.

## Common Errors to Avoid

1. **Raw statistical analysis vs. manual definitions**: Computing average fees across all True vs. all False rows gives misleading results. Rules apply to different merchant segments — the manual's directional statements are authoritative.

2. **"Highest volume where fees do NOT become cheaper"**: This asks for the maximum/ceiling tier — the answer is `>5m`. Reasoning: at `>5m` there is no higher volume tier, so fees cannot decrease further. The pricing curve flattens at this ceiling. Do NOT interpret this as the tier just before the cheapest tier (which would be `1m-5m`).

3. **Volume tier string ordering**: String sort puts `>5m` before `<100k` alphabetically. Always use the logical numeric order: `<100k` < `100k-1m` < `1m-5m` < `>5m`.

4. **Null fields**: A null/NaN value in a fee rule field means the rule applies universally for that field — not that data is missing.

5. **Absence evidence**: When a fee concept is mentioned causally in the manual (e.g., "retrying may cause downgrades") but no fee amount or rule field exists for it, the correct answer is "Not Applicable" — not 0 or unknown.

6. **is_credit direction**: The column exists in both `payments.csv` (transaction-level) and `fees.json` (rule-level). Credit = more expensive. Debit (False) = cheaper. Don't conflate "more expensive" with "True".
