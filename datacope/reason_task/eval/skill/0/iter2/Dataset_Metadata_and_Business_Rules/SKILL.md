---
name: Dataset_Metadata_and_Business_Rules
description: >
  Answers questions about the structure, metadata, field definitions, and business rules of the dabstep payment processing dataset.
  Use this skill when questions ask about: column names, field meanings, fee rule structure, which factors affect fees, fee formula, boolean factor effects on cost, volume/fraud/capture-delay tier meanings, or whether specific fee types exist. Also use for "Not Applicable" determination when a business concept is mentioned but no corresponding fee or data field exists.
---

# Dataset_Metadata_and_Business_Rules Skill

## Dataset Overview

The dabstep dataset consists of payment processing data for a merchant payment processor:

| File | Content |
|------|---------|
| `payments.csv` | Transaction records |
| `payments-readme.md` | Column documentation for payments.csv |
| `manual.md` | Business rules, fee structure, field definitions |
| `fees.json` | 1000 fee rules with matching criteria and amounts |
| `merchant_data.json` | Merchant info (MCC, account_type, acquirers, capture_delay) |
| `merchant_category_codes.csv` | MCC codes and descriptions |
| `acquirer_countries.csv` | Acquirer country codes |

---

## Step 1: Always Read Documentation First

Before analyzing any data, read both reference documents:

```python
with open('data/manual.md', 'r') as f:
    manual = f.read()
print(manual)

with open('data/payments-readme.md', 'r') as f:
    readme = f.read()
print(readme)
```

These files are the authoritative source for all definitions and business rules. Many questions can be answered directly from them without loading fee data.

---

## payments.csv Column Reference

Key columns (from payments-readme.md):

| Column | Type | Description |
|--------|------|-------------|
| `psp_reference` | ID | Unique payment identifier |
| `merchant` | Categorical | Merchant name |
| `card_scheme` | Categorical | MasterCard, Visa, Amex, Other |
| `is_credit` | Categorical | Credit or Debit card indicator |
| `eur_amount` | Numeric | Payment amount in euros |
| `ip_country` | Categorical | Country from shopper's IP |
| `issuing_country` | Categorical | Card-issuing country |
| `device_type` | Categorical | Windows, Linux, MacOS, iOS, Android, Other |
| `shopper_interaction` | Categorical | Ecommerce or POS |
| `has_fraudulent_dispute` | Boolean | **Fraud indicator** - fraudulent dispute from issuing bank |
| `is_refused_by_adyen` | Boolean | Adyen refusal indicator |
| `aci` | Categorical | Authorization Characteristics Indicator |
| `acquirer_country` | Categorical | Location of acquiring bank |

**Critical**: `has_fraudulent_dispute` is THE fraud column. "Fraud" = this column.

---

## fees.json Structure and Business Rules

### Fee Rule Fields

```python
import json
with open('data/fees.json', 'r') as f:
    fees = json.load(f)
# Each rule has: ID, card_scheme, account_type, capture_delay,
# monthly_fraud_level, monthly_volume, merchant_category_code,
# is_credit, aci, fixed_amount, rate, intracountry
```

**Fee formula**: `fee = fixed_amount + rate * transaction_value / 10000`

**Null means "applies to all"**: A `null` value in any field means that fee rule applies regardless of that field's value.

### Boolean Fee Fields (Critical for Analysis)

| Field | True Meaning | False Meaning | Cost Impact |
|-------|-------------|---------------|-------------|
| `is_credit` | Credit card transaction | Debit card transaction | **True = MORE expensive** |
| `intracountry` | Domestic (issuer country = acquirer country) | International transaction | **True = CHEAPER** (domestic) |

**From manual.md (authoritative)**:
- `is_credit`: "Typically credit transactions are more expensive (higher fee)"
- `intracountry`: "False are for international transactions...typically are more expensive"

### Therefore:
- **Factors cheaper when True**: `intracountry` only
- **Factors cheaper when False**: `is_credit` only
- **Factors cheaper when decreased**: `monthly_fraud_level` (less fraud → cheaper), `is_credit` (True→False = cheaper)

### Ordered/Tiered Fee Fields

**monthly_volume** (higher volume → cheaper fees per manual):
- `'<100k'` → `'100k-1m'` → `'1m-5m'` → `'>5m'`

**monthly_fraud_level** (lower fraud → cheaper fees):
- `'<7.2%'` → `'7.2%-7.7%'` → `'7.7%-8.3%'` → `'>8.3%'`

**capture_delay** (faster = MORE expensive, not cheaper):
- `'immediate'` (most expensive) → `'<3'` → `'3-5'` → `'>5'` → `'manual'`

---

## Authorization Characteristics Indicator (ACI)

| Code | Description |
|------|-------------|
| A | Card present - Non-authenticated |
| B | Card Present - Authenticated |
| C | Tokenized card with mobile device |
| D | Card Not Present - Card On File |
| E | Card Not Present - Recurring Bill Payment |
| F | Card Not Present - 3-D Secure |
| G | Card Not Present - Non-3-D Secure |

---

## Account Types

| Type | Description |
|------|-------------|
| R | Enterprise - Retail |
| D | Enterprise - Digital |
| H | Enterprise - Hospitality |
| F | Platform - Franchise |
| S | Platform - SaaS |
| O | Other |

---

## Task-Specific Strategies

### "What column indicates X?"
→ Look in `payments-readme.md` first. For fraud: `has_fraudulent_dispute`.

### "Does a specific fee (e.g., retry fee, penalty fee) exist?"
→ Check `fees.json` field names and `manual.md`. Fee rules only contain: `fixed_amount` and `rate`. If no such fee type exists in the data, answer **"Not Applicable"**.

```python
# Check available fields
df = pd.DataFrame(fees)
print(df.columns.tolist())
# Check for specific concept in manual
import re
matches = re.findall(r'.*retry.*', manual, re.IGNORECASE)
print(matches)
```

Note: manual.md mentions "excessive retrying causes downgrades" but there is NO explicit retry fee amount in fees.json → "Not Applicable".

### "What boolean factors lead to cheaper fees if set to True/False?"
→ Use manual.md's explicit statements, **not** empirical data analysis (confounding variables make raw data misleading).

- True → cheaper: **intracountry** (domestic is cheaper)
- False → cheaper: **is_credit** (debit is cheaper than credit)

### "Which factors lead to cheaper fees when decreased?"
Use manual.md rules:
- `monthly_fraud_level` ↓ → cheaper ✓ (less fraud = less risk = lower fees)
- `is_credit` ↓ (True→False) → cheaper ✓ (debit < credit cost)
- `monthly_volume` ↓ → **more expensive** ✗ (higher volume = better rates)
- `capture_delay` ↓ (faster) → **more expensive** ✗ ("faster capture = more expensive")
- `intracountry` is boolean (True/False), not meaningfully "decreased"

### "What volume tier marks a threshold for cheaper fees?"
→ Analyze fees.json statistically by monthly_volume tier:

```python
import pandas as pd
import json

with open('data/fees.json') as f:
    fees = json.load(f)
df = pd.DataFrame(fees)

volume_order = ['<100k', '100k-1m', '1m-5m', '>5m']
for vol in volume_order:
    sub = df[df['monthly_volume'] == vol]
    print(f"{vol}: count={len(sub)}, median_rate={sub['rate'].median():.1f}, median_fixed={sub['fixed_amount'].median():.4f}")
```

Compare median rate and fixed_amount across tiers to find where fees stop improving.

**Example pattern** (from actual data):
- `<100k`: median_rate=50, median_fixed=0.05
- `100k-1m`: median_rate=49.5, median_fixed=0.07
- `1m-5m`: median_rate=52, median_fixed=0.08 ← fees got WORSE
- `>5m`: median_rate=59, median_fixed=0.07 ← rate still not cheaper

So "the highest volume at which fees do NOT become cheaper" = `>5m` (the highest tier where fees are still not lower than all prior tiers).

### "What are the possible values for field X?"

```python
import json, pandas as pd
with open('data/fees.json') as f:
    fees = json.load(f)
df = pd.DataFrame(fees)
print(df['monthly_fraud_level'].unique())  # or any other field
print(df['capture_delay'].unique())
print(df['monthly_volume'].unique())
```

---

## Key Business Rules (from manual.md)

1. **Fee formula**: `fee = fixed_amount + rate * transaction_value / 10000`
2. **Fraud definition**: ratio of fraudulent volume over total volume (monthly)
3. **PIN limit**: 3 consecutive incorrect PIN attempts → card temporarily blocked
4. **PCI DSS non-compliance penalty**: EUR 5,000 to EUR 100,000 per month
5. **No excessive retry fee** defined in fees.json (downgrades mentioned, not fees)
6. **Monthly volumes computed** in natural months (day 1 to last day of month)
7. **Local acquiring** (intracountry=True) → reduced fees and friction

---

## Answer Format Guidelines

- Single column name → just the column name (e.g., `has_fraudulent_dispute`)
- List of factors → comma-separated (e.g., `monthly_fraud_level, is_credit`)
- No applicable answer → `Not Applicable`
- Volume/tier value → use exact string from data (e.g., `>5m`, `1m-5m`)
- Dollar amount → number only if clearly defined; otherwise `Not Applicable`
