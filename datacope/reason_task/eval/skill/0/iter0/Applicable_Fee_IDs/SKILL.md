---
name: Applicable_Fee_IDs
description: Solve Applicable_Fee_IDs questions in the dabstep dataset. Use this skill whenever a question asks which fee IDs apply to a merchant, a specific account_type/aci combination, a specific day/month/year, or when reverse-looking up which merchants are affected by a given fee ID.
---

# Applicable Fee IDs — Solution Guide

## Dataset Overview

- `fees.json`: 1000 fee rules, each with filtering fields and formula fields (`fixed_amount`, `rate`)
- `merchant_data.json`: Merchant properties — `account_type`, `capture_delay` (can be numeric string like "1"), `acquirer` (list), `merchant_category_code`
- `payments.csv`: Transaction-level data with `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `has_fraudulent_dispute`
- `acquirer_countries.csv`: Acquirer → country_code mapping (rarely needed directly)
- `manual.md`: Domain definitions — ALWAYS read first

## Fee Rule Matching Logic

A fee rule applies to a transaction/merchant when **all** specified fields match.

### Null/Empty = "Applies to All"
Per the manual: `null` means the rule applies to all values.  
**Empty list `[]` behaves the same as null** for list fields (`account_type`, `aci`, `merchant_category_code`).

```python
def field_matches_list(rule_list, value):
    """True if rule_list is null/empty OR value is in rule_list."""
    return not rule_list or value in rule_list

def field_matches_scalar(rule_val, value):
    """True if rule_val is null OR equals value."""
    return rule_val is None or rule_val == value
```

### capture_delay Mapping
Merchant data stores numeric days; fee rules use categorical strings:
```python
def map_capture_delay(raw):
    if raw in ('immediate', 'manual'):
        return raw
    n = int(raw)          # e.g. "1", "2", "7"
    if n < 3:   return '<3'
    if n <= 5:  return '3-5'
    return '>5'
```

### intracountry
Compute per transaction from payments.csv:
```python
payments['intracountry'] = payments['issuing_country'] == payments['acquirer_country']
```
In fee rules, `intracountry` is stored as `1.0` (True) or `0.0` (False) or `null` (all).

### monthly_volume Ranges
```
'<100k'   → monthly_vol_eur < 100_000
'100k-1m' → 100_000 ≤ vol < 1_000_000
'1m-5m'   → 1_000_000 ≤ vol < 5_000_000
'>5m'     → vol ≥ 5_000_000
```

### monthly_fraud_level Ranges
```
'<7.2%'     → fraud_pct < 7.2
'7.2%-7.7%' → 7.2 ≤ pct < 7.7
'7.7%-8.3%' → 7.7 ≤ pct < 8.3
'>8.3%'     → pct ≥ 8.3
```
Fraud rate = sum(eur_amount where has_fraudulent_dispute==True) / sum(eur_amount) × 100 for the merchant in that **natural calendar month**.

## Complete fee_applies Function

```python
def fee_applies(fee, card_scheme, is_credit, aci, intracountry,
                account_type, capture_delay_mapped, mcc,
                monthly_vol, monthly_fraud_pct):
    if not field_matches_scalar(fee['card_scheme'], card_scheme):
        return False
    if not field_matches_list(fee['account_type'], account_type):
        return False
    if not field_matches_scalar(fee['capture_delay'], capture_delay_mapped):
        return False
    if not field_matches_list(fee['merchant_category_code'], mcc):
        return False
    if not field_matches_scalar(fee['is_credit'], is_credit):
        return False
    if not field_matches_list(fee['aci'], aci):
        return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != intracountry:
        return False
    if not matches_monthly_volume(fee['monthly_volume'], monthly_vol):
        return False
    if not matches_monthly_fraud(fee['monthly_fraud_level'], monthly_fraud_pct):
        return False
    return True
```

## Task Types and Solutions

---

### Type 1: Pure Attribute Query
**"What fee IDs apply to account_type = X and aci = Y?"**

No transaction data needed. Filter `fees.json` directly:

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

target_account_type = 'F'   # from question
target_aci = 'A'             # from question

result = [
    fee['ID'] for fee in fees
    if field_matches_list(fee['account_type'], target_account_type)
    and field_matches_list(fee['aci'], target_aci)
]
print(', '.join(str(x) for x in sorted(result)))
```

**Critical**: Both empty `[]` and `null` mean "applies to all values". Do NOT skip fees because they have empty lists.

---

### Type 2: Merchant + Specific Day
**"For the Nth day of year 2023, what Fee IDs apply to Merchant X?"**

```python
import pandas as pd, json

# Load data
payments = pd.read_csv('payments.csv')
with open('merchant_data.json') as f:
    merchants = json.load(f)
with open('fees.json') as f:
    fees = json.load(f)

# Merchant properties
merch = next(m for m in merchants if m['merchant'] == 'MerchantName')
account_type = merch['account_type']
mcc = merch['merchant_category_code']
capture_delay = map_capture_delay(merch['capture_delay'])

# Determine month for monthly stats (day N → which calendar month)
from datetime import datetime, timedelta
target_day = 10   # day_of_year
target_date = datetime(2023, 1, 1) + timedelta(days=target_day - 1)
target_month = target_date.month

# Monthly volume and fraud for that calendar month
payments['date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(payments['day_of_year'] - 1, unit='D')
payments['month'] = payments['date'].dt.month
monthly_txns = payments[(payments['merchant'] == merch['merchant']) &
                         (payments['year'] == 2023) &
                         (payments['month'] == target_month)]
monthly_vol = monthly_txns['eur_amount'].sum()
fraud_vol = monthly_txns[monthly_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
monthly_fraud_pct = (fraud_vol / monthly_vol * 100) if monthly_vol > 0 else 0.0

# Transaction profiles for the specific day
day_txns = payments[(payments['merchant'] == merch['merchant']) &
                    (payments['year'] == 2023) &
                    (payments['day_of_year'] == target_day)].copy()
day_txns['intracountry'] = day_txns['issuing_country'] == day_txns['acquirer_country']

# Use ACTUAL transaction profiles — do NOT generate Cartesian products
unique_profiles = day_txns[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()

applicable_ids = set()
for _, row in unique_profiles.iterrows():
    for fee in fees:
        if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                       row['intracountry'], account_type, capture_delay, mcc,
                       monthly_vol, monthly_fraud_pct):
            applicable_ids.add(fee['ID'])

print(', '.join(str(x) for x in sorted(applicable_ids)))
```

**Key**: Use actual unique (card_scheme, is_credit, aci, intracountry) combinations from real transactions — not Cartesian products of each dimension separately.

---

### Type 3: Merchant + Time Period (Month or Year)
**"What were the applicable Fee IDs for Merchant X in [month/year] 2023?"**

For a **single month**: same as Type 2 but filter all transactions for that month, compute monthly stats for that month, and collect unique profiles across the whole month.

For a **full year**: iterate over each calendar month, compute per-month stats and profiles, collect the **union** of applicable fees across all months.

```python
# Year-level: union across months
all_applicable = set()
for month_num in range(1, 13):
    month_txns = payments[(payments['merchant'] == merch['merchant']) &
                           (payments['year'] == 2023) &
                           (payments['month'] == month_num)]
    if len(month_txns) == 0:
        continue
    
    monthly_vol = month_txns['eur_amount'].sum()
    fraud_vol = month_txns[month_txns['has_fraudulent_dispute']]['eur_amount'].sum()
    monthly_fraud_pct = (fraud_vol / monthly_vol * 100) if monthly_vol > 0 else 0.0
    
    month_txns = month_txns.copy()
    month_txns['intracountry'] = month_txns['issuing_country'] == month_txns['acquirer_country']
    unique_profiles = month_txns[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()
    
    for _, row in unique_profiles.iterrows():
        for fee in fees:
            if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                           row['intracountry'], account_type, capture_delay, mcc,
                           monthly_vol, monthly_fraud_pct):
                all_applicable.add(fee['ID'])

print(', '.join(str(x) for x in sorted(all_applicable)))
```

---

### Type 4: Reverse Lookup — Which Merchants Are Affected by Fee X?
**"In 2023, which merchants were affected by the Fee with ID 709?"**

```python
# Get the specific fee rule
target_fee = next(f for f in fees if f['ID'] == target_fee_id)

# Filter payments for matching transaction-level conditions from the fee rule
filtered = payments.copy()
filtered['intracountry'] = filtered['issuing_country'] == filtered['acquirer_country']

if target_fee['card_scheme'] is not None:
    filtered = filtered[filtered['card_scheme'] == target_fee['card_scheme']]
if target_fee['is_credit'] is not None:
    filtered = filtered[filtered['is_credit'] == target_fee['is_credit']]
if target_fee['aci']:  # non-empty list
    filtered = filtered[filtered['aci'].isin(target_fee['aci'])]
if target_fee['intracountry'] is not None:
    filtered = filtered[filtered['intracountry'] == bool(target_fee['intracountry'])]

# For merchant-level filters (account_type, mcc, capture_delay), check via merchant_data
# Build merchant lookup
merch_lookup = {m['merchant']: m for m in merchants}

affected_merchants = set()
for merch_name in filtered['merchant'].unique():
    merch = merch_lookup.get(merch_name)
    if merch is None:
        continue
    if not field_matches_list(target_fee['account_type'], merch['account_type']):
        continue
    if not field_matches_list(target_fee['merchant_category_code'], merch['merchant_category_code']):
        continue
    cap = map_capture_delay(merch['capture_delay'])
    if not field_matches_scalar(target_fee['capture_delay'], cap):
        continue
    affected_merchants.add(merch_name)

print(', '.join(sorted(affected_merchants)))
```

Note: For reverse lookup with monthly_volume/fraud conditions, you would need to compute per-merchant per-month stats and check if conditions are ever met within the year.

## Output Format

- Return a **sorted, comma-separated list** of integer fee IDs
- If no fees match, return an **empty string** (not "Not Applicable")
- Example: `3, 4, 5, 7, 8, 9, 11`

## Common Mistakes to Avoid

1. **Using Cartesian products** instead of actual transaction profiles — leads to over-counting fees
2. **Treating `[]` as "no match"** — empty lists mean "applies to all"
3. **Ignoring `monthly_volume`/`monthly_fraud_level`** — these restrict which fees apply; always compute per-month stats
4. **Wrong capture_delay mapping** — merchant "1" or "2" maps to `<3`, not `immediate`; "7" maps to `>5`
5. **Wrong intracountry comparison** — fees store `1.0`/`0.0`, convert with `bool()` before comparing
6. **Using yearly totals for monthly stats** — always compute monthly stats using the natural calendar month
