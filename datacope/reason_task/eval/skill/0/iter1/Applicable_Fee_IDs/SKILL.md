---
name: Applicable_Fee_IDs
description: Solve Applicable_Fee_IDs questions in the dabstep dataset. Use this skill whenever a question asks which fee IDs apply to a merchant, a specific account_type/aci combination, a specific day/month/year, or when reverse-looking up which merchants are affected by a given fee ID.
---

# Applicable Fee IDs — Solution Guide

## Critical: Submit Immediately After Computing

**The #1 failure mode is computing the correct answer and then entering a verification loop that exhausts all turns.** Once you have the sorted list of applicable fee IDs, format and print the final answer, then submit it. Do NOT spend additional turns verifying individual fees or checking why specific fees don't apply.

## Dataset Overview

- `fees.json`: 1000 fee rules, each with filtering fields (`card_scheme`, `account_type`, `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `merchant_category_code`, `is_credit`, `aci`, `intracountry`) and formula fields (`fixed_amount`, `rate`)
- `merchant_data.json`: 30 merchants with `account_type`, `capture_delay` (string: "1", "2", "7", "immediate", "manual"), `acquirer` (list), `merchant_category_code`
- `payments.csv`: Transaction-level data with `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `has_fraudulent_dispute`
- `manual.md`: Domain definitions — read first if needed

## Fee Rule Matching Logic

A fee rule applies when **all** specified fields match.

### Null/Empty = "Applies to All"
`null` AND empty list `[]` both mean the rule applies to all values of that field.

```python
def field_matches_list(rule_list, value):
    return not rule_list or value in rule_list

def field_matches_scalar(rule_val, value):
    return rule_val is None or rule_val == value
```

### capture_delay Mapping
Merchant data stores raw strings; fee rules use categorical strings:
```python
def map_capture_delay(raw):
    if raw in ('immediate', 'manual'):
        return raw
    n = int(raw)          # merchant_data has "1", "2", "7"
    if n < 3:   return '<3'
    if n <= 5:  return '3-5'
    return '>5'
```

### intracountry
Computed per transaction: `issuing_country == acquirer_country`  
In fee rules: `1.0` = True (domestic), `0.0` = False (international), `null` = all.

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
Fraud rate = `sum(eur_amount where has_fraudulent_dispute==True) / sum(eur_amount) × 100` for the merchant in that **natural calendar month**.

## Complete fee_applies Function

```python
def matches_monthly_volume(rule_vol, monthly_vol):
    if rule_vol is None: return True
    if rule_vol == '<100k':   return monthly_vol < 100_000
    if rule_vol == '100k-1m': return 100_000 <= monthly_vol < 1_000_000
    if rule_vol == '1m-5m':   return 1_000_000 <= monthly_vol < 5_000_000
    return monthly_vol >= 5_000_000  # '>5m'

def matches_monthly_fraud(rule_fraud, fraud_pct):
    if rule_fraud is None: return True
    if rule_fraud == '<7.2%':     return fraud_pct < 7.2
    if rule_fraud == '7.2%-7.7%': return 7.2 <= fraud_pct < 7.7
    if rule_fraud == '7.7%-8.3%': return 7.7 <= fraud_pct < 8.3
    return fraud_pct >= 8.3  # '>8.3%'

def fee_applies(fee, card_scheme, is_credit, aci, intracountry,
                account_type, capture_delay_mapped, mcc,
                monthly_vol, monthly_fraud_pct):
    if not field_matches_scalar(fee['card_scheme'], card_scheme): return False
    if not field_matches_list(fee['account_type'], account_type): return False
    if not field_matches_scalar(fee['capture_delay'], capture_delay_mapped): return False
    if not field_matches_list(fee['merchant_category_code'], mcc): return False
    if not field_matches_scalar(fee['is_credit'], is_credit): return False
    if not field_matches_list(fee['aci'], aci): return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != intracountry: return False
    if not matches_monthly_volume(fee['monthly_volume'], monthly_vol): return False
    if not matches_monthly_fraud(fee['monthly_fraud_level'], monthly_fraud_pct): return False
    return True
```

## Task Types and Solutions

### Type 1: Pure Attribute Query
**"What fee IDs apply to account_type = X and aci = Y?"**

No transaction data needed. Filter `fees.json` directly:

```python
import json
with open('fees.json') as f: fees = json.load(f)

result = [
    fee['ID'] for fee in fees
    if field_matches_list(fee['account_type'], 'F')   # target account_type
    and field_matches_list(fee['aci'], 'A')            # target aci
]
print(', '.join(str(x) for x in sorted(result)))
```

---

### Type 2: Merchant + Specific Day
**"For the Nth of the year 2023, what Fee IDs apply to Merchant X?"**

```python
import pandas as pd, json

with open('merchant_data.json') as f: merchants = json.load(f)
with open('fees.json') as f: fees = json.load(f)
payments = pd.read_csv('payments.csv')

# Merchant properties
merch = next(m for m in merchants if m['merchant'] == 'MerchantName')
account_type = merch['account_type']
mcc = merch['merchant_category_code']
capture_delay = map_capture_delay(merch['capture_delay'])

# Add month column (works for 2023 non-leap year)
merch_payments = payments[(payments['merchant'] == merch['merchant']) & (payments['year'] == 2023)].copy()
merch_payments['month'] = (pd.to_datetime('2023-01-01') +
                            pd.to_timedelta(merch_payments['day_of_year'] - 1, unit='D')).dt.month

# Target day and its calendar month
target_day = 200
target_month = merch_payments[merch_payments['day_of_year'] == target_day]['month'].iloc[0]

# Monthly stats for that calendar month
month_txns = merch_payments[merch_payments['month'] == target_month]
monthly_vol = month_txns['eur_amount'].sum()
fraud_vol = month_txns[month_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
monthly_fraud_pct = (fraud_vol / monthly_vol * 100) if monthly_vol > 0 else 0.0

# Unique profiles from the specific day
day_txns = merch_payments[merch_payments['day_of_year'] == target_day].copy()
day_txns['intracountry'] = day_txns['issuing_country'] == day_txns['acquirer_country']
unique_profiles = day_txns[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()

# Find applicable fees
applicable_ids = set()
for _, row in unique_profiles.iterrows():
    for fee in fees:
        if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                       row['intracountry'], account_type, capture_delay, mcc,
                       monthly_vol, monthly_fraud_pct):
            applicable_ids.add(fee['ID'])

# Print and submit immediately
result = ', '.join(str(x) for x in sorted(applicable_ids))
print(result)
```

**Key**: Use actual unique (card_scheme, is_credit, aci, intracountry) combinations from real transactions — NOT Cartesian products.

---

### Type 3a: Merchant + Full Year
**"What were the applicable Fee IDs for Merchant X in 2023?"**

Iterate over each calendar month, compute per-month stats, collect **union** of applicable fees:

```python
# (Load data and merchant properties same as Type 2)
merch_payments['month'] = (pd.to_datetime('2023-01-01') +
                            pd.to_timedelta(merch_payments['day_of_year'] - 1, unit='D')).dt.month

all_applicable = set()
for month_num in range(1, 13):
    month_txns = merch_payments[merch_payments['month'] == month_num]
    if len(month_txns) == 0:
        continue
    monthly_vol = month_txns['eur_amount'].sum()
    fraud_vol = month_txns[month_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
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

### Type 3b: Merchant + Specific Month
**"What were the applicable Fee IDs for Merchant X in October 2023?"**

Same as Type 3a but filter for one month only. Parse month name to number:

```python
month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
             'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
target_month = month_map['October']  # from question

# (Same as Type 3a loop but only for target_month)
```

---

### Type 4: Reverse Lookup
**"In 2023, which merchants were affected by Fee with ID 709?"**

```python
target_fee = next(f for f in fees if f['ID'] == target_fee_id)

filtered = payments.copy()
filtered['intracountry'] = filtered['issuing_country'] == filtered['acquirer_country']

if target_fee['card_scheme'] is not None:
    filtered = filtered[filtered['card_scheme'] == target_fee['card_scheme']]
if target_fee['is_credit'] is not None:
    filtered = filtered[filtered['is_credit'] == target_fee['is_credit']]
if target_fee['aci']:
    filtered = filtered[filtered['aci'].isin(target_fee['aci'])]
if target_fee['intracountry'] is not None:
    filtered = filtered[filtered['intracountry'] == bool(target_fee['intracountry'])]

merch_lookup = {m['merchant']: m for m in merchants}
affected = set()
for merch_name in filtered['merchant'].unique():
    merch = merch_lookup.get(merch_name)
    if not merch: continue
    if not field_matches_list(target_fee['account_type'], merch['account_type']): continue
    if not field_matches_list(target_fee['merchant_category_code'], merch['merchant_category_code']): continue
    if not field_matches_scalar(target_fee['capture_delay'], map_capture_delay(merch['capture_delay'])): continue
    affected.add(merch_name)

print(', '.join(sorted(affected)))
```

## Output Format

- Return a **sorted, comma-separated list** of integer fee IDs: `3, 4, 5, 7, 8, 9, 11`
- If no fees match, return an **empty string** (not "Not Applicable")

## Common Mistakes to Avoid

1. **Verification loops** — compute answer, print it, submit; do not spend turns verifying individual fee matches
2. **Cartesian products** instead of actual transaction profiles — over-counts fees
3. **Treating `[]` as "no match"** — empty lists mean "applies to all", same as null
4. **Wrong capture_delay mapping** — merchant "1" or "2" → `<3`; "7" → `>5`
5. **Wrong intracountry comparison** — fee stores `1.0`/`0.0`, convert with `bool()` before comparing
6. **Yearly totals for monthly stats** — always compute per natural calendar month
7. **Monthly stats from wrong month** — for a specific day query, monthly stats come from the calendar month containing that day (e.g., day 200 = July → use July's monthly stats)
