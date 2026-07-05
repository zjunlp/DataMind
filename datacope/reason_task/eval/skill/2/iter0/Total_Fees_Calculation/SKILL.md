---
name: Total_Fees_Calculation
description: >
  Calculates total payment processing fees for a merchant over a specified period (day, month, or full year)
  in the dabstep dataset. Use this skill whenever the question asks for "total fees" a merchant "paid" or
  "should pay," covering any time window. Involves joining payments.csv with fees.json using multi-field rule
  matching (card_scheme, account_type, capture_delay, MCC, is_credit, aci, intracountry, monthly_fraud_level,
  monthly_volume). Always triggers for fee aggregation questions in the payment processing domain.
---

# Total Fees Calculation

## Problem Pattern

Given a merchant name and a time period (specific day, month, or full year), calculate the **sum of fees** for all applicable transactions. The answer is a number rounded to 2 decimal places.

**Key insight**: Many transactions will have no matching fee rule (fee = 0). This is expected — not all transaction type combinations are covered by the fee schedule.

---

## Algorithm (implement in Python)

### Step 1: Load data

```python
import json, pandas as pd
from datetime import date, timedelta

with open('fees.json') as f: fees = json.load(f)
with open('merchant_data.json') as f: merchant_data = json.load(f)
merchant_map = {m['merchant']: m for m in merchant_data}
acquirer_countries = pd.read_csv('acquirer_countries.csv')
payments = pd.read_csv('payments.csv')

def doy_to_month(doy, year):
    return (date(year, 1, 1) + timedelta(days=doy - 1)).month

payments['month'] = payments.apply(lambda r: doy_to_month(r['day_of_year'], r['year']), axis=1)
```

### Step 2: Get merchant attributes

```python
m = merchant_map[merchant_name]
merchant_at = m['account_type']          # e.g. "R"
merchant_mcc = m['merchant_category_code']  # e.g. 5942
merchant_cd = map_capture_delay(m['capture_delay'])  # see below
```

**Capture delay mapping** — merchant stores numeric days; fee rules use category strings:

```python
def map_capture_delay(cd):
    if cd in ('immediate', 'manual'): return cd
    n = int(cd)
    return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
```

### Step 3: Filter target transactions

**Always exclude refused transactions** (`is_refused_by_adyen == False`):

```python
# For a specific day (e.g. day_of_year=10):
txs = payments[(payments['merchant'] == merchant_name) &
               (payments['year'] == year) &
               (payments['day_of_year'] == target_day) &
               (payments['is_refused_by_adyen'] == False)]

# For a specific month:
txs = payments[(payments['merchant'] == merchant_name) &
               (payments['year'] == year) &
               (payments['month'] == target_month) &
               (payments['is_refused_by_adyen'] == False)]
```

### Step 4: Compute per-month stats (used for fee rule matching)

Monthly stats are computed **per natural calendar month** using **all merchant transactions** (not just target day/filtered):

```python
monthly_txs = payments[(payments['merchant'] == merchant_name) &
                       (payments['year'] == year) &
                       (payments['month'] == month)]
monthly_volume = monthly_txs['eur_amount'].sum()
fraud_volume = monthly_txs[monthly_txs['has_fraudulent_dispute'] == True]['eur_amount'].sum()
monthly_fraud_rate = fraud_volume / monthly_volume if monthly_volume > 0 else 0
```

When calculating fees for a full year, compute these stats **separately for each month**.

### Step 5: Find the applicable fee rule per transaction

```python
def find_rule(tx, merchant_at, merchant_cd, merchant_mcc, monthly_volume, monthly_fraud_rate):
    ic = (tx['issuing_country'] == tx['acquirer_country'])  # intracountry flag
    
    candidates = []
    for rule in fees:
        # Mandatory exact match
        if rule['card_scheme'] != tx['card_scheme']: continue
        
        # List fields: empty [] means "applies to all"
        if rule['account_type'] and merchant_at not in rule['account_type']: continue
        if rule['merchant_category_code'] and merchant_mcc not in rule['merchant_category_code']: continue
        if rule['aci'] and tx['aci'] not in rule['aci']: continue
        
        # Scalar fields: None means "applies to all"
        if rule['capture_delay'] is not None and rule['capture_delay'] != merchant_cd: continue
        if rule['is_credit'] is not None and rule['is_credit'] != tx['is_credit']: continue
        if rule['intracountry'] is not None:
            if rule['intracountry'] != (1.0 if ic else 0.0): continue
        
        # Range fields: None means "applies to all"
        if rule['monthly_fraud_level'] is not None:
            lo, hi = parse_range(rule['monthly_fraud_level'], pct=True)
            if not (lo <= monthly_fraud_rate < hi): continue
        if rule['monthly_volume'] is not None:
            lo, hi = parse_range(rule['monthly_volume'])
            if not (lo <= monthly_volume < hi): continue
        
        candidates.append(rule)
    
    if not candidates: return None
    # Pick most specific (most non-null/non-empty fields)
    return max(candidates, key=lambda r: sum([
        bool(r['account_type']), r['capture_delay'] is not None,
        bool(r['merchant_category_code']), r['is_credit'] is not None,
        bool(r['aci']), r['intracountry'] is not None,
        r['monthly_fraud_level'] is not None, r['monthly_volume'] is not None
    ]))
```

### Step 6: Parse range strings

```python
def parse_range(s, pct=False):
    def val(v):
        v = v.strip().replace('%', '')
        if v.endswith('m'): return float(v[:-1]) * 1e6
        if v.endswith('k'): return float(v[:-1]) * 1e3
        return float(v) / 100.0 if pct else float(v)
    if s.startswith('<'): return (0.0, val(s[1:]))
    if s.startswith('>'): return (val(s[1:]), float('inf'))
    a, b = s.split('-')
    return (val(a), val(b))
```

### Step 7: Calculate fee and sum

```python
# Fee formula from manual: fee = fixed_amount + rate * transaction_value / 10000
total_fee = 0.0
for _, row in txs.iterrows():
    rule = find_rule(row.to_dict(), merchant_at, merchant_cd, merchant_mcc,
                     monthly_volume, monthly_fraud_rate)
    if rule:
        total_fee += rule['fixed_amount'] + rule['rate'] * row['eur_amount'] / 10000

print(round(total_fee, 2))
```

---

## Critical Rules

**intracountry**: `True` if `issuing_country == acquirer_country` in the payments row. The `acquirer_country` column is **directly in payments.csv** — no lookup needed.

**Empty list `[]`** in fee rule fields (account_type, merchant_category_code, aci) means the rule applies to ALL values of that field. This is functionally equivalent to `null`.

**No matching rule → fee = 0**. This is expected and correct; many transaction type combinations are not covered by the fee schedule.

**Monthly stats scope**: Always use the full natural calendar month, not just the target day's data. For full-year queries, loop over each month (1–12) and recompute stats per month.

**Monthly fraud level boundary**: Use `lo <= rate < hi` (inclusive lower, exclusive upper).

**Monthly volume boundary**: Use `lo <= volume < hi`.

---

## Question Patterns

| Question Pattern | Filter |
|---|---|
| "For the Nth of year Y" | `day_of_year == N, year == Y` |
| "In [Month] [Year]" | `month == M, year == Y` |
| "In year Y" (full year) | loop month 1–12 for year Y |

---

## Data Schema Quick Reference

- `payments.csv`: psp_reference, merchant, card_scheme, year, day_of_year, is_credit, eur_amount, issuing_country, acquirer_country, aci, has_fraudulent_dispute, is_refused_by_adyen, month (computed)
- `merchant_data.json`: merchant, capture_delay (numeric string or 'immediate'/'manual'), acquirer (list), merchant_category_code (int), account_type (single letter)
- `fees.json`: ID, card_scheme, account_type (list), capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code (list), is_credit, aci (list), fixed_amount, rate, intracountry
- Fee formula: `fee = fixed_amount + rate * eur_amount / 10000`
