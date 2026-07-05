---
name: Total_Fees_Calculation
description: >
  Skill for computing total payment processing fees for a merchant over a specific day, date range, or month
  in the dabstep dataset. Use this skill whenever the question asks for "total fees", "fees paid", or
  "fees charged" for a merchant over some time period. The computation requires matching each transaction
  to a fee rule from fees.json using multiple merchant and transaction properties, then summing
  fixed_amount + rate * eur_amount / 10000 for each matched transaction.
---

# Total Fees Calculation — dabstep Dataset

## Problem Summary
Compute total payment processing fees (in euros) for a given merchant over a specified period (a specific day of year, a month, or a full year). Round the final answer to 2 decimal places.

## Data Files Required
- `payments.csv` — transactions (filter by `merchant`, `year`, `day_of_year`)
- `fees.json` — 1000 fee rules with conditions and fee parameters
- `merchant_data.json` — per-merchant properties: `account_type`, `capture_delay`, `merchant_category_code`
- `manual.md` — read for domain definitions

## Fee Formula
```
fee = fixed_amount + rate * eur_amount / 10000
```
Sum fees across all transactions that match a fee rule. Transactions with no matching rule contribute 0.

---

## Step-by-Step Algorithm

### Step 1: Load Merchant Properties
From `merchant_data.json`, for the target merchant extract:
- `account_type`: one of `R`, `D`, `H`, `F`, `S`, `O`
- `capture_delay`: raw value (numeric like `"1"` or `"7"`, or categorical like `"immediate"`, `"manual"`)
- `merchant_category_code` (MCC): integer

**Map numeric `capture_delay` to fee rule category:**
```python
def map_capture_delay(cd):
    try:
        n = int(cd)
        return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
    except (ValueError, TypeError):
        return cd  # already a string category
```

### Step 2: Determine the Query Period and Month Boundaries
**2023 is NOT a leap year.** Day-of-year to month mapping:

| Month | day_of_year range |
|-------|-------------------|
| Jan   | 1 – 31   |
| Feb   | 32 – 59  |
| Mar   | 60 – 90  |
| Apr   | 91 – 120 |
| May   | 121 – 151|
| Jun   | 152 – 181|
| Jul   | 182 – 212|
| Aug   | 213 – 243|
| Sep   | 244 – 273|
| Oct   | 274 – 304|
| Nov   | 305 – 334|
| Dec   | 335 – 365|

- "Nth day of the year" → `day_of_year == N`; its month is determined by the table above.
- "Month X 2023" → use the full month range.
- "Full year 2023" → process each month separately (different monthly stats per month).

### Step 3: Compute Monthly Statistics (Full Natural Month)
Monthly stats must be computed from **ALL transactions in the full natural month** (not just the queried day). This is critical for fee rule matching.

```python
MONTH_RANGES = {
    1:(1,31), 2:(32,59), 3:(60,90), 4:(91,120),
    5:(121,151), 6:(152,181), 7:(182,212), 8:(213,243),
    9:(244,273), 10:(274,304), 11:(305,334), 12:(335,365)
}

month_txns = payments[
    (payments['merchant'] == MERCHANT) &
    (payments['year'] == YEAR) &
    (payments['day_of_year'].between(MONTH_START, MONTH_END))
]
monthly_vol = month_txns['eur_amount'].sum()
fraud_vol = month_txns[month_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_rate = fraud_vol / monthly_vol if monthly_vol > 0 else 0
```

**Categorize into fee rule buckets:**
```python
def categorize_volume(v):
    if v < 100_000:    return '<100k'
    elif v < 1_000_000: return '100k-1m'
    elif v < 5_000_000: return '1m-5m'
    else:               return '>5m'

def categorize_fraud(f):
    if f < 0.072:   return '<7.2%'
    elif f < 0.077: return '7.2%-7.7%'
    elif f < 0.083: return '7.7%-8.3%'
    else:           return '>8.3%'
```

### Step 4: Match Each Transaction to a Fee Rule

**The single most important rule: empty list `[]` in any fee rule field means "applies to all" (same semantics as `null`).**

In `fees.json`: `account_type: []`, `merchant_category_code: []`, `aci: []` all mean "no restriction on this field".

```python
def rule_specificity(rule):
    """Higher = more specific. Used as tiebreaker when multiple rules match."""
    return sum([
        bool(rule['account_type']),
        rule['capture_delay'] is not None,
        rule['monthly_fraud_level'] is not None,
        rule['monthly_volume'] is not None,
        bool(rule['merchant_category_code']),
        rule['is_credit'] is not None,
        bool(rule['aci']),
        rule['intracountry'] is not None,
    ])

def find_best_rule(txn, acct_type, cap_delay, mcc, fraud_level, vol_category, fees):
    """Find the best matching fee rule for a transaction."""
    # intracountry: use columns from payments.csv directly
    intracountry = (txn['issuing_country'] == txn['acquirer_country'])
    matches = []
    for rule in fees:
        # card_scheme: exact match required
        if rule['card_scheme'] != txn['card_scheme']:
            continue
        # account_type: [] or null means all
        if rule['account_type'] and acct_type not in rule['account_type']:
            continue
        # capture_delay: null means all
        if rule['capture_delay'] is not None and rule['capture_delay'] != cap_delay:
            continue
        # monthly_fraud_level: null means all
        if rule['monthly_fraud_level'] is not None and rule['monthly_fraud_level'] != fraud_level:
            continue
        # monthly_volume: null means all
        if rule['monthly_volume'] is not None and rule['monthly_volume'] != vol_category:
            continue
        # merchant_category_code: [] means all
        if rule['merchant_category_code'] and mcc not in rule['merchant_category_code']:
            continue
        # is_credit: null means all
        if rule['is_credit'] is not None and rule['is_credit'] != bool(txn['is_credit']):
            continue
        # aci: [] means all
        if rule['aci'] and txn['aci'] not in rule['aci']:
            continue
        # intracountry: null means all; 0.0=False, 1.0=True
        if rule['intracountry'] is not None and bool(rule['intracountry']) != intracountry:
            continue
        matches.append(rule)
    if not matches:
        return None  # no matching rule → no fee for this transaction
    # Most specific rule wins; break ties with lowest ID
    return sorted(matches, key=lambda r: (-rule_specificity(r), r['ID']))[0]
```

### Step 5: Sum All Transaction Fees

```python
total_fees = 0.0
for _, txn in target_transactions.iterrows():
    rule = find_best_rule(txn, acct_type, cap_delay, mcc, fraud_level, vol_category, fees)
    if rule:
        total_fees += rule['fixed_amount'] + rule['rate'] * txn['eur_amount'] / 10000

print(round(total_fees, 2))
```

---

## Complete Python Template (Single Day or Month)

```python
import json
import pandas as pd

# Load data (update paths as provided in the problem)
payments = pd.read_csv('<path>/payments.csv')
with open('<path>/fees.json') as f:
    fees = json.load(f)
with open('<path>/merchant_data.json') as f:
    merchant_data = json.load(f)

MERCHANT = '<merchant_name>'
YEAR = 2023

# Date range for the QUERY (e.g., single day or full month):
TARGET_DAYS = [<day_of_year>]              # e.g., [10] for day 10; list(range(60,91)) for March
# Date range of the MONTH containing the target (for monthly stats):
MONTH_START, MONTH_END = <start>, <end>   # e.g., 1, 31 for January

MONTH_RANGES = {
    1:(1,31), 2:(32,59), 3:(60,90), 4:(91,120),
    5:(121,151), 6:(152,181), 7:(182,212), 8:(213,243),
    9:(244,273), 10:(274,304), 11:(305,334), 12:(335,365)
}

def map_capture_delay(cd):
    try:
        n = int(cd)
        return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
    except (ValueError, TypeError):
        return cd

def categorize_volume(v):
    if v < 100_000:    return '<100k'
    elif v < 1_000_000: return '100k-1m'
    elif v < 5_000_000: return '1m-5m'
    else:               return '>5m'

def categorize_fraud(f):
    if f < 0.072:   return '<7.2%'
    elif f < 0.077: return '7.2%-7.7%'
    elif f < 0.083: return '7.7%-8.3%'
    else:           return '>8.3%'

def rule_specificity(rule):
    return sum([bool(rule['account_type']),
                rule['capture_delay'] is not None,
                rule['monthly_fraud_level'] is not None,
                rule['monthly_volume'] is not None,
                bool(rule['merchant_category_code']),
                rule['is_credit'] is not None,
                bool(rule['aci']),
                rule['intracountry'] is not None])

def find_best_rule(txn, acct_type, cap_delay, mcc, fraud_level, vol_category):
    intracountry = (txn['issuing_country'] == txn['acquirer_country'])
    matches = []
    for rule in fees:
        if rule['card_scheme'] != txn['card_scheme']: continue
        if rule['account_type'] and acct_type not in rule['account_type']: continue
        if rule['capture_delay'] is not None and rule['capture_delay'] != cap_delay: continue
        if rule['monthly_fraud_level'] is not None and rule['monthly_fraud_level'] != fraud_level: continue
        if rule['monthly_volume'] is not None and rule['monthly_volume'] != vol_category: continue
        if rule['merchant_category_code'] and mcc not in rule['merchant_category_code']: continue
        if rule['is_credit'] is not None and rule['is_credit'] != bool(txn['is_credit']): continue
        if rule['aci'] and txn['aci'] not in rule['aci']: continue
        if rule['intracountry'] is not None and bool(rule['intracountry']) != intracountry: continue
        matches.append(rule)
    if not matches: return None
    return sorted(matches, key=lambda r: (-rule_specificity(r), r['ID']))[0]

# Merchant properties
merchant = next(m for m in merchant_data if m['merchant'] == MERCHANT)
acct_type = merchant['account_type']
cap_delay = map_capture_delay(merchant['capture_delay'])
mcc = merchant['merchant_category_code']

# Monthly stats (full natural month)
month_txns = payments[
    (payments['merchant'] == MERCHANT) & (payments['year'] == YEAR) &
    (payments['day_of_year'].between(MONTH_START, MONTH_END))
]
monthly_vol = month_txns['eur_amount'].sum()
fraud_vol = month_txns[month_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_rate = fraud_vol / monthly_vol if monthly_vol > 0 else 0
fraud_level = categorize_fraud(fraud_rate)
vol_category = categorize_volume(monthly_vol)

# Target transactions and fee computation
target_txns = payments[
    (payments['merchant'] == MERCHANT) & (payments['year'] == YEAR) &
    (payments['day_of_year'].isin(TARGET_DAYS))
]
total_fees = 0.0
for _, txn in target_txns.iterrows():
    rule = find_best_rule(txn, acct_type, cap_delay, mcc, fraud_level, vol_category)
    if rule:
        total_fees += rule['fixed_amount'] + rule['rate'] * txn['eur_amount'] / 10000

print(round(total_fees, 2))
```

## Full Year Variation
For a full year query, monthly stats differ per month. Precompute them and apply per-transaction:

```python
# Precompute monthly stats for all 12 months
monthly_stats = {}
for month, (ms, me) in MONTH_RANGES.items():
    mt = payments[(payments['merchant']==MERCHANT) & (payments['year']==YEAR) &
                  (payments['day_of_year'].between(ms, me))]
    mv = mt['eur_amount'].sum()
    fv = mt[mt['has_fraudulent_dispute']==True]['eur_amount'].sum()
    fr = fv/mv if mv > 0 else 0
    monthly_stats[month] = (categorize_fraud(fr), categorize_volume(mv))

def get_month(day):
    for m, (s, e) in MONTH_RANGES.items():
        if s <= day <= e: return m
    return None

all_txns = payments[(payments['merchant']==MERCHANT) & (payments['year']==YEAR)]
total_fees = 0.0
for _, txn in all_txns.iterrows():
    m = get_month(txn['day_of_year'])
    fl, vc = monthly_stats[m]
    rule = find_best_rule(txn, acct_type, cap_delay, mcc, fl, vc)
    if rule:
        total_fees += rule['fixed_amount'] + rule['rate'] * txn['eur_amount'] / 10000
print(round(total_fees, 2))
```

---

## Key Facts About the Data

- **Merchants**: Crossfit_Hanna, Belles_cookbook_store, Golfclub_Baron_Friso, Martinis_Fine_Steakhouse, Rafa_AI
- **Years in data**: 2023 only
- **Card schemes**: TransactPlus, GlobalCard, NexPay, SwiftCharge
- **Empty list `[]` = "all values"** — true for `account_type`, `merchant_category_code`, `aci` in fee rules
- **MCC coverage**: Many merchant MCCs (5942, 7372, 7993, 7997, etc.) are NOT in any explicit MCC list in fees.json; only rules with `merchant_category_code: []` match these merchants
- **`intracountry`**: Always compute from `payments.csv` using `issuing_country == acquirer_country` — ignore merchant_data's acquirer list for this
- **Fraud**: `has_fraudulent_dispute == True` → `has_fraudulent_dispute` is a boolean column; fraud rate = sum(fraudulent eur_amount) / sum(total eur_amount) for the month
- **Transactions with no matching rule**: These have fee = 0 (excluded from total)

## Common Pitfalls
1. **Omitting `monthly_fraud_level`/`monthly_volume` from rule matching** — these fields significantly narrow the candidates
2. **Treating `[]` as "no match"** — an empty list in a fee rule means "applies to all values"
3. **Computing monthly stats from only the queried day** — must use the entire natural month
4. **Using merchant_data's `acquirer` to determine intracountry** — use `acquirer_country` from `payments.csv` directly
5. **Missing the numeric-to-category mapping for `capture_delay`** — `"1"` → `"<3"`, `"7"` → `">5"`, `"4"` → `"3-5"`
