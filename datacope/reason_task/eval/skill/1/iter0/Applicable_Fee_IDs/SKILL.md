---
name: applicable-fee-ids
description: Solve questions about which fee IDs apply to a payment merchant, transaction characteristics, or time period in the dabstep dataset. Use this skill for any question asking "which fee IDs apply to X", "what are the applicable fee IDs for merchant Y", "which merchants are affected by fee Z", or any query involving filtering fees.json based on payment characteristics or merchant attributes.
---

# Applicable Fee IDs — Solution Guide

## Dataset Files

| File | Purpose |
|------|---------|
| `fees.json` | 1000 fee rules, each with conditions and an `ID` |
| `merchant_data.json` | Merchant attributes: `account_type`, `capture_delay`, `merchant_category_code`, `acquirer` list |
| `payments.csv` | Actual transactions: `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `has_fraudulent_dispute`, `day_of_year`, `year`, `merchant` |
| `acquirer_countries.csv` | Maps acquirer name → `country_code` |
| `manual.md` | Domain definitions (account types, ACI codes, fee field descriptions) |

---

## Fee Rule Matching Logic

A fee applies when **every** non-null/non-empty condition in the fee record matches the transaction or merchant characteristic.

### Null/Empty = Applies to All
```python
# For list fields (account_type, aci, merchant_category_code):
#   [] or None  → matches all values
# For scalar fields (capture_delay, is_credit, intracountry, monthly_volume, monthly_fraud_level):
#   None        → matches all values

def list_matches(fee_list, value):
    """Empty list or None means 'applies to all'."""
    return not fee_list or value in fee_list

def scalar_matches(fee_val, value):
    return fee_val is None or fee_val == value
```

### intracountry Field
- In `fees.json`, stored as float: `0.0` = international (False), `1.0` = domestic (True), `None` = both
- In `payments.csv`, compute: `intracountry = (issuing_country == acquirer_country)`

```python
def intracountry_matches(fee_val, tx_val):
    if fee_val is None:
        return True
    return bool(fee_val) == tx_val
```

### capture_delay Mapping
Merchant data has numeric strings ("1", "2", "7") or text ("immediate", "manual"). Map to fee categories:

```python
def map_capture_delay(merchant_delay):
    if merchant_delay in ('immediate', 'manual'):
        return merchant_delay
    try:
        n = int(merchant_delay)
        if n < 3:    return '<3'
        elif n <= 5: return '3-5'
        else:        return '>5'
    except:
        return merchant_delay
```

### Monthly Volume and Fraud Brackets
Fee conditions reference monthly aggregates. Compute for the relevant month(s):

```python
def get_monthly_bracket(txs):
    """txs: DataFrame filtered to the relevant month for this merchant."""
    total_vol = txs['eur_amount'].sum()
    fraud_vol = txs[txs['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    fraud_pct = (fraud_vol / total_vol * 100) if total_vol > 0 else 0

    if total_vol < 100_000:       vol_bracket = '<100k'
    elif total_vol < 1_000_000:   vol_bracket = '100k-1m'
    elif total_vol < 5_000_000:   vol_bracket = '1m-5m'
    else:                          vol_bracket = '>5m'

    if fraud_pct < 7.2:            fraud_bracket = '<7.2%'
    elif fraud_pct < 7.7:          fraud_bracket = '7.2%-7.7%'
    elif fraud_pct < 8.3:          fraud_bracket = '7.7%-8.3%'
    else:                           fraud_bracket = '>8.3%'

    return vol_bracket, fraud_bracket
```

**Which month to use:**
- Day query (e.g., "day 200 of 2023"): use the natural calendar month containing that day
- Month query (e.g., "October 2023"): use that month's transactions
- Full-year query (e.g., "in 2023"): skip monthly constraints (treat as null) — iterate across all yearly transactions

---

## Question Types and Solutions

### Type 1: Simple Attribute Filter
**"What fee IDs apply to account_type = F and aci = A?"**

No payment data needed. Filter `fees.json` directly:

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

# Extract the target values from the question
target_account_type = 'F'
target_aci = 'A'

matching = []
for fee in fees:
    acct_ok = not fee['account_type'] or target_account_type in fee['account_type']
    aci_ok  = not fee['aci']          or target_aci          in fee['aci']
    if acct_ok and aci_ok:
        matching.append(fee['ID'])

print(', '.join(str(x) for x in sorted(matching)))
```

---

### Type 2: Merchant + Period Query
**"What fee IDs apply to Merchant_X on day/month/year Y?"**

```python
import pandas as pd
import json

# Load data
with open('fees.json') as f:          fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('payments.csv')

# --- 1. Get merchant attributes ---
merchant = next(m for m in merchants if m['merchant'] == 'Merchant_X')
acct_type = merchant['account_type']           # e.g., 'R'
mcc       = merchant['merchant_category_code'] # e.g., 5942
cap_delay = map_capture_delay(merchant['capture_delay'])  # see above

# --- 2. Filter payments for the period ---
# For day 200 of 2023:
period_tx = payments[(payments['merchant'] == 'Merchant_X') &
                     (payments['year'] == 2023) &
                     (payments['day_of_year'] == 200)].copy()

# --- 3. Compute intracountry for each transaction ---
period_tx['intracountry'] = period_tx['issuing_country'] == period_tx['acquirer_country']

# --- 4. Get actual (card_scheme, is_credit, aci, intracountry) combinations ---
combos = period_tx[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()

# --- 5. Compute monthly stats (for the month containing day 200 = July) ---
month_tx = payments[(payments['merchant'] == 'Merchant_X') &
                    (payments['year'] == 2023) &
                    (payments['day_of_year'] >= 182) &  # July 1
                    (payments['day_of_year'] <= 212)]   # July 31
vol_bracket, fraud_bracket = get_monthly_bracket(month_tx)

# --- 6. Match fees ---
def fee_applies(fee, cs, cr, ac, ic, acct, cap, mcc_val, vol_b, fraud_b):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:       return False
    if fee['account_type'] and acct not in fee['account_type']:           return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != cap:  return False
    if fee['merchant_category_code'] and mcc_val not in fee['merchant_category_code']: return False
    if fee['is_credit'] is not None and fee['is_credit'] != cr:           return False
    if fee['aci'] and ac not in fee['aci']:                                return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != ic: return False
    if fee['monthly_volume'] is not None and fee['monthly_volume'] != vol_b:       return False
    if fee['monthly_fraud_level'] is not None and fee['monthly_fraud_level'] != fraud_b: return False
    return True

applicable = set()
for _, row in combos.iterrows():
    for fee in fees:
        if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                       row['intracountry'], acct_type, cap_delay, mcc,
                       vol_bracket, fraud_bracket):
            applicable.add(fee['ID'])

print(', '.join(str(x) for x in sorted(applicable)))
```

**Month day-range reference (non-leap year 2023):**
```
Jan: 1–31    Feb: 32–59   Mar: 60–90   Apr: 91–120
May: 121–151 Jun: 152–181 Jul: 182–212 Aug: 213–243
Sep: 244–273 Oct: 274–304 Nov: 305–334 Dec: 335–365
```

For full-year queries, **skip monthly constraints** (remove the `monthly_volume` and `monthly_fraud_level` checks from `fee_applies`), because no single month represents the full year.

---

### Type 3: Reverse Lookup — Fee → Merchants
**"In 2023, which merchants were affected by fee ID 709?"**

```python
import json, pandas as pd

with open('fees.json') as f:
    fees = json.load(f)

fee = next(f for f in fees if f['ID'] == 709)
# Fee 709: card_scheme='GlobalCard', is_credit=False, aci=['A','B','C']

payments = pd.read_csv('payments.csv')
mask = (payments['year'] == 2023)

# Apply non-null conditions from the fee rule
if fee['card_scheme']:
    mask &= (payments['card_scheme'] == fee['card_scheme'])
if fee['is_credit'] is not None:
    mask &= (payments['is_credit'] == fee['is_credit'])
if fee['aci']:
    mask &= (payments['aci'].isin(fee['aci']))
# account_type, MCC, capture_delay are merchant-level — handled via merchant_data.json if needed
# intracountry, monthly_* conditions are usually not applied in reverse lookups unless specified

affected = sorted(payments[mask]['merchant'].unique())
print(', '.join(affected))
```

---

## Critical Pitfalls

1. **Monthly constraints are often the deciding factor** — fees with `monthly_volume='1m-5m'` do NOT apply to a merchant with `<100k` monthly volume. Always compute actual monthly stats and filter.

2. **capture_delay mapping**: "1" or "2" → `'<3'`, "7" → `'>5'`; do NOT skip this mapping. A fee with `capture_delay='<3'` won't match a merchant with `capture_delay='manual'`.

3. **intracountry can be True for some merchants** — merchants with multiple acquirers in EU countries (e.g., acquirer in NL) will have intracountry=True transactions when issuing_country matches acquirer_country. Always compute from actual transaction data, not hardcoded False.

4. **Empty list `[]` = applies to all**, same as `None`. Never treat `[]` as "no match".

5. **Use actual transaction combinations** from payments.csv for the specific period, not all theoretical permutations. A fee only applies if a matching transaction actually occurred in the queried period.

6. **Format the answer** as a comma-separated list of integers sorted in ascending order. Return an empty string if no fees apply.
