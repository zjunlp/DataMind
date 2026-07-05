---
name: Applicable_Fee_IDs
description: >
  Solve dabstep dataset questions about identifying applicable fee IDs from fees.json.
  Use this skill for any question asking which fee IDs apply to given conditions,
  a specific merchant on a specific day/month/year, or which merchants are affected
  by a specific fee ID.
---

# Applicable Fee IDs Skill

## Dataset Files
- `fees.json`: 1000 fee rules with matching criteria
- `merchant_data.json`: merchant account_type, capture_delay, MCC, acquirers (list of dicts)
- `payments.csv`: transactions with columns: merchant, card_scheme, year, day_of_year, is_credit, eur_amount, issuing_country, acquirer_country, aci, has_fraudulent_dispute
- `acquirer_countries.csv`: maps acquirer name to country_code (rarely needed—payments.csv already has acquirer_country)

## Fee Rule Structure (fees.json)
Each fee has: `ID`, `card_scheme`, `account_type` (list), `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `merchant_category_code` (list), `is_credit`, `aci` (list), `fixed_amount`, `rate`, `intracountry`

**Critical matching rule**: An **empty list `[]`** for `account_type`, `aci`, or `merchant_category_code` means "applies to all values"—same as `null`. There are NO null values for these list fields in fees.json; only empty lists and populated lists exist.

A fee rule applies when ALL of these match:
| Field | Match condition |
|-------|----------------|
| `card_scheme` | null → all; otherwise exact match |
| `account_type` | `[]` or null → all; otherwise merchant's account_type in list |
| `capture_delay` | null → all; otherwise exact string match |
| `monthly_fraud_level` | null → all; otherwise computed monthly fraud category must match |
| `monthly_volume` | null → all; otherwise computed monthly volume category must match |
| `merchant_category_code` | `[]` or null → all; otherwise merchant's MCC in list |
| `is_credit` | null → all; otherwise must match transaction's is_credit |
| `aci` | `[]` or null → all; otherwise transaction's ACI in list |
| `intracountry` | null → all; 0.0 → international (issuing≠acquirer); 1.0 → domestic (issuing==acquirer) |

## Question Types & Approaches

### Type 1: Simple Attribute Filter
**Pattern**: "What fee IDs apply to account_type = X and aci = Y?"

Filter fees.json directly—no payments data needed:
```python
import json
with open('fees.json') as f:
    fees = json.load(f)

account_type = 'F'
aci = 'A'

result = [
    fee['ID'] for fee in fees
    if (not fee['account_type'] or account_type in fee['account_type'])
    and (not fee['aci'] or aci in fee['aci'])
]
print(', '.join(str(x) for x in sorted(result)))
```
Apply the same pattern for other simple field combinations (card_scheme, is_credit, etc.).

---

### Type 2: Merchant + Date (Day/Month/Year) → Applicable Fee IDs
**Pattern**: "For the Nth day of 2023, what Fee IDs apply to [Merchant]?" or "What Fee IDs applied in [Month] 2023?"

**Step 1: Get merchant characteristics**
```python
with open('merchant_data.json') as f:
    merchants = json.load(f)  # list of dicts
merchant = next(m for m in merchants if m['merchant'] == 'Belles_cookbook_store')
account_type = merchant['account_type']      # e.g. 'R'
mcc = merchant['merchant_category_code']     # e.g. 5942
capture_delay_raw = merchant['capture_delay']  # e.g. '1', '7', 'immediate', 'manual'
```

**Step 2: Map capture_delay to fee rule format**
```python
def map_capture_delay(raw):
    try:
        days = int(raw)
        if days < 3: return '<3'
        elif days <= 5: return '3-5'
        else: return '>5'
    except ValueError:
        return raw  # already 'immediate', 'manual', '<3', '3-5', '>5'
capture_delay = map_capture_delay(capture_delay_raw)
```

**Step 3: Determine the time window and filter payments**
```python
import pandas as pd, datetime
payments = pd.read_csv('payments.csv')

# For a specific day (day_of_year=200, year=2023):
txns = payments[(payments['merchant']=='Belles_cookbook_store') &
                (payments['year']==2023) & (payments['day_of_year']==200)]

# For October 2023:
oct_start = datetime.date(2023, 10, 1).timetuple().tm_yday   # 274
oct_end   = datetime.date(2023, 10, 31).timetuple().tm_yday  # 304
txns = payments[(payments['merchant']=='Belles_cookbook_store') &
                (payments['year']==2023) &
                (payments['day_of_year'] >= oct_start) &
                (payments['day_of_year'] <= oct_end)]

# For all of 2023:
txns = payments[(payments['merchant']=='Belles_cookbook_store') & (payments['year']==2023)]
```

**Step 4: Compute monthly volume & fraud categories for the relevant month(s)**

Monthly stats must be computed for the calendar month(s) containing the transactions. For a specific day query, compute for the month that day falls in. For month-wide query, compute for that month. For year-wide query, compute per-month and union results.

```python
def volume_category(eur):
    if eur < 100_000:   return '<100k'
    elif eur < 1_000_000: return '100k-1m'
    elif eur < 5_000_000: return '1m-5m'
    else:               return '>5m'

def fraud_category(rate_pct):
    if rate_pct < 7.2:    return '<7.2%'
    elif rate_pct < 7.7:  return '7.2%-7.7%'
    elif rate_pct <= 8.3: return '7.7%-8.3%'
    else:                 return '>8.3%'

# For a single month:
total_vol = txns['eur_amount'].sum()
fraud_vol = txns[txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_rate = (fraud_vol / total_vol * 100) if total_vol > 0 else 0
vol_cat = volume_category(total_vol)
fraud_cat = fraud_category(fraud_rate)
```

**Step 5: Identify applicable fee IDs**
```python
def fee_applies(fee, cs, is_credit, aci, intracountry, vol_cat, fraud_cat,
                account_type, mcc, capture_delay):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:
        return False
    if fee['account_type'] and account_type not in fee['account_type']:
        return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != capture_delay:
        return False
    if fee['monthly_fraud_level'] is not None and fee['monthly_fraud_level'] != fraud_cat:
        return False
    if fee['monthly_volume'] is not None and fee['monthly_volume'] != vol_cat:
        return False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']:
        return False
    if fee['is_credit'] is not None and fee['is_credit'] != is_credit:
        return False
    if fee['aci'] and aci not in fee['aci']:
        return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != intracountry:
        return False
    return True

with open('fees.json') as f:
    fees = json.load(f)

applicable = set()
for _, row in txns.iterrows():
    intracountry = (row['issuing_country'] == row['acquirer_country'])
    for fee in fees:
        if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                       intracountry, vol_cat, fraud_cat, account_type, mcc, capture_delay):
            applicable.add(fee['ID'])

print(', '.join(str(x) for x in sorted(applicable)))
```

**For year-wide queries**, loop over each month (1–12), compute monthly stats for that month's transactions, then union all applicable fee IDs:
```python
applicable = set()
for month in range(1, 13):
    m_start = datetime.date(2023, month, 1).timetuple().tm_yday
    m_end = (datetime.date(2023, month+1, 1) - datetime.timedelta(days=1)).timetuple().tm_yday \
            if month < 12 else 365
    month_txns = txns[(txns['day_of_year'] >= m_start) & (txns['day_of_year'] <= m_end)]
    if month_txns.empty:
        continue
    total_vol = month_txns['eur_amount'].sum()
    fraud_vol = month_txns[month_txns['has_fraudulent_dispute']==True]['eur_amount'].sum()
    fraud_rate = (fraud_vol / total_vol * 100) if total_vol > 0 else 0
    v_cat = volume_category(total_vol)
    f_cat = fraud_category(fraud_rate)
    for _, row in month_txns.iterrows():
        intracountry = (row['issuing_country'] == row['acquirer_country'])
        for fee in fees:
            if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'],
                           intracountry, v_cat, f_cat, account_type, mcc, capture_delay):
                applicable.add(fee['ID'])
```

---

### Type 3: Reverse Lookup — Which Merchants Does Fee X Affect?
**Pattern**: "Which merchants were affected by Fee ID 709 in 2023?"

```python
fee = next(f for f in fees if f['ID'] == 709)
# Filter transactions matching the fee's criteria
subset = payments[payments['year'] == 2023].copy()
if fee['card_scheme']:
    subset = subset[subset['card_scheme'] == fee['card_scheme']]
if fee['is_credit'] is not None:
    subset = subset[subset['is_credit'] == fee['is_credit']]
if fee['aci']:
    subset = subset[subset['aci'].isin(fee['aci'])]
if fee['intracountry'] is not None:
    subset = subset[(subset['issuing_country'] == subset['acquirer_country']) == bool(fee['intracountry'])]
# merchant-level filters (account_type, mcc, capture_delay) require joining merchant_data
merchants_affected = sorted(subset['merchant'].unique())
print(', '.join(merchants_affected))
```
If the fee has account_type, mcc, or capture_delay constraints, also filter merchants from merchant_data before joining.

---

## Output Format
- Applicable fee IDs: comma-separated integers in ascending order: `3, 4, 5, 7, ...`
- Empty result: empty string `""`
- If question is irrelevant: `Not Applicable`

## Common Pitfalls
- **Empty list ≠ null for lists**: In fees.json, empty `[]` for account_type/aci/merchant_category_code means "all"—treat it the same as null
- **capture_delay conversion**: Merchant's numeric delay (e.g., "1", "7") must be mapped to fee rule strings ("<3", "3-5", ">5")
- **Monthly stats are required**: Fees with `monthly_fraud_level` or `monthly_volume` constraints only apply when the merchant's actual monthly stats match—never skip this calculation
- **intracountry is float in fees.json**: `0.0` = international, `1.0` = domestic; compare with `bool(fee['intracountry'])`
- **merchant_data.json is a list**: Access with `next(m for m in merchants if m['merchant'] == name)`, not dict-style lookup
- **payments.csv has `acquirer_country` directly**: No need to join acquirer_countries.csv for intracountry calculation
