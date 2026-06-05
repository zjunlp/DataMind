---
name: applicable-fee-ids
description: Solve questions about which fee IDs apply to a payment merchant, transaction characteristics, or time period in the dabstep dataset. Use this skill for any question asking "which fee IDs apply to X", "what are the applicable fee IDs for merchant Y", "which merchants are affected by fee Z", or any query involving filtering fees.json based on payment characteristics or merchant attributes.
---

# Applicable Fee IDs — Solution Guide

## CRITICAL: Turn Efficiency

**DO NOT read `manual.md` or `payments-readme.md`** — all domain knowledge needed is already in this skill. Reading them wastes 2–4 turns and risks hitting the turn limit without producing an answer.

**Execute all logic in a single Python code block.** Splitting into multiple steps wastes turns. Aim to complete the full analysis in 1–2 code blocks and output the answer immediately.

---

## Dataset Files

| File | Purpose |
|------|---------|
| `fees.json` | 1000 fee rules, each with conditions and an `ID` |
| `merchant_data.json` | Merchant attributes: `account_type`, `capture_delay`, `merchant_category_code`, `acquirer` list |
| `payments.csv` | Actual transactions: `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `has_fraudulent_dispute`, `day_of_year`, `year`, `merchant` |

> `acquirer_countries.csv` is NOT needed. Do NOT use it.

---

## Fee Rule Matching Logic

A fee applies when **every** non-null/non-empty condition in the fee record matches the transaction or merchant characteristic.

### Null/Empty = Applies to All
- List fields (`account_type`, `aci`, `merchant_category_code`): `[]` or `None` → matches all values
- Scalar fields (`capture_delay`, `is_credit`, `intracountry`, `monthly_volume`, `monthly_fraud_level`): `None` → matches all values

### intracountry
- In `fees.json`: `0.0` = international, `1.0` = domestic, `None` = both
- **Compute directly**: `intracountry = (issuing_country == acquirer_country)` from `payments.csv`
- **Never use `acquirer_countries.csv`** — per-transaction `acquirer_country` column in `payments.csv` is the correct source

### capture_delay Mapping
```python
def map_capture_delay(raw):
    if raw in ('immediate', 'manual'): return raw
    try:
        n = int(raw)
        if n < 3:    return '<3'
        elif n <= 5: return '3-5'
        else:        return '>5'
    except: return raw
```

### Monthly Volume and Fraud Brackets
```python
def get_monthly_bracket(month_txs):
    total_vol = month_txs['eur_amount'].sum()
    fraud_vol = month_txs[month_txs['has_fraudulent_dispute'] == True]['eur_amount'].sum()
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

**Natural calendar months (2023, non-leap year):**
```
Jan: 1–31    Feb: 32–59   Mar: 60–90   Apr: 91–120
May: 121–151 Jun: 152–181 Jul: 182–212 Aug: 213–243
Sep: 244–273 Oct: 274–304 Nov: 305–334 Dec: 335–365
```

---

## Complete Single-Run Templates

### Type 1: Merchant + Day Query
**"What fee IDs apply to Merchant_X on day N of 2023?"**

Run this entire block at once:
```python
import pandas as pd, json

# ── CONFIG ──────────────────────────────────────
MERCHANT = 'Belles_cookbook_store'
DAY      = 200
MONTH_START, MONTH_END = 182, 212  # July; look up in day-range table above
# ─────────────────────────────────────────────────

with open('/path/to/fees.json') as f:          fees = json.load(f)
with open('/path/to/merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('/path/to/payments.csv')

merchant  = next(m for m in merchants if m['merchant'] == MERCHANT)
acct_type = merchant['account_type']
mcc       = merchant['merchant_category_code']

def map_capture_delay(raw):
    if raw in ('immediate', 'manual'): return raw
    try:
        n = int(raw)
        return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
    except: return raw

cap_delay = map_capture_delay(merchant['capture_delay'])

period_tx = payments[(payments['merchant'] == MERCHANT) &
                     (payments['year'] == 2023) &
                     (payments['day_of_year'] == DAY)].copy()
period_tx['intracountry'] = period_tx['issuing_country'] == period_tx['acquirer_country']
combos = period_tx[['card_scheme','is_credit','aci','intracountry']].drop_duplicates()

month_tx = payments[(payments['merchant'] == MERCHANT) &
                    (payments['year'] == 2023) &
                    (payments['day_of_year'] >= MONTH_START) &
                    (payments['day_of_year'] <= MONTH_END)]
total_vol = month_tx['eur_amount'].sum()
fraud_vol = month_tx[month_tx['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_pct = (fraud_vol / total_vol * 100) if total_vol > 0 else 0
vol_bracket   = '<100k' if total_vol < 100_000 else ('100k-1m' if total_vol < 1_000_000 else ('1m-5m' if total_vol < 5_000_000 else '>5m'))
fraud_bracket = '<7.2%' if fraud_pct < 7.2 else ('7.2%-7.7%' if fraud_pct < 7.7 else ('7.7%-8.3%' if fraud_pct < 8.3 else '>8.3%'))

def fee_applies(fee, cs, cr, ac, ic):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:                    return False
    if fee['account_type'] and acct_type not in fee['account_type']:                   return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != cap_delay:         return False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']:     return False
    if fee['is_credit'] is not None and fee['is_credit'] != cr:                        return False
    if fee['aci'] and ac not in fee['aci']:                                             return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != ic:            return False
    if fee['monthly_volume'] is not None and fee['monthly_volume'] != vol_bracket:     return False
    if fee['monthly_fraud_level'] is not None and fee['monthly_fraud_level'] != fraud_bracket: return False
    return True

applicable = set()
for _, row in combos.iterrows():
    for fee in fees:
        if fee_applies(fee, row['card_scheme'], row['is_credit'], row['aci'], row['intracountry']):
            applicable.add(fee['ID'])

print(', '.join(str(x) for x in sorted(applicable)))
```

### Type 2: Merchant + Month Query
**"What fee IDs apply to Merchant_X in [month] 2023?"**

Same as Type 1 but:
- `DAY` → use month range for `day_of_year` filter: `period_tx = payments[... & (day_of_year >= MONTH_START) & (day_of_year <= MONTH_END)]`
- `month_tx` = same as `period_tx`
- Apply monthly constraints normally

### Type 3: Merchant + Full-Year Query
**"What fee IDs apply to Merchant_X in 2023?"**

Same as Type 1 but:
- `period_tx = payments[(payments['merchant'] == MERCHANT) & (payments['year'] == 2023)]`
- **Skip `monthly_volume` and `monthly_fraud_level` checks** (no single month represents the full year)

```python
def fee_applies_yearly(fee, cs, cr, ac, ic):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:                return False
    if fee['account_type'] and acct_type not in fee['account_type']:               return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != cap_delay:     return False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']: return False
    if fee['is_credit'] is not None and fee['is_credit'] != cr:                    return False
    if fee['aci'] and ac not in fee['aci']:                                         return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != ic:        return False
    # monthly_volume and monthly_fraud_level intentionally skipped
    return True
```

### Type 4: Simple Attribute Filter (no payments data needed)
**"What fee IDs apply to account_type=F and aci=A?"**

```python
import json
with open('/path/to/fees.json') as f: fees = json.load(f)
matching = [fee['ID'] for fee in fees
            if (not fee['account_type'] or 'F' in fee['account_type'])
            and (not fee['aci'] or 'A' in fee['aci'])]
print(', '.join(str(x) for x in sorted(matching)))
```

### Type 5: Reverse Lookup — Fee → Merchants
**"Which merchants were affected by fee ID 709 in 2023?"**

```python
import pandas as pd, json
with open('/path/to/fees.json') as f: fees = json.load(f)
with open('/path/to/merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('/path/to/payments.csv')

target_fee = next(f for f in fees if f['ID'] == 709)

# Build merchant lookup for attribute-based conditions
mdata = {m['merchant']: m for m in merchants}

def map_capture_delay(raw):
    if raw in ('immediate', 'manual'): return raw
    try:
        n = int(raw)
        return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
    except: return raw

mask = (payments['year'] == 2023)
if target_fee['card_scheme']:      mask &= (payments['card_scheme'] == target_fee['card_scheme'])
if target_fee['is_credit'] is not None: mask &= (payments['is_credit'] == target_fee['is_credit'])
if target_fee['aci']:              mask &= (payments['aci'].isin(target_fee['aci']))
if target_fee['intracountry'] is not None:
    mask &= ((payments['issuing_country'] == payments['acquirer_country']) == bool(target_fee['intracountry']))

candidate_tx = payments[mask]

# Filter by merchant-level conditions
affected = []
for merchant_name in candidate_tx['merchant'].unique():
    m = mdata.get(merchant_name)
    if m is None: continue
    if target_fee['account_type'] and m['account_type'] not in target_fee['account_type']: continue
    if target_fee['capture_delay'] is not None and map_capture_delay(m['capture_delay']) != target_fee['capture_delay']: continue
    if target_fee['merchant_category_code'] and m['merchant_category_code'] not in target_fee['merchant_category_code']: continue
    affected.append(merchant_name)

print(', '.join(sorted(affected)))
```

---

## Critical Pitfalls

1. **Output immediately after fee matching** — do NOT run post-computation verification. The computed result is the answer. Every extra turn risks hitting the turn limit.

2. **intracountry from `payments.csv` only** — use `issuing_country == acquirer_country` per transaction. Never use `acquirer_countries.csv`.

3. **Monthly constraints are decisive** — a fee with `monthly_volume='1m-5m'` does NOT apply to a merchant with `<100k` monthly volume. Always compute actual monthly stats from transaction data.

4. **capture_delay mapping is required** — `'1'` or `'2'` → `'<3'`; `'7'` → `'>5'`. A fee with `capture_delay='<3'` won't match a merchant with `capture_delay='manual'`.

5. **Empty list `[]` = applies to all** — same as `None`. Never treat `[]` as "no match".

6. **Use actual transaction combos** from `payments.csv` for the specific period. A fee only applies if a matching transaction actually occurred.

7. **Format**: comma-separated integers sorted ascending. Empty string if no fees apply.
