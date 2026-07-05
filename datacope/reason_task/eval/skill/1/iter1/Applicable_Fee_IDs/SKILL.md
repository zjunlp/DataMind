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
| `manual.md` | Domain definitions (account types, ACI codes, fee field descriptions) |

> `acquirer_countries.csv` is NOT needed for Applicable_Fee_IDs questions. Do NOT use it to compute intracountry.

---

## Fee Rule Matching Logic

A fee applies when **every** non-null/non-empty condition in the fee record matches the transaction or merchant characteristic.

### Null/Empty = Applies to All
- List fields (`account_type`, `aci`, `merchant_category_code`): `[]` or `None` → matches all values
- Scalar fields (`capture_delay`, `is_credit`, `intracountry`, `monthly_volume`, `monthly_fraud_level`): `None` → matches all values

### intracountry Field
- In `fees.json`: `0.0` = international (False), `1.0` = domestic (True), `None` = both
- **Always compute from `payments.csv` directly**: `intracountry = (issuing_country == acquirer_country)`
- **NEVER look up acquirer country from `acquirer_countries.csv`** — the `acquirer_country` column in `payments.csv` already contains the per-transaction acquiring country. Using `acquirer_countries.csv` will produce wrong combinations and incorrect results.

### capture_delay Mapping
Merchant data has numeric strings or text. Map to fee categories:

```python
def map_capture_delay(raw):
    if raw in ('immediate', 'manual'):
        return raw
    try:
        n = int(raw)
        if n < 3:    return '<3'
        elif n <= 5: return '3-5'
        else:        return '>5'
    except:
        return raw
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

**Natural calendar months** (per manual): always use full natural months (Jan 1–31, Feb 1–28, etc.).

**Month day-range reference (non-leap year 2023):**
```
Jan: 1–31    Feb: 32–59   Mar: 60–90   Apr: 91–120
May: 121–151 Jun: 152–181 Jul: 182–212 Aug: 213–243
Sep: 244–273 Oct: 274–304 Nov: 305–334 Dec: 335–365
```

---

## Question Types and Solutions

### Type 1: Merchant + Day Query
**"What fee IDs apply to Merchant_X on day N of 2023?"**

```python
import pandas as pd, json

with open('fees.json') as f:          fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('payments.csv')

# Merchant attributes
merchant = next(m for m in merchants if m['merchant'] == 'MERCHANT_NAME')
acct_type = merchant['account_type']
mcc       = merchant['merchant_category_code']
cap_delay = map_capture_delay(merchant['capture_delay'])

# Filter transactions for the target day
DAY = N  # e.g., 200 for day 200
period_tx = payments[(payments['merchant'] == 'MERCHANT_NAME') &
                     (payments['year'] == 2023) &
                     (payments['day_of_year'] == DAY)].copy()

# Compute intracountry from payments.csv columns directly
period_tx['intracountry'] = period_tx['issuing_country'] == period_tx['acquirer_country']

# Unique (card_scheme, is_credit, aci, intracountry) combos that actually occurred
combos = period_tx[['card_scheme', 'is_credit', 'aci', 'intracountry']].drop_duplicates()

# Monthly stats: use the natural calendar month containing DAY
MONTH_START, MONTH_END = S, E  # look up in the day-range table above
month_tx = payments[(payments['merchant'] == 'MERCHANT_NAME') &
                    (payments['year'] == 2023) &
                    (payments['day_of_year'] >= MONTH_START) &
                    (payments['day_of_year'] <= MONTH_END)]
vol_bracket, fraud_bracket = get_monthly_bracket(month_tx)

# Match fees
def fee_applies(fee, cs, cr, ac, ic):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:              return False
    if fee['account_type'] and acct_type not in fee['account_type']:             return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != cap_delay:   return False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']: return False
    if fee['is_credit'] is not None and fee['is_credit'] != cr:                  return False
    if fee['aci'] and ac not in fee['aci']:                                       return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != ic:      return False
    if fee['monthly_volume'] is not None and fee['monthly_volume'] != vol_bracket:      return False
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
- `period_tx` = all transactions in the target month (use the day range from the table)
- `month_tx` = same as `period_tx` (month query → month stats use the same period)
- Apply monthly constraints normally

### Type 3: Merchant + Full-Year Query
**"What fee IDs apply to Merchant_X in 2023?"**

Same as Type 1 but:
- `period_tx` = all transactions in 2023 (`year == 2023`)
- **Skip `monthly_volume` and `monthly_fraud_level` checks entirely** — no single month represents the full year, so treat these fee conditions as non-applicable (skip them, don't filter on them)

```python
def fee_applies_yearly(fee, cs, cr, ac, ic):
    if fee['card_scheme'] is not None and fee['card_scheme'] != cs:              return False
    if fee['account_type'] and acct_type not in fee['account_type']:             return False
    if fee['capture_delay'] is not None and fee['capture_delay'] != cap_delay:   return False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']: return False
    if fee['is_credit'] is not None and fee['is_credit'] != cr:                  return False
    if fee['aci'] and ac not in fee['aci']:                                       return False
    if fee['intracountry'] is not None and bool(fee['intracountry']) != ic:      return False
    # monthly_volume and monthly_fraud_level are skipped for full-year queries
    return True
```

### Type 4: Simple Attribute Filter (No payment data needed)
**"What fee IDs apply to account_type=F and aci=A?"**

```python
import json
with open('fees.json') as f: fees = json.load(f)

matching = [fee['ID'] for fee in fees
            if (not fee['account_type'] or 'F' in fee['account_type'])
            and (not fee['aci'] or 'A' in fee['aci'])]
print(', '.join(str(x) for x in sorted(matching)))
```

### Type 5: Reverse Lookup — Fee → Merchants
**"Which merchants were affected by fee ID 709 in 2023?"**

```python
with open('fees.json') as f: fees = json.load(f)
fee = next(f for f in fees if f['ID'] == 709)
payments = pd.read_csv('payments.csv')
mask = (payments['year'] == 2023)
if fee['card_scheme']:      mask &= (payments['card_scheme'] == fee['card_scheme'])
if fee['is_credit'] is not None: mask &= (payments['is_credit'] == fee['is_credit'])
if fee['aci']:              mask &= (payments['aci'].isin(fee['aci']))
# For merchant-level conditions (account_type, MCC, capture_delay), join merchant_data.json
affected = sorted(payments[mask]['merchant'].unique())
print(', '.join(affected))
```

---

## Critical Pitfalls

1. **intracountry MUST come from `payments.csv['acquirer_country']`**, NOT from `acquirer_countries.csv`. The `acquirer_country` column in each payment row is the actual acquiring country for that transaction. Using `acquirer_countries.csv` to look up the merchant's acquirer and hardcoding its country as the acquirer country will produce wrong transaction combinations and wrong fee matches.

2. **Monthly constraints are decisive** — a fee with `monthly_volume='1m-5m'` does NOT apply to a merchant with `<100k` monthly volume. Always compute actual monthly stats from transaction data.

3. **capture_delay mapping is required** — `'1'` or `'2'` → `'<3'`; `'7'` → `'>5'`. A fee with `capture_delay='<3'` won't match a merchant with `capture_delay='manual'`.

4. **Empty list `[]` = applies to all**, same as `None`. Never treat `[]` as "no match".

5. **Use actual transaction combos** from `payments.csv` for the specific period, not all theoretical permutations. A fee only applies if a matching transaction actually occurred.

6. **Output the answer as soon as fee matching is complete.** Avoid excessive verification loops — they waste turns and risk hitting context limits before producing a result.

7. **Format**: comma-separated integers sorted ascending. Empty string if no fees apply.
