---
name: Fee_Delta_and_Impact_Simulation
description: Solve dabstep Fee_Delta_and_Impact_Simulation questions: computing fee deltas when a fee's rate changes, identifying which merchants are affected by fee rule changes, and computing fee deltas when a merchant's MCC changes. Use when asked about fee impact, delta payments, rate changes, MCC code changes, or which merchants would be affected by modifying a fee rule.
---

# Fee Delta and Impact Simulation

This skill covers three question types:
1. **Rate-change delta**: How much more/less would a merchant pay if a specific fee's rate changed?
2. **Affected-merchant identification**: Which merchants would be impacted if a fee rule's applicability changed?
3. **MCC-change delta**: How much more/less would a merchant pay if its MCC code changed?

## Data Files

| File | Purpose |
|------|---------|
| `fees.json` | 1000 fee rules with matching criteria and rates |
| `payments.csv` | 138,236 transactions (all year 2023) |
| `merchant_data.json` | 30 merchants with account_type, MCC, capture_delay, acquirers |
| `acquirer_countries.csv` | Maps acquirer names → country codes |
| `manual.md` | Domain definitions (read first) |

## Key Domain Rules

### Fee Formula
```
fee = fixed_amount + rate * transaction_value / 10000
```
**"Relative fee"** = the `rate` field. When a question says "relative fee changed to X", it means the `rate` changes from its current value to X.

### Fee Rule Matching

A transaction/merchant matches a fee rule when ALL conditions hold:

| Fee field | Condition |
|-----------|-----------|
| `card_scheme` | Exact match with transaction's card_scheme |
| `account_type` | `[]` (empty list) = applies to all; else merchant's account_type must be in the list |
| `merchant_category_code` | `[]` (empty list) = applies to all; else merchant's MCC must be in the list |
| `capture_delay` | `null` = applies to all; else see mapping below |
| `monthly_fraud_level` | `null` = applies to all; else see calculation below |
| `monthly_volume` | `null` = applies to all; else see calculation below |
| `is_credit` | `null` = applies to all; else must equal transaction's is_credit |
| `aci` | `[]` (empty list) or `null` = applies to all; else transaction's aci must be in the list |
| `intracountry` | `null` = applies to all; `1.0` = issuing_country == acquirer_country; `0.0` = issuing_country != acquirer_country |

**Critical**: Empty list `[]` for `account_type`, `merchant_category_code`, and `aci` means "applies to all values".

**When multiple fees match a transaction**: Use the **first matching fee** — iterate fees.json in order (fees are sorted by ascending ID) and return the FIRST fee that satisfies ALL criteria. Do NOT pick the most-specific one or sum them all.

### Capture Delay Mapping

Merchant `capture_delay` values must be mapped to fee rule categories:
```
'immediate' → matches fee rule 'immediate'
'manual'    → matches fee rule 'manual'
numeric n   → '<3' if n < 3 (e.g., '1', '2')
              '3-5' if 3 ≤ n ≤ 5
              '>5' if n > 5 (e.g., '7')
```

### Monthly Fraud Level Calculation

Compute per merchant per natural calendar month (using day_of_year):
```python
fraud_volume = sum(eur_amount where has_fraudulent_dispute == True)
total_volume = sum(eur_amount)
fraud_ratio = fraud_volume / total_volume * 100  # as percentage

# Map to fee rule category:
# '<7.2%'    : ratio < 7.2
# '7.2%-7.7%': 7.2 <= ratio <= 7.7
# '7.7%-8.3%': 7.7 <= ratio <= 8.3
# '>8.3%'    : ratio > 8.3
```

### Monthly Volume Calculation

Compute per merchant per natural calendar month:
```python
monthly_total = sum(eur_amount)  # in euros

# Map to fee rule category:
# '<100k'  : total < 100,000
# '100k-1m': 100,000 <= total <= 1,000,000
# '1m-5m'  : 1,000,000 <= total <= 5,000,000
# '>5m'    : total > 5,000,000
```

### Month → day_of_year (2023, non-leap year)

| Month | day_of_year range |
|-------|-------------------|
| January | 1–31 |
| February | 32–59 |
| March | 60–90 |
| April | 91–120 |
| May | 121–151 |
| June | 152–181 |
| July | 182–212 |
| August | 213–243 |
| September | 244–273 |
| October | 274–304 |
| November | 305–334 |
| December | 335–365 |

### Intracountry

Use `acquirer_country` directly from `payments.csv` (already resolved). Compare with `issuing_country`:
```python
is_intracountry = (row['issuing_country'] == row['acquirer_country'])
```

---

## Question Type 1: Rate-Change Delta

**Question pattern**: "In [period] what delta would [merchant] pay if the relative fee of the fee with ID=[X] changed to [Y]?"

### Step-by-step Algorithm

```python
import json, pandas as pd

# 1. Load fee rule
with open('fees.json') as f:
    fees = json.load(f)
fee = next(f for f in fees if f['ID'] == X)
old_rate = fee['rate']
new_rate = Y  # from question

# 2. Load merchant properties
with open('merchant_data.json') as f:
    merchants = json.load(f)
merchant = next(m for m in merchants if m['merchant'] == MERCHANT_NAME)

# 3. Load payments and filter by merchant + time period
payments = pd.read_csv('payments.csv')
mask = payments['merchant'] == MERCHANT_NAME
# For a specific month: also filter day_of_year range
# For a full year: no day_of_year filter needed

# 4. Check merchant-level criteria (checked once, not per transaction)
merchant_matches = True
if fee['account_type']:
    merchant_matches &= merchant['account_type'] in fee['account_type']
if fee['merchant_category_code']:
    merchant_matches &= merchant['merchant_category_code'] in fee['merchant_category_code']
if fee['capture_delay'] is not None:
    merchant_matches &= capture_delay_matches(merchant['capture_delay'], fee['capture_delay'])

if not merchant_matches:
    print("Delta = 0 (merchant does not match fee rule)")
else:
    # 5. Filter transactions by fee's transaction-level criteria
    if fee['is_credit'] is not None:
        mask &= payments['is_credit'] == fee['is_credit']
    if fee['aci']:  # non-empty list
        mask &= payments['aci'].isin(fee['aci'])
    if fee['intracountry'] is not None:
        if fee['intracountry'] == 1.0:
            mask &= payments['issuing_country'] == payments['acquirer_country']
        else:
            mask &= payments['issuing_country'] != payments['acquirer_country']

    # 6. Handle monthly conditions (if fee has monthly_fraud_level or monthly_volume)
    # If both are null: just sum all matching transactions
    # If either is non-null: must filter month by month

    matching = payments[mask]
    total_amount = matching['eur_amount'].sum()
    delta = (new_rate - old_rate) * total_amount / 10000
    print(f"Delta = {delta:.14f}")
```

**Capture delay helper:**
```python
def capture_delay_matches(merchant_val, fee_val):
    if fee_val is None:
        return True
    if merchant_val in ('immediate', 'manual'):
        return merchant_val == fee_val
    n = int(merchant_val)
    if fee_val == '<3': return n < 3
    if fee_val == '3-5': return 3 <= n <= 5
    if fee_val == '>5': return n > 5
    return False
```

### Important: Monthly Conditions for Rate-Change

When the fee has `monthly_fraud_level` or `monthly_volume` != null, these are **merchant-level, month-level** conditions. For a single-month query, calculate for that month. For a full-year query, the fee may apply in some months but not others — filter transactions month-by-month.

---

## Question Type 2: Affected Merchants (Account Type Change)

**Question pattern**: "During 2023, imagine if the Fee with ID [X] was only applied to account type [T], which merchants would have been affected by this change?"

### Interpretation

"Affected" = merchants whose fee X application STATUS CHANGES:
- Currently: if `account_type = []` (applies to all), ALL matching merchants use fee X
- After change: only account_type T merchants use fee X
- Affected = merchants whose current status ≠ new status AND who have transactions that would be matched by other fee criteria

### Step-by-step Algorithm

```python
# 1. Load fee rule
fee = next(f for f in fees if f['ID'] == X)

# 2. For each merchant, check if fee status changes
affected = []
for m in merchants:
    # Check merchant-level criteria (MCC, capture_delay) - excluding account_type
    if fee['merchant_category_code'] and m['merchant_category_code'] not in fee['merchant_category_code']:
        continue
    if fee['capture_delay'] is not None and not capture_delay_matches(m['capture_delay'], fee['capture_delay']):
        continue

    # Check if merchant has any matching transactions (transaction-level filters)
    m_payments = payments[payments['merchant'] == m['merchant']]
    mask = m_payments['card_scheme'] == fee['card_scheme']
    if fee['is_credit'] is not None:
        mask &= m_payments['is_credit'] == fee['is_credit']
    if fee['aci']:
        mask &= m_payments['aci'].isin(fee['aci'])
    if fee['intracountry'] is not None:
        if fee['intracountry'] == 1.0:
            mask &= m_payments['issuing_country'] == m_payments['acquirer_country']
        else:
            mask &= m_payments['issuing_country'] != m_payments['acquirer_country']

    if mask.sum() == 0:
        continue  # No matching transactions

    # Check if account_type status changes
    currently_applies = not fee['account_type'] or m['account_type'] in fee['account_type']
    would_apply = m['account_type'] == T  # after the change

    if currently_applies != would_apply:
        affected.append(m['merchant'])

print(', '.join(sorted(affected)))
```

---

## Question Type 3: MCC-Change Delta

**Question pattern**: "Imagine the merchant [M] had changed its MCC code to [NEW_MCC] before 2023 started, what amount delta will it have to pay in fees for the year 2023?"

### Algorithm

For each transaction, find the first matching fee under the OLD MCC and NEW MCC. Delta = total_fees(new_mcc) - total_fees(old_mcc).

**Write this as a single complete script — do not split into multiple exploration steps to avoid hitting turn limits.**

```python
import json
import pandas as pd

# Load data
with open('fees.json') as f:
    fees = json.load(f)
with open('merchant_data.json') as f:
    merchants = json.load(f)
payments = pd.read_csv('payments.csv')

# Merchant info
merchant = next(m for m in merchants if m['merchant'] == MERCHANT_NAME)
old_mcc = merchant['merchant_category_code']
new_mcc = NEW_MCC  # from question
acct_type = merchant['account_type']
cap_delay = str(merchant['capture_delay'])

# Filter transactions for this merchant
df = payments[payments['merchant'] == MERCHANT_NAME].copy()

# Map day_of_year to month
def day_to_month(day):
    boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    for i in range(1, len(boundaries)):
        if day <= boundaries[i]:
            return i
    return 12

df['month'] = df['day_of_year'].apply(day_to_month)

# Precompute monthly fraud and volume categories
def get_fraud_cat(ratio):
    if ratio < 7.2: return '<7.2%'
    if ratio <= 7.7: return '7.2%-7.7%'
    if ratio <= 8.3: return '7.7%-8.3%'
    return '>8.3%'

def get_vol_cat(vol):
    if vol < 100000: return '<100k'
    if vol <= 1000000: return '100k-1m'
    if vol <= 5000000: return '1m-5m'
    return '>5m'

monthly_cats = {}
for month in range(1, 13):
    md = df[df['month'] == month]
    fv = md[md['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    tv = md['eur_amount'].sum()
    ratio = (fv / tv * 100) if tv > 0 else 0
    monthly_cats[month] = (get_fraud_cat(ratio), get_vol_cat(tv))

def capture_delay_matches(merchant_val, fee_val):
    if fee_val is None: return True
    if merchant_val in ('immediate', 'manual'): return merchant_val == fee_val
    n = int(merchant_val)
    if fee_val == '<3': return n < 3
    if fee_val == '3-5': return 3 <= n <= 5
    if fee_val == '>5': return n > 5
    return False

def find_first_matching_fee(tx, mcc, fraud_cat, vol_cat):
    """Return the FIRST fee (by ID order) matching all criteria, or None."""
    cs = tx['card_scheme']
    ic = tx['is_credit']
    aci = tx['aci']
    intra = (tx['issuing_country'] == tx['acquirer_country'])
    for fee in fees:  # fees.json is sorted by ascending ID
        if fee['card_scheme'] != cs: continue
        if fee['account_type'] and acct_type not in fee['account_type']: continue
        if not capture_delay_matches(cap_delay, fee['capture_delay']): continue
        if fee['monthly_fraud_level'] is not None and fee['monthly_fraud_level'] != fraud_cat: continue
        if fee['monthly_volume'] is not None and fee['monthly_volume'] != vol_cat: continue
        if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']: continue
        if fee['is_credit'] is not None and fee['is_credit'] != ic: continue
        if fee['aci'] and aci not in fee['aci']: continue
        if fee['intracountry'] is not None:
            if fee['intracountry'] == 1.0 and not intra: continue
            if fee['intracountry'] == 0.0 and intra: continue
        return fee
    return None

# Calculate total fees under both MCC scenarios
total_old = 0.0
total_new = 0.0

for _, tx in df.iterrows():
    month = int(tx['month'])
    fraud_cat, vol_cat = monthly_cats[month]
    amount = tx['eur_amount']

    fee_old = find_first_matching_fee(tx, old_mcc, fraud_cat, vol_cat)
    if fee_old:
        total_old += fee_old['fixed_amount'] + fee_old['rate'] * amount / 10000

    fee_new = find_first_matching_fee(tx, new_mcc, fraud_cat, vol_cat)
    if fee_new:
        total_new += fee_new['fixed_amount'] + fee_new['rate'] * amount / 10000

delta = total_new - total_old
print(f"{delta:.6f}")
```

---

## Common Pitfalls

1. **Empty list ≠ null for account_type/MCC/aci**: Both mean "applies to all", but the data uses `[]` not `null` for these list fields.

2. **Intracountry uses payments.csv acquirer_country directly** — do NOT look up acquirer_countries.csv per transaction (the column is already resolved in payments.csv).

3. **Capture delay numeric mapping**: merchant value "1" or "2" maps to '<3', "7" maps to '>5'. Always convert the numeric string to int before comparing.

4. **Delta sign**: `delta = (new_fees - old_fees)`. Positive = merchant pays more; negative = merchant pays less.

5. **Month filtering**: January = day_of_year 1–31, December = 335–365. Use day_of_year column directly.

6. **Monthly conditions are merchant+month scoped**: Calculate fraud level / volume from ALL of that merchant's transactions in that natural month (not just the filtered subset).

7. **Rate-change output format**: Round to 14 decimal places. Example: `f"{delta:.14f}"`.

8. **MCC-change output format**: Round to 6 decimal places. Example: `f"{delta:.6f}"`.

9. **First-match rule**: When multiple fee rules match a transaction, always pick the first one by ID order (fees.json is sorted ascending by ID). Do NOT pick the most specific one or sum multiple fees.

10. **Efficiency**: Write the complete MCC-change computation in ONE script. Avoid splitting into multiple exploration/verification steps — the calculation over 20,000–30,000 transactions is fast when written as a single loop.
