---
name: Fee_Delta_and_Impact_Simulation
description: Solve dabstep Fee_Delta_and_Impact_Simulation questions: computing fee deltas when a fee's rate changes, and identifying which merchants are affected by fee rule changes. Use when asked about fee impact, delta payments, rate changes, or which merchants would be affected by modifying a fee rule.
---

# Fee Delta and Impact Simulation

This skill covers two question types in the dabstep dataset:
1. **Rate-change delta**: How much more/less would a merchant pay if a specific fee's rate changed?
2. **Affected-merchant identification**: Which merchants would be impacted if a fee rule's applicability changed?

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

**Delta** = fees paid under new rate minus fees paid under old rate:
```
delta = (new_rate - old_rate) * sum(matching_eur_amounts) / 10000
```

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

**Critical**: Empty list `[]` for `account_type`, `merchant_category_code`, and `aci` means "applies to all values" — same as null for those fields.

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
fraud_ratio = fraud_volume / total_volume  # as percentage

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

# 4. Check if this fee can possibly apply given monthly conditions
# If fee has monthly_fraud_level or monthly_volume constraints,
# compute these per month first and check if they match
# If they don't match for any month in the period, delta = 0

# 5. Filter transactions by fee's transaction-level criteria
mask &= payments['card_scheme'] == fee['card_scheme']

if fee['is_credit'] is not None:
    mask &= payments['is_credit'] == fee['is_credit']

if fee['aci']:  # non-empty list
    mask &= payments['aci'].isin(fee['aci'])

if fee['intracountry'] is not None:
    if fee['intracountry'] == 1.0:
        mask &= payments['issuing_country'] == payments['acquirer_country']
    else:  # 0.0
        mask &= payments['issuing_country'] != payments['acquirer_country']

# merchant-level criteria (checked once, not per transaction):
# account_type: fee['account_type'] empty OR merchant's type in list
# merchant_category_code: fee['merchant_category_code'] empty OR merchant's MCC in list
# capture_delay: fee['capture_delay'] is None OR matches merchant's capture_delay

# Check merchant-level filters once
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

### Important: Monthly Conditions

When the fee has `monthly_fraud_level` or `monthly_volume` != null, these are **merchant-level, month-level** conditions. For a single-month query, calculate for that month. For a full-year query, the fee may apply in some months but not others — you must filter transactions month-by-month.

**If fee has no monthly constraints** (both null): apply the filter once across the whole period.

**Delta can be 0** if:
- The merchant's account type, MCC, or capture_delay doesn't match the fee rule
- The monthly fraud/volume level doesn't match in the queried period
- No transactions match the other criteria (card_scheme, is_credit, aci, intracountry)

---

## Question Type 2: Affected Merchants (Account Type Change)

**Question pattern**: "During 2023, imagine if the Fee with ID [X] was only applied to account type [T], which merchants would have been affected by this change?"

### Interpretation

"Affected" = merchants whose fee X application STATUS CHANGES:
- Currently: if `account_type = []` (applies to all), ALL matching merchants use fee X
- After change: only account_type T merchants use fee X
- Affected = merchants that CURRENTLY use fee X AND are NOT account_type T (they LOSE the fee)

Note: merchants gaining the fee (if the rule was previously more restricted) would also be affected, but in practice the fee usually starts broad (empty list) and narrows.

### Step-by-step Algorithm

```python
# 1. Load fee rule and identify target account type T (from question)
fee = next(f for f in fees if f['ID'] == X)

# 2. Find merchants that currently match fee X's OTHER criteria
# (exclude account_type from matching — we'll check that separately)
with open('merchant_data.json') as f:
    merchants = json.load(f)

# 3. Filter payments for matching transactions
payments = pd.read_csv('payments.csv')
# Apply transaction-level filters (card_scheme, is_credit, aci, intracountry)
mask = payments['card_scheme'] == fee['card_scheme']
if fee['is_credit'] is not None:
    mask &= payments['is_credit'] == fee['is_credit']
if fee['aci']:
    mask &= payments['aci'].isin(fee['aci'])
if fee['intracountry'] is not None:
    if fee['intracountry'] == 1.0:
        mask &= payments['issuing_country'] == payments['acquirer_country']
    else:
        mask &= payments['issuing_country'] != payments['acquirer_country']

# 4. For each merchant, check if they have matching transactions
# AND whether their account_type != T (they'd lose the fee)
affected = []
for m in merchants:
    m_payments = payments[mask & (payments['merchant'] == m['merchant'])]
    
    # Check merchant-level criteria (MCC, capture_delay)
    if fee['merchant_category_code'] and m['merchant_category_code'] not in fee['merchant_category_code']:
        continue
    if fee['capture_delay'] is not None and not capture_delay_matches(m['capture_delay'], fee['capture_delay']):
        continue
    
    # Check if they have transactions matching the fee
    if len(m_payments) == 0:
        continue
    
    # Check if account_type would CHANGE their status
    currently_applies = not fee['account_type'] or m['account_type'] in fee['account_type']
    would_apply = m['account_type'] == T  # after the change
    
    if currently_applies != would_apply:
        affected.append(m['merchant'])

print(', '.join(sorted(affected)))
```

---

## Common Pitfalls

1. **Empty list ≠ null for account_type/MCC**: Both mean "applies to all", but the data uses `[]` not `null` for these list fields.

2. **Intracountry uses payments.csv acquirer_country directly** — do NOT look up acquirer_countries.csv per transaction (the column is already resolved in payments.csv).

3. **Capture delay numeric mapping**: merchant value "1" or "2" maps to '<3', "7" maps to '>5'. Always convert the numeric string to int before comparing.

4. **Delta sign**: `delta = (new_rate - old_rate) * amount / 10000`. Positive = merchant pays more; negative = merchant pays less.

5. **Month filtering**: January = day_of_year 1–31, December = 335–365. Use day_of_year column directly.

6. **Monthly conditions are merchant+month scoped**: Calculate fraud level / volume from ALL of that merchant's transactions in that natural month (not just the filtered subset).

7. **Output format**: Round to 14 decimal places. Example: `f"{delta:.14f}"`.

8. **MCC change questions** require full fee matching per transaction (for each transaction: find which fee rule applies under old MCC vs new MCC). This is substantially more complex and requires a complete fee-matching function.
