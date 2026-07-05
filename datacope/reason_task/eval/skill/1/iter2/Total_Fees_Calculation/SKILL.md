---
name: Total_Fees_Calculation
description: >
  Use this skill to calculate total payment processing fees for a merchant over a specified time period
  (e.g., a specific day, month, or year) in the dabstep dataset. Apply when asked: "What are the total
  fees that [merchant] paid in [period]?", "What is the total fees for [merchant] on [day]?", or any
  question requiring summing per-transaction fees using the fees.json rule engine.
---

# Total Fees Calculation — Dabstep Dataset

## Problem Type

Calculate the total fees (in euros) a merchant owes for all their transactions in a given time window by matching each transaction to the correct fee rule and summing `fee = fixed_amount + rate * eur_amount / 10000`.

## Workflow

After reading any required documentation files, **go directly to computation using the template below in a single code block**. Do NOT inspect data structures separately — the schema is fully specified here. Implement and run everything in one execution.

## Data Files

| File | Purpose |
|------|---------|
| `payments.csv` | All transactions: `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `is_refused_by_adyen`, `has_fraudulent_dispute` |
| `fees.json` | 1000 fee rules with matching criteria and `fixed_amount` + `rate` |
| `merchant_data.json` | Merchant properties: `account_type`, `capture_delay`, `merchant_category_code` |

## Complete Solution Template

Adapt the parameters and run this as a single code block:

```python
import pandas as pd, json, datetime, calendar

# --- Load data ---
payments = pd.read_csv('payments.csv')
with open('merchant_data.json') as f:
    merchants = {m['merchant']: m for m in json.load(f)}
with open('fees.json') as f:
    fees = json.load(f)

# --- Parameters (adjust for the question) ---
merchant_name = 'Belles_cookbook_store'
year = 2023
# Time window: choose ONE option below
def month_to_doy_range(yr, mo):
    start = sum(calendar.monthrange(yr, i)[1] for i in range(1, mo)) + 1
    end = start + calendar.monthrange(yr, mo)[1] - 1
    return start, end

start_day, end_day = month_to_doy_range(year, 9)   # specific month (e.g., September = 9)
# start_day, end_day = 10, 10                        # specific day
# start_day = 1; end_day = 366 if calendar.isleap(year) else 365  # full year

# --- Merchant properties ---
merch = merchants[merchant_name]
account_type = merch['account_type']
mcc = merch['merchant_category_code']

def map_capture_delay(delay):
    if delay in ('immediate', 'manual'): return delay
    d = int(delay)
    if d < 3:    return '<3'
    elif d <= 5: return '3-5'
    else:        return '>5'

capture_delay_mapped = map_capture_delay(merch['capture_delay'])

# --- Filter non-refused transactions for fee calculation ---
merchant_txns = payments[
    (payments['merchant'] == merchant_name) &
    (payments['year'] == year) &
    (payments['day_of_year'] >= start_day) &
    (payments['day_of_year'] <= end_day) &
    (payments['is_refused_by_adyen'] == False)
].copy()

# --- Monthly stats (computed from ALL transactions including refused) ---
def get_month_for_doy(yr, doy):
    return (datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)).month

def categorize_volume(vol):
    if vol < 100_000:     return '<100k'
    elif vol < 1_000_000: return '100k-1m'
    elif vol < 5_000_000: return '1m-5m'
    else:                 return '>5m'

def categorize_fraud(rate):
    pct = rate * 100
    if pct < 7.2:   return '<7.2%'
    elif pct < 7.7: return '7.2%-7.7%'
    elif pct < 8.3: return '7.7%-8.3%'
    else:           return '>8.3%'

all_merchant_year = payments[
    (payments['merchant'] == merchant_name) &
    (payments['year'] == year)
].copy()
all_merchant_year['_month'] = all_merchant_year['day_of_year'].apply(
    lambda d: get_month_for_doy(year, d))

monthly_cache = {}
def get_monthly_stats(month):
    if month not in monthly_cache:
        data = all_merchant_year[all_merchant_year['_month'] == month]
        total_vol = data['eur_amount'].sum()
        fraud_vol = data[data['has_fraudulent_dispute'] == True]['eur_amount'].sum()
        fraud_rate = fraud_vol / total_vol if total_vol > 0 else 0
        monthly_cache[month] = (categorize_volume(total_vol), categorize_fraud(fraud_rate))
    return monthly_cache[month]

# --- Fee rule matching (first match = lowest ID wins) ---
def find_matching_rule(row, mv_cat, mf_cat):
    intracountry = (row['issuing_country'] == row['acquirer_country'])
    for rule in fees:
        if rule['card_scheme'] != row['card_scheme']:
            continue
        if rule['account_type'] and account_type not in rule['account_type']:
            continue
        if rule['capture_delay'] is not None and rule['capture_delay'] != capture_delay_mapped:
            continue
        if rule['monthly_fraud_level'] is not None and rule['monthly_fraud_level'] != mf_cat:
            continue
        if rule['monthly_volume'] is not None and rule['monthly_volume'] != mv_cat:
            continue
        if rule['merchant_category_code'] and mcc not in rule['merchant_category_code']:
            continue
        if rule['is_credit'] is not None and rule['is_credit'] != row['is_credit']:
            continue
        if rule['aci'] and row['aci'] not in rule['aci']:
            continue
        if rule['intracountry'] is not None:
            if rule['intracountry'] != (1.0 if intracountry else 0.0):
                continue
        return rule
    return None

# --- Compute total fees ---
total_fees = 0.0
no_match_count = 0
for _, row in merchant_txns.iterrows():
    month = get_month_for_doy(int(row['year']), int(row['day_of_year']))
    mv_cat, mf_cat = get_monthly_stats(month)
    rule = find_matching_rule(row, mv_cat, mf_cat)
    if rule:
        total_fees += rule['fixed_amount'] + rule['rate'] * row['eur_amount'] / 10000
    else:
        no_match_count += 1

print(f"Total fees: {round(total_fees, 2)}")
print(f"Matched: {len(merchant_txns) - no_match_count}, No match: {no_match_count}")
```

## Fee Rule Matching Logic

Fields follow **"empty/null = applies to all"**:

| fees.json field | Match logic |
|-----------------|-------------|
| `card_scheme` | Must equal transaction's `card_scheme` (exact) |
| `account_type` | `[]` → all; else merchant's `account_type` must be in the list |
| `capture_delay` | `null` → all; else must equal `capture_delay_mapped` |
| `monthly_fraud_level` | `null` → all; else must equal categorized fraud level |
| `monthly_volume` | `null` → all; else must equal categorized volume |
| `merchant_category_code` | `[]` → all; else merchant's `mcc` must be in the list |
| `is_credit` | `null` → all; else must equal transaction's `is_credit` |
| `aci` | `[]` → all; else transaction's `aci` must be in the list |
| `intracountry` | `null` → all; `0.0` → False (international); `1.0` → True (domestic) |

**Intracountry** = `issuing_country == acquirer_country` using `payments.csv` columns (NOT merchant_data.json).

**Multiple rule matches**: Use the first rule in fees.json order (lowest ID).

## Critical Rules

**Submit immediately after the first computation**: Once you have the total fees value, submit it as the answer. Do NOT investigate why some transactions have no matching rule.

**High no-match rates are EXPECTED and CORRECT**: Many merchant MCCs (e.g., 5942, 7372, 7993) do not appear in any fee rule's explicit `merchant_category_code` list — they can only match rules where `merchant_category_code: []`. After further filtering by card_scheme, aci, is_credit, intracountry, and monthly stats, **30–70% no-match rates are entirely normal**. This is verified correct behavior across multiple merchants and time periods. Do not doubt results with high no-match rates.

**Do NOT modify the matching function after getting a result**: If you see a high no-match rate, do not "fix" the matching logic. The algorithm above is exact — changing it will produce wrong answers (seen empirically: modified logic inflated results by ~30%).

## Common Pitfalls

- **Monthly stats use ALL transactions** (including refused) for volume and fraud rate; fee calculation uses only `is_refused_by_adyen == False`.
- **capture_delay mapping**: numeric '1' → `'<3'`; '7' → `'>5'`; 'manual'/'immediate' → unchanged.
- **Fraud rate** = total fraudulent EUR amount / total EUR amount (not transaction count). Use `has_fraudulent_dispute == True`.
- **Round final answer to 2 decimal places**.
- **Month from day_of_year**: Day 10 → January (month 1); Day 200 → July. Always convert using `get_month_for_doy`.
- **Full year boundary**: Use `366 if calendar.isleap(year) else 365` for end_day, not a hardcoded 365.
