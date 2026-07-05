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

Calculate the total fees (in euros) a merchant owes for all their transactions in a given time window, by matching each transaction to the correct fee rule and summing `fee = fixed_amount + rate * eur_amount / 10000`.

## Data Files

| File | Purpose |
|------|---------|
| `payments.csv` | All transactions: `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `aci`, `issuing_country`, `acquirer_country`, `eur_amount`, `is_refused_by_adyen`, `has_fraudulent_dispute` |
| `fees.json` | 1000 fee rules with matching criteria and `fixed_amount` + `rate` |
| `merchant_data.json` | Merchant properties: `account_type`, `capture_delay`, `merchant_category_code` |
| `manual.md` | Domain definitions (read first for context) |

## Step-by-Step Solution

### Step 1: Filter Transactions

Filter `payments.csv` for the target merchant and time period. **Exclude refused transactions** (`is_refused_by_adyen == True`):

```python
import pandas as pd, json, datetime, calendar

payments = pd.read_csv('payments.csv')
merchant_txns = payments[
    (payments['merchant'] == merchant_name) &
    (payments['year'] == year) &
    (payments['day_of_year'] >= start_day) &
    (payments['day_of_year'] <= end_day) &
    (payments['is_refused_by_adyen'] == False)
].copy()
```

**Time period → day_of_year mapping**:
- Specific day N of year Y: `day_of_year == N`
- A calendar month (e.g., April 2023): compute start/end using `calendar.monthrange`

```python
# Month to day_of_year range
def month_to_doy_range(year, month):
    start = sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + 1
    end = start + calendar.monthrange(year, month)[1] - 1
    return start, end
```

### Step 2: Get Merchant Properties

```python
with open('merchant_data.json') as f:
    merchants = {m['merchant']: m for m in json.load(f)}

m = merchants[merchant_name]
account_type = m['account_type']          # e.g. 'R', 'D', 'F', 'H', 'S'
mcc = m['merchant_category_code']         # e.g. 5942
raw_capture_delay = m['capture_delay']    # e.g. '1', '7', 'manual', 'immediate'
```

**Map numeric capture_delay to fee rule category**:

```python
def map_capture_delay(delay):
    if delay in ('immediate', 'manual'):
        return delay
    d = int(delay)
    if d < 3:   return '<3'
    elif d <= 5: return '3-5'
    else:        return '>5'

capture_delay_mapped = map_capture_delay(raw_capture_delay)
```

### Step 3: Compute Monthly Statistics (Per Transaction Month)

The fee rule may filter on `monthly_fraud_level` and `monthly_volume`, computed over the **full calendar month** of each transaction.

```python
def get_month_for_doy(year, doy):
    return (datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)).month

def compute_monthly_stats(merchant_name, year, month, payments_df):
    """All transactions (including refused) for the merchant in that month."""
    monthly = payments_df[
        (payments_df['merchant'] == merchant_name) &
        (payments_df['year'] == year)
    ].copy()
    monthly['month'] = monthly['day_of_year'].apply(lambda d: get_month_for_doy(year, d))
    data = monthly[monthly['month'] == month]
    total_vol = data['eur_amount'].sum()
    fraud_vol = data[data['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    fraud_rate = fraud_vol / total_vol if total_vol > 0 else 0
    return total_vol, fraud_rate

def categorize_volume(vol):
    if vol < 100_000:   return '<100k'
    elif vol < 1_000_000: return '100k-1m'
    elif vol < 5_000_000: return '1m-5m'
    else:                 return '>5m'

def categorize_fraud(rate):
    pct = rate * 100
    if pct < 7.2:   return '<7.2%'
    elif pct < 7.7: return '7.2%-7.7%'
    elif pct < 8.3: return '7.7%-8.3%'
    else:           return '>8.3%'
```

Cache monthly stats since many transactions share the same month:

```python
monthly_cache = {}
def get_stats(merchant, year, month):
    key = (merchant, year, month)
    if key not in monthly_cache:
        monthly_cache[key] = compute_monthly_stats(merchant, year, month, payments)
    return monthly_cache[key]
```

### Step 4: Fee Rule Matching

For each transaction, find all matching rules from `fees.json`. Fields follow **"empty/null = applies to all"**:

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

**Intracountry** = `issuing_country == acquirer_country` from `payments.csv`.

```python
with open('fees.json') as f:
    fees = json.load(f)

def find_matching_rules(row, account_type, capture_delay_mapped, mcc, mv_cat, mf_cat):
    intracountry = row['issuing_country'] == row['acquirer_country']
    matches = []
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
        matches.append(rule)
    return matches
```

### Step 5: Calculate and Sum Fees

```python
total_fees = 0.0
for _, row in merchant_txns.iterrows():
    month = get_month_for_doy(int(row['year']), int(row['day_of_year']))
    vol, fraud_rate = get_stats(merchant_name, int(row['year']), month)
    mv_cat = categorize_volume(vol)
    mf_cat = categorize_fraud(fraud_rate)
    
    rules = find_matching_rules(row, account_type, capture_delay_mapped, mcc, mv_cat, mf_cat)
    
    if rules:
        rule = rules[0]  # Use first matching rule (lowest ID)
        total_fees += rule['fixed_amount'] + rule['rate'] * row['eur_amount'] / 10000

print(round(total_fees, 2))
```

**When no rule matches**: the transaction contributes 0 to the total (this is common when the merchant's MCC doesn't appear in any rule with the right combination of conditions).

**When multiple rules match**: use `rules[0]` — the first rule in fees.json order (lowest ID).

## Common Pitfalls

- **Do not exclude refused transactions from monthly stats** — use all transactions when computing monthly volume and fraud rate, but only non-refused transactions for fee calculation.
- **capture_delay "1" → '<3'**, "7" → '>5'** — always map numeric values to the correct category.
- **intracountry uses `acquirer_country` from `payments.csv`**, not from `merchant_data.json`.
- **Empty list `[]` in fees.json means "all"** — same semantic as `null` for list-type fields.
- **Fraud rate** = total fraudulent EUR amount / total EUR amount (not transaction count).
- Round final answer to **2 decimal places**.
