---
name: fee-delta-and-impact-simulation
description: >
  Solves fee delta and impact simulation problems in the dabstep payment processing dataset.
  Use this skill for questions about: (1) how much a merchant's fees change if a specific fee rule's rate is modified, (2) which merchants are affected if a fee's account type restriction changes, and (3) how a merchant's total fees change if its MCC code changes. Trigger whenever the question involves "delta", "fee with ID", "relative fee changed to", "account type", "affected by this change", or "MCC code changed".
---

# Fee Delta and Impact Simulation

## Dataset Files

| File | Key Fields |
|------|-----------|
| `payments.csv` | `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `eur_amount`, `issuing_country`, `acquirer_country`, `aci`, `has_fraudulent_dispute`, `is_refused_by_adyen` |
| `fees.json` | `ID`, `card_scheme`, `account_type`, `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `merchant_category_code`, `is_credit`, `aci`, `fixed_amount`, `rate`, `intracountry` |
| `merchant_data.json` | `merchant`, `account_type`, `capture_delay`, `acquirer` (list), `merchant_category_code` |
| `acquirer_countries.csv` | `acquirer`, `country_code` |

## Core Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

**"Relative fee"** always refers to the `rate` field.

## Fee Matching Rules

A fee rule applies to a transaction if ALL conditions match:

| Rule Field | Null/Empty? | Match Logic |
|-----------|-------------|-------------|
| `card_scheme` | never null | exact match with transaction |
| `account_type` | `[]` = all types | merchant's account_type must be in list (or list is empty) |
| `capture_delay` | `None` = all | merchant's mapped capture_delay must equal rule value |
| `merchant_category_code` | `[]` = all MCCs | merchant's MCC must be in list (or list is empty) |
| `is_credit` | `None` = all | exact match with transaction |
| `aci` | `[]` = all ACI | transaction's aci must be in list (or list is empty) |
| `intracountry` | `None` = all | `True` if issuing_country==acquirer_country else `False` |
| `monthly_fraud_level` | `None` = all | merchant's monthly fraud ratio must fall in the range |
| `monthly_volume` | `None` = all | merchant's monthly total EUR must fall in the range |

**Capture delay mapping** (merchant numeric days → fee rule category):
- `<3` (days < 3): merchant values "1", "2"
- `3-5` (3 ≤ days ≤ 5): merchant values "3", "4", "5"
- `>5` (days > 5): merchant value "7" or higher
- `immediate` / `manual`: exact string match

## Question Type 1: Fee Rate Delta

**Question pattern:** "In [period], what delta would [merchant] pay if the relative fee of fee ID=[X] changed to [Y]?"

**Algorithm:**
1. Load the target fee by ID. Note `current_rate` and compute `rate_change = new_rate - current_rate`.
2. Load merchant data: `account_type`, `capture_delay`, `merchant_category_code`.
3. Map merchant's `capture_delay` to fee rule category (see table above).
4. Load payments filtered by merchant and time period.
5. For each applicable month, compute:
   - **monthly_fraud_level** = sum(eur_amount where has_fraudulent_dispute=True) / sum(eur_amount)
   - **monthly_volume** = sum(eur_amount) for that month
6. Filter transactions where ALL fee criteria match:
   ```python
   mask = (
       (payments['card_scheme'] == fee['card_scheme']) &
       # is_credit: only filter if fee's is_credit is not None
       # aci: only filter if fee's aci list is non-empty
       # intracountry: only filter if fee's intracountry is not None
       # account_type: check merchant's type is in fee's list if non-empty
       # capture_delay: check mapped category matches fee's value if not None
       # MCC: check merchant's MCC is in fee's list if non-empty
       # monthly conditions: per-month check
   )
   ```
7. Compute: `delta = rate_change * qualifying_transactions['eur_amount'].sum() / 10000`
8. Return delta rounded to 14 decimal places. If no qualifying transactions exist (e.g., monthly conditions never met, or merchant account type excluded), delta = 0.00000000000000.

**Note on simplification for fee rate delta:** You do NOT need complex fee selection logic for this question type. Just filter directly by the specific fee's criteria — you're computing the impact of changing THIS fee's rate on transactions that THIS fee governs.

**Key implementation:**
```python
import json, pandas as pd

# Load data
fees = json.load(open('fees.json'))
merchants = json.load(open('merchant_data.json'))
payments = pd.read_csv('payments.csv')

# Find target fee
fee = next(f for f in fees if f['ID'] == target_fee_id)
old_rate = fee['rate']
rate_change = new_rate - old_rate

# Get merchant attributes
merchant = next(m for m in merchants if m['merchant'] == target_merchant)
account_type = merchant['account_type']
mcc = merchant['merchant_category_code']
raw_delay = merchant['capture_delay']

# Map capture_delay
def map_capture_delay(cd):
    try:
        d = int(cd)
        if d < 3: return '<3'
        elif d <= 5: return '3-5'
        else: return '>5'
    except ValueError:
        return cd  # 'immediate' or 'manual'
mapped_delay = map_capture_delay(raw_delay)

# Filter merchant payments in time period
df = payments[(payments['merchant'] == target_merchant) & time_period_filter]

# Build filter mask
def matches_fee(df, fee, account_type, mapped_delay, mcc):
    mask = df['card_scheme'] == fee['card_scheme']
    if fee['is_credit'] is not None:
        mask &= df['is_credit'] == fee['is_credit']
    if fee['aci']:  # non-empty list
        mask &= df['aci'].isin(fee['aci'])
    if fee['intracountry'] is not None:
        is_intra = df['issuing_country'] == df['acquirer_country']
        mask &= is_intra == bool(fee['intracountry'])
    if fee['account_type'] and account_type not in fee['account_type']:
        return mask & False  # no transactions qualify
    if fee['capture_delay'] is not None and fee['capture_delay'] != mapped_delay:
        return mask & False
    if fee['merchant_category_code'] and mcc not in fee['merchant_category_code']:
        return mask & False
    return mask

# Handle monthly conditions
if fee['monthly_fraud_level'] is None and fee['monthly_volume'] is None:
    qualifying = df[matches_fee(df, fee, account_type, mapped_delay, mcc)]
else:
    qualifying_rows = []
    for month, group in df.groupby('month'):  # derive month from day_of_year
        # Check monthly conditions
        total_vol = group['eur_amount'].sum()
        fraud_vol = group.loc[group['has_fraudulent_dispute'], 'eur_amount'].sum()
        fraud_ratio = fraud_vol / total_vol if total_vol > 0 else 0
        if not check_monthly_conditions(fee, fraud_ratio, total_vol):
            continue
        qualifying_rows.append(group[matches_fee(group, fee, account_type, mapped_delay, mcc)])
    qualifying = pd.concat(qualifying_rows) if qualifying_rows else pd.DataFrame()

delta = rate_change * qualifying['eur_amount'].sum() / 10000
```

### Time Period Mapping

- "In the year 2023": all transactions with `year == 2023`
- "In [Month] 2023": filter by `day_of_year` range (2023 is not a leap year):

```python
import datetime
def get_month_range(month_num):  # month_num 1=Jan, 12=Dec
    start = datetime.date(2023, month_num, 1).timetuple().tm_yday
    if month_num == 12:
        end = 365
    else:
        end = datetime.date(2023, month_num + 1, 1).timetuple().tm_yday - 1
    return start, end
# Jan: 1-31, Feb: 32-59, Mar: 60-90, Apr: 91-120, May: 121-151, Jun: 152-181
# Jul: 182-212, Aug: 213-243, Sep: 244-273, Oct: 274-304, Nov: 305-334, Dec: 335-365
```

### Deriving Month for Monthly Condition Checks

```python
import datetime
# Assign month to each payment row
def day_of_year_to_month(doy, year=2023):
    return datetime.date(year, 1, 1).replace() + datetime.timedelta(days=doy-1)
# Or simpler: use pd.to_datetime with day_of_year
payments['date'] = pd.to_datetime(payments['day_of_year'], format='%j', origin=str(payments['year'].iloc[0]))
payments['month'] = payments['day_of_year'].apply(
    lambda d: (datetime.date(2023, 1, 1) + datetime.timedelta(days=d-1)).month
)
```

### Monthly Condition Ranges

```python
def check_monthly_fraud_level(fee_range, ratio_pct):
    """ratio_pct is already multiplied by 100 (percentage)"""
    if fee_range == '<7.2%': return ratio_pct < 7.2
    if fee_range == '7.2%-7.7%': return 7.2 <= ratio_pct < 7.7
    if fee_range == '7.7%-8.3%': return 7.7 <= ratio_pct <= 8.3
    if fee_range == '>8.3%': return ratio_pct > 8.3
    return True

def check_monthly_volume(fee_range, vol_eur):
    if fee_range == '<100k': return vol_eur < 100_000
    if fee_range == '100k-1m': return 100_000 <= vol_eur < 1_000_000
    if fee_range == '1m-5m': return 1_000_000 <= vol_eur < 5_000_000
    if fee_range == '>5m': return vol_eur >= 5_000_000
    return True
```

---

## Question Type 2: Account Type Impact ("Which merchants affected?")

**Question pattern:** "During [year], imagine if the Fee with ID [X] was only applied to account type [Y], which merchants would have been affected by this change?"

**What "affected" means:** Merchants that CURRENTLY have the fee applied (because the fee's `account_type=[]` means all types) but would NO LONGER have it applied when restricted to type Y.

**Algorithm:**
1. Load fee by ID. Note it typically has `account_type: []` (applies to all).
2. Load merchant `account_type` for all merchants.
3. Filter all 2023 payments by the fee's transaction-level criteria (card_scheme, is_credit, aci, etc.) AND by the merchant's capture_delay/MCC constraints. **Do not filter by account_type at this stage.**
4. Find unique merchants with qualifying transactions.
5. From those merchants, keep only those with `account_type != Y` — these merchants currently have the fee but would LOSE it under the restriction.
6. Return merchants sorted alphabetically, comma-separated.

```python
# Filter payments by fee criteria (ignoring account_type)
mask = (payments['year'] == 2023) & (payments['card_scheme'] == fee['card_scheme'])
if fee['is_credit'] is not None:
    mask &= payments['is_credit'] == fee['is_credit']
if fee['aci']:
    mask &= payments['aci'].isin(fee['aci'])
if fee['intracountry'] is not None:
    is_intra = payments['issuing_country'] == payments['acquirer_country']
    mask &= is_intra == bool(fee['intracountry'])
# Also check capture_delay and MCC per merchant (important for fees with non-null capture_delay)

qualifying = payments[mask]
affected_merchants = []
for merchant_name in qualifying['merchant'].unique():
    m = merchant_map[merchant_name]
    # Check merchant-level criteria
    if fee['account_type'] and m['account_type'] not in fee['account_type']:
        continue  # fee doesn't currently apply to this merchant
    if fee['capture_delay'] is not None:
        if map_capture_delay(m['capture_delay']) != fee['capture_delay']:
            continue
    if fee['merchant_category_code'] and m['merchant_category_code'] not in fee['merchant_category_code']:
        continue
    # This merchant currently has the fee applied; check if they'd lose it
    if m['account_type'] != target_account_type:
        affected_merchants.append(merchant_name)

print(', '.join(sorted(affected_merchants)))
```

**Important:** Merchants with `account_type == Y` are NOT affected because the fee still applies to them.

---

## Question Type 3: MCC Change Simulation

**Question pattern:** "Imagine merchant [X] had changed its MCC to [Y] before 2023, what amount delta will it have to pay in fees for the year 2023?"

This requires computing the difference in total fees under two fee selection scenarios (current MCC vs. new MCC). Fee selection needs to find the best-matching rule for each transaction.

**Fee selection priority:** When multiple rules match a transaction, use the rule with the most specific constraints (most non-null/non-empty fields among: account_type, capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code, is_credit, aci, intracountry). Ties broken by lower fee ID.

**Algorithm:**
1. Load all data.
2. For each transaction, find the best matching fee under current MCC, then under new MCC.
3. Compute: `fee = fixed_amount + rate * amount / 10000` for each.
4. `delta = sum(new_fees) - sum(current_fees)`

```python
def specificity_score(rule):
    score = 0
    if rule['account_type']: score += 1
    if rule['capture_delay'] is not None: score += 1
    if rule['monthly_fraud_level'] is not None: score += 1
    if rule['monthly_volume'] is not None: score += 1
    if rule['merchant_category_code']: score += 1
    if rule['is_credit'] is not None: score += 1
    if rule['aci']: score += 1
    if rule['intracountry'] is not None: score += 1
    return score

def find_best_fee(transaction, fees, merchant_info, monthly_stats):
    candidates = []
    for rule in fees:
        if not rule_matches(rule, transaction, merchant_info, monthly_stats):
            continue
        candidates.append((specificity_score(rule), -rule['ID'], rule))
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x[0], x[1]))[2]
```

---

## Common Pitfalls

1. **Empty list vs null**: `[]` for account_type/aci/MCC means "all" (same as null). Don't treat it as "none match."
2. **intracountry**: Use `payments['acquirer_country']` directly—it already contains the correct country. Don't try to look it up from acquirer_countries.csv (the payments file has it directly).
3. **Capture delay categories**: Merchant's numeric capture_delay (e.g., "7") is NOT the same as the fee rule's value (">5"). Always map before comparing.
4. **Delta sign**: delta = (new_rate - old_rate) * amount / 10000. Negative means merchant pays less; positive means more.
5. **Monthly conditions**: Calculate fraud level and volume across ALL the merchant's transactions for that month (not just card-scheme-filtered ones).
6. **"Affected" means LOSING the fee**: When a fee is restricted to a specific account type, merchants of OTHER types are affected (they lose it). F-type merchants keep it—they are NOT affected.
7. **Year boundaries in 2023**: day_of_year=1 is Jan 1, day_of_year=365 is Dec 31 (2023 is not a leap year). January = days 1-31, February = 32-59, March = 60-90, etc.

