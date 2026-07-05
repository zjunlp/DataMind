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
| `payments.csv` | `merchant`, `year`, `day_of_year`, `card_scheme`, `is_credit`, `eur_amount`, `issuing_country`, `acquirer_country`, `aci`, `has_fraudulent_dispute` |
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

**`intracountry` in fees.json is stored as float (0.0/1.0/None), not bool.** Use `bool(rule['intracountry'])` when comparing.

**Capture delay mapping** (merchant numeric days → fee rule category):
- `<3` (days < 3): merchant values "1", "2"
- `3-5` (3 ≤ days ≤ 5): merchant values "3", "4", "5"
- `>5` (days > 5): merchant value "7" or higher
- `immediate` / `manual`: exact string match

```python
def map_capture_delay(cd):
    try:
        d = int(cd)
        if d < 3: return '<3'
        elif d <= 5: return '3-5'
        else: return '>5'
    except ValueError:
        return cd  # 'immediate' or 'manual'
```

## Monthly Condition Ranges

```python
def check_monthly_fraud(fee_range, ratio_pct):
    """ratio_pct is percentage (already * 100)"""
    if fee_range is None: return True
    if fee_range == '<7.2%': return ratio_pct < 7.2
    if fee_range == '7.2%-7.7%': return 7.2 <= ratio_pct < 7.7
    if fee_range == '7.7%-8.3%': return 7.7 <= ratio_pct <= 8.3
    if fee_range == '>8.3%': return ratio_pct > 8.3
    return True

def check_monthly_vol(fee_range, vol_eur):
    if fee_range is None: return True
    if fee_range == '<100k': return vol_eur < 100_000
    if fee_range == '100k-1m': return 100_000 <= vol_eur < 1_000_000
    if fee_range == '1m-5m': return 1_000_000 <= vol_eur < 5_000_000
    if fee_range == '>5m': return vol_eur >= 5_000_000
    return True
```

**Monthly stats** are always computed over ALL transactions for the merchant in the period (not filtered by card scheme or other criteria). Compute once and reuse for all fee matching.

---

## Question Type 1: Fee Rate Delta

**Question pattern:** "In [period], what delta would [merchant] pay if the relative fee of fee ID=[X] changed to [Y]?"

**Algorithm:**
1. Load the target fee by ID. Compute `rate_change = new_rate - current_rate`.
2. Load merchant attributes: `account_type`, mapped `capture_delay`, `merchant_category_code`.
3. Filter payments by merchant and time period.
4. For each month, compute fraud_ratio and total_volume over ALL merchant transactions for that month.
5. Filter transactions where ALL fee criteria match (including monthly conditions checked per-month).
6. `delta = rate_change * qualifying['eur_amount'].sum() / 10000`
7. Return rounded to 14 decimal places. If no qualifying transactions: `0.00000000000000`.

**Implementation:**
```python
import json, pandas as pd, datetime

fees = json.load(open('fees.json'))
merchants = json.load(open('merchant_data.json'))
payments = pd.read_csv('payments.csv')

fee = next(f for f in fees if f['ID'] == target_fee_id)
rate_change = new_rate - fee['rate']

merchant = next(m for m in merchants if m['merchant'] == target_merchant)
account_type = merchant['account_type']
mcc = merchant['merchant_category_code']
mapped_delay = map_capture_delay(merchant['capture_delay'])

df = payments[(payments['merchant'] == target_merchant) & time_period_filter].copy()
df['month'] = df['day_of_year'].apply(
    lambda d: (datetime.date(2023, 1, 1) + datetime.timedelta(days=d-1)).month)

# Check merchant-level criteria first (fast exit)
if fee['account_type'] and account_type not in fee['account_type']:
    delta = 0.0
elif fee['capture_delay'] is not None and fee['capture_delay'] != mapped_delay:
    delta = 0.0
elif fee['merchant_category_code'] and mcc not in fee['merchant_category_code']:
    delta = 0.0
else:
    mask = df['card_scheme'] == fee['card_scheme']
    if fee['is_credit'] is not None:
        mask &= df['is_credit'] == fee['is_credit']
    if fee['aci']:
        mask &= df['aci'].isin(fee['aci'])
    if fee['intracountry'] is not None:
        mask &= (df['issuing_country'] == df['acquirer_country']) == bool(fee['intracountry'])
    
    if fee['monthly_fraud_level'] is None and fee['monthly_volume'] is None:
        qualifying = df[mask]
    else:
        parts = []
        for month, grp in df[mask].groupby('month'):
            all_m = df[df['month'] == month]
            vol = all_m['eur_amount'].sum()
            fraud_ratio = all_m.loc[all_m['has_fraudulent_dispute'], 'eur_amount'].sum() / vol * 100 if vol > 0 else 0
            if check_monthly_fraud(fee['monthly_fraud_level'], fraud_ratio) and check_monthly_vol(fee['monthly_volume'], vol):
                parts.append(grp)
        qualifying = pd.concat(parts) if parts else pd.DataFrame()
    
    delta = rate_change * qualifying['eur_amount'].sum() / 10000

print(f"{delta:.14f}")
```

### Time Period Mapping (2023 is not a leap year)

```python
# Year: year == 2023
# Month: compute day_of_year range
import datetime
def get_month_range(month_num):
    start = datetime.date(2023, month_num, 1).timetuple().tm_yday
    end = (datetime.date(2023, month_num + 1, 1).timetuple().tm_yday - 1) if month_num < 12 else 365
    return start, end
# Jan:1-31, Feb:32-59, Mar:60-90, Apr:91-120, May:121-151, Jun:152-181
# Jul:182-212, Aug:213-243, Sep:244-273, Oct:274-304, Nov:305-334, Dec:335-365
```

---

## Question Type 2: Account Type Impact ("Which merchants affected?")

**Question pattern:** "During [year], imagine if the Fee with ID [X] was only applied to account type [Y], which merchants would have been affected by this change?"

**"Affected"** = merchants that currently have the fee applied but would LOSE it when restricted to type Y (i.e., merchants whose account_type ≠ Y).

**Algorithm:**
1. Load fee by ID (typically has `account_type: []`).
2. Filter payments by year and fee's transaction-level criteria (card_scheme, is_credit, aci, intracountry).
3. For each unique merchant in qualifying payments, check merchant-level criteria (account_type, capture_delay, MCC).
4. Among merchants where the fee currently applies, keep those with `account_type != target_type`.
5. Return sorted alphabetically, comma-separated.

```python
mask = (payments['year'] == 2023) & (payments['card_scheme'] == fee['card_scheme'])
if fee['is_credit'] is not None: mask &= payments['is_credit'] == fee['is_credit']
if fee['aci']: mask &= payments['aci'].isin(fee['aci'])
if fee['intracountry'] is not None:
    mask &= (payments['issuing_country'] == payments['acquirer_country']) == bool(fee['intracountry'])

qualifying = payments[mask]
affected = []
for merchant_name in qualifying['merchant'].unique():
    m = merchant_map[merchant_name]
    if fee['account_type'] and m['account_type'] not in fee['account_type']: continue
    if fee['capture_delay'] is not None and map_capture_delay(m['capture_delay']) != fee['capture_delay']: continue
    if fee['merchant_category_code'] and m['merchant_category_code'] not in fee['merchant_category_code']: continue
    if m['account_type'] != target_account_type:
        affected.append(merchant_name)

print(', '.join(sorted(affected)))
```

---

## Question Type 3: MCC Change Simulation

**Question pattern:** "Imagine merchant [X] had changed its MCC to [Y] before 2023, what amount delta will it have to pay in fees for the year 2023?"

### CRITICAL: Run the full computation directly — do NOT test individual transactions first

Many individual transactions will legitimately return no matching fee under either MCC — this is normal, not a bug. In particular:
- Some MCCs have **zero specific fee rules** (only empty-MCC catch-all rules may apply).
- Even among empty-MCC rules, a specific transaction's combination of ACI, is_credit, intracountry, and capture_delay may match nothing.
- If you test 5 sample transactions and ALL return None, **this does not indicate an implementation error**. The correct response is to run the full computation over all thousands of transactions and sum the result.
- Debugging individual transactions will consume your entire turn budget without producing an answer. Trust the algorithm and run it end-to-end.

**CRITICAL: Never return 'Not Applicable'.** Always compute and return the numerical delta (even if it is 0).

**Algorithm:**
1. Load merchant attributes: `account_type`, mapped `capture_delay`, current `merchant_category_code`.
2. Filter 2023 payments for the merchant.
3. Compute monthly stats (fraud_ratio, total_volume) over ALL the merchant's transactions per month.
4. For each transaction, find the best-matching fee under both current MCC and new MCC.
5. `delta = sum(new_fees) - sum(current_fees)` — transactions with no match contribute 0.
6. Return rounded to 6 decimals.

**Fee selection (specificity scoring):** Among all matching rules, choose the one with the most specific constraints (highest specificity score). Ties broken by **lower fee ID**.

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
```

**Complete implementation (copy and run directly — do not build incrementally):**
```python
import json, pandas as pd, datetime

fees = json.load(open('fees.json'))
merchants = json.load(open('merchant_data.json'))
payments = pd.read_csv('payments.csv')

merchant = next(m for m in merchants if m['merchant'] == target_merchant)
account_type = merchant['account_type']
mapped_delay = map_capture_delay(merchant['capture_delay'])
current_mcc = merchant['merchant_category_code']
new_mcc = <new_mcc_value>

df = payments[(payments['merchant'] == target_merchant) & (payments['year'] == 2023)].copy()
df['month'] = df['day_of_year'].apply(
    lambda d: (datetime.date(2023, 1, 1) + datetime.timedelta(days=d-1)).month)

# Compute monthly stats over ALL merchant transactions
monthly_dict = {}
for month, grp in df.groupby('month'):
    vol = grp['eur_amount'].sum()
    fraud = grp.loc[grp['has_fraudulent_dispute'], 'eur_amount'].sum()
    monthly_dict[int(month)] = {'vol': vol, 'fraud_ratio': fraud / vol * 100 if vol > 0 else 0}

def rule_matches(rule, txn, mcc_to_use):
    if rule['card_scheme'] != txn['card_scheme']: return False
    if rule['is_credit'] is not None and rule['is_credit'] != txn['is_credit']: return False
    if rule['aci'] and txn['aci'] not in rule['aci']: return False
    if rule['intracountry'] is not None:
        if (txn['issuing_country'] == txn['acquirer_country']) != bool(rule['intracountry']): return False
    if rule['account_type'] and account_type not in rule['account_type']: return False
    if rule['capture_delay'] is not None and rule['capture_delay'] != mapped_delay: return False
    if rule['merchant_category_code'] and mcc_to_use not in rule['merchant_category_code']: return False
    stats = monthly_dict.get(int(txn['month']), {'vol': 0, 'fraud_ratio': 0})
    if not check_monthly_fraud(rule['monthly_fraud_level'], stats['fraud_ratio']): return False
    if not check_monthly_vol(rule['monthly_volume'], stats['vol']): return False
    return True

def find_best_fee(txn, mcc):
    best, best_score = None, (-1, float('inf'))
    for rule in fees:
        if rule_matches(rule, txn, mcc):
            s = (specificity_score(rule), -rule['ID'])
            if s > best_score:
                best_score = s
                best = rule
    return best  # None means no match → contributes 0

total_current = 0.0
total_new = 0.0
for _, row in df.iterrows():
    txn = row.to_dict()
    fee_c = find_best_fee(txn, current_mcc)
    fee_n = find_best_fee(txn, new_mcc)
    if fee_c:
        total_current += fee_c['fixed_amount'] + fee_c['rate'] * txn['eur_amount'] / 10000
    if fee_n:
        total_new += fee_n['fixed_amount'] + fee_n['rate'] * txn['eur_amount'] / 10000

delta = total_new - total_current
print(round(delta, 6))
```

---

## Common Pitfalls

1. **Empty list vs null**: `[]` for account_type/aci/MCC means "all" (same as null). Don't treat it as "none match."
2. **intracountry**: Use `payments['acquirer_country']` directly—it already contains the correct country. Do NOT look it up from acquirer_countries.csv.
3. **intracountry stored as float**: In fees.json, `intracountry` values are `0.0`, `1.0`, or `None`. Always use `bool(rule['intracountry'])` when comparing.
4. **Capture delay categories**: Merchant's numeric capture_delay (e.g., "7") must be mapped before comparing to fee rule's category (">5"). Never compare raw values.
5. **Delta sign**: delta = new_fees - old_fees (or new_total - current_total). Negative means merchant pays less; positive means more.
6. **Monthly stats scope**: Calculate fraud level and volume across ALL the merchant's transactions for that month, not just transactions matching the current fee criteria.
7. **No matching fee is normal**: For Type 3 (MCC change), many transactions may have no matching fee under either MCC. Treat them as 0-fee transactions. NEVER return 'Not Applicable'.
8. **MCC with 0 specific rules**: Some MCCs have no specific fee rules at all (e.g., a merchant's current MCC may not appear in any fee rule's `merchant_category_code` list). Only empty-MCC rules (which match all MCCs) may apply. This is correct behavior — do not debug why no MCC-specific rules match.
9. **"Affected" means LOSING the fee** (Type 2): Merchants of type Y keep the fee — they are NOT affected. Only merchants of other types are affected.
10. **Year boundaries in 2023**: day_of_year=1 is Jan 1, day_of_year=365 is Dec 31 (2023 is not a leap year).
11. **Sample transactions returning None is expected**: When testing individual sample transactions, it is normal to find that many (or even all) of the first few transactions have no matching fee. This does NOT mean the implementation is wrong. Never spend multiple turns debugging why sample transactions return None — just run the complete computation over all transactions and trust the aggregate result.
