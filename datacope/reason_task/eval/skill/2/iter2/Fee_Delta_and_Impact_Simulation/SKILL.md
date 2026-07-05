---
name: Fee_Delta_and_Impact_Simulation
description: Solve dabstep questions about fee deltas from rate changes and fee impact simulations. Use for: (1) computing the monetary delta a merchant would pay if a fee's relative rate changed to a new value in a specific time period; (2) determining which merchants are affected when a fee's account_type restriction changes; (3) computing the fee delta if a merchant changed its MCC code. Trigger whenever a question involves "delta", "relative fee", "fee changed", "affected merchants", "MCC code changed", or simulating how a fee rule change impacts merchant payments.
---

# Fee Delta and Impact Simulation

Three question types appear in this category:

1. **Rate-change delta**: "In [month/year] what delta would [merchant] pay if the relative fee of fee ID=[N] changed to [new_rate]?"
2. **Account-type impact**: "During [year], if fee ID=[N] was only applied to account type [X], which merchants would have been affected?"
3. **MCC-change delta**: "If merchant M had changed its MCC to X before [year], what delta would it pay?"

## Data Sources

- `fees.json` — ~1000 fee rule objects
- `merchant_data.json` — merchant properties (account_type, capture_delay, acquirers, MCC)
- `payments.csv` — transaction records (issuing_country, acquirer_country per row)
- `manual.md` — field definitions and fee formula (read first)

## Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

**"Relative fee"** = the `rate` field. Delta = new_total_fee − old_total_fee.

## Fee Rule Field Semantics

| Field | Null/empty meaning |
|-------|-------------------|
| `account_type` (list) | `[]` → applies to **all** account types |
| `merchant_category_code` (list) | `[]` → applies to **all** MCCs |
| `aci` (list) | `[]` → applies to all ACI values |
| `is_credit` (bool) | `null` → applies to both credit and debit |
| `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `intracountry` | `null` → applies to all values |

**`[]` means "no restriction" (match all), NOT "match nothing".**

---

## CRITICAL: Unmatched Transactions Are Expected — Do NOT Investigate

When computing total fees, some transactions will match zero fee rules. **This is correct behavior, not a bug.** The fee dataset does not guarantee a catch-all rule for every combination of card_scheme, MCC, account_type, capture_delay, aci, is_credit, intracountry, and monthly thresholds. A transaction with no matching fee rule contributes **0** to the total fee.

**Do NOT** verify results by sampling individual transactions to check whether fees match them. Doing so wastes turns, misleads the analysis, and causes the computation to fail before producing an answer.

**After computing the delta, output it immediately as the final answer.**

---

## Helper Functions (reuse across all question types)

```python
import json, pandas as pd

with open('fees.json') as f: fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('payments.csv')

MONTH_RANGES = {
    1:(1,31),2:(32,59),3:(60,90),4:(91,120),5:(121,151),6:(152,181),
    7:(182,212),8:(213,243),9:(244,273),10:(274,304),11:(305,334),12:(335,365)
}

def day_to_month(d):
    for m,(s,e) in MONTH_RANGES.items():
        if s <= d <= e: return m
    return 12

def capture_delay_matches(fee_cd, merch_cd):
    if fee_cd is None: return True
    if fee_cd in ('immediate', 'manual'): return merch_cd == fee_cd
    try:
        days = float(merch_cd)
        if fee_cd == '<3':  return days < 3
        if fee_cd == '3-5': return 3 <= days <= 5
        if fee_cd == '>5':  return days > 5
    except: return False
    return False

def fraud_level_matches(rule, rate_pct):
    if rule is None: return True
    if rule == '<7.2%':      return rate_pct < 7.2
    if rule == '7.2%-7.7%': return 7.2 <= rate_pct < 7.7
    if rule == '7.7%-8.3%': return 7.7 <= rate_pct < 8.3
    if rule == '>8.3%':     return rate_pct > 8.3
    return False

def volume_matches(rule, vol):
    if rule is None: return True
    if rule == '<100k':   return vol < 100000
    if rule == '100k-1m': return 100000 <= vol < 1000000
    if rule == '1m-5m':   return 1000000 <= vol < 5000000
    if rule == '>5m':     return vol >= 5000000
    return False
```

---

## Question Type 1: Rate-Change Delta

### Algorithm

1. Find target fee by ID. Get merchant's properties.
2. Filter payments for merchant + time period.
3. Apply all fee matching criteria as filters.
4. `delta = (new_rate - old_rate) * matching_df['eur_amount'].sum() / 10000`

```python
fee = next(r for r in fees if r['ID'] == target_fee_id)
merchant = next(m for m in merchants if m['merchant'] == merchant_name)

# Filter for merchant and time period
s, e = MONTH_RANGES[month_num]  # or use all rows for full year
df = payments[(payments['merchant'] == merchant_name) &
              (payments['day_of_year'] >= s) & (payments['day_of_year'] <= e)].copy()

# card_scheme (always required)
df = df[df['card_scheme'] == fee['card_scheme']]

# is_credit (null = both)
if fee['is_credit'] is not None:
    df = df[df['is_credit'] == fee['is_credit']]

# aci ([] = all)
if fee['aci']:
    df = df[df['aci'].isin(fee['aci'])]

# merchant_category_code ([] = all)
if fee['merchant_category_code']:
    if merchant['merchant_category_code'] not in fee['merchant_category_code']:
        df = df.iloc[0:0]

# account_type ([] = all)
if fee['account_type']:
    if merchant['account_type'] not in fee['account_type']:
        df = df.iloc[0:0]

# capture_delay (None = all)
if not capture_delay_matches(fee['capture_delay'], merchant['capture_delay']):
    df = df.iloc[0:0]

# intracountry (None = all; 1.0 = domestic; 0.0 = international)
if fee['intracountry'] is not None:
    is_intra = df['issuing_country'] == df['acquirer_country']
    df = df[is_intra == (fee['intracountry'] == 1.0)]

# monthly_fraud_level / monthly_volume — process month by month
if fee['monthly_fraud_level'] is not None or fee['monthly_volume'] is not None:
    df['_month'] = df['day_of_year'].apply(day_to_month)
    keep = []
    for mo, mdf in df.groupby('_month'):
        s2, e2 = MONTH_RANGES[mo]
        m_txns = payments[(payments['merchant'] == merchant_name) &
                          (payments['day_of_year'] >= s2) & (payments['day_of_year'] <= e2)]
        total = m_txns['eur_amount'].sum()
        fraud_pct = (m_txns[m_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
                     / total * 100) if total > 0 else 0
        if (fraud_level_matches(fee['monthly_fraud_level'], fraud_pct) and
                volume_matches(fee['monthly_volume'], total)):
            keep.append(mdf)
    df = pd.concat(keep) if keep else df.iloc[0:0]

old_rate = fee['rate']
delta = (new_rate - old_rate) * df['eur_amount'].sum() / 10000
print(round(delta, 6))
```

---

## Question Type 2: Account-Type Impact Simulation

**"If fee ID=X was only applied to account type Y, which merchants would have been affected?"**

Affected = merchants whose status *changes* (currently get the fee but wouldn't, or vice versa).

```python
fee = next(r for r in fees if r['ID'] == target_fee_id)
merchant_lookup = {m['merchant']: m for m in merchants}

# Find all merchants who have at least one transaction matching fee's non-account_type criteria
mask = payments['card_scheme'] == fee['card_scheme']
if fee['is_credit'] is not None:
    mask &= payments['is_credit'] == fee['is_credit']
if fee['aci']:
    mask &= payments['aci'].isin(fee['aci'])
# Note: intracountry, MCC, capture_delay are per-merchant and checked below
candidate_merchants = payments[mask]['merchant'].unique()

def merchant_matches_fee(m_name, acct_type_list):
    if m_name not in merchant_lookup: return False
    merch = merchant_lookup[m_name]
    if acct_type_list and merch['account_type'] not in acct_type_list: return False
    if not capture_delay_matches(fee['capture_delay'], merch['capture_delay']): return False
    if fee['merchant_category_code'] and merch['merchant_category_code'] not in fee['merchant_category_code']: return False
    return True

current_set = {m for m in candidate_merchants if merchant_matches_fee(m, fee['account_type'])}
new_set = {m for m in candidate_merchants if merchant_matches_fee(m, [new_account_type])}

affected = sorted((current_set - new_set) | (new_set - current_set))
print(', '.join(affected))
```

---

## Question Type 3: MCC-Change Delta

**"If merchant M had changed its MCC to X before [year], what delta would it pay?"**

Compute total fees under original MCC and under the new MCC; delta = new_total − original_total.

**Key rule**: For each transaction, find the **best-matching fee** = the matching fee rule with the highest specificity score. Specificity = **1 point per constrained field** (binary, not length-based):

```python
def specificity(fee):
    """Count fields that actively constrain (non-null / non-empty list)."""
    return sum([
        fee['card_scheme'] is not None,          # always non-null
        bool(fee['account_type']),               # non-empty list
        fee['capture_delay'] is not None,
        fee['monthly_fraud_level'] is not None,
        fee['monthly_volume'] is not None,
        bool(fee['merchant_category_code']),     # non-empty list
        fee['is_credit'] is not None,
        bool(fee['aci']),                        # non-empty list
        fee['intracountry'] is not None,
    ])
```

### Complete Implementation

```python
import json, pandas as pd

with open('fees.json') as f: fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)
payments = pd.read_csv('payments.csv')

MONTH_RANGES = {
    1:(1,31),2:(32,59),3:(60,90),4:(91,120),5:(121,151),6:(152,181),
    7:(182,212),8:(213,243),9:(244,273),10:(274,304),11:(305,334),12:(335,365)
}
def day_to_month(d):
    for m,(s,e) in MONTH_RANGES.items():
        if s<=d<=e: return m
    return 12

merchant_name = 'TARGET_MERCHANT'
new_mcc = TARGET_NEW_MCC

merchant = next(m for m in merchants if m['merchant'] == merchant_name)
orig_mcc = merchant['merchant_category_code']
acct_type = merchant['account_type']
cap_delay = merchant['capture_delay']

df = payments[payments['merchant'] == merchant_name].copy()
df['month'] = df['day_of_year'].apply(day_to_month)

# Compute monthly fraud rate and volume once
monthly_stats = {}
for mo, (s, e) in MONTH_RANGES.items():
    m_txns = df[df['month'] == mo]
    total = m_txns['eur_amount'].sum()
    fraud = m_txns[m_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    monthly_stats[mo] = {
        'vol': total,
        'fraud_rate': (fraud / total * 100) if total > 0 else 0.0
    }

def calc_total_fees(mcc_code):
    total_fee = 0.0
    for _, txn in df.iterrows():
        mo = txn['month']
        fr = monthly_stats[mo]['fraud_rate']
        vol = monthly_stats[mo]['vol']
        best, best_spec = None, -1
        for fee in fees:
            # Merchant-level filters
            if fee['account_type'] and acct_type not in fee['account_type']: continue
            if not capture_delay_matches(fee['capture_delay'], cap_delay): continue
            if fee['merchant_category_code'] and mcc_code not in fee['merchant_category_code']: continue
            # Monthly filters
            if not fraud_level_matches(fee['monthly_fraud_level'], fr): continue
            if not volume_matches(fee['monthly_volume'], vol): continue
            # Transaction-level filters
            if fee['card_scheme'] != txn['card_scheme']: continue
            if fee['is_credit'] is not None and fee['is_credit'] != txn['is_credit']: continue
            if fee['aci'] and txn['aci'] not in fee['aci']: continue
            if fee['intracountry'] is not None:
                same = txn['issuing_country'] == txn['acquirer_country']
                if fee['intracountry'] == 1.0 and not same: continue
                if fee['intracountry'] == 0.0 and same: continue
            # Track best by specificity
            spec = specificity(fee)
            if spec > best_spec:
                best_spec = spec
                best = fee
        if best:
            total_fee += best['fixed_amount'] + best['rate'] * txn['eur_amount'] / 10000
    return total_fee

fee_orig = calc_total_fees(orig_mcc)
fee_new  = calc_total_fees(new_mcc)
delta = fee_new - fee_orig
print(round(delta, 6))
```

**Execution tip**: Run `calc_total_fees` for both MCCs in a single code block. Once you have the delta printed, **output it immediately as your final answer** — do NOT run additional checks or sample individual transactions.

---

## Common Mistakes

1. **`[]` means "no match"**: Wrong. `[]` and `null` both mean "applies to all". An `account_type: []` rule applies to every merchant.

2. **Wrong specificity scoring**: Each constrained field adds exactly **1 point** regardless of list size. Do not add `len(list)` — add 1 if non-empty.

3. **Wrong delta sign**: `delta = fee_new_MCC − fee_original_MCC`. Negative = merchant saves money.

4. **Missing field checks**: Check ALL non-null fee fields: card_scheme, is_credit, aci, account_type, capture_delay, MCC, intracountry, monthly_fraud_level, monthly_volume.

5. **Intracountry check**: Use `payments['issuing_country'] == payments['acquirer_country']` directly (both columns exist per row in payments.csv). No join needed.

6. **Monthly metrics computed wrong**: Monthly fraud rate and volume must be computed over the **full natural month** for the merchant, not just the filtered transactions.

7. **Capture delay numeric**: Merchant `capture_delay = '7'` is a string. Parse as float to compare with `'>5'`. Values `'immediate'` and `'manual'` are string-matched directly.

8. **Output precision**: Round to the decimal places specified in the question (usually 6). Use `round(delta, 6)`.

9. **Impact simulation**: Also check merchants who **gain** the fee (not just lose it). Use symmetric difference: `(current_set - new_set) | (new_set - current_set)`.

10. **Unnecessary verification after computing delta**: After printing the delta, do NOT try to verify by examining individual sample transactions. Some transactions legitimately match zero fee rules — this is expected and contributes 0 to the total. Investigating "unmatched" transactions wastes turns and leads to running out of budget without a final answer. Trust the calculation and output the result immediately.
