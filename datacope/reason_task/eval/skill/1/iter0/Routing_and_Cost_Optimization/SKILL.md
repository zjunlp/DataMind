---
name: Routing_and_Cost_Optimization
description: Solve payment routing and cost optimization problems in the dabstep dataset. Use this skill for questions about which card scheme to steer merchant traffic to (for minimum or maximum fees), or which Authorization Characteristics Indicator (ACI) to incentivize for fraudulent transactions to minimize fees. Always invoke this skill when the question asks about steering traffic, optimal card scheme selection, or ACI optimization with fee comparison.
---

# Routing and Cost Optimization

## Problem Types

### Type 1: Card Scheme Routing
"Which card scheme should merchant X steer traffic to in order to pay min/max fees in [month/year]?"

Compute total fees per card scheme for ALL merchant transactions in the period, then pick the scheme with min/max total.

### Type 2: ACI Optimization for Fraudulent Transactions
"If fraudulent transactions were moved to a different ACI, what would be the preferred choice for lowest fees?"

Compute total fees per candidate ACI for FRAUDULENT transactions only, then pick the ACI with lowest total that covers ALL fraudulent transactions.

## Answer Format
`{card_scheme_or_ACI}:{total_fee_rounded_to_2_decimals}`
Examples: `GlobalCard:2800.62` or `C:17.82`

## Month-to-Day Mapping (2023, non-leap year)
```
Jan:1-31, Feb:32-59, Mar:60-90,  Apr:91-120,  May:121-151, Jun:152-181
Jul:182-212, Aug:213-243, Sep:244-273, Oct:274-304, Nov:305-334, Dec:335-365
Full year: 1-365
```

## Core Setup

```python
import json
import pandas as pd

payments = pd.read_csv('payments.csv')
with open('fees.json') as f:
    fees = json.load(f)
with open('merchant_data.json') as f:
    merchant_data = json.load(f)

# Get merchant characteristics
merchant_name = 'MerchantName'
merch = next(m for m in merchant_data if m['merchant'] == merchant_name)
account_type = merch['account_type']
mcc = merch['merchant_category_code']

def normalize_capture_delay(delay):
    """Map numeric or named capture_delay to fee rule category."""
    if delay in ('immediate', 'manual'):
        return delay
    n = int(delay)
    if n < 3: return '<3'
    elif n <= 5: return '3-5'
    else: return '>5'

capture_delay_norm = normalize_capture_delay(merch['capture_delay'])

# Filter transactions for the period
DAY_START, DAY_END = 32, 59  # e.g., February
merch_txns = payments[
    (payments['merchant'] == merchant_name) &
    (payments['day_of_year'] >= DAY_START) &
    (payments['day_of_year'] <= DAY_END)
].copy()
merch_txns['intracountry'] = merch_txns['issuing_country'] == merch_txns['acquirer_country']

# Monthly metrics (from ALL merchant transactions in the period)
monthly_vol = merch_txns['eur_amount'].sum()
fraud_vol = merch_txns[merch_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_pct = (fraud_vol / monthly_vol) * 100 if monthly_vol > 0 else 0

def get_vol_cat(vol):
    if vol < 100000: return '<100k'
    elif vol < 1000000: return '100k-1m'
    elif vol < 5000000: return '1m-5m'
    else: return '>5m'

def get_fraud_cat(pct):
    if pct < 7.2: return '<7.2%'
    elif pct < 7.7: return '7.2%-7.7%'
    elif pct < 8.3: return '7.7%-8.3%'
    else: return '>8.3%'

vol_cat = get_vol_cat(monthly_vol)
fraud_cat = get_fraud_cat(fraud_pct)
```

## Fee Rule Matching

**CRITICAL**: Empty list `[]` means "match all" (same as `null`). Never treat `[]` as "match nothing".

```python
def rule_matches(rule, card_scheme, aci, is_credit, intracountry,
                 acct=account_type, cap=capture_delay_norm, m=mcc,
                 vc=vol_cat, fc=fraud_cat):
    if rule['card_scheme'] != card_scheme: return False
    if rule['account_type'] and acct not in rule['account_type']: return False
    if rule['capture_delay'] is not None and rule['capture_delay'] != cap: return False
    if rule['merchant_category_code'] and m not in rule['merchant_category_code']: return False
    if rule['is_credit'] is not None and rule['is_credit'] != is_credit: return False
    if rule['aci'] and aci not in rule['aci']: return False
    if rule['intracountry'] is not None and bool(rule['intracountry']) != intracountry: return False
    if rule['monthly_fraud_level'] is not None and rule['monthly_fraud_level'] != fc: return False
    if rule['monthly_volume'] is not None and rule['monthly_volume'] != vc: return False
    return True

def get_min_fee_for_combo(card_scheme, aci, is_credit, intracountry, amount):
    """Minimum fee rule fee for a given transaction context. Returns None if no rule matches."""
    applicable = [r for r in fees if rule_matches(r, card_scheme, aci, is_credit, intracountry)]
    if not applicable: return None
    return min(r['fixed_amount'] + r['rate'] * amount / 10000 for r in applicable)
```

## Type 1: Card Scheme Routing

```python
all_schemes = ['GlobalCard', 'NexPay', 'SwiftCharge', 'TransactPlus']
n = len(merch_txns)

scheme_totals = {}
for scheme in all_schemes:
    total, matched = 0, 0
    for _, txn in merch_txns.iterrows():
        fee = get_min_fee_for_combo(scheme, txn['aci'], txn['is_credit'],
                                    txn['intracountry'], txn['eur_amount'])
        if fee is not None:
            total += fee; matched += 1
    scheme_totals[scheme] = {'total': total, 'matched': matched}
    print(f"{scheme}: {round(total,2)} ({matched}/{n} matched)")

# Prefer schemes with full coverage; for min/max pick accordingly
full = {k: v for k, v in scheme_totals.items() if v['matched'] == n}
candidates = full if full else scheme_totals
# For min fees: use min(); for max fees: use max()
best = min(candidates, key=lambda k: candidates[k]['total'])  # change to max() for 'maximum fees'
print(f"Answer: {best}:{round(scheme_totals[best]['total'], 2)}")
```

## Type 2: ACI Optimization

```python
fraud_txns = merch_txns[merch_txns['has_fraudulent_dispute'] == True].copy()
print(f"Fraudulent transactions: {len(fraud_txns)}, current ACIs: {fraud_txns['aci'].unique()}")

current_acis = set(fraud_txns['aci'].unique())
candidate_acis = [a for a in ['A','B','C','D','E','F','G'] if a not in current_acis]

n_fraud = len(fraud_txns)
aci_totals = {}
for target_aci in candidate_acis:
    total, matched = 0, 0
    for _, txn in fraud_txns.iterrows():
        fee = get_min_fee_for_combo(txn['card_scheme'], target_aci, txn['is_credit'],
                                    txn['intracountry'], txn['eur_amount'])
        if fee is not None:
            total += fee; matched += 1
    aci_totals[target_aci] = {'total': total, 'matched': matched}
    print(f"ACI {target_aci}: total={round(total,4)}, matched={matched}/{n_fraud}")

full = {k: v for k, v in aci_totals.items() if v['matched'] == n_fraud}
candidates = full if full else aci_totals
best_aci = min(candidates, key=lambda k: candidates[k]['total'])
print(f"Answer: {best_aci}:{round(aci_totals[best_aci]['total'], 2)}")
```

## Yearly Questions (2023)

For full-year questions, compute monthly metrics per natural month and aggregate:

```python
MONTHS = [(1,31),(32,59),(60,90),(91,120),(121,151),(152,181),
          (182,212),(213,243),(244,273),(274,304),(305,334),(335,365)]

all_schemes = ['GlobalCard', 'NexPay', 'SwiftCharge', 'TransactPlus']
scheme_totals_yearly = {s: 0 for s in all_schemes}

for d_start, d_end in MONTHS:
    m_txns = payments[
        (payments['merchant'] == merchant_name) &
        (payments['day_of_year'] >= d_start) &
        (payments['day_of_year'] <= d_end)
    ].copy()
    if len(m_txns) == 0:
        continue
    m_txns['intracountry'] = m_txns['issuing_country'] == m_txns['acquirer_country']
    mv = m_txns['eur_amount'].sum()
    fv = m_txns[m_txns['has_fraudulent_dispute']==True]['eur_amount'].sum()
    vc = get_vol_cat(mv)
    fc = get_fraud_cat((fv/mv)*100 if mv > 0 else 0)
    
    # Override vol_cat/fraud_cat for this month's matching
    for scheme in all_schemes:
        for _, txn in m_txns.iterrows():
            applicable = [r for r in fees if rule_matches(r, scheme, txn['aci'],
                          txn['is_credit'], txn['intracountry'], vc=vc, fc=fc)]
            if applicable:
                scheme_totals_yearly[scheme] += min(
                    r['fixed_amount'] + r['rate'] * txn['eur_amount'] / 10000
                    for r in applicable)

best = min(scheme_totals_yearly, key=lambda k: scheme_totals_yearly[k])
print(f"Answer: {best}:{round(scheme_totals_yearly[best], 2)}")
```

## Key Pitfalls

1. **`[]` = match all**: Empty list and `null` both mean "applies to all". This is the most common source of bugs — if you treat `[]` as "no match", you'll find zero matching rules for many transactions.

2. **capture_delay normalization**: merchant_data stores numeric strings like `"7"` or `"2"`. Map to fee-rule categories: `int < 3` → `"<3"`, `int 3-5` → `"3-5"`, `int > 5` → `">5"`. Named values (`"immediate"`, `"manual"`) stay unchanged.

3. **MCC absent from fee rules**: Some MCCs (like 7997) don't appear in any rule's specific MCC list. Only rules with empty `[]` MCC list apply — these are the "catch-all" rules.

4. **intracountry in payments.csv**: `acquirer_country` is already a 2-letter country code. Compute `intracountry = issuing_country == acquirer_country` directly.

5. **Multiple matching rules → minimum fee**: When multiple rules match a transaction, the merchant is charged the minimum applicable fee.

6. **Scope of monthly metrics**: Use ALL merchant transactions in the period (not just fraudulent) to compute `monthly_vol` and `fraud_pct`.
