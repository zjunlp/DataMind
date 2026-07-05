---
name: Routing_and_Cost_Optimization
description: >
  Solves payment routing and cost optimization problems in the dabstep dataset.
  Use when the question asks which card scheme a merchant should steer traffic to
  (for minimum or maximum fees), or which ACI (Authorization Characteristics Indicator)
  to incentivize for fraudulent transactions to achieve the lowest possible fees.
  Triggers on: "steer traffic", "card scheme", "minimum fees", "maximum fees", "ACI",
  "incentivize", "fraudulent transactions", "lowest possible fees", payment routing.
---

# Routing and Cost Optimization Skill

## Problem Types

**Type 1 – Card Scheme Routing**: "To which card scheme should merchant X steer traffic to pay minimum/maximum fees in [month/year]?"

**Type 2 – ACI Optimization**: "For merchant X in [month/year], if fraudulent transactions were moved to a different ACI to achieve the lowest fees, what would be the preferred ACI?"

**Output format** (both types): `{card_scheme_or_ACI}:{fee_rounded_to_2_decimals}`
- Type 1 example: `SwiftCharge:37.21`
- Type 2 example: `C:17.82`

---

## Required Data Files

| File | Purpose |
|------|---------|
| `manual.md` | Fee formula, field definitions (READ FIRST) |
| `payments-readme.md` | Column descriptions for payments.csv |
| `payments.csv` | Transaction data |
| `fees.json` | 1000 fee rules |
| `merchant_data.json` | account_type, capture_delay, mcc, acquirers |

**Always read `manual.md` and `payments-readme.md` first.**

---

## Step-by-Step Algorithm

### Step 1: Load Merchant Characteristics

```python
import json, pandas as pd

with open('fees.json') as f: fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)

merchant = next(m for m in merchants if m['merchant'] == MERCHANT_NAME)
account_type = merchant['account_type']
mcc = merchant['merchant_category_code']
```

Map `capture_delay` to fee rule categories:
```python
def map_capture_delay(cd):
    if cd in ('immediate', 'manual'): return cd
    days = int(cd)
    if days < 3: return '<3'
    elif days <= 5: return '3-5'
    else: return '>5'

capture_delay = map_capture_delay(merchant['capture_delay'])
```

### Step 2: Filter Transactions

```python
payments = pd.read_csv('payments.csv')
```

**Day-of-year ranges for 2023 (non-leap year):**
| Month | Start | End |
|-------|-------|-----|
| January | 1 | 31 |
| February | 32 | 59 |
| March | 60 | 90 |
| April | 91 | 120 |
| May | 121 | 151 |
| June | 152 | 181 |
| July | 182 | 212 |
| August | 213 | 243 |
| September | 244 | 273 |
| October | 274 | 304 |
| November | 305 | 334 |
| December | 335 | 365 |

For full-year questions use the full range 1–365.

```python
period_txns = payments[
    (payments['merchant'] == MERCHANT_NAME) &
    (payments['day_of_year'] >= DOY_START) &
    (payments['day_of_year'] <= DOY_END)
].copy()
period_txns['intracountry'] = period_txns['issuing_country'] == period_txns['acquirer_country']
```

### Step 3: Fee Rule Matching Function

```python
def rule_matches(rule, account_type, capture_delay, mcc, is_credit, intracountry, aci):
    """
    Match a fee rule to a transaction's characteristics.
    Empty list [] for account_type/aci/merchant_category_code means 'applies to all'.
    """
    if rule['account_type'] and account_type not in rule['account_type']:
        return False
    if rule['capture_delay'] is not None and rule['capture_delay'] != capture_delay:
        return False
    if rule['merchant_category_code'] and mcc not in rule['merchant_category_code']:
        return False
    if rule['is_credit'] is not None and rule['is_credit'] != is_credit:
        return False
    if rule['intracountry'] is not None and bool(rule['intracountry']) != intracountry:
        return False
    if rule['aci'] and aci not in rule['aci']:
        return False
    return True
```

**Fee formula:** `fee = fixed_amount + rate * eur_amount / 10000`

**When multiple rules match:** use the **minimum fee** among all matching rules.

**Critical note on monthly metrics:** Do NOT filter fee rules by `monthly_fraud_level` or `monthly_volume`. These fields add additional context but should not restrict which rules are considered applicable — always pick the minimum available fee across all matching rules regardless of monthly tier.

**ACI 'G':** No explicit rules in fees.json target ACI 'G'. When computing alternatives, ACI G transactions have limited rule coverage.

### Step 4A: Card Scheme Routing (Type 1)

```python
fees_by_scheme = {}
for _, txn in period_txns.iterrows():
    scheme = txn['card_scheme']
    is_credit = bool(txn['is_credit'])
    aci = txn['aci']
    intracountry = bool(txn['intracountry'])
    amount = txn['eur_amount']

    matches = [
        rule['fixed_amount'] + rule['rate'] * amount / 10000
        for rule in fees
        if rule['card_scheme'] == scheme and
           rule_matches(rule, account_type, capture_delay, mcc, is_credit, intracountry, aci)
    ]
    if matches:
        fees_by_scheme[scheme] = fees_by_scheme.get(scheme, 0) + min(matches)

# For minimum fees:
answer_scheme = min(fees_by_scheme, key=fees_by_scheme.get)
# For maximum fees:
answer_scheme = max(fees_by_scheme, key=fees_by_scheme.get)

answer_fee = round(fees_by_scheme[answer_scheme], 2)
print(f"{answer_scheme}:{answer_fee}")
```

### Step 4B: ACI Optimization for Fraudulent Transactions (Type 2)

The question asks which **alternative** ACI gives the lowest total fees for the fraudulent transactions.

```python
fraud_txns = period_txns[period_txns['has_fraudulent_dispute'] == True].copy()

aci_options = ['A', 'B', 'C', 'D', 'E', 'F']  # Exclude current ACI (usually 'G')
aci_totals = {}
aci_no_rule = {}

for target_aci in aci_options:
    total = 0.0
    missing = 0
    for _, txn in fraud_txns.iterrows():
        scheme = txn['card_scheme']
        is_credit = bool(txn['is_credit'])
        intracountry = bool(txn['intracountry'])
        amount = txn['eur_amount']

        matches = [
            rule['fixed_amount'] + rule['rate'] * amount / 10000
            for rule in fees
            if rule['card_scheme'] == scheme and
               rule_matches(rule, account_type, capture_delay, mcc, is_credit, intracountry, target_aci)
        ]
        if matches:
            total += min(matches)
        else:
            missing += 1

    aci_totals[target_aci] = total
    aci_no_rule[target_aci] = missing

# Only consider ACIs that cover ALL fraudulent transactions (no missing rules)
valid_acis = {a: t for a, t in aci_totals.items() if aci_no_rule[a] == 0}
best_aci = min(valid_acis, key=valid_acis.get)
answer_fee = round(valid_acis[best_aci], 2)
print(f"{best_aci}:{answer_fee}")
```

---

## Common Pitfalls

1. **Empty list ≠ "no match"**: `[]` for `account_type`, `aci`, `merchant_category_code` means "applies to all".
2. **Capture delay must be mapped**: Numeric strings like `"7"` → `">5"`, `"2"` → `"<3"`, `"1"` → `"<3"`.
3. **Intracountry from payments.csv**: Compute `issuing_country == acquirer_country` from the `acquirer_country` column in payments.csv directly (no need to use acquirer_countries.csv).
4. **Multiple matching rules**: Always take the **minimum fee** among all matching rules for each transaction.
5. **ACI coverage check**: For Type 2, only report ACIs that have matching fee rules for every fraudulent transaction. Exclude ACIs with missing coverage.
6. **ACI 'G'** is the current ACI for most fraudulent transactions and typically has limited/no explicit fee rules. Explore ACIs A–F as alternatives.

---

## Answer Format

- **Card scheme question**: `{card_scheme}:{total_fee_rounded_to_2_decimals}`
  - Example: `SwiftCharge:37.21`
- **ACI question**: `{ACI}:{total_fee_rounded_to_2_decimals}`
  - Example: `C:17.82`
- If no applicable data: `Not Applicable`
