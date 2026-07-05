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

## CRITICAL: Submit Immediately

Once you compute fees per scheme/ACI and identify the min/max, **print the answer and submit immediately**. Do NOT:
- Investigate why some transactions have no matching rules
- Verify individual transaction fee calculations
- Debug rule-matching for specific transactions

**It is completely normal for 50–90% of transactions to have no matching fee rule.** ACI='G' appears in payments but never in any fee rule's `aci` list — it can only match rules where `aci: []`. Many scheme/ACI/account_type combinations simply have no rules; those transactions legitimately contribute 0. This is not a bug.

---

## CRITICAL DISTINCTION: Coverage Requirements

| Problem Type | Coverage Requirement |
|---|---|
| **Type 1** (card scheme routing) | **NO coverage requirement.** Sum fees for matched transactions only; skip unmatched (contribute 0). Pick min/max regardless of how many transactions had no matching rules. |
| **Type 2** (ACI optimization) | **MUST have 100% coverage.** Only consider ACIs that have matching rules for ALL fraudulent transactions. Exclude ACIs with any missing rules. |

**Do NOT apply the 100% coverage constraint to Type 1.** This is the most common mistake. A scheme with only 10% of transactions matched is still valid — sum those fees and include the scheme in the comparison.

**Never output `Not Applicable`** solely because some transactions lack matching rules. Only return `Not Applicable` if the merchant has zero transactions in the specified period.

---

## Required Data Files

| File | Purpose |
|------|---------|
| `manual.md` | Fee formula, field definitions (READ FIRST) |
| `payments-readme.md` | Column descriptions for payments.csv |
| `payments.csv` | Transaction data |
| `fees.json` | 1000 fee rules |
| `merchant_data.json` | account_type, capture_delay, mcc, acquirers |

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
    try:
        n = int(cd)
        return '<3' if n < 3 else ('3-5' if n <= 5 else '>5')
    except (ValueError, TypeError):
        return cd  # 'immediate' or 'manual' returned as-is

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

For full-year questions use the full range 1–365 (or filter by `year == 2023`).

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
    Empty list [] or None for account_type/aci/merchant_category_code means 'applies to all'.
    intracountry in fees.json: 1.0 = domestic, 0.0 = international, None = all.
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

**No matching rules:** skip the transaction (contributes 0 to that scheme/ACI's total).

**monthly_fraud_level / monthly_volume fields:** Ignore for rule matching — these fields do NOT restrict which rules apply.

### Step 4A: Card Scheme Routing (Type 1)

Iterate over ALL merchant transactions for the period. Use each transaction's **actual `card_scheme`** field. Accumulate fees by scheme.

```python
fees_by_scheme = {}
matched_by_scheme = {}
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
        matched_by_scheme[scheme] = matched_by_scheme.get(scheme, 0) + 1

# For minimum fees:
answer_scheme = min(fees_by_scheme, key=fees_by_scheme.get)
# For maximum fees:
answer_scheme = max(fees_by_scheme, key=fees_by_scheme.get)

answer_fee = round(fees_by_scheme[answer_scheme], 2)
print(f"{answer_scheme}:{answer_fee}")
# SUBMIT THIS IMMEDIATELY — do not continue debugging
```

**Key points for Type 1:**
- Compare total fees per scheme based on that scheme's actual transactions.
- Schemes with very few matched transactions naturally have lower totals — this is expected and valid.
- Do NOT require 100% coverage. Do NOT exclude schemes with unmatched transactions.

### Step 4B: ACI Optimization for Fraudulent Transactions (Type 2)

The question asks which **alternative** ACI gives the lowest total fees for the fraudulent transactions.

```python
fraud_txns = period_txns[period_txns['has_fraudulent_dispute'] == True].copy()

aci_options = ['A', 'B', 'C', 'D', 'E', 'F']  # Current fraudulent ACIs are usually 'G'
aci_totals = {}
aci_missing = {}

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
    aci_missing[target_aci] = missing

# For Type 2: ONLY consider ACIs with zero missing rules (100% coverage required)
valid_acis = {a: t for a, t in aci_totals.items() if aci_missing[a] == 0}
best_aci = min(valid_acis, key=valid_acis.get)
answer_fee = round(valid_acis[best_aci], 2)
print(f"{best_aci}:{answer_fee}")
# SUBMIT THIS IMMEDIATELY
```

---

## Common Pitfalls

1. **Empty list ≠ "no match"**: `[]` for `account_type`, `aci`, `merchant_category_code` means "applies to all values".
2. **Capture delay must be mapped**: Numeric strings `"1"`, `"2"` → `"<3"`; `"7"` → `">5"`; `"immediate"`, `"manual"` stay as-is.
3. **Intracountry from payments.csv**: Compute `issuing_country == acquirer_country` directly. In fees.json: `1.0` = domestic, `0.0` = international, `None` = all.
4. **Multiple matching rules**: Always take the **minimum fee** among all matching rules per transaction.
5. **Do NOT apply coverage constraint to Type 1**: Only Type 2 (ACI) excludes options with missing rules. Type 1 sums whatever fees are computable and picks the min/max.
6. **ACI 'G' in Type 2**: No explicit rules target ACI 'G'. Fraudulent transactions use 'G'; explore alternatives A–F only.
7. **Do not investigate unmatched transactions**: High unmatched counts (50–90%) are normal. Just sum matched fees and report the min/max.

---

## Answer Format

- **Card scheme question**: `{card_scheme}:{total_fee_rounded_to_2_decimals}`
  - Example: `SwiftCharge:37.21`
- **ACI question**: `{ACI}:{total_fee_rounded_to_2_decimals}`
  - Example: `C:17.82`
- Return `Not Applicable` only if the merchant has zero transactions in the specified period.
