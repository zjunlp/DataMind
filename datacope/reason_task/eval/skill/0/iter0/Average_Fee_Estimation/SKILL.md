---
name: Average_Fee_Estimation
description: Solve Average_Fee_Estimation questions in the dabstep dataset. Use this skill when asked about the average fee a card scheme charges for a given transaction value, or which card scheme is cheapest/most expensive in an average scenario. Handles filtering by credit type, account type, and merchant category code (MCC).
---

# Average Fee Estimation Skill

## Dataset Files

- `fees.json` — fee rules (1000 rules, card schemes: GlobalCard, NexPay, SwiftCharge, TransactPlus)
- `merchant_category_codes.csv` — MCC code ↔ description lookup
- `manual.md` — domain rules and fee formula

> **Do not confuse card schemes**: `payments.csv` uses MasterCard/Visa/Amex/Other; `fees.json` uses GlobalCard/NexPay/SwiftCharge/TransactPlus. All fee questions reference `fees.json`.

## Core Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

`fixed_amount` is in EUR; `rate` is in basis points (per 10,000); `transaction_value` in EUR.

## Rule-Matching Logic

Each rule in `fees.json` may restrict by: `card_scheme`, `account_type`, `merchant_category_code`, `is_credit`, `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `aci`, `intracountry`.

**Critical matching semantics (verified from data):**

| Field value | Meaning |
|---|---|
| `null` | applies to all values |
| `[]` (empty list) | applies to all values (same as null) |
| Non-empty list | must contain the target value |

This applies to both `account_type` and `merchant_category_code`. An empty list `[]` is **not** "no match" — it means the rule imposes no restriction on that field.

Python filter pattern:
```python
def matches(rule, card_scheme=None, account_type=None, mcc=None, is_credit=None):
    if card_scheme and rule['card_scheme'] != card_scheme:
        return False
    if account_type is not None:
        acct = rule['account_type']
        if acct and account_type not in acct:  # non-empty and doesn't contain target
            return False
    if mcc is not None:
        mcc_list = rule['merchant_category_code']
        if mcc_list and mcc not in mcc_list:  # non-empty and doesn't contain target
            return False
    if is_credit is not None:
        if rule['is_credit'] is not None and rule['is_credit'] != is_credit:
            return False
    return True
```

The key: `if acct and target not in acct` — Python's truthiness treats `[]` as falsy, so empty list passes the filter (= applies to all).

## Task Types and Solution Patterns

### Type 1: Average fee for a card scheme + credit transactions

**Example**: "For credit transactions, what would be the average fee that GlobalCard would charge for 50 EUR?"

```python
import json
fees = json.load(open('fees.json'))
rules = [r for r in fees
         if r['card_scheme'] == 'GlobalCard'
         and r['is_credit'] in [True, None]]
avg = sum(r['fixed_amount'] + r['rate'] * 50 / 10000 for r in rules) / len(rules)
print(round(avg, 6))
```

### Type 2: Average fee for card scheme + account type

**Example**: "For account type R, what would be the average fee that NexPay would charge for 1000 EUR?"

```python
rules = [r for r in fees
         if r['card_scheme'] == 'NexPay'
         and (not r['account_type'] or 'R' in r['account_type'])]
avg = sum(r['fixed_amount'] + r['rate'] * 1000 / 10000 for r in rules) / len(rules)
print(round(avg, 6))
```

### Type 3: Average fee for card scheme + account type + MCC description

**Example**: "For account type H and MCC 'Drinking Places...', what is the average fee GlobalCard charges for 50 EUR?"

```python
import pandas as pd
# Step 1: Resolve MCC description → code
mcc_df = pd.read_csv('merchant_category_codes.csv')
match = mcc_df[mcc_df['description'].str.contains('Drinking Places', case=False)]
mcc_code = int(match.iloc[0]['mcc'])  # e.g., 5813

# Step 2: Filter rules
rules = [r for r in fees
         if r['card_scheme'] == 'GlobalCard'
         and (not r['account_type'] or 'H' in r['account_type'])
         and (not r['merchant_category_code'] or mcc_code in r['merchant_category_code'])]

# Step 3: Average
avg = sum(r['fixed_amount'] + r['rate'] * 50 / 10000 for r in rules) / len(rules)
print(round(avg, 6))
```

### Type 4: Which card scheme is cheapest/most expensive? ("average scenario")

**Example**: "In the average scenario, which card scheme would provide the cheapest fee for 10 EUR?"

Average ALL rules for each card scheme (no filter), compare averages:

```python
from collections import defaultdict
scheme_fees = defaultdict(list)
for r in fees:
    fee = r['fixed_amount'] + r['rate'] * 10 / 10000
    scheme_fees[r['card_scheme']].append(fee)

avgs = {s: sum(fs)/len(fs) for s, fs in scheme_fees.items()}
cheapest = min(avgs, key=avgs.get)      # for "cheapest"
most_exp = max(avgs, key=avgs.get)     # for "most expensive"
print(cheapest)
```

**Known results** (from fees.json, all transaction values):
- Cheapest average: **GlobalCard** (consistently lowest average fee at all transaction values)
- Most expensive average: **NexPay** (consistently highest average fee at all transaction values)

## Output Format

- Numeric fee answers: return as a float **rounded to 6 decimal places**
- Card scheme name answers: return the exact name string (e.g., "GlobalCard", "NexPay")
- If truly not applicable: return `'Not Applicable'`

## Common Mistakes to Avoid

1. **Empty list bug**: Never use `mcc is None or code in mcc` — this misses `[]` rules. Use Python truthiness: `not mcc or code in mcc`.
2. **Wrong card schemes**: `fees.json` has GlobalCard/NexPay/SwiftCharge/TransactPlus, NOT MasterCard/Visa.
3. **is_credit None exclusion**: When filtering for credit, include rules where `is_credit` is `None` (applies to all).
4. **MCC from payments.csv**: Always look up MCC codes from `merchant_category_codes.csv`, not from `payments.csv`.
5. **Single rule vs. average**: These questions ask for the **average** across all applicable rules, not a single matching rule.
