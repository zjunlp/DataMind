---
name: Average_Fee_Estimation
description: Solve dabstep Average_Fee_Estimation problems. Use this skill for questions asking about average payment processing fees, which card scheme is cheapest/most expensive in an average scenario, and fee calculations based on card scheme, account type, MCC, or transaction type filters.
---

# Average Fee Estimation — Dabstep Dataset

## Core Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

Fee rules live in `fees.json` (1000 rules). Each rule has: `card_scheme`, `account_type` (list), `merchant_category_code` (list), `is_credit` (bool/null), `aci` (list), `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `intracountry`, `fixed_amount`, `rate`.

**Empty list `[]` = null = applies to all values of that field.** A rule with `account_type: []` applies to every account type. Same for `merchant_category_code: []`, `aci: []`.

## Question Types

### Type 1 — Credit/Debit Average
*"For credit transactions, what would be the average fee that [scheme] would charge for a transaction value of X EUR?"*

```python
import json
with open('fees.json') as f:
    fees = json.load(f)

scheme = 'GlobalCard'  # from question
txn = 50               # from question
is_credit = True       # from question ("credit" → True, "debit" → False)

applicable = [r for r in fees
              if r['card_scheme'] == scheme
              and (r['is_credit'] is None or r['is_credit'] == is_credit)]

fee_list = [r['fixed_amount'] + r['rate'] * txn / 10000 for r in applicable]
avg = sum(fee_list) / len(fee_list)
print(round(avg, 6))
```

### Type 2 — Account Type Only Average
*"For account type R, what would be the average fee that [scheme] would charge for a transaction value of X EUR?"*

```python
scheme = 'NexPay'
account_type = 'R'
txn = 1000

applicable = [r for r in fees
              if r['card_scheme'] == scheme
              and (not r['account_type'] or account_type in r['account_type'])]

fee_list = [r['fixed_amount'] + r['rate'] * txn / 10000 for r in applicable]
avg = sum(fee_list) / len(fee_list)
print(round(avg, 6))
```

### Type 3 — Account Type + MCC Average
*"For account type H and the MCC description: Drinking Places..., what would be the average fee that [scheme] would charge for X EUR?"*

First, look up the MCC numeric code from `merchant_category_codes.csv`:

```python
import pandas as pd
mcc_df = pd.read_csv('merchant_category_codes.csv')
# Search by partial description match
row = mcc_df[mcc_df['description'].str.contains('Drinking Places', case=False)]
mcc_code = int(row['mcc'].iloc[0])  # e.g., 5813
```

Then filter:

```python
scheme = 'GlobalCard'
account_type = 'H'
txn = 50

applicable = [r for r in fees
              if r['card_scheme'] == scheme
              and (not r['account_type'] or account_type in r['account_type'])
              and (not r['merchant_category_code'] or mcc_code in r['merchant_category_code'])]

fee_list = [r['fixed_amount'] + r['rate'] * txn / 10000 for r in applicable]
avg = sum(fee_list) / len(fee_list)
print(round(avg, 6))
```

### Type 4 — Average Scenario Comparison
*"In the average scenario, which card scheme would provide the cheapest/most expensive fee for a transaction value of X EUR?"*

Average ALL rules for each card scheme (no additional filtering):

```python
from collections import defaultdict

txn = 50  # from question
scheme_fees = defaultdict(list)
for r in fees:
    scheme_fees[r['card_scheme']].append(r['fixed_amount'] + r['rate'] * txn / 10000)

avg_by_scheme = {s: sum(v)/len(v) for s, v in scheme_fees.items()}

# cheapest:
answer = min(avg_by_scheme, key=avg_by_scheme.get)
# most expensive:
answer = max(avg_by_scheme, key=avg_by_scheme.get)
print(answer)
```

## Output Format

- Numerical answers: **round to 6 decimal places** (e.g., `0.315937`)
- Card scheme name answers: exact name as in fees.json (e.g., `GlobalCard`, `NexPay`, `SwiftCharge`, `TransactPlus`)
- Format: `<answer>VALUE</answer>`

## Key Rules Summary

| Field | null or `[]` means | Non-empty means |
|-------|-------------------|-----------------|
| `account_type` | applies to all account types | must match one in list |
| `merchant_category_code` | applies to all MCCs | must match one in list |
| `aci` | applies to all ACI values | must match one in list |
| `is_credit` | applies to credit and debit | must match exactly |
| `capture_delay` | applies to all | must match exactly |
| `monthly_fraud_level` | applies to all | must match exactly |
| `monthly_volume` | applies to all | must match exactly |
| `intracountry` | applies to all | must match exactly |

## MCC Lookup

When the question gives an MCC *description* instead of a numeric code, look it up:
```python
import pandas as pd
mcc_df = pd.read_csv('merchant_category_codes.csv')
# Use partial match for robustness:
keyword = 'Drinking Places'  # distinctive keyword from description
mcc_code = int(mcc_df[mcc_df['description'].str.contains(keyword, case=False)]['mcc'].iloc[0])
```

## Validation Checklist

1. Used the correct fee formula? `fixed_amount + rate * txn / 10000`
2. Included rules where the list field is empty `[]` (= applies to all)?
3. For "credit transactions": filtered `is_credit == True or is_credit is None`?
4. For "average scenario": used ALL rules per scheme, no additional filters?
5. Rounded to 6 decimals?
6. MCC code looked up from CSV (not guessed)?
