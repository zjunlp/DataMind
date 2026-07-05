---
name: Average_Fee_Estimation
description: Solve dabstep Average_Fee_Estimation problems. Use this skill for questions asking about average payment processing fees, which card scheme is cheapest/most expensive in an average scenario, and fee calculations based on card scheme, account type, MCC description, credit/debit type, or any combination of these filters.
---

# Average Fee Estimation — Dabstep Dataset

## Core Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

Fee rules live in `fees.json` (1000 rules). Key fields per rule:
- `card_scheme`: string — `GlobalCard`, `NexPay`, `SwiftCharge`, `TransactPlus`
- `account_type`: list — `[]` means applies to ALL account types (R, D, H, F, S, O)
- `merchant_category_code`: list — `[]` means applies to ALL MCCs
- `aci`: list — `[]` means applies to ALL ACI values
- `is_credit`: bool or `null` — `null` means applies to BOTH credit and debit
- `intracountry`: float or `null` — `1.0` = domestic, `0.0` = international, `null` = both
- `capture_delay`, `monthly_fraud_level`, `monthly_volume`: string or `null` — `null` = applies to all
- `fixed_amount`: float (EUR), `rate`: integer

**Filter rule:** For list fields, `[]` = applies to all → `not r['account_type']` is `True` for `[]`. For scalar/bool fields, `None` = applies to all → `r['is_credit'] is None`.

## Question Types

### Type A — Average Scenario Comparison
*"In the average scenario, which card scheme would provide the cheapest/most expensive fee for X EUR?"*

Use **ALL 1000 rules** grouped by scheme — **no additional filtering whatsoever**.

```python
import json
from collections import defaultdict

with open('fees.json') as f:
    fees = json.load(f)

txn = 100  # from question

scheme_fees = defaultdict(list)
for r in fees:
    scheme_fees[r['card_scheme']].append(r['fixed_amount'] + r['rate'] * txn / 10000)

avg_by_scheme = {s: sum(v)/len(v) for s, v in scheme_fees.items()}

answer = min(avg_by_scheme, key=avg_by_scheme.get)  # cheapest
# answer = max(avg_by_scheme, key=avg_by_scheme.get)  # most expensive
print(answer)
```

### Type B — Filtered Average (specific scheme + conditions)
*"For [account type / credit / debit / MCC / combination], what is the average fee that [scheme] would charge for X EUR?"*

Filter rules matching the scheme and all stated conditions, then compute the average.

**Filter logic by field type:**
| Field | Empty/null | Non-empty/non-null |
|-------|-----------|-------------------|
| `account_type` (list) | `not r['account_type']` → True, include | `account_type in r['account_type']` |
| `merchant_category_code` (list) | `not r['merchant_category_code']` → True, include | `mcc_code in r['merchant_category_code']` |
| `aci` (list) | `not r['aci']` → True, include | `aci_val in r['aci']` |
| `is_credit` (bool/null) | `r['is_credit'] is None` → include | `r['is_credit'] == True/False` |
| `capture_delay` (str/null) | `r['capture_delay'] is None` → include | `r['capture_delay'] == value` |

```python
import json, pandas as pd

with open('fees.json') as f:
    fees = json.load(f)

scheme = 'GlobalCard'
account_type = 'H'
txn = 50
# is_credit = True        # True for "credit", False for "debit"

# If MCC is given as description, look it up first:
mcc_df = pd.read_csv('merchant_category_codes.csv')
keyword = 'Drinking Places'  # distinctive keyword from the question's MCC description
mcc_code = int(mcc_df[mcc_df['description'].str.contains(keyword, case=False)]['mcc'].iloc[0])

applicable = [r for r in fees
              if r['card_scheme'] == scheme
              and (not r['account_type'] or account_type in r['account_type'])
              and (not r['merchant_category_code'] or mcc_code in r['merchant_category_code'])
              # and (r['is_credit'] is None or r['is_credit'] == is_credit)  # add if question specifies
             ]

fee_list = [r['fixed_amount'] + r['rate'] * txn / 10000 for r in applicable]
print(round(sum(fee_list) / len(fee_list), 6))
```

Add or remove filter conditions based on exactly what the question specifies. **Only filter on fields explicitly mentioned in the question.**

## Output Format

- Numerical answers: **round to 6 decimal places** — `round(avg, 6)`
- Card scheme name: exact case from fees.json — `GlobalCard`, `NexPay`, `SwiftCharge`, `TransactPlus`
- Wrap in: `<answer>VALUE</answer>`

## Validation Checklist

1. Formula: `fixed_amount + rate * txn / 10000`?
2. Empty-list rules included? `not r['account_type']` → `True` for `[]`?
3. Null scalar fields included? `r['is_credit'] is None` → include?
4. "Average scenario": ALL rules used, NO extra filters?
5. MCC code looked up from CSV, not guessed?
6. Rounded to 6 decimal places?
7. At least 1 applicable rule found? (if 0, re-check filter logic)
