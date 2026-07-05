---
name: Average_Fee_Estimation
description: Solve dabstep questions that ask for average payment processing fees. Use this skill for: (1) computing the average fee a specific card scheme charges for a given transaction value, filtered by credit/debit type, account type, or MCC description; (2) determining which card scheme provides the cheapest or most expensive fee in an average scenario. Trigger whenever a question involves average fee, fee estimation, card scheme fee, or fee comparison in the dabstep payment dataset.
---

# Average Fee Estimation

Two question types appear in this category:
1. **Specific average fee**: "What is the average fee that card scheme X would charge for a transaction value of Y EUR [for credit transactions / account type Z / MCC description D]?"
2. **Best/worst card scheme**: "In the average scenario, which card scheme is cheapest/most expensive for a transaction value of Y EUR?"

## Data Sources

Only two files are needed — do **not** read `manual.md`, `payments-readme.md`, or `payments.csv` for this category:
- `fees.json` — array of 1000 fee rule objects
- `merchant_category_codes.csv` — columns `mcc` (int) and `description` (string); needed only when question specifies an MCC by text description

## Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

The **average fee** is the arithmetic mean across all matching rules.

## Null / Empty = "Applies to All"

| Field type | "Applies to all" value |
|-----------|----------------------|
| Scalar (`is_credit`, `capture_delay`, etc.) | `null` |
| List (`account_type`, `merchant_category_code`, `aci`) | `[]` (empty list) |

When the question specifies a filter value, **include a rule if** its field is null/empty **OR** explicitly contains that value.

## Account Type Codes

| Code | Meaning |
|------|---------|
| R | Enterprise - Retail |
| D | Enterprise - Digital |
| H | Enterprise - Hospitality |
| F | Platform - Franchise |
| S | Platform - SaaS |
| O | Other |

---

## Question Type 1: Specific Average Fee

Filter `fees.json` to applicable rules, compute each rule's fee, then average.

### Filter logic for stated constraints

- **`is_credit`** (when question says "credit transactions"):
  Include rules where `is_credit is True` **or** `is_credit is None`

- **`account_type`** (when question specifies account type, e.g., "account type H"):
  Include rules where `account_type == []` **or** specified code is in `account_type`

- **`merchant_category_code`** (when question specifies an MCC description):
  1. Look up 4-digit MCC in `merchant_category_codes.csv` matching the description text
  2. Include rules where `merchant_category_code == []` **or** that MCC is in the list

- **Unspecified fields**: Do **not** filter — include rules with any value for that field.

### Pattern A: Credit filter only

```python
import json
with open('fees.json') as f:
    fees = json.load(f)

transaction_value = 50
matching = [r for r in fees
            if r['card_scheme'] == 'GlobalCard'
            and (r['is_credit'] is None or r['is_credit'] == True)]

fee_values = [r['fixed_amount'] + r['rate'] * transaction_value / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Pattern B: Account type filter

```python
matching = [r for r in fees
            if r['card_scheme'] == 'NexPay'
            and (r['account_type'] == [] or 'R' in r['account_type'])]

fee_values = [r['fixed_amount'] + r['rate'] * transaction_value / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Pattern C: Account type + MCC description filter

```python
import pandas as pd
mcc_df = pd.read_csv('merchant_category_codes.csv')
mcc_code = int(mcc_df[mcc_df['description'].str.contains('Drinking Places', case=False)]['mcc'].values[0])

matching = [r for r in fees
            if r['card_scheme'] == 'GlobalCard'
            and (r['account_type'] == [] or 'H' in r['account_type'])
            and (r['merchant_category_code'] == [] or mcc_code in r['merchant_category_code'])]

fee_values = [r['fixed_amount'] + r['rate'] * transaction_value / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Output format

Single float rounded to **6 decimal places**. Example: `0.560694`

---

## Question Type 2: Which Card Scheme is Cheapest / Most Expensive

"In the average scenario" means average across **all** fee rules for each card scheme — apply **no filtering** by is_credit, account_type, or any other field.

```python
import json
from collections import defaultdict

with open('fees.json') as f:
    fees = json.load(f)

transaction_value = 10  # EUR from question

scheme_fees = defaultdict(list)
for rule in fees:
    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000
    scheme_fees[rule['card_scheme']].append(fee)

avg_by_scheme = {scheme: sum(f) / len(f) for scheme, f in scheme_fees.items()}

cheapest = min(avg_by_scheme, key=avg_by_scheme.get)       # for "cheapest" questions
most_expensive = max(avg_by_scheme, key=avg_by_scheme.get) # for "most expensive" questions
```

### Output format

Single string — the card scheme name exactly as it appears in the data: `NexPay`, `GlobalCard`, `SwiftCharge`, or `TransactPlus`.

---

## Common Mistakes

1. **Missing null/empty rules**: Rules with `is_credit = null` apply to credit transactions; rules with `account_type = []` apply to all account types. Omitting them produces the wrong count and a wrong average.

2. **Only including explicitly-listed rules**: A rule with `account_type = []` covers account type F even though F is not in the list. Always include empty-list rules.

3. **Wrong MCC column name**: The description column in `merchant_category_codes.csv` is named `description` (not `edited_description`). Use `mcc_df['description'].str.contains(...)`.

4. **Wrong "average scenario" interpretation**: For cheapest/most expensive scheme questions, do **not** filter by specific transaction characteristics — average over ALL rules per scheme.

5. **Precision**: Numeric answers require exactly 6 decimal places.

---

## Verified Examples

| Task | Query | Approach | Answer |
|------|-------|----------|--------|
| 1277 | GlobalCard, credit, 50 EUR | is_credit=True or None → 144 rules | 0.315937 |
| 1279 | SwiftCharge, credit, 50 EUR | is_credit=True or None → 156 rules | 0.338686 |
| 1281 | GlobalCard, credit, 100 EUR | is_credit=True or None → 144 rules | 0.560694 |
| 1538 | NexPay, acct=R, 1000 EUR | acct=R or [] → 167 rules | 5.625868 |
| 1570 | NexPay, acct=D, 1000 EUR | acct=D or [] → 167 rules | 5.504371 |
| 1574 | NexPay, acct=D, 5000 EUR | acct=D or [] → 167 rules | 27.250479 |
| 1577 | GlobalCard, acct=D, 1234 EUR | acct=D or [] → 221 rules | 6.580365 |
| 1341 | GlobalCard, acct=H, MCC=5813, 50 EUR | acct=H or [], MCC=5813 or [] → 74 rules | 0.369324 |
| 1347 | SwiftCharge, acct=H, MCC=5813, 100 EUR | acct=H or [], MCC=5813 or [] → 86 rules | 0.626512 |
| 1505 | Cheapest for 10 EUR | all rules per scheme (no filter) | GlobalCard |
| 1508 | Most expensive for 50 EUR | all rules per scheme (no filter) | NexPay |
| 1509 | Cheapest for 100 EUR | all rules per scheme (no filter) | GlobalCard |
| 1515 | Cheapest for 5000 EUR | all rules per scheme (no filter) | GlobalCard |
