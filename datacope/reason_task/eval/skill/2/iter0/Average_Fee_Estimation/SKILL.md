---
name: Average_Fee_Estimation
description: Solve questions about estimating the average fee a card scheme would charge for a transaction. Use this skill for questions asking about average fees per card scheme (e.g., "what would be the average fee that GlobalCard would charge for a transaction value of 50 EUR?"), comparing which card scheme is cheapest or most expensive, or filtering by credit/debit status, account type, or merchant category code description.
---

# Average Fee Estimation

## Core Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

All values in EUR. The **average fee** is the arithmetic mean of fees computed across all matching rules in `fees.json`.

## Data Files

- `fees.json`: 1000 fee rules (card_scheme, account_type, merchant_category_code, aci, is_credit, capture_delay, monthly_fraud_level, monthly_volume, intracountry, fixed_amount, rate)
- `merchant_category_codes.csv`: columns `mcc` (int) and `description` (string) — use this to look up MCC code from a description

## Null / Empty = "Applies to All"

For every field in `fees.json`:
- **Scalar fields** (is_credit, capture_delay, intracountry, etc.): `null` → rule applies to all values
- **List fields** (account_type, merchant_category_code, aci): empty list `[]` → rule applies to all values

When the question specifies a filter value, **include a rule if** the rule's field is null/empty OR the field explicitly contains that value.

## Question Type Patterns

### Type 1: Credit/debit filter only
> "For credit transactions, what is the average fee that card scheme X would charge for Y EUR?"

```python
import json
with open('fees.json') as f:
    fees = json.load(f)

matching = [r for r in fees
            if r['card_scheme'] == 'X'
            and (r['is_credit'] is None or r['is_credit'] == True)]  # True for credit

fee_values = [r['fixed_amount'] + r['rate'] * Y / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Type 2: Account type filter
> "For account type T, what is the average fee that card scheme X would charge for Y EUR?"

```python
matching = [r for r in fees
            if r['card_scheme'] == 'X'
            and (r['account_type'] == [] or 'T' in r['account_type'])]

fee_values = [r['fixed_amount'] + r['rate'] * Y / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Type 3: Account type + MCC description filter
> "For account type T and the MCC description: [description], what is the average fee that card scheme X would charge for Y EUR?"

First look up the MCC code:
```python
import pandas as pd
mcc_df = pd.read_csv('merchant_category_codes.csv')
# Match by substring or full description
mcc_code = mcc_df[mcc_df['description'].str.contains('keyword', case=False)]['mcc'].values[0]
```

Then filter:
```python
matching = [r for r in fees
            if r['card_scheme'] == 'X'
            and (r['account_type'] == [] or 'T' in r['account_type'])
            and (r['merchant_category_code'] == [] or mcc_code in r['merchant_category_code'])]

fee_values = [r['fixed_amount'] + r['rate'] * Y / 10000 for r in matching]
answer = round(sum(fee_values) / len(fee_values), 6)
```

### Type 4: Cheapest / most expensive card scheme
> "In the average scenario, which card scheme would provide the cheapest/most expensive fee for Y EUR?"

Average ALL rules per card scheme (no filtering):
```python
from collections import defaultdict
scheme_fees = defaultdict(list)
for r in fees:
    fee = r['fixed_amount'] + r['rate'] * Y / 10000
    scheme_fees[r['card_scheme']].append(fee)

scheme_avg = {s: sum(fl)/len(fl) for s, fl in scheme_fees.items()}
cheapest = min(scheme_avg, key=scheme_avg.get)
most_expensive = max(scheme_avg, key=scheme_avg.get)
```

## Implementation Notes

1. **Empty list `[]` vs null**: In `fees.json`, list fields use `[]` (not null) to mean "all". Check with `r['account_type'] == []`, not `r['account_type'] is None`.

2. **MCC lookup**: Use `str.contains()` with a distinctive keyword from the description. The `mcc` column is an integer.

3. **No other data files needed**: For average fee questions, only `fees.json` and (for MCC) `merchant_category_codes.csv` are required. Do not filter by payments.csv data.

4. **Output format**: Round to 6 decimal places. If the question asks for a card scheme name (cheapest/most expensive), return the scheme name string.

5. **"Not Applicable" rule**: Only use if the question is genuinely unanswerable (e.g., no rules match, undefined scenario). For "average scenario" questions, average ALL rules across the card scheme without additional filtering.

## Verified Examples

| Task | Query | Approach | Answer |
|------|-------|----------|--------|
| 1277 | GlobalCard, credit, 50 EUR | is_credit=True or null | 0.315937 |
| 1279 | SwiftCharge, credit, 50 EUR | is_credit=True or null | 0.338686 |
| 1281 | GlobalCard, credit, 100 EUR | is_credit=True or null | 0.560694 |
| 1538 | NexPay, acct_type=R, 1000 EUR | acct=R or [] | 5.625868 |
| 1570 | NexPay, acct_type=D, 1000 EUR | acct=D or [] | 5.504371 |
| 1574 | NexPay, acct_type=D, 5000 EUR | acct=D or [] | 27.250479 |
| 1577 | GlobalCard, acct_type=D, 1234 EUR | acct=D or [] | 6.580365 |
| 1341 | GlobalCard, acct=H, MCC=5813, 50 EUR | acct=H or [], MCC=5813 or [] | 0.369324 |
| 1347 | SwiftCharge, acct=H, MCC=5813, 100 EUR | acct=H or [], MCC=5813 or [] | 0.626512 |
| 1508 | Most expensive for 50 EUR | all rules per scheme | NexPay |
| 1510 | Most expensive for 100 EUR | all rules per scheme | NexPay |
| 1515 | Cheapest for 5000 EUR | all rules per scheme | GlobalCard |
