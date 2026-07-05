---
name: Highest_Cost_Scenario_Identification
description: Solve dabstep dataset questions that ask to identify the most expensive Merchant Category Code (MCC) or the most expensive Authorization Characteristics Indicator (ACI) for a given transaction. Use this skill when the question asks "what is the most expensive MCC for a transaction of X euros" or "what is the most expensive ACI for a credit transaction of X euros on [CardScheme]".
---

# Highest Cost Scenario Identification

This skill covers two question types in the dabstep dataset, both requiring fee calculation from `fees.json`.

## Fee Formula

From `manual.md`:
```
fee = fixed_amount + rate * transaction_value / 10000
```

Null fields in a fee rule apply to all possible values of that field. An empty list `[]` means the rule does NOT apply to any value of that dimension.

## Data Files

- `fees.json`: 1000 fee rules with fields: `card_scheme`, `account_type`, `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `merchant_category_code`, `is_credit`, `aci`, `fixed_amount`, `rate`, `intracountry`
- `manual.md`: domain definitions (ACI values A–F, account types, fee formula)
- `merchant_category_codes.csv`: MCC descriptions only — do NOT use as a filter

---

## Question Type 1: Most Expensive MCC

**Pattern**: "What is the most expensive MCC for a transaction of X euros, in general?"

**Algorithm**:
1. Load `fees.json`
2. For every rule, compute `fee = fixed_amount + rate * X / 10000`
3. For each MCC in the rule's `merchant_category_code` list, track the maximum fee seen
4. Find the global maximum fee across all MCCs
5. Return all MCCs (sorted ascending) that achieve this maximum fee

**Critical**: Include ALL MCCs from `fees.json` fee rules. Do NOT filter to only MCCs present in `merchant_category_codes.csv` — some valid MCCs in the fee rules (e.g., 3003, 7231) are absent from that CSV but are correct answers.

**Output format**: Comma-separated list of MCC integers, sorted ascending.

**Example code**:
```python
import json

with open('fees.json') as f:
    fees = json.load(f)

transaction_value = X  # from question
mcc_max_fees = {}

for rule in fees:
    mcc_list = rule['merchant_category_code']
    if not mcc_list:  # skip empty lists
        continue
    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000
    for mcc in mcc_list:
        if mcc not in mcc_max_fees or fee > mcc_max_fees[mcc]:
            mcc_max_fees[mcc] = fee

max_fee = max(mcc_max_fees.values())
result = sorted([mcc for mcc, f in mcc_max_fees.items() if f == max_fee])
print(', '.join(str(m) for m in result))
```

---

## Question Type 2: Most Expensive ACI

**Pattern**: "For a credit transaction of X euros on [CardScheme], what would be the most expensive ACI? In the case of a draw between multiple ACIs, return the ACI with the lowest alphabetical order."

**Algorithm**:
1. Load `fees.json`
2. Filter rules: `card_scheme == CardScheme` AND (`is_credit == True` OR `is_credit == None`)
3. For each filtered rule, skip if `aci` list is empty (`[]`)
4. Compute `fee = fixed_amount + rate * X / 10000`
5. For each ACI in the rule's `aci` list, track the maximum fee seen
6. Find the global maximum fee across all ACIs
7. Return the ACI(s) achieving this maximum; if tied, return the alphabetically lowest

**Output format**: A list with one letter, e.g., `['C']`. The answer is the ACI letter, NOT the fee amount.

**Card schemes in dataset**: `GlobalCard`, `NexPay`, `TransactPlus`, `SwiftCharge`

**ACI values**: A, B, C, D, E, F (G appears in the manual but not in any fee rule in this dataset)

**Example code**:
```python
import json

with open('fees.json') as f:
    fees = json.load(f)

card_scheme = 'GlobalCard'  # from question
transaction_value = 1       # from question
aci_max_fees = {}

for rule in fees:
    if rule['card_scheme'] != card_scheme:
        continue
    # is_credit: True = credit only, False = debit only, None = both
    if rule['is_credit'] is not None and rule['is_credit'] != True:
        continue
    aci_list = rule['aci']
    if not aci_list:  # skip empty aci lists
        continue
    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000
    for aci in aci_list:
        if aci not in aci_max_fees or fee > aci_max_fees[aci]:
            aci_max_fees[aci] = fee

max_fee = max(aci_max_fees.values())
winners = sorted([aci for aci, f in aci_max_fees.items() if f == max_fee])
answer = winners[0]  # lowest alphabetical in case of tie
print([answer])
```

---

## Common Mistakes to Avoid

1. **MCC filtering**: Never exclude an MCC just because it's absent from `merchant_category_codes.csv`. The fee rules are the authoritative source.
2. **ACI answer format**: The answer must be the ACI letter (e.g., `C`), NOT the computed fee amount (e.g., `0.23`). The `0.23` example in the prompt instructions is just a format example, not related to ACI questions.
3. **Empty lists**: Both `merchant_category_code: []` and `aci: []` mean "this dimension is not constrained to any specific value" — skip these rules when iterating over MCCs/ACIs specifically.
4. **is_credit filtering for ACI questions**: Include rules with `is_credit=None` (applies to all), not just `is_credit=True`.
5. **Tie-breaking for ACI**: When multiple ACIs have the same maximum fee, return the one that comes first alphabetically.
