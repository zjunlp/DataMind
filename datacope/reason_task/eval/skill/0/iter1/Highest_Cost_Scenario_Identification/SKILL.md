---
name: Highest_Cost_Scenario_Identification
description: Solve dabstep dataset questions that ask to identify the most expensive Merchant Category Code (MCC) or the most expensive Authorization Characteristics Indicator (ACI) for a given transaction. Use this skill when the question asks "what is the most expensive MCC for a transaction of X euros" or "what is the most expensive ACI for a credit transaction of X euros on [CardScheme]".
---

# Highest Cost Scenario Identification

This skill covers two question types in the dabstep dataset, both requiring fee calculation from `fees.json`.

## Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

In `fees.json`: an empty list `[]` for `aci` or `merchant_category_code` means the rule does NOT apply to any specific value of that field — skip it when iterating. A `null` value on other fields (e.g. `is_credit`, `card_scheme`) means the rule applies to ALL values of that field.

**Data facts (verified from the actual files):**
- `fees.json` has 1000 rules; `card_scheme` is never null; `aci` is either `[]` or a list of letters (never null)
- `is_credit`: `True` = credit only, `False` = debit only, `null/None` = both
- ACI values present: A, B, C, D, E, F only (G does not appear in any fee rule)
- Card schemes: `GlobalCard`, `NexPay`, `TransactPlus`, `SwiftCharge`

---

## Question Type 1: Most Expensive MCC

**Pattern**: "What is the most expensive MCC for a transaction of X euros, in general?"

**Algorithm**:
1. Load `fees.json`
2. For every rule, skip if `merchant_category_code` is empty (`[]`)
3. Compute `fee = fixed_amount + rate * X / 10000`
4. For each MCC in the rule's list, track the maximum fee seen across all rules
5. Return all MCCs (sorted ascending) that tie for the global maximum fee

**Critical**: No card_scheme or is_credit filter for "in general" MCC questions. Include ALL MCCs from `fees.json` — do NOT restrict to MCCs listed in `merchant_category_codes.csv`.

**Output format**: Comma-separated MCC integers, sorted ascending. Example: `3000, 3001, 7512`

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

transaction_value = X  # from question
mcc_max_fees = {}

for rule in fees:
    mcc_list = rule['merchant_category_code']
    if not mcc_list:  # skip [] (no MCC applies)
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
2. Filter rules: `card_scheme == CardScheme` AND `is_credit` is `True` or `None`
3. Skip rules where `aci` list is empty (`[]`)
4. Compute `fee = fixed_amount + rate * X / 10000`
5. For each ACI in the rule's list, track the maximum fee seen
6. Return the ACI with the highest fee; if tied, return the alphabetically lowest

**Output format**: A Python list containing one letter string. Example: `['E']`

> **CRITICAL — do NOT return the fee amount as the answer.** The task instructions show `e.g., <answer>0.23</answer>` as a generic format placeholder — that `0.23` is irrelevant to ACI questions. Your answer must be the ACI letter (`['A']`, `['B']`, ... `['F']`), never a numeric fee value.

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

card_scheme = 'NexPay'      # from question
transaction_value = 1       # from question
aci_max_fees = {}

for rule in fees:
    if rule['card_scheme'] != card_scheme:
        continue
    # None means applies to all (credit and debit); include for credit transactions
    if rule['is_credit'] is not None and rule['is_credit'] != True:
        continue
    aci_list = rule['aci']
    if not aci_list:  # skip [] (no ACI applies)
        continue
    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000
    for aci in aci_list:
        if aci not in aci_max_fees or fee > aci_max_fees[aci]:
            aci_max_fees[aci] = fee

max_fee = max(aci_max_fees.values())
winners = sorted([aci for aci, f in aci_max_fees.items() if f == max_fee])
answer = winners[0]  # lowest alphabetical in case of tie
print([answer])      # output as a list, e.g. ['E']
```

---

## Common Mistakes

1. **MCC filtering**: Never exclude an MCC just because it's absent from `merchant_category_codes.csv`. The fee rules are the sole source of truth.
2. **ACI answer is a letter, not a fee**: Return the ACI letter in a list (e.g., `['E']`). The `0.23` seen in the task instructions is only a generic format example — it has no connection to the correct ACI answer.
3. **Empty lists `[]`**: Skip rules where `merchant_category_code` or `aci` is `[]`; these rules don't apply to any specific MCC or ACI.
4. **is_credit for ACI questions**: Include rules with `is_credit=None` (applies to all transaction types), not only `is_credit=True`.
5. **Tie-breaking for ACI**: When multiple ACIs share the maximum fee, return the alphabetically first one (e.g., `'A'` before `'B'`).
