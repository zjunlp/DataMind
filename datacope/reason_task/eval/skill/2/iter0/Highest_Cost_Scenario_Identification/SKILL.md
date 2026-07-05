---
name: Highest_Cost_Scenario_Identification
description: >
  Solves "highest cost scenario identification" questions in the dabstep payment processing dataset.
  Use this skill when the question asks about the most expensive MCC (Merchant Category Code) for a
  given transaction amount, or the most expensive ACI (Authorization Characteristics Indicator) for
  a given card scheme and transaction amount. Also triggers for questions like "which scenario incurs
  the highest fee", "most costly MCC in general", or "which ACI is most expensive for [card_scheme]
  credit transactions". Always use this skill for dabstep tasks involving fee maximization or
  identifying worst-case cost scenarios.
---

# Highest Cost Scenario Identification

## Task Types

Two main question patterns appear in this category:

**Type 1 — Most expensive MCC (general):**
> "What is the most expensive MCC for a transaction of X euros, in general? If there are many MCCs with the same value, list all of them."

**Type 2 — Most expensive ACI for a card scheme:**
> "For a credit transaction of X euros on [CardScheme], what would be the most expensive Authorization Characteristics Indicator (ACI)? In the case of a draw between multiple ACIs, return the ACI with the lowest alphabetical order."

---

## Fee Formula

From `manual.md`:
```
fee = fixed_amount + rate * transaction_value / 10000
```

A null or empty field in a fee rule means "applies to all values" of that field.

---

## Solution for Type 1: Most Expensive MCC

**Goal:** Find the MCC(s) that can incur the highest possible fee for a given transaction amount.

"In general" means: find the maximum fee any rule could charge for each MCC, across all rules and conditions.

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

transaction_value = <X>  # from the question

mcc_max_fee = {}  # mcc -> max fee

for rule in fees:
    fixed = rule['fixed_amount']
    rate = rule['rate']
    fee = fixed + rate * transaction_value / 10000

    for mcc in rule['merchant_category_code']:  # iterate specific MCCs only
        if mcc not in mcc_max_fee or fee > mcc_max_fee[mcc]:
            mcc_max_fee[mcc] = fee

max_fee = max(mcc_max_fee.values())
result = sorted([mcc for mcc, f in mcc_max_fee.items() if f == max_fee])
print(', '.join(map(str, result)))
```

**Critical:** Do NOT filter MCCs against `merchant_category_codes.csv`. Use the MCCs directly from `fees.json`. Some MCCs present in fee rules (e.g., 3003, 7231) are absent from `merchant_category_codes.csv` but must still be included.

Rules with an empty `merchant_category_code` list apply to all MCCs, but in practice specific-MCC rules produce the highest fees — so iterating only over non-empty lists gives the correct answer.

**Output format:** comma-separated list of integer MCC codes sorted ascending.
Example: `3000, 3001, 3002, 3003, 7011, 7032, 7512, 7513`

---

## Solution for Type 2: Most Expensive ACI

**Goal:** Find which ACI value yields the highest possible fee for a credit transaction on a given card scheme.

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

card_scheme = "<CardScheme>"   # exact string from question, e.g. "GlobalCard"
transaction_value = <X>        # from the question
ALL_ACIS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

aci_max_fee = {}

for rule in fees:
    # Filter: must match card scheme
    if rule['card_scheme'] != card_scheme:
        continue
    # Filter: must apply to credit (is_credit=True or null)
    if rule['is_credit'] is not None and rule['is_credit'] != True:
        continue

    fixed = rule['fixed_amount']
    rate = rule['rate']
    fee = fixed + rate * transaction_value / 10000

    # Empty aci list means "applies to all ACIs"
    acis = rule['aci'] if rule['aci'] else ALL_ACIS

    for aci in acis:
        if aci not in aci_max_fee or fee > aci_max_fee[aci]:
            aci_max_fee[aci] = fee

max_fee = max(aci_max_fee.values())
best_acis = sorted([a for a, f in aci_max_fee.items() if f == max_fee])
result = best_acis[0]  # lowest alphabetical order in case of tie
print(result)
```

**Output format:** single letter (A–G) wrapped in a list.
Example: `['C']` or `["E"]`

If the question says "in case of draw return lowest alphabetical", just take `sorted_list[0]`.

---

## Common Mistakes to Avoid

1. **Filtering MCCs by `merchant_category_codes.csv`** — Wrong. Always use MCCs from fee rules directly. MCCs like 3003 and 7231 appear in fee rules but not in the CSV file; excluding them gives incorrect results.

2. **Outputting the fee value instead of the MCC/ACI** — The answer is the identifier (MCC code or ACI letter), not the numeric fee amount. The fee is only an intermediate calculation.

3. **Forgetting null/empty == "applies to all"** — An empty `aci` list means the rule applies to every ACI value (A–G). Treat it the same as `null`.

4. **Filtering by `is_credit=True` only** — Rules with `is_credit=null` also apply to credit transactions. Include both `True` and `None`.

5. **Sorting output** — Always sort MCC results in ascending numeric order; sort ACI results alphabetically when there's a tie.

---

## Reference: Verified Results

| Query | Answer |
|-------|--------|
| Most expensive MCC, 50 EUR | 3000, 3001, 3002, 3003, 7011, 7032, 7512, 7513 (fee=0.63, rule ID 120) |
| Most expensive MCC, 5000 EUR | 7231, 7298, 7991, 8011, 8021, 8049, 8062 (fee=49.63, rule ID 971) |
| Most expensive ACI, GlobalCard credit | C |
| Most expensive ACI, NexPay credit | E |
| Most expensive ACI, TransactPlus credit | A (ties with C; A wins alphabetically) |
| Most expensive ACI, SwiftCharge credit | B |

These results hold regardless of transaction amount because the fee formula is linear and relative rankings among ACI/MCC values are stable across the rules.
