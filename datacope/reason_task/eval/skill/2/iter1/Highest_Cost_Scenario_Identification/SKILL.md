---
name: Highest_Cost_Scenario_Identification
description: Identify the most expensive MCC (Merchant Category Code) or ACI (Authorization Characteristics Indicator) for a given transaction scenario in the dabstep payment processing dataset. Use this skill when asked which MCC or ACI results in the highest fee for a specific transaction amount, card scheme, and/or credit/debit type. Always apply this skill for questions about "most expensive", "highest cost", or "maximum fee" scenarios involving MCCs or ACIs.
---

# Highest Cost Scenario Identification

This skill covers two question patterns in the dabstep dataset, both requiring you to find the scenario that maximizes the payment processing fee.

## Fee Formula

From `manual.md`:
```
fee = fixed_amount + rate * transaction_value / 10000
```

All fee rules are in `fees.json`. Key rule fields:
- `card_scheme`: card network name (e.g. "GlobalCard", "NexPay", "TransactPlus", "SwiftCharge")
- `is_credit`: `true` (credit only), `false` (debit only), `null` (applies to both)
- `aci`: list of ACI letters this rule covers; `null` means all ACIs; **empty list `[]` means no ACI matches** (rule is ACI-agnostic/not ACI-specific — treat as applying to all ACIs when empty)
- `merchant_category_code`: list of integer MCC codes this rule covers
- `fixed_amount`, `rate`: fee components

**Critical null semantics**: `null` in any field = applies to all values of that field.

---

## Question Type 1: Most Expensive MCC

**Pattern**: "What is the most expensive MCC for a transaction of X euros, in general?"

### Algorithm

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

transaction_value = X  # euros, from the question

mcc_max_fee = {}  # mcc_code -> max fee seen

for rule in fees:
    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000
    for mcc in rule['merchant_category_code']:
        if mcc not in mcc_max_fee or fee > mcc_max_fee[mcc]:
            mcc_max_fee[mcc] = fee

overall_max = max(mcc_max_fee.values())
result_mccs = sorted([mcc for mcc, f in mcc_max_fee.items() if f == overall_max])
print(', '.join(str(m) for m in result_mccs))
```

### Critical Rules

- **Do NOT filter MCCs by `merchant_category_codes.csv`**. Use all MCCs that appear in `fees.json` rules. The fees.json is the authoritative source for which MCCs have fee rules.
- "In general" means: find the maximum possible fee over ALL rules, with no restriction on card_scheme, is_credit, aci, account_type, capture_delay, etc.
- If multiple MCCs tie at the maximum fee, list all of them sorted numerically.

### Common Error to Avoid

Wrong: Filtering MCCs to only those in `merchant_category_codes.csv` (this incorrectly excludes valid MCCs like 3003, 7231 that appear in fees.json but not in the MCC reference file).

---

## Question Type 2: Most Expensive ACI for a Specific Card Scheme

**Pattern**: "For a credit transaction of X euros on [CardScheme], what would be the most expensive ACI? In the case of a draw, return the ACI with the lowest alphabetical order."

### Algorithm

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

card_scheme = "GlobalCard"   # from the question
transaction_value = 1.0       # euros, from the question
ALL_ACIS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

aci_max_fee = {}  # aci -> max fee

for rule in fees:
    # Filter by card scheme
    if rule['card_scheme'] != card_scheme:
        continue
    # Filter by credit: keep if is_credit is True or None (null = applies to both)
    if rule['is_credit'] == False:
        continue

    fee = rule['fixed_amount'] + rule['rate'] * transaction_value / 10000

    # Determine which ACIs this rule covers
    rule_acis = rule['aci']
    if rule_acis is None:
        rule_acis = ALL_ACIS       # null = all ACIs
    elif len(rule_acis) == 0:
        rule_acis = ALL_ACIS       # empty list = applies to all ACIs (no ACI restriction)

    for aci in rule_acis:
        if aci not in aci_max_fee or fee > aci_max_fee[aci]:
            aci_max_fee[aci] = fee

# Find the ACI(s) with the highest fee
if not aci_max_fee:
    print("Not Applicable")
else:
    max_fee = max(aci_max_fee.values())
    best_acis = sorted([a for a, f in aci_max_fee.items() if f == max_fee])
    # Tiebreak: alphabetically lowest
    answer = best_acis[0]
    print(f"['{answer}']")
```

### Critical Rules

- **Filter `is_credit`**: For "credit transaction" questions, keep rules where `is_credit is True` or `is_credit is None`. Exclude rules where `is_credit == False`.
- **ACI null vs empty list**: Both `null` and `[]` in the `aci` field mean the rule applies to all ACIs. Do NOT skip rules with empty aci lists.
- **Output format**: The answer is the ACI **letter** (e.g. `C`, `E`), NOT the fee value. Return in list format: `['C']`.
- **Tiebreak**: If multiple ACIs share the maximum fee, return the one that comes first alphabetically.

### Common Errors to Avoid

1. **Outputting the fee amount** (e.g. "0.23") instead of the ACI letter — the question asks WHICH ACI, not what the fee is.
2. **Treating empty `aci` list as "no ACIs"** — empty list means all ACIs apply (same as null).
3. **Forgetting null is_credit rules** — rules with `is_credit: null` apply to credit transactions too.

---

## Output Format

- MCC questions: comma-separated list of integer MCC codes, sorted numerically. E.g.: `3000, 3001, 3002, 3003, 7011, 7032, 7512, 7513`
- ACI questions: single letter in a list. E.g.: `['C']`
- If no applicable rules exist: `Not Applicable`

## Validation Checklist

Before finalizing your answer:
1. Did you use the correct fee formula: `fixed_amount + rate * transaction_value / 10000`?
2. For MCC questions: did you search ALL fee rules without filtering by card scheme or credit type?
3. For ACI questions: did you filter by the correct card scheme AND include `is_credit=null` rules?
4. Is your output the MCC codes / ACI letters (not the fee amount)?
5. Are ties handled correctly (all MCCs listed / alphabetically first ACI)?
