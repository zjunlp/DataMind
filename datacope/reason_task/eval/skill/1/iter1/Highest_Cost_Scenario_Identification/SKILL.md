---
name: Highest_Cost_Scenario_Identification
description: Identifies the highest-cost scenario in payment processing fee analysis for the dabstep dataset. Use this skill when asked to find the most expensive MCC (Merchant Category Code) for a given transaction amount, or the most expensive ACI (Authorization Characteristics Indicator) for a given card scheme and transaction amount. Triggers on questions like "what is the most expensive MCC", "which ACI leads to highest fees", "find the costliest transaction scenario", or any query asking to identify maximum-fee scenarios in payment processing.
---

# Highest Cost Scenario Identification

## Dataset Overview

The dabstep dataset models payment processing fees with these key files:
- `fees.json`: 1000 fee rules, each specifying conditions and fee parameters
- `merchant_category_codes.csv`: MCC descriptions (not exhaustive — fee rules may reference MCCs not in this file)
- `manual.md`: Domain definitions including ACI codes (A–G) and fee formula

**Fee formula**: `fee = fixed_amount + rate * transaction_amount / 10000`

## Two Question Types

### Type 1: Most Expensive MCC
*"What is the most expensive MCC for a transaction of X euros, in general?"*

Find which MCC(s) can incur the highest fee across all fee rules.

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

amount = 50  # transaction amount in euros

mcc_max_fee = {}
for rule in fees:
    mcc_list = rule['merchant_category_code']
    if not mcc_list:  # empty list means rule applies to all MCCs, skip for MCC ranking
        continue
    fee = rule['fixed_amount'] + rule['rate'] * amount / 10000
    for mcc in mcc_list:
        mcc_max_fee[mcc] = max(mcc_max_fee.get(mcc, 0), fee)

max_fee = max(mcc_max_fee.values())
result_mccs = sorted([mcc for mcc, f in mcc_max_fee.items() if f == max_fee])
# Output as comma-separated sorted list
print(', '.join(str(m) for m in result_mccs))
```

**Critical**: Do NOT filter MCCs against `merchant_category_codes.csv`. Some MCCs in fee rules (e.g., 3003, 7231) are absent from the description file but are valid and must be included in results.

### Type 2: Most Expensive ACI
*"For a credit transaction of X euros on [CardScheme], what would be the most expensive ACI?"*

Find which ACI letter leads to the highest possible fee for the specified card scheme and credit type.

```python
import json

with open('fees.json') as f:
    fees = json.load(f)

card_scheme = 'GlobalCard'   # exact match required
is_credit = True
amount = 1  # transaction amount in euros
ALL_ACIS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

aci_max_fee = {}
for rule in fees:
    # Filter by card scheme
    if rule['card_scheme'] != card_scheme:
        continue
    # Filter by credit status: null means applies to both credit and debit
    if rule['is_credit'] is not None and rule['is_credit'] != is_credit:
        continue
    
    fee = rule['fixed_amount'] + rule['rate'] * amount / 10000
    # Empty aci list means rule applies to ALL ACI values
    acis = rule['aci'] if rule['aci'] else ALL_ACIS
    for aci in acis:
        aci_max_fee[aci] = max(aci_max_fee.get(aci, 0), fee)

max_fee = max(aci_max_fee.values())
best_acis = sorted([a for a, f in aci_max_fee.items() if f == max_fee])
result = best_acis[0]  # alphabetically first when tied
print(result)  # Output: single letter like 'C'
```

## Key Rules and Pitfalls

### Null/Empty field semantics (from manual.md)
| Field | Null/Empty means |
|-------|-----------------|
| `is_credit: null` | Applies to both credit and debit |
| `aci: []` | Applies to all ACI values (A–G) |
| `merchant_category_code: []` | Applies to all MCCs (skip for MCC ranking) |
| `account_type: []` | Applies to all account types |
| Other null fields | Apply to all values of that field |

### Common errors to avoid
1. **Filtering MCCs by merchant_category_codes.csv** — wrong. MCC 3003 and 7231 exist in fee rules but not in the CSV. Always use the MCCs from `fees.json` directly.
2. **Outputting fee amount instead of ACI letter** — for ACI questions the answer is a letter (e.g., `C`), not the fee value (e.g., `0.23`).
3. **Ignoring null is_credit rules for ACI questions** — rules with `is_credit: null` apply to credit transactions and must be included.
4. **Forgetting empty aci list** — an empty `aci` list in a rule means it covers ALL ACIs; don't skip these rules.

### "In general" interpretation
"Most expensive in general" means the worst-case maximum fee across all possible rule configurations. For each MCC or ACI, find the highest fee achievable under any applicable fee rule — you do not need to match all rule conditions, just find the global maximum.

## Output Format
- **MCC questions**: Comma-separated sorted list of MCC integers, e.g., `3000, 3001, 3002`
- **ACI questions**: Single letter in a list, e.g., `['C']` or `["C"]` — do NOT output the fee amount

## ACI Reference (from manual.md)
| ACI | Description |
|-----|-------------|
| A | Card Present - Non-authenticated |
| B | Card Present - Authenticated |
| C | Tokenized card with mobile device |
| D | Card Not Present - Card On File |
| E | Card Not Present - Recurring Bill Payment |
| F | Card Not Present - 3-D Secure |
| G | Card Not Present - Non-3-D Secure |

## Card Schemes in Dataset
GlobalCard, NexPay, SwiftCharge, TransactPlus
