---
name: Routing_and_Cost_Optimization
description: Solve payment routing and cost optimization problems on the dabstep dataset. Use this skill for questions like "which card scheme should merchant X steer traffic to for min/max fees?" or "which ACI should fraudulent transactions be moved to for lowest fees?" These questions require matching fee rules against transaction data for a merchant over a specific time period.
---

# Routing and Cost Optimization

This skill handles two types of optimization questions on the dabstep payment dataset:

1. **Card scheme routing**: "Which card scheme should merchant X steer traffic to in order to pay min/max fees in period Y?"
2. **ACI optimization**: "For merchant X in period Y, which ACI should fraudulent transactions be moved to for lowest possible fees?"

Answer format is always: `{selection}:{fee}` (e.g., `SwiftCharge:37.21` or `C:18.58`), where fee is the **total fee** across relevant transactions rounded to 2 decimals.

---

## Data Files

- `payments.csv`: Transaction records (138,236 rows, year=2023 only). Key columns: `merchant`, `card_scheme`, `day_of_year`, `is_credit`, `eur_amount`, `issuing_country`, `acquirer_country`, `aci`, `has_fraudulent_dispute`
- `fees.json`: 1,000 fee rules (card schemes: NexPay, GlobalCard, SwiftCharge, TransactPlus)
- `merchant_data.json`: List of merchant objects with `merchant`, `capture_delay`, `acquirer`, `merchant_category_code`, `account_type`
- `acquirer_countries.csv`: Maps acquirer name → `country_code`
- `manual.md`: Definitions (read first for domain context)

---

## Step-by-Step Algorithm

### Step 1: Load data

```python
import json, pandas as pd

df = pd.read_csv('payments.csv')
with open('fees.json') as f: fees = json.load(f)
with open('merchant_data.json') as f: merchants = json.load(f)
acq_countries = pd.read_csv('acquirer_countries.csv')

merchant_info = next(m for m in merchants if m['merchant'] == MERCHANT_NAME)
```

### Step 2: Filter transactions by period

`payments.csv` uses `day_of_year` (1–365 for 2023). Convert months:

```python
import pandas as pd

def get_month_range(month):
    start = pd.Timestamp(2023, month, 1).dayofyear
    end = (pd.Timestamp(2023, month+1, 1).dayofyear - 1) if month < 12 else 365
    return start, end

# For a specific month:
start, end = get_month_range(MONTH_NUMBER)  # Jan=1, Feb=2, ..., Dec=12
df_period = df[(df['merchant'] == MERCHANT_NAME) &
               (df['day_of_year'] >= start) & (df['day_of_year'] <= end)]

# For full year:
df_period = df[df['merchant'] == MERCHANT_NAME]
```

Month boundaries for 2023 (non-leap year):
- Jan:1–31, Feb:32–59, Mar:60–90, Apr:91–120, May:121–151, Jun:152–181
- Jul:182–212, Aug:213–243, Sep:244–273, Oct:274–304, Nov:305–334, Dec:335–365

### Step 3: Compute period-level merchant metrics

These are needed to match fee rules with `monthly_volume` and `monthly_fraud_level` fields:

```python
total_vol = df_period['eur_amount'].sum()
fraud_vol = df_period[df_period['has_fraudulent_dispute']]['eur_amount'].sum()
fraud_rate = fraud_vol / total_vol  # ratio, not percentage

def get_volume_category(vol):
    if vol < 100_000: return '<100k'
    elif vol < 1_000_000: return '100k-1m'
    elif vol < 5_000_000: return '1m-5m'
    else: return '>5m'

def get_fraud_category(rate):
    if rate < 0.072: return '<7.2%'
    elif rate < 0.077: return '7.2%-7.7%'
    elif rate < 0.083: return '7.7%-8.3%'
    else: return '>8.3%'

vol_cat = get_volume_category(total_vol)
fraud_cat = get_fraud_category(fraud_rate)
```

### Step 4: Prepare merchant-level matching fields

```python
def get_capture_delay_category(capture_delay):
    """Convert merchant capture_delay string to fee rule category."""
    if capture_delay in ('manual', 'immediate'):
        return capture_delay
    try:
        days = int(capture_delay)
        if days < 3: return '<3'
        elif days <= 5: return '3-5'
        else: return '>5'
    except:
        return capture_delay

cap_cat = get_capture_delay_category(merchant_info['capture_delay'])
mcc = merchant_info['merchant_category_code']
account_type = merchant_info['account_type']
```

### Step 5: Fee rule matching function (CRITICAL)

**Important rules for null/empty handling:**
- `null` in any field → matches all values of that field
- Empty list `[]` in `account_type`, `merchant_category_code`, or `aci` → also matches all (same as null)

```python
def match_fee_rule(rule, card_scheme, is_credit, aci, intracountry,
                   account_type, cap_cat, mcc, vol_cat, fraud_cat):
    # card_scheme: exact match required
    if rule['card_scheme'] != card_scheme:
        return False
    # account_type: [] or null = all; otherwise merchant's account_type must be in list
    if rule['account_type'] and account_type not in rule['account_type']:
        return False
    # capture_delay: null = all; otherwise must match merchant's category
    if rule['capture_delay'] is not None and rule['capture_delay'] != cap_cat:
        return False
    # monthly_fraud_level: null = all; otherwise must match computed fraud category
    if rule['monthly_fraud_level'] is not None and rule['monthly_fraud_level'] != fraud_cat:
        return False
    # monthly_volume: null = all; otherwise must match computed volume category
    if rule['monthly_volume'] is not None and rule['monthly_volume'] != vol_cat:
        return False
    # merchant_category_code: [] or null = all; otherwise MCC must be in list
    if rule['merchant_category_code'] and mcc not in rule['merchant_category_code']:
        return False
    # is_credit: null = all; otherwise must match transaction's is_credit
    if rule['is_credit'] is not None and rule['is_credit'] != is_credit:
        return False
    # aci: [] or null = all; otherwise transaction's ACI must be in list
    if rule['aci'] and aci not in rule['aci']:
        return False
    # intracountry: null = all; True = domestic (issuer==acquirer country); False = cross-border
    if rule['intracountry'] is not None:
        if bool(rule['intracountry']) != intracountry:
            return False
    return True

def calc_fee(rule, amount):
    """Fee formula: fixed_amount + rate * transaction_value / 10000"""
    return rule['fixed_amount'] + rule['rate'] * amount / 10000
```

**intracountry** per transaction: `intracountry = (row['issuing_country'] == row['acquirer_country'])`

### Step 6a: Card scheme routing question

Sum fees for each card scheme over the period:

```python
scheme_fees = {}
for scheme in df_period['card_scheme'].unique():
    df_s = df_period[df_period['card_scheme'] == scheme]
    total = 0.0
    for _, row in df_s.iterrows():
        intra = (row['issuing_country'] == row['acquirer_country'])
        rules = [r for r in fees if match_fee_rule(
            r, row['card_scheme'], row['is_credit'], row['aci'], intra,
            account_type, cap_cat, mcc, vol_cat, fraud_cat
        )]
        if rules:
            total += min(calc_fee(r, row['eur_amount']) for r in rules)
        # transactions with no matching rule contribute 0
    scheme_fees[scheme] = total

# For minimum: best = min(scheme_fees, key=scheme_fees.get)
# For maximum: best = max(scheme_fees, key=scheme_fees.get)
best_scheme = min(scheme_fees, key=scheme_fees.get)  # or max
answer = f"{best_scheme}:{scheme_fees[best_scheme]:.2f}"
```

### Step 6b: ACI optimization question (fraudulent transactions)

Only operate on fraudulent transactions. Try each alternative ACI (A–F). Note: **ACI 'G' has no fee rules** — it's the current state being moved away from.

```python
fraud_df = df_period[df_period['has_fraudulent_dispute']]
n_fraud = len(fraud_df)

aci_results = {}
for test_aci in ['A', 'B', 'C', 'D', 'E', 'F']:
    total_fee = 0.0
    no_rule_count = 0
    for _, row in fraud_df.iterrows():
        intra = (row['issuing_country'] == row['acquirer_country'])
        rules = [r for r in fees if match_fee_rule(
            r, row['card_scheme'], row['is_credit'], test_aci, intra,
            account_type, cap_cat, mcc, vol_cat, fraud_cat
        )]
        if rules:
            total_fee += min(calc_fee(r, row['eur_amount']) for r in rules)
        else:
            no_rule_count += 1
    aci_results[test_aci] = {'total': total_fee, 'no_rule': no_rule_count}

# Only consider ACIs with rules for ALL fraudulent transactions
valid = {a: v['total'] for a, v in aci_results.items() if v['no_rule'] == 0}
best_aci = min(valid, key=valid.get)  # lowest fee
answer = f"{best_aci}:{valid[best_aci]:.2f}"
```

---

## Key Gotchas

1. **MCC not found in fee rules**: Many merchant MCCs (e.g., 7997, 7993, 7372) do NOT appear in any fee rule's `merchant_category_code` list. These merchants are covered by rules with empty `[]` MCC lists — treat `[]` as "matches all MCCs" (same as null).

2. **Capture delay conversion**: Merchant `capture_delay` values can be numeric strings like `"1"`, `"2"`, `"7"`. Convert them: `<3`, `3-5`, `>5`. String values `"manual"` and `"immediate"` match directly.

3. **No matching rule → fee = 0**: When no rule matches a transaction, it contributes 0 to the total. For ACI optimization, an ACI with ANY unmatched transaction is excluded from consideration.

4. **ACI 'G' has no fee rules**: Fee rules only cover ACIs A–F. Transactions with current ACI 'G' simply won't match any rule. For ACI optimization questions, the goal is to find an alternative to the current ACI.

5. **Take minimum fee when multiple rules match**: If several rules all match a transaction, use the one with the lowest fee.

6. **Monthly metrics use the full analysis period**: For a monthly question, compute `vol_cat` and `fraud_cat` from all transactions of that merchant in that month. For a full-year question, use all transactions in 2023.

7. **`intracountry` is a float in fees.json** (1.0 = True, 0.0 = False): Cast with `bool(rule['intracountry'])` before comparing.

---

## Performance Tip

For large datasets (full year = ~27k transactions), the per-transaction loop can be slow. Consider vectorizing or pre-filtering rules:

```python
# Pre-filter rules by card_scheme, account_type, capture_delay, mcc, vol_cat, fraud_cat
# (merchant-level fields that don't change per transaction)
# Then only check is_credit, aci, intracountry per transaction
```
