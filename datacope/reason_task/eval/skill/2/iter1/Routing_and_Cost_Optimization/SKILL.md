---
name: Routing_and_Cost_Optimization
description: Solve payment routing and cost optimization problems in the dabstep dataset. Use this skill when asked to determine the optimal card scheme or Authorization Characteristics Indicator (ACI) for a merchant to minimize or maximize fees, or when questions involve routing fraudulent transactions to a different ACI. Applies to monthly or annual fee optimization across merchants and time periods.
---

# Routing and Cost Optimization

Solve payment routing optimization problems: find which card scheme or ACI minimizes/maximizes fees for a merchant over a time period.

## Question Types

1. **Card Scheme Routing**: "Which card scheme should merchant X steer traffic to in [month/year] to pay [minimum/maximum] fees?"
2. **ACI Optimization**: "For merchant X in [period], if we move fraudulent transactions to a different ACI, what is the preferred choice considering [lowest] fees?"

**Answer format**: `{option}:{total_fee_rounded_to_2_decimals}` — e.g., `GlobalCard:142.37` or `B:102.59`

## Required Datasets

| File | Purpose |
|------|---------|
| `payments.csv` | Transactions (merchant, card_scheme, aci, is_credit, eur_amount, issuing_country, acquirer_country, has_fraudulent_dispute, day_of_year, year) |
| `fees.json` | Fee rules (1000 rules with matching criteria + fixed_amount + rate) |
| `merchant_data.json` | Merchant profile (account_type, capture_delay, merchant_category_code, acquirer list) |

Always read `manual.md` and `payments-readme.md` first for domain definitions.

## Fee Calculation Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

When multiple fee rules match a transaction, **select the one producing the lowest fee**.

## CRITICAL: Acquirer Country for Intracountry

**Always use the `acquirer_country` column from `payments.csv` directly** — one per transaction row.

**NEVER** compute acquirer country by looking up the merchant's acquirer list in `acquirer_countries.csv`. The `acquirer_country` in `payments.csv` may differ significantly from the acquirer_countries.csv mapping. For example, a merchant's acquirer mapped to 'FR' in acquirer_countries.csv might have 'NL' in the payments.csv `acquirer_country` column for all its transactions. Using the wrong source causes incorrect intracountry flags and wrong fee totals.

```python
# CORRECT
intracountry = row['issuing_country'] == row['acquirer_country']  # from payments.csv

# WRONG - do NOT do this
acquirer_country = acq_countries[acq_countries['acquirer'] == merchant_acquirer]['country_code'].values[0]
intracountry = row['issuing_country'] == acquirer_country  # this may be wrong
```

## Fee Rule Matching Logic

Match ALL of these criteria. A rule applies when:

| Field | Rule applies when |
|-------|------------------|
| `card_scheme` | Exactly matches transaction's card scheme |
| `account_type` | `[]` (empty list) or `null` → all; else merchant's type must be in list |
| `capture_delay` | `null` → all; else must match mapped merchant capture delay |
| `merchant_category_code` | `[]` (empty list) or `null` → all; else merchant's MCC must be in list |
| `is_credit` | `null` → all; else must match transaction's `is_credit` |
| `aci` | `[]` (empty list) or `null` → all; else transaction/target ACI must be in list |
| `intracountry` | `null` → all; `1.0`/`true` → domestic only; `0.0`/`false` → international only |
| `monthly_fraud_level` | `null` → all; else merchant's period fraud % must fall in range |
| `monthly_volume` | `null` → all; else merchant's period EUR volume must fall in range |

**Empty list `[]` = applies to all** (same as null). This is the most common mistake to avoid.

## Capture Delay Mapping

Merchant's `capture_delay` in `merchant_data.json` may be numeric (days) or a string bracket:
- `"0"` or `"immediate"` → `immediate`
- `"1"` or `"2"` (1-2 days) → `<3`
- `"3"`, `"4"`, `"5"` (3-5 days) → `3-5`
- `"6"`, `"7"`, or any number > 5 → `>5`
- `"manual"` → `manual`

## Range Parsing

**Monthly fraud level** (fraudulent_volume / total_volume × 100):
```
'<7.2%'    → fraud_pct < 7.2
'7.2%-7.7%' → 7.2 <= fraud_pct <= 7.7
'7.7%-8.3%' → 7.7 <= fraud_pct <= 8.3
'>8.3%'    → fraud_pct > 8.3
```

**Monthly volume** (sum of eur_amount in EUR):
```
'<100k'   → volume < 100_000
'100k-1m' → 100_000 <= volume <= 1_000_000
'1m-5m'   → 1_000_000 < volume <= 5_000_000
'>5m'     → volume > 5_000_000
```

## Step-by-Step Solution Process

### Step 1: Load data
```python
import pandas as pd, json
payments = pd.read_csv('.../payments.csv')
fees = json.load(open('.../fees.json'))
merchants = json.load(open('.../merchant_data.json'))
```

### Step 2: Filter transactions for the period
```python
# For a monthly question (e.g., September 2023):
payments['date'] = pd.to_datetime(payments['year'].astype(str) + '-' +
                                   payments['day_of_year'].astype(str), format='%Y-%j')
payments['month'] = payments['date'].dt.month
subset = payments[(payments['merchant'] == merchant_name) & (payments['month'] == target_month)]

# For an annual question (e.g., year 2023):
subset = payments[(payments['merchant'] == merchant_name) & (payments['year'] == 2023)]
```

### Step 3: Compute period metrics
```python
total_volume = subset['eur_amount'].sum()
fraud_volume = subset[subset['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_pct = fraud_volume / total_volume * 100
```

For **annual** questions, compute these metrics over the **entire year** (not per natural month). The fee rule fields `monthly_volume` and `monthly_fraud_level` are matched against the full-period aggregated volume and fraud rate for the queried time window.

### Step 4: Determine intracountry per transaction
```python
# Use acquirer_country from payments.csv directly — do NOT look up from acquirer_countries.csv
subset = subset.copy()
subset['intracountry'] = subset['issuing_country'] == subset['acquirer_country']
```

### Step 5: Build the fee matching function
```python
def matches_rule(rule, account_type, capture_delay, fraud_pct, volume, mcc,
                 is_credit, aci, intracountry):
    # account_type
    at = rule.get('account_type') or []
    if at and account_type not in at:
        return False
    # capture_delay
    cd = rule.get('capture_delay')
    if cd is not None and cd != capture_delay:
        return False
    # monthly_fraud_level
    mfl = rule.get('monthly_fraud_level')
    if mfl:
        if mfl == '<7.2%' and not fraud_pct < 7.2: return False
        elif mfl == '7.2%-7.7%' and not (7.2 <= fraud_pct <= 7.7): return False
        elif mfl == '7.7%-8.3%' and not (7.7 <= fraud_pct <= 8.3): return False
        elif mfl == '>8.3%' and not fraud_pct > 8.3: return False
    # monthly_volume
    mv = rule.get('monthly_volume')
    if mv:
        if mv == '<100k' and not volume < 100_000: return False
        elif mv == '100k-1m' and not (100_000 <= volume <= 1_000_000): return False
        elif mv == '1m-5m' and not (1_000_000 < volume <= 5_000_000): return False
        elif mv == '>5m' and not volume > 5_000_000: return False
    # merchant_category_code
    mccs = rule.get('merchant_category_code') or []
    if mccs and mcc not in mccs:
        return False
    # is_credit
    ic = rule.get('is_credit')
    if ic is not None and ic != is_credit:
        return False
    # aci
    rule_aci = rule.get('aci') or []
    if rule_aci and aci not in rule_aci:
        return False
    # intracountry
    intra = rule.get('intracountry')
    if intra is not None and bool(intra) != intracountry:
        return False
    return True

def calc_fee(rule, amount):
    return rule['fixed_amount'] + rule['rate'] * amount / 10000
```

### Step 6: Calculate total fees per option

**Card-scheme routing**: iterate over transactions already on each scheme.
**ACI optimization**: iterate over only fraudulent transactions, testing each ACI (A–F).

```python
# Card-scheme routing
options = subset['card_scheme'].unique()

results = {}
for option in options:
    txns = subset[subset['card_scheme'] == option]
    total_fee = 0.0
    no_rule_count = 0
    for _, row in txns.iterrows():
        matching = [r for r in fees
                    if r['card_scheme'] == row['card_scheme']
                    and matches_rule(r, account_type, capture_delay, fraud_pct, volume,
                                     mcc, row['is_credit'], row['aci'], row['intracountry'])]
        if matching:
            total_fee += min(calc_fee(r, row['eur_amount']) for r in matching)
        else:
            no_rule_count += 1  # contributes 0 to total fee
    results[option] = {'total_fee': total_fee, 'no_rule': no_rule_count}

# ACI optimization (fraudulent transactions only)
fraud_txns = subset[subset['has_fraudulent_dispute'] == True]
aci_options = ['A', 'B', 'C', 'D', 'E', 'F']  # G has no fee rules

for aci_target in aci_options:
    total_fee = 0.0
    no_rule_count = 0
    for _, row in fraud_txns.iterrows():
        matching = [r for r in fees
                    if r['card_scheme'] == row['card_scheme']
                    and matches_rule(r, account_type, capture_delay, fraud_pct, volume,
                                     mcc, row['is_credit'], aci_target, row['intracountry'])]
        if matching:
            total_fee += min(calc_fee(r, row['eur_amount']) for r in matching)
        else:
            no_rule_count += 1
    results[aci_target] = {'total_fee': total_fee, 'no_rule': no_rule_count}
```

### Step 7: Select and format answer

**For card-scheme questions**: Pick the scheme with min/max total fee across all schemes (transactions without matching rules contribute 0 — do NOT filter by coverage).

**For ACI questions**: Only consider ACIs with **full coverage** (no transactions without a matching rule). Among fully-covered ACIs, pick the one with lowest total fee.
```python
# Card scheme: no coverage filter
best = min(results, key=lambda k: results[k]['total_fee'])  # or max for maximum

# ACI: filter to full coverage first
full_coverage = {k: v for k, v in results.items() if v['no_rule'] == 0}
best = min(full_coverage, key=lambda k: full_coverage[k]['total_fee'])
fee = round(full_coverage[best]['total_fee'], 2)
answer = f"{best}:{fee}"
```

## Common Pitfalls

- **Using acquirer_countries.csv for intracountry**: The `acquirer_country` in payments.csv is per-transaction and authoritative. Never override it with a lookup from acquirer_countries.csv — the two can differ for the same merchant.
- **Treating `[]` as "no match"**: Empty list in `account_type`, `aci`, or `merchant_category_code` means "applies to all".
- **Missing ACI coverage check**: For ACI optimization, ACIs with any uncovered transaction must be excluded even if their partial fee sum is lower.
- **Applying ACI coverage filter to card-scheme routing**: For card-scheme routing, transactions with no matching rule just contribute 0 — do not exclude entire schemes.
- **Not using minimum fee**: When multiple rules match, always use the rule producing the **lowest fee**.
- **Capture delay numeric values**: `capture_delay: "7"` means 7 days → maps to `>5`.
- **Annual period metrics**: For annual questions, compute `monthly_volume` and `monthly_fraud_level` categories using the **full year's aggregated totals**, not per-month breakdowns.
