---
name: Fee_Delta_and_Impact_Simulation
description: >
  Use this skill for dabstep payment-processing fee questions that involve
  simulating a change to a fee rule and computing the resulting impact.
  Trigger whenever a question asks: "what delta would <merchant> pay if the
  relative fee of fee ID=<X> changed to <Y>?", "which merchants would be
  affected if fee <X> was only applied to account type <Y>?", or "what amount
  delta will <merchant> pay if its MCC changed to <X>?".
---

# Fee Delta and Impact Simulation

## Problem Types

1. **Rate-change delta** – "In [month/year], what delta would [merchant] pay if
   the relative fee of fee ID=X changed to Y?"
2. **Merchant-impact filter** – "If fee ID=X was only applied to account type Y,
   which merchants would be affected?"
3. **MCC-change delta** – "Imagine [merchant] changed its MCC to X, what amount
   delta will it pay?"

---

## Datasets

| File | Key fields |
|------|-----------|
| `fees.json` | ID, card_scheme, account_type, capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code, is_credit, aci, fixed_amount, **rate**, intracountry |
| `merchant_data.json` | merchant, account_type, capture_delay, merchant_category_code, acquirer |
| `payments.csv` | merchant, card_scheme, year, day_of_year, is_credit, eur_amount, issuing_country, acquirer_country, aci, has_fraudulent_dispute |
| `acquirer_countries.csv` | acquirer, country_code |
| `manual.md` | domain definitions (read first) |

---

## Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
delta = (new_rate - old_rate) * SUM(applicable_eur_amount) / 10000
```

- **"relative fee"** always refers to the `rate` field.
- Delta sign: positive = merchant pays more; negative = merchant pays less.

---

## Fee-Rule Matching Logic

A fee rule applies to a transaction **only when all conditions below are met**.

### Empty list `[]` vs `null`

In `fees.json`, list fields (`account_type`, `aci`, `merchant_category_code`)
never contain `null`; they use `[]` to mean **"applies to all values"**
(equivalent to `null` in scalar fields). Scalar fields use `null` for "all".

### Per-merchant checks (done once, not per transaction)

| Field | Applies if fee value is | Merchant matches when |
|-------|------------------------|-----------------------|
| `account_type` | `[]` | always | 
| `account_type` | `["H","R",…]` | merchant's account_type is in the list |
| `capture_delay` | `null` | always |
| `capture_delay` | `"<3"` | merchant capture_delay is numeric and < 3 |
| `capture_delay` | `"3-5"` | merchant capture_delay is numeric and 3 ≤ x ≤ 5 |
| `capture_delay` | `">5"` | merchant capture_delay is numeric and > 5 |
| `capture_delay` | `"immediate"` | merchant capture_delay == "immediate" |
| `capture_delay` | `"manual"` | merchant capture_delay == "manual" |
| `merchant_category_code` | `[]` | always |
| `merchant_category_code` | `[5812,…]` | merchant's MCC is in the list |

> Merchant capture_delay values in the data are "immediate", "manual", or a
> string number of days (e.g. "1", "2", "7").

### Per-transaction checks

| Field | Applies if fee value is | Transaction matches when |
|-------|------------------------|--------------------------|
| `card_scheme` | any string | `payments.card_scheme == fee.card_scheme` |
| `is_credit` | `null` | always |
| `is_credit` | `True`/`False` | `payments.is_credit == fee.is_credit` |
| `aci` | `[]` | always |
| `aci` | `["A","B",…]` | `payments.aci` is in the list |
| `intracountry` | `null` | always |
| `intracountry` | `1.0` | `payments.issuing_country == payments.acquirer_country` |
| `intracountry` | `0.0` | `payments.issuing_country != payments.acquirer_country` |

### Per-month merchant checks (compute from payments.csv)

Compute for the **same month and merchant** as the question:

| Field | Applies if fee value is | Check |
|-------|------------------------|-------|
| `monthly_fraud_level` | `null` | always |
| `monthly_fraud_level` | `"<7.2%"` | fraud_rate < 7.2 % |
| `monthly_fraud_level` | `"7.2%-7.7%"` | 7.2 ≤ fraud_rate < 7.7 % |
| `monthly_fraud_level` | `"7.7%-8.3%"` | 7.7 ≤ fraud_rate < 8.3 % |
| `monthly_fraud_level` | `">8.3%"` | fraud_rate > 8.3 % |
| `monthly_volume` | `null` | always |
| `monthly_volume` | `"<100k"` | total_volume < 100 000 |
| `monthly_volume` | `"100k-1m"` | 100 000 ≤ total_volume < 1 000 000 |
| `monthly_volume` | `"1m-5m"` | 1 000 000 ≤ total_volume < 5 000 000 |
| `monthly_volume` | `">5m"` | total_volume ≥ 5 000 000 |

```python
# fraud_rate and monthly_volume for a merchant in a given month
month_txns = payments[(payments['merchant'] == merchant_name) &
                       (payments['day_of_year'] >= month_start) &
                       (payments['day_of_year'] <= month_end)]
total_volume = month_txns['eur_amount'].sum()
fraud_volume = month_txns[month_txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
fraud_rate = (fraud_volume / total_volume * 100) if total_volume > 0 else 0
```

---

## Date Handling

All payments in the dataset are year 2023. Use `day_of_year` for month filtering.

```python
import datetime
def month_range(month: int) -> tuple:
    """Return (first_day_of_year, last_day_of_year) for a 2023 month."""
    start = datetime.date(2023, month, 1).timetuple().tm_yday
    import calendar
    last_day = calendar.monthrange(2023, month)[1]
    end = datetime.date(2023, month, last_day).timetuple().tm_yday
    return start, end
```

| Month | day_of_year range |
|-------|------------------|
| January | 1 – 31 |
| February | 32 – 59 |
| July | 182 – 212 |
| September | 244 – 273 |
| December | 335 – 365 |
| Full year | all rows |

---

## Problem Type 1: Rate-Change Delta

### Algorithm

```python
import pandas as pd, json

# 1. Load data
payments = pd.read_csv('payments.csv')
fees = json.load(open('fees.json'))
merchants = json.load(open('merchant_data.json'))

# 2. Locate the fee and merchant
fee = next(f for f in fees if f['ID'] == fee_id)
merchant = next(m for m in merchants if m['merchant'] == merchant_name)
old_rate = fee['rate']

# 3. Check merchant-level criteria (disqualify whole merchant early)
# account_type
if fee['account_type'] and merchant['account_type'] not in fee['account_type']:
    print("Fee does not apply – account_type mismatch"); delta = 0.0
# capture_delay
# merchant capture_delay is string; map to category:
def capture_delay_matches(fee_cd, merchant_cd):
    if fee_cd is None: return True
    if fee_cd in ('immediate', 'manual'): return merchant_cd == fee_cd
    try:
        days = float(merchant_cd)
        if fee_cd == '<3':   return days < 3
        if fee_cd == '3-5':  return 3 <= days <= 5
        if fee_cd == '>5':   return days > 5
    except ValueError: return False
    return False

if not capture_delay_matches(fee['capture_delay'], merchant['capture_delay']):
    print("Fee does not apply – capture_delay mismatch"); delta = 0.0

# 4. Filter payments to the relevant time window
# (for a month question, filter day_of_year; for full year, use all)
txns = payments[payments['merchant'] == merchant_name]
# … apply day_of_year filter here …

# 5. Check monthly fraud_level and monthly_volume (if fee requires them)
if fee['monthly_fraud_level'] is not None or fee['monthly_volume'] is not None:
    total_vol = txns['eur_amount'].sum()
    fraud_vol = txns[txns['has_fraudulent_dispute'] == True]['eur_amount'].sum()
    fraud_rate_pct = (fraud_vol / total_vol * 100) if total_vol > 0 else 0
    # compare to thresholds and set a flag

# 6. Filter transactions by transaction-level criteria
mask = txns['card_scheme'] == fee['card_scheme']
if fee['is_credit'] is not None:
    mask &= txns['is_credit'] == fee['is_credit']
if fee['aci']:          # non-empty list
    mask &= txns['aci'].isin(fee['aci'])
if fee['intracountry'] is not None:
    same = txns['issuing_country'] == txns['acquirer_country']
    if fee['intracountry'] == 1.0:
        mask &= same
    else:
        mask &= ~same

applicable_txns = txns[mask]

# 7. Compute delta
total_amount = applicable_txns['eur_amount'].sum()
delta = (new_rate - old_rate) * total_amount / 10000
print(f"{delta:.14f}")
```

### Common Pitfalls

- **Sign**: delta = (new_rate − old_rate) × amount / 10000.  
  Negative means merchant saves money; positive means they pay more.
- **Empty list `[]`** for list fields means "no restriction" (match all values).  
  Do NOT interpret as "matches nothing".
- **`is_credit` is boolean** in payments.csv; compare with `== True/False`.
- **`intracountry`** uses `payments.issuing_country` vs `payments.acquirer_country`,  
  not the acquirer_countries.csv lookup (that CSV maps acquirer names, but the
  column in payments.csv already contains the country code directly).
- For a **monthly question with monthly_fraud_level/monthly_volume** constraints:
  compute fraud rate from the same month's transactions.  
  If outside the fee's range → zero applicable transactions → delta = 0.
- For a **full-year question with monthly constraints**: process each calendar
  month separately; only include months where the constraint is satisfied.

---

## Problem Type 2: Merchant-Impact Filter

"If fee ID=X was **only applied to account type Y**, which merchants would be affected?"

**Interpretation**: The fee currently applies to ALL account types (account_type=[]).
The change restricts it to only account type Y. Find all merchants that would gain
or lose the fee as a result.

### Algorithm

```python
# 1. Find the fee and its current criteria (besides account_type)
# 2. Identify merchants currently receiving the fee:
#    - Merchant's capture_delay matches the fee's capture_delay
#    - Merchant has at least one transaction matching card_scheme, is_credit, aci
# 3. Identify merchants that WOULD receive the fee after the change:
#    - Same as above, PLUS account_type == Y
# 4. Affected = (currently_receiving XOR would_receive) 
#    i.e., merchants who LOSE the fee (account_type != Y and currently receiving it)
#    plus merchants who GAIN the fee (account_type == Y and not currently receiving it)
```

In practice most questions have the new restriction remove the fee from non-Y
merchants, so the answer is: merchants with account_type ≠ Y who currently have
the fee applied.

---

## Problem Type 3: MCC-Change Delta

"Imagine merchant M changed its MCC to X before [year], what delta in fees?"

This requires computing total fees under original MCC vs. hypothetical MCC, then
subtracting. The full algorithm needs to:

1. For each transaction of M in the period, find the **best-matching fee rule**
   under the original MCC.
2. Repeat with the hypothetical MCC.
3. Delta = total_fee_new_MCC − total_fee_original_MCC.

"Best-matching" means the rule that satisfies all criteria and has the most
specific (non-null / non-empty) constraints. Use a specificity score:
count the number of non-null and non-empty fields that match.

---

## Output Format

- Numerical delta: rounded to 14 decimal places, e.g. `-1.37683100000000`
- Merchant list: comma-separated (order not critical), e.g.
  `Belles_cookbook_store, Crossfit_Hanna, Golfclub_Baron_Friso`
- If no transactions match → `0.00000000000000`
- If truly not applicable → `Not Applicable`
