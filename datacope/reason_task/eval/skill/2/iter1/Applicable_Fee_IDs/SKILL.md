---
name: Applicable_Fee_IDs
description: Solve "Applicable Fee IDs" questions in the dabstep payment dataset. Use this skill whenever a question asks which fee IDs apply to a merchant, transaction, or combination of payment attributes (account_type, aci, card_scheme, capture_delay, intracountry, MCC, etc.). Covers single-day queries ("what fees apply on day 200 for MerchantX?"), single-month queries ("what fees applied in October 2023?"), and full-year queries ("what fees applied in 2023?").
---

# Applicable Fee IDs — Dabstep Dataset

## Query Types

**Type A — Single Day**: "For the Nth of year 2023, what are the Fee IDs applicable to [Merchant]?"
- Monthly stats = full calendar month containing day N
- Transaction filter = only transactions on day N

**Type B — Single Month**: "What were the applicable Fee IDs for [Merchant] in [Month] 2023?"
- Monthly stats = that calendar month
- Transaction filter = all transactions in that month

**Type C — Full Year**: "What are the applicable fee IDs for [Merchant] in 2023?"
- Iterate over each calendar month; union all applicable IDs across all months

---

## Data Files

| File | Purpose |
|------|---------|
| `fees.json` | 1000 fee rules; each rule is a dict with matching conditions + `fixed_amount` + `rate` |
| `merchant_data.json` | Per-merchant: `account_type`, `capture_delay`, `merchant_category_code`, `acquirer` |
| `payments.csv` | 138 236 transactions; key columns: `merchant`, `card_scheme`, `year`, `day_of_year`, `is_credit`, `eur_amount`, `issuing_country`, `acquirer_country`, `aci`, `has_fraudulent_dispute` |
| `manual.md` | Domain definitions |

---

## Fee Rule Structure

```json
{
  "ID": 42,
  "card_scheme": "GlobalCard",       // always a specific scheme (never null)
  "account_type": ["F", "S"],        // list or [] (empty = all)
  "capture_delay": "<3",             // string or null
  "monthly_fraud_level": ">8.3%",   // string or null
  "monthly_volume": "100k-1m",       // string or null
  "merchant_category_code": [5812],  // list or [] (empty = all)
  "is_credit": true,                 // bool or null
  "aci": ["A", "C"],                 // list or [] (empty = all)
  "intracountry": true               // bool or null (also stored as 1.0/0.0)
}
```

**Null / empty-list semantics (critical):** `null` or `[]` means the rule applies to **all** values of that field.

---

## Core Matching Code

```python
import json, pandas as pd, datetime

fees = json.load(open("fees.json"))
merchants = json.load(open("merchant_data.json"))
payments = pd.read_csv("payments.csv")

# --- Merchant lookup ---
m = next(x for x in merchants if x["merchant"] == "MerchantName")
account_type = m["account_type"]
mcc = m["merchant_category_code"]
capture_delay = map_capture_delay(m["capture_delay"])

# --- Category helpers ---
def map_capture_delay(raw):
    if raw in ("immediate", "manual"): return raw
    days = int(raw)
    if days < 3: return "<3"
    if days <= 5: return "3-5"
    return ">5"

def volume_category(eur):
    if eur < 100_000: return "<100k"
    if eur < 1_000_000: return "100k-1m"
    if eur < 5_000_000: return "1m-5m"
    return ">5m"

def fraud_category(rate_pct):
    if rate_pct < 7.2: return "<7.2%"
    if rate_pct < 7.7: return "7.2%-7.7%"
    if rate_pct <= 8.3: return "7.7%-8.3%"
    return ">8.3%"

# --- Fee matching ---
def fee_matches(fee, card_scheme, is_credit, aci, intracountry,
                vol_cat, fraud_cat, account_type, mcc, capture_delay):
    if fee["card_scheme"] != card_scheme:                         return False
    if fee["account_type"] and account_type not in fee["account_type"]: return False
    if fee["capture_delay"] is not None and fee["capture_delay"] != capture_delay: return False
    if fee["monthly_volume"] is not None and fee["monthly_volume"] != vol_cat: return False
    if fee["monthly_fraud_level"] is not None and fee["monthly_fraud_level"] != fraud_cat: return False
    if fee["merchant_category_code"] and mcc not in fee["merchant_category_code"]: return False
    if fee["is_credit"] is not None and fee["is_credit"] != is_credit: return False
    if fee["aci"] and aci not in fee["aci"]:                      return False
    if fee["intracountry"] is not None and bool(fee["intracountry"]) != intracountry: return False
    return True
```

---

## Capture Delay Mapping

| Merchant value | Fee rule category |
|---------------|------------------|
| `"immediate"` | `"immediate"` |
| `"manual"` | `"manual"` |
| `"1"` or `"2"` (days < 3) | `"<3"` |
| `"3"`, `"4"`, `"5"` | `"3-5"` |
| `"7"` or any value > 5 | `">5"` |

---

## Monthly Stats Computation

Monthly stats (volume + fraud rate) are always computed over a **full calendar month**. Use `datetime` to get exact day-of-year ranges:

```python
def month_day_range(year, month):
    start = datetime.date(year, month, 1).timetuple().tm_yday
    if month < 12:
        end = (datetime.date(year, month+1, 1) - datetime.timedelta(1)).timetuple().tm_yday
    else:
        end = datetime.date(year, 12, 31).timetuple().tm_yday
    return start, end

def compute_monthly_stats(df, merchant, year, month):
    s, e = month_day_range(year, month)
    txns = df[(df["merchant"]==merchant) & (df["year"]==year) &
              (df["day_of_year"]>=s) & (df["day_of_year"]<=e)]
    if txns.empty: return None, None, txns
    vol = txns["eur_amount"].sum()
    fraud_vol = txns[txns["has_fraudulent_dispute"]==True]["eur_amount"].sum()
    return volume_category(vol), fraud_category(fraud_vol/vol*100 if vol>0 else 0), txns
```

---

## Full Workflow

### Type A: Single Day

```python
# Determine containing month, compute stats for that month
day_date = datetime.date(2023, 1, 1) + datetime.timedelta(days=N-1)
month = day_date.month
vol_cat, fraud_cat, month_txns = compute_monthly_stats(payments, merchant_name, 2023, month)

# Filter transactions for the specific day only
day_txns = payments[(payments["merchant"]==merchant_name) &
                    (payments["year"]==2023) & (payments["day_of_year"]==N)]

applicable = set()
if vol_cat and not day_txns.empty:
    day_txns = day_txns.copy()
    day_txns["intracountry"] = day_txns["issuing_country"] == day_txns["acquirer_country"]
    for _, row in day_txns.iterrows():
        for fee in fees:
            if fee_matches(fee, row["card_scheme"], row["is_credit"], row["aci"],
                          bool(row["intracountry"]), vol_cat, fraud_cat,
                          account_type, mcc, capture_delay):
                applicable.add(fee["ID"])
```

### Type B: Single Month

```python
vol_cat, fraud_cat, txns = compute_monthly_stats(payments, merchant_name, 2023, month_number)

applicable = set()
if vol_cat and not txns.empty:
    txns = txns.copy()
    txns["intracountry"] = txns["issuing_country"] == txns["acquirer_country"]
    for _, row in txns.iterrows():
        for fee in fees:
            if fee_matches(fee, row["card_scheme"], row["is_credit"], row["aci"],
                          bool(row["intracountry"]), vol_cat, fraud_cat,
                          account_type, mcc, capture_delay):
                applicable.add(fee["ID"])
```

### Type C: Full Year

```python
applicable = set()
for month in range(1, 13):
    vol_cat, fraud_cat, txns = compute_monthly_stats(payments, merchant_name, 2023, month)
    if vol_cat is None or txns.empty: continue
    txns = txns.copy()
    txns["intracountry"] = txns["issuing_country"] == txns["acquirer_country"]
    for _, row in txns.iterrows():
        for fee in fees:
            if fee_matches(fee, row["card_scheme"], row["is_credit"], row["aci"],
                          bool(row["intracountry"]), vol_cat, fraud_cat,
                          account_type, mcc, capture_delay):
                applicable.add(fee["ID"])
```

---

## Output Format

Return fee IDs as a **sorted, comma-separated list**:
```
29, 36, 51, 64, 65, 89, 107, ...
```

If no fees match, return an empty string `""`.

---

## Common Pitfalls

1. **`[]` means "all values apply"** — never treat empty account_type, aci, or merchant_category_code lists as "no match".
2. **card_scheme is never null** — always an exact match (GlobalCard, NexPay, TransactPlus, SwiftCharge).
3. **Volume/fraud matching is categorical** — compute the merchant's category string, then compare with the fee rule's string using equality. Do NOT parse ranges numerically.
4. **capture_delay mapping** — convert merchant's raw value (e.g. `"1"`) to the fee-rule category string (`"<3"`).
5. **intracountry type** — fees.json stores as `1.0`/`0.0`; always wrap with `bool()` before comparing.
6. **Monthly stats scope** — always compute volume/fraud over the full calendar month, even for single-day queries.
7. **No transactions → no applicable fees** — return `""`.
