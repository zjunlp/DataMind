---
name: dataset-metadata-and-business-rules
description: >
  Answer questions about the dabstep payment processing dataset's metadata, schema,
  column definitions, and business rules governing fees. Use this skill whenever
  questions ask about column names/meanings, fee rule factors and their directions
  (e.g., "which boolean factor makes fees cheaper when True"), business rule
  definitions (e.g., capture delay, fraud level, volume tiers), whether a concept
  exists in the dataset (e.g., "is there an excessive retry fee?"), or how fee
  parameters interact (intracountry, is_credit, monthly_fraud_level, monthly_volume,
  capture_delay). Also applies to questions about ACI codes, account types, MCC codes,
  and any metadata from payments-readme.md or manual.md.
---

# Dataset Metadata and Business Rules

## Dataset Files

| File | Contents |
|------|----------|
| `payments-readme.md` | Column definitions for `payments.csv` |
| `manual.md` | Business rules, fee factor semantics, ACI codes, account types |
| `fees.json` | 1000 fee rules; fields: ID, card_scheme, account_type, capture_delay, monthly_fraud_level, monthly_volume, merchant_category_code, is_credit, aci, fixed_amount, rate, intracountry |
| `payments.csv` | Transaction data (see payments-readme.md for columns) |
| `merchant_data.json` | Per-merchant: merchant_category_code, account_type, capture_delay, acquirers |
| `merchant_category_codes.csv` | MCC codes and descriptions |
| `acquirer_countries.csv` | country_code of acquirers |

## Fee Formula

```
fee = fixed_amount + rate * transaction_value / 10000
```

`null` in any fee rule field means the rule applies to all values of that field.

---

## Step-by-Step Approach

### Step 1: Read the authoritative sources first

Always start by reading the relevant documentation:

```python
with open('path/to/manual.md', 'r') as f:
    manual = f.read()
with open('path/to/payments-readme.md', 'r') as f:
    readme = f.read()
```

For column/field definitions → `payments-readme.md`  
For fee rules and business logic → `manual.md` (especially Section 5)  
For actual fee rule data → `fees.json`

### Step 2: Identify question type

| Question Type | Primary Source | Strategy |
|---------------|----------------|----------|
| Column name/definition | payments-readme.md | Direct lookup |
| Does concept X exist? | manual.md + fees.json + payments-readme.md | Exhaustive search |
| Boolean factor effect (True/False) | manual.md Section 5 | Use manual semantics |
| Which factors decrease → cheaper? | manual.md Section 5 | Use manual semantics |
| Volume tier values / thresholds | fees.json | Query data |
| Fee rule field values | fees.json | Query data |
| ACI code meanings | manual.md Section 4 | Direct lookup |
| Account type codes | manual.md Section 2 | Direct lookup |

### Step 3: Apply business rules from manual.md

**Trust the manual's explicit directional statements** about fee effects, even if raw data statistics appear to contradict them (confounding variables exist in the data).

**Boolean fields and their cost direction (from manual Section 5):**

| Field | True means | False means | Cheaper when |
|-------|-----------|------------|--------------|
| `is_credit` | Credit transaction → typically more expensive | Debit → cheaper | **False** |
| `intracountry` | Domestic (issuer country = acquirer country) → cheaper | International → typically more expensive | **True** |

**Numeric/categorical fields and cost direction:**

| Field | Direction | Rule |
|-------|-----------|------|
| `monthly_fraud_level` | Higher fraud → more expensive | Decrease → cheaper |
| `monthly_volume` | Higher volume → cheaper (economies of scale) | Increase → cheaper |
| `capture_delay` | Faster capture → more expensive | Slowing down → cheaper |

Volume tiers in fees.json: `<100k`, `100k-1m`, `1m-5m`, `>5m`  
Fraud level tiers: `<7.2%`, `7.2%-7.7%`, `7.7%-8.3%`, `>8.3%`  
Capture delay values: `immediate`, `<3`, `3-5`, `>5`, `manual`

### Step 4: For "Not Applicable" determination

If a question asks about a concept (e.g., "retry fee", "chargeback amount") that may or may not exist:
1. Search manual.md for the concept
2. Search fees.json fields and values
3. Search payments-readme.md
4. Search payments.csv column headers

Only answer `Not Applicable` after exhaustively confirming the concept doesn't exist in any source. The manual mentions "excessive retrying" causes downgrades but defines **no monetary excessive retry fee**.

### Step 5: Answer formatting

Match the exact format requested and use exact strings from the source files:
- Column names: use exact case as in payments-readme.md (e.g., `has_fraudulent_dispute`, not `Has_Fraudulent_Dispute`)
- Volume tiers: use exact strings from fees.json (e.g., `>5m`, `100k-1m`)
- Multiple values: comma-separated list (e.g., `monthly_fraud_level, is_credit`)
- Non-existent concepts: `Not Applicable`

---

## Key Business Rule Summaries

**Fraud indicator column:** `has_fraudulent_dispute` (Boolean - from payments-readme.md)

**ACI values:** A (Card present non-auth), B (Card present auth), C (Tokenized mobile), D (Card not present COF), E (Recurring), F (3D Secure), G (Non-3D Secure)

**Account types:** R (Enterprise Retail), D (Enterprise Digital), H (Enterprise Hospitality), F (Platform Franchise), S (Platform SaaS), O (Other)

**Boolean factors → cheaper if True:** `intracountry`  
**Boolean factors → cheaper if False:** `is_credit`  
**Factors → cheaper if decreased:** `monthly_fraud_level`, `is_credit`

---

## Common Pitfall: Data vs. Manual

Raw statistical comparison of fee rates by field value can be misleading because fee rules vary across many dimensions (card_scheme, MCC, ACI, etc.). For questions about **which direction** a factor pushes fees, rely on the manual's explicit statements rather than naive aggregate statistics from fees.json. Use fees.json data when you need **specific values, tier labels, or counts**.
