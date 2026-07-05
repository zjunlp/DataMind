---
name: sec-10k-company-analysis
description: >
  Comprehensive analysis of public companies using SEC EDGAR 10-K financial data stored in a SQLite database.
  Use this skill whenever the task involves analyzing a company by CIK (Central Index Key), querying SEC financial
  data, exploring a financial database with tables like companies, filings, financial_facts, or producing a
  structured financial analysis report. Covers schema navigation, metric discovery, industry-specific exploration,
  and producing a complete financial summary.
---

# SEC 10-K Company Analysis

## Database Schema

The SQLite database has **5 tables** with these exact column names (wrong column names are a common failure):

### `companies` (primary key: `cik`)
Key columns: `cik`, `name`, `sic`, `sic_description`, `entity_type`, `category`, `fiscal_year_end`,
`state_of_incorporation`, `phone`, `description`, `website`, `former_names`, `owner_org`

### `company_addresses`
Columns: `cik`, `address_type` ("business"/"mailing"), `street1`, `city`, `state_or_country`, `zip_code`

### `company_tickers`
Columns: `cik`, `ticker`, `exchange`

### `filings` — use column `form` (NOT `form_type`)
Key columns: `cik`, `accession_number`, `filing_date`, `report_date`, `form`, `core_type`, `size`, `is_xbrl`

### `financial_facts` — use column `fact_name` (NOT `tag`), `form_type` (NOT `form`)
Key columns: `cik`, `fact_name`, `fact_value`, `unit`, `fact_category`, `fiscal_year`, `fiscal_period`,
`end_date`, `accession_number`, `form_type`, `filed_date`, `dimension_segment`, `dimension_geography`

**`fiscal_period` values**: `FY` (annual), `Q1`, `Q2`, `Q3`, `Q4`
**`fact_category` values**: `us-gaap`, `dei`, `ifrs-full`

---

## Analysis Workflow

### Step 1: Database Discovery
```sql
-- Always start by understanding structure
get_database_info()
describe_table("companies")
describe_table("financial_facts")
```

### Step 2: Company Basics
```sql
SELECT * FROM companies WHERE cik = '<CIK>'
SELECT * FROM company_tickers WHERE cik = '<CIK>'
SELECT * FROM company_addresses WHERE cik = '<CIK>'
```

### Step 3: Filing History
```sql
-- Note: column is `form`, not `form_type`
SELECT form, filing_date, report_date, accession_number
FROM filings WHERE cik = '<CIK>'
ORDER BY filing_date DESC LIMIT 20

-- Count by form type
SELECT form, COUNT(*) as count FROM filings
WHERE cik = '<CIK>' GROUP BY form ORDER BY count DESC
```

### Step 4: Discover Available Financial Metrics
```sql
-- Get all fact names for this company (always LIMIT to avoid huge result)
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' ORDER BY fact_name LIMIT 100
```

### Step 5: Core Financial Metrics (Annual Data)
```sql
-- Get key annual metrics — filter by form_type='10-K' for clean annual data
SELECT fact_name, fact_value, unit, fiscal_year, fiscal_period, end_date
FROM financial_facts
WHERE cik = '<CIK>'
  AND form_type = '10-K'
  AND fact_name IN (
    'Assets', 'Liabilities', 'StockholdersEquity',
    'NetIncomeLoss', 'OperatingIncomeLoss',
    'CashAndCashEquivalentsAtCarryingValue',
    'EarningsPerShareBasic', 'EarningsPerShareDiluted',
    'EntityCommonStockSharesOutstanding'
  )
ORDER BY end_date DESC LIMIT 50
```

### Step 6: Revenue Discovery (Critical — many companies use non-standard names)
Start with these candidates; if empty, search with LIKE:
```sql
-- Try common revenue names
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN (
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'RevenueFromContractWithCustomerIncludingAssessedTax',
    'SalesRevenueNet',
    'RevenuesNetOfInterestExpense'
  )
ORDER BY end_date DESC LIMIT 20

-- If empty, discover the actual revenue metric name
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%')
ORDER BY fact_name LIMIT 20
```

### Step 7: Cash Flow Analysis
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN (
    'NetCashProvidedByUsedInOperatingActivities',
    'NetCashProvidedByUsedInInvestingActivities',
    'NetCashProvidedByUsedInFinancingActivities',
    'CapitalExpendituresIncurredButNotYetPaid',
    'PaymentsToAcquirePropertyPlantAndEquipment'
  )
ORDER BY end_date DESC LIMIT 15
```

### Step 8: Debt & Capital Structure
```sql
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND fact_name LIKE '%Debt%' LIMIT 30

SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN ('LongTermDebt', 'DebtCurrent', 'LongTermDebtCurrent',
                    'DebtInstrumentCarryingAmount', 'LongTermDebtNoncurrent')
ORDER BY end_date DESC LIMIT 20
```

### Step 9: Industry-Specific Metrics
After identifying the company's SIC code, query relevant industry metrics:

**REIT / Real Estate (SIC 6500-6799)**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%RealEstate%' OR fact_name LIKE '%NumberOfReal%'
       OR fact_name LIKE '%NumberOfUnit%' OR fact_name LIKE '%FundsFrom%'
       OR fact_name LIKE '%Rental%')
ORDER BY end_date DESC LIMIT 30
```

**Pharmaceutical / Life Sciences (SIC 2830-2836)**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN ('ResearchAndDevelopmentExpense',
    'SellingGeneralAndAdministrativeExpense',
    'AllocatedShareBasedCompensationExpense',
    'CommonStockDividendsPerShareDeclared')
ORDER BY end_date DESC LIMIT 20
```

**Retail / Consumer (SIC 5200-5999)**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN ('CostOfGoodsSold', 'GrossProfit', 'InventoryNet',
    'AdvertisingExpense', 'NumberOfStores')
ORDER BY end_date DESC LIMIT 20
```

**Utilities (SIC 4900-4999)**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN ('PublicUtilitiesPropertyPlantAndEquipmentNet',
    'RegulatoryAssetsCurrent', 'RegulatoryAssetsNoncurrent')
ORDER BY end_date DESC LIMIT 20
```

**Financial Services / Banks (SIC 6000-6499)**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K'
  AND (fact_name LIKE '%Interest%' OR fact_name LIKE '%Loan%'
       OR fact_name LIKE '%Deposit%')
ORDER BY end_date DESC LIMIT 30
```

**Technology / Manufacturing**:
```sql
SELECT fact_name, fact_value, unit, fiscal_year, end_date
FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN ('ResearchAndDevelopmentExpense', 'GoodwillAndIntangible',
    'PropertyPlantAndEquipmentNet', 'InventoryNet')
ORDER BY end_date DESC LIMIT 20
```

---

## Common Pitfalls

| Mistake | Correct Approach |
|---------|-----------------|
| `SELECT DISTINCT tag FROM financial_facts` | `SELECT DISTINCT fact_name FROM financial_facts` |
| `GROUP BY form_type FROM filings` | `GROUP BY form FROM filings` (filings uses `form`) |
| `SELECT * FROM filings` (no LIMIT) | Always add `LIMIT` to avoid huge results |
| Assuming `Revenues` metric exists | Try multiple revenue names; use LIKE fallback |
| Missing parentheses: `WHERE fact_name LIKE '%A%' OR fact_name LIKE '%B%'` | Use: `WHERE (fact_name LIKE '%A%' OR fact_name LIKE '%B%')` |
| Only checking `fiscal_period='FY'` for revenue | Some companies report differently; also check `form_type='10-K'` |

---

## Output Structure

Always produce a comprehensive final report with a **"FINISH:"** prefix. Include:

```
FINISH:

## Company Overview
- Name, CIK, Ticker (Exchange), SIC code and description
- Entity type, Filer category, State of incorporation
- Fiscal year end, Address, Phone
- Former names (if any)

## Financial Performance (Most Recent FY + 3-year trend)
- Revenue: [values by year]
- Gross Profit / Operating Income
- Net Income
- EPS (Basic and Diluted)

## Balance Sheet (Most Recent)
- Total Assets
- Total Liabilities
- Stockholders' Equity
- Cash & Cash Equivalents
- Long-term Debt

## Cash Flow (Most Recent Annual)
- Operating / Investing / Financing

## Industry-Specific Metrics
[Metrics relevant to company's sector]

## SEC Filing Activity
- Total filings, key form types and counts
- Most recent 10-K date

## Key Observations
- Trends, notable changes, financial health indicators
```

---

## Multi-Year Trend Query Pattern

To get clean multi-year comparisons from 10-K filings:
```sql
SELECT
  end_date,
  MAX(CASE WHEN fact_name = 'Assets' THEN fact_value END) AS Assets,
  MAX(CASE WHEN fact_name = 'NetIncomeLoss' THEN fact_value END) AS NetIncome,
  MAX(CASE WHEN fact_name = 'StockholdersEquity' THEN fact_value END) AS Equity
FROM financial_facts
WHERE cik = '<CIK>'
  AND form_type = '10-K'
  AND fiscal_period = 'FY'
  AND fact_name IN ('Assets', 'NetIncomeLoss', 'StockholdersEquity')
GROUP BY end_date
ORDER BY end_date DESC
LIMIT 10
```
