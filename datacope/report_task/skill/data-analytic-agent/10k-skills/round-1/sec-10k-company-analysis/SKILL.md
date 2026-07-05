---
name: sec-10k-company-analysis
description: >
  Comprehensive analysis of public companies using SEC EDGAR 10-K financial data stored in a SQLite database.
  Use this skill whenever the task involves analyzing a company by CIK (Central Index Key), querying SEC financial
  data, exploring a financial database with tables like companies, filings, financial_facts, or producing a
  structured financial analysis report. Covers schema navigation, metric discovery, industry-specific exploration,
  multi-year trend analysis, capital returns analysis, long-term obligations, and producing an insightful
  financial summary that connects metrics across dimensions.
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
-- Column is `form`, not `form_type`
SELECT form, COUNT(*) as count FROM filings
WHERE cik = '<CIK>' GROUP BY form ORDER BY count DESC

SELECT form, filing_date, report_date, accession_number
FROM filings WHERE cik = '<CIK>' AND form = '10-K'
ORDER BY filing_date DESC LIMIT 20
```

### Step 4: Discover Available Financial Metrics
```sql
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' ORDER BY fact_name LIMIT 100
```

### Step 5: Core Annual Metrics — use PIVOT queries for multi-year trends
Fetch 10–15 years of data. Long historical ranges reveal structural shifts, merger impacts,
and cyclical patterns that short windows miss.

```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'Assets' THEN fact_value END) AS Assets,
  MAX(CASE WHEN fact_name = 'Liabilities' THEN fact_value END) AS Liabilities,
  MAX(CASE WHEN fact_name = 'StockholdersEquity' THEN fact_value END) AS Equity,
  MAX(CASE WHEN fact_name = 'NetIncomeLoss' THEN fact_value END) AS NetIncome,
  MAX(CASE WHEN fact_name = 'OperatingIncomeLoss' THEN fact_value END) AS OperatingIncome,
  MAX(CASE WHEN fact_name = 'EarningsPerShareDiluted' THEN fact_value END) AS DilutedEPS,
  MAX(CASE WHEN fact_name = 'CashAndCashEquivalentsAtCarryingValue' THEN fact_value END) AS Cash
FROM financial_facts
WHERE cik = '<CIK>'
  AND form_type = '10-K'
  AND fiscal_period = 'FY'
  AND fact_name IN (
    'Assets', 'Liabilities', 'StockholdersEquity',
    'NetIncomeLoss', 'OperatingIncomeLoss',
    'EarningsPerShareBasic', 'EarningsPerShareDiluted',
    'CashAndCashEquivalentsAtCarryingValue'
  )
GROUP BY end_date
ORDER BY end_date DESC
LIMIT 15
```

### Step 6: Revenue Discovery (many companies use non-standard names)
```sql
-- Try common names first
SELECT fact_name, fact_value, fiscal_year, end_date
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN (
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'RevenueFromContractWithCustomerIncludingAssessedTax',
    'SalesRevenueNet', 'RevenuesNetOfInterestExpense'
  )
ORDER BY end_date DESC LIMIT 20

-- If empty, discover the actual name
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%')
ORDER BY fact_name LIMIT 30
```

### Step 7: Cash Flow Analysis
```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'NetCashProvidedByUsedInOperatingActivities' THEN fact_value END) AS OperatingCF,
  MAX(CASE WHEN fact_name = 'NetCashProvidedByUsedInInvestingActivities' THEN fact_value END) AS InvestingCF,
  MAX(CASE WHEN fact_name = 'NetCashProvidedByUsedInFinancingActivities' THEN fact_value END) AS FinancingCF,
  MAX(CASE WHEN fact_name = 'PaymentsToAcquirePropertyPlantAndEquipment' THEN fact_value END) AS Capex
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN (
    'NetCashProvidedByUsedInOperatingActivities',
    'NetCashProvidedByUsedInInvestingActivities',
    'NetCashProvidedByUsedInFinancingActivities',
    'PaymentsToAcquirePropertyPlantAndEquipment'
  )
GROUP BY end_date ORDER BY end_date DESC LIMIT 15
```

### Step 8: Debt & Capital Structure
```sql
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND fact_name LIKE '%Debt%' LIMIT 30

SELECT end_date,
  MAX(CASE WHEN fact_name = 'LongTermDebt' THEN fact_value END) AS LTDebt,
  MAX(CASE WHEN fact_name = 'LongTermDebtNoncurrent' THEN fact_value END) AS LTDebtNoncurrent,
  MAX(CASE WHEN fact_name = 'DebtCurrent' THEN fact_value END) AS CurrentDebt,
  MAX(CASE WHEN fact_name = 'InterestExpense' THEN fact_value END) AS InterestExpense
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN ('LongTermDebt', 'LongTermDebtNoncurrent', 'DebtCurrent', 'InterestExpense')
GROUP BY end_date ORDER BY end_date DESC LIMIT 15
```

### Step 9: Capital Returns (Share Repurchases, Dividends, Shares Outstanding)
This dimension is critical for understanding capital allocation maturity and EPS trajectory.
Declining share counts combined with net income growth creates compounding EPS expansion.

```sql
-- Discover available capital return metrics
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Repurchase%' OR fact_name LIKE '%Treasury%'
       OR fact_name LIKE '%Dividend%' OR fact_name LIKE '%SharesOut%')
LIMIT 30

SELECT end_date,
  MAX(CASE WHEN fact_name = 'PaymentsForRepurchaseOfCommonStock' THEN fact_value END) AS Buybacks,
  MAX(CASE WHEN fact_name = 'TreasuryStockValue' THEN fact_value END) AS TreasuryStock,
  MAX(CASE WHEN fact_name = 'CommonStockDividendsPerShareCashPaid' THEN fact_value END) AS DividendPerShare,
  MAX(CASE WHEN fact_name = 'EntityCommonStockSharesOutstanding' THEN fact_value END) AS SharesOutstanding
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN (
    'PaymentsForRepurchaseOfCommonStock', 'TreasuryStockValue',
    'CommonStockDividendsPerShareCashPaid', 'EntityCommonStockSharesOutstanding'
  )
GROUP BY end_date ORDER BY end_date DESC LIMIT 15
```

### Step 10: Goodwill, Intangibles, and Long-Term Obligations
Acquisition-driven companies carry substantial goodwill; impairments signal overvaluation.
Environmental, pension, and asset retirement obligations represent hidden long-term cash needs.

```sql
-- Goodwill and intangibles (key for M&A-heavy companies)
SELECT end_date,
  MAX(CASE WHEN fact_name = 'Goodwill' THEN fact_value END) AS Goodwill,
  MAX(CASE WHEN fact_name = 'IntangibleAssetsNetExcludingGoodwill' THEN fact_value END) AS Intangibles,
  MAX(CASE WHEN fact_name = 'GoodwillImpairmentLoss' THEN fact_value END) AS GoodwillImpairment,
  MAX(CASE WHEN fact_name = 'AmortizationOfIntangibleAssets' THEN fact_value END) AS Amortization
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN ('Goodwill', 'IntangibleAssetsNetExcludingGoodwill',
                    'GoodwillImpairmentLoss', 'AmortizationOfIntangibleAssets')
GROUP BY end_date ORDER BY end_date DESC LIMIT 15

-- Discover long-term obligation types for this company
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Environmental%'
       OR fact_name LIKE '%AssetRetirement%'
       OR fact_name LIKE '%Pension%'
       OR fact_name LIKE '%PostRetirement%'
       OR fact_name LIKE '%OperatingLease%')
LIMIT 40
```

### Step 11: Industry-Specific Metrics
After confirming the SIC code, discover what unique metrics exist for this company using LIKE patterns.
Then query the ones found with the pivot pattern from Steps 5–10.

**REIT / Real Estate (SIC 6500–6799)**:
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%RealEstate%' OR fact_name LIKE '%FundsFrom%'
       OR fact_name LIKE '%Rental%' OR fact_name LIKE '%LandAvailable%'
       OR fact_name LIKE '%NumberOfReal%' OR fact_name LIKE '%NumberOfUnit%')
LIMIT 30
```

**Oil & Gas / Mining (SIC 1000–1499, 1311, 2900)**:
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%AssetRetirement%' OR fact_name LIKE '%Depletion%'
       OR fact_name LIKE '%Exploration%' OR fact_name LIKE '%Environmental%'
       OR fact_name LIKE '%Proved%' OR fact_name LIKE '%Accretion%')
LIMIT 30
```

**Defense / Aerospace (SIC 3720–3812)**:
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%RemainingPerformance%' OR fact_name LIKE '%ContractWith%'
       OR fact_name LIKE '%Unbilled%' OR fact_name LIKE '%CustomerAdvance%'
       OR fact_name LIKE '%CostsInExcess%' OR fact_name LIKE '%InventoryNet%')
LIMIT 30
```

**Pharmaceutical / Biotech (SIC 2830–2836)**:
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Research%' OR fact_name LIKE '%Development%'
       OR fact_name LIKE '%Collaboration%' OR fact_name LIKE '%Milestone%')
LIMIT 30
```

**Financial Services / Banks (SIC 6000–6499)**:
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Interest%' OR fact_name LIKE '%Loan%'
       OR fact_name LIKE '%Deposit%' OR fact_name LIKE '%AllowanceFor%')
LIMIT 30
```

**Industrial / Technology (SIC 3000–3999, 7370–7379)**:
Focus on R&D expense, PP&E, inventory, operating segments, and acquisition-related goodwill.

---

## Common Pitfalls

| Mistake | Correct Approach |
|---------|-----------------|
| `SELECT DISTINCT tag FROM financial_facts` | Use `fact_name`, not `tag` |
| `GROUP BY form_type FROM filings` | `filings` uses `form`, not `form_type` |
| `SELECT * FROM filings` (no LIMIT) | Always add `LIMIT` to avoid huge results |
| Assuming `Revenues` exists | Try multiple names; use LIKE fallback |
| Only fetching 3-year trends | Extend to 10–15 years — structural patterns require it |
| Skipping capital returns (buybacks, dividends) | Always check Step 9 — drives EPS trajectory |
| Skipping environmental/ARO/pension discovery | These are material for industrial, energy, defense companies |
| Missing parentheses in OR conditions | `WHERE (fact_name LIKE '%A%' OR fact_name LIKE '%B%')` |
| Querying specific fact names before discovery | Run Step 4 first to know what's available |

---

## Analytical Synthesis

Strong analysis connects data across dimensions — not just listing each metric in isolation.
After gathering data, identify and explain these linkages:

**Capital allocation narrative**: How did improving operating cash flow change priorities over time?
(e.g., debt-heavy growth → debt reduction → share repurchases → EPS expansion)

**Operating leverage**: Is revenue growing faster or slower than operating income?
Compute: operating margin = OperatingIncome / Revenue for each year.

**Debt and coverage**: Is debt growth supported by earnings and cash flow?
Compute: interest coverage = OperatingIncome / InterestExpense; debt-to-equity trend.

**Balance sheet composition**: What drives asset growth — organic PP&E, acquisitions (goodwill), or financial assets?
Note goodwill as % of total assets; flag if goodwill > 40% as acquisition concentration risk.

**Shareholder returns mechanics**: Declining share count × rising net income → compounding diluted EPS.
Connect buyback amounts to share count reduction to EPS trajectory explicitly.

**Historical inflection points**: Identify years where metrics shifted sharply (mergers, downturns, business model changes).
Long-term data (10+ years) often reveals these better than short windows.

Always include specific dollar amounts, year ranges, and percentage changes in insights.

---

## Output Structure

Always produce a comprehensive final report with a **"FINISH:"** prefix. Aim for 5–10 year trends.
Include specific dollar amounts, percentages, and multi-dimensional analytical observations.

```
FINISH:

## Company Overview
- Name, CIK, Ticker (Exchange), SIC code and description
- Entity type, Filer category, State of incorporation
- Fiscal year end, Address, Phone, Website
- Former names (if any)

## Financial Performance (5–10 year trend)
- Revenue: [values by year with % YoY change]
- Operating Income and margin (%)
- Net Income
- EPS (Diluted, multi-year trend)

## Balance Sheet Composition (5-year trend)
- Total Assets vs. Liabilities vs. Stockholders' Equity
- Cash & Cash Equivalents
- Goodwill / Intangibles (if significant — note % of total assets)
- Long-term Debt (with interest expense trend)

## Cash Flow & Capital Allocation (5-year trend)
- Operating / Investing / Financing cash flows
- Capital expenditures
- Share repurchases (annual amounts)
- Dividends per share (trend)
- Shares outstanding (trend — connects to EPS impact)

## Long-Term Obligations (where applicable)
- Environmental loss contingencies
- Asset retirement obligations
- Pension / post-retirement liabilities
- Operating lease liabilities

## Industry-Specific Metrics
[Sector-relevant metrics with historical trend and interpretation]

## SEC Filing Activity
- Total filings, key form types and counts
- Most recent 10-K date

## Key Analytical Observations
- Capital allocation evolution with dollar amounts and timeframes
- Operating leverage trends (revenue vs. profit growth rates)
- Debt trajectory and interest coverage
- Shareholder returns mechanics (buybacks → share count → EPS)
- Industry-specific strategic observations
- Any historical inflection points (mergers, impairments, crises)
```

---

## Multi-Year Pivot Query Pattern

Use this pattern throughout the workflow — it's more efficient than separate queries
and produces clean year-by-year comparison tables:

```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'MetricA' THEN fact_value END) AS MetricA,
  MAX(CASE WHEN fact_name = 'MetricB' THEN fact_value END) AS MetricB
FROM financial_facts
WHERE cik = '<CIK>'
  AND form_type = '10-K'
  AND fiscal_period = 'FY'
  AND fact_name IN ('MetricA', 'MetricB')
GROUP BY end_date
ORDER BY end_date DESC
LIMIT 15
```
