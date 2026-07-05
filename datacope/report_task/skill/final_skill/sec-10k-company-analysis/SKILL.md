---
name: sec-10k-company-analysis
description: >
  Comprehensive analysis of public companies using SEC EDGAR 10-K financial data stored in a SQLite database.
  Use this skill whenever the task involves analyzing a company by CIK (Central Index Key), querying SEC financial
  data, exploring a financial database with tables like companies, filings, financial_facts, or producing a
  structured financial analysis report. Covers schema navigation, metric discovery, industry-specific exploration,
  multi-year trend analysis, income statement structure, capital returns analysis, long-term obligations, and
  producing an insightful financial summary that connects metrics across dimensions.
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
Call `get_database_info()` then `describe_table()` for `companies` and `financial_facts` to confirm column names.

### Step 2: Company Basics
```sql
SELECT * FROM companies WHERE cik = '<CIK>';
SELECT * FROM company_tickers WHERE cik = '<CIK>';
SELECT * FROM company_addresses WHERE cik = '<CIK>';
```

### Step 3: Filing History
```sql
-- Column is `form`, not `form_type`
SELECT form, COUNT(*) as count FROM filings WHERE cik = '<CIK>' GROUP BY form ORDER BY count DESC;
SELECT form, filing_date, report_date FROM filings WHERE cik = '<CIK>' AND form = '10-K'
ORDER BY filing_date DESC LIMIT 20;
```

### Step 4: Discover Available Financial Metrics
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>' ORDER BY fact_name LIMIT 100;
```

### Step 5: Core Annual Metrics — use PIVOT queries for multi-year trends
Fetch **10–15 years** of data. Long ranges reveal structural shifts, merger impacts, and cyclical patterns.

```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'Assets' THEN fact_value END) AS Assets,
  MAX(CASE WHEN fact_name = 'AssetsCurrent' THEN fact_value END) AS CurrentAssets,
  MAX(CASE WHEN fact_name = 'Liabilities' THEN fact_value END) AS Liabilities,
  MAX(CASE WHEN fact_name = 'LiabilitiesCurrent' THEN fact_value END) AS CurrentLiabilities,
  MAX(CASE WHEN fact_name = 'StockholdersEquity' THEN fact_value END) AS Equity,
  MAX(CASE WHEN fact_name = 'NetIncomeLoss' THEN fact_value END) AS NetIncome,
  MAX(CASE WHEN fact_name = 'OperatingIncomeLoss' THEN fact_value END) AS OperatingIncome,
  MAX(CASE WHEN fact_name = 'IncomeTaxExpenseBenefit' THEN fact_value END) AS TaxExpense,
  MAX(CASE WHEN fact_name = 'EarningsPerShareDiluted' THEN fact_value END) AS DilutedEPS,
  MAX(CASE WHEN fact_name = 'CashAndCashEquivalentsAtCarryingValue' THEN fact_value END) AS Cash
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN (
    'Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent', 'StockholdersEquity',
    'NetIncomeLoss', 'OperatingIncomeLoss', 'IncomeTaxExpenseBenefit',
    'EarningsPerShareBasic', 'EarningsPerShareDiluted',
    'CashAndCashEquivalentsAtCarryingValue'
  )
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

If `Liabilities` is null for all years, compute it as `Assets − Equity` inline.

### Step 6: Revenue Discovery (many companies use non-standard names)
```sql
-- Try common names first
SELECT fact_name, fact_value, fiscal_year, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
  AND fact_name IN (
    'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
    'RevenueFromContractWithCustomerIncludingAssessedTax',
    'SalesRevenueNet', 'RevenuesNetOfInterestExpense'
  )
ORDER BY end_date DESC LIMIT 20;

-- If empty, discover the actual name
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%')
ORDER BY fact_name LIMIT 30;
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
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

### Step 8: Debt & Capital Structure
```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>' AND fact_name LIKE '%Debt%' LIMIT 30;

SELECT end_date,
  MAX(CASE WHEN fact_name = 'LongTermDebt' THEN fact_value END) AS LTDebt,
  MAX(CASE WHEN fact_name = 'LongTermDebtNoncurrent' THEN fact_value END) AS LTDebtNoncurrent,
  MAX(CASE WHEN fact_name = 'DebtCurrent' THEN fact_value END) AS CurrentDebt,
  MAX(CASE WHEN fact_name = 'InterestExpense' THEN fact_value END) AS InterestExpense
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN ('LongTermDebt', 'LongTermDebtNoncurrent', 'DebtCurrent', 'InterestExpense')
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

### Step 9: Capital Returns (Share Repurchases, Dividends, Shares Outstanding)
Declining share counts combined with net income growth creates compounding EPS expansion.

```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Repurchase%' OR fact_name LIKE '%Treasury%'
       OR fact_name LIKE '%Dividend%' OR fact_name LIKE '%SharesOut%') LIMIT 30;

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
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

### Step 10: Goodwill, Intangibles, and Long-Term Obligations
Acquisition-driven companies carry substantial goodwill; impairments signal overvaluation.

```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'Goodwill' THEN fact_value END) AS Goodwill,
  MAX(CASE WHEN fact_name = 'IntangibleAssetsNetExcludingGoodwill' THEN fact_value END) AS Intangibles,
  MAX(CASE WHEN fact_name = 'GoodwillImpairmentLoss' THEN fact_value END) AS GoodwillImpairment,
  MAX(CASE WHEN fact_name = 'AmortizationOfIntangibleAssets' THEN fact_value END) AS Amortization
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN ('Goodwill', 'IntangibleAssetsNetExcludingGoodwill',
                    'GoodwillImpairmentLoss', 'AmortizationOfIntangibleAssets')
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;

-- Discover long-term obligation types
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%Environmental%' OR fact_name LIKE '%AssetRetirement%'
       OR fact_name LIKE '%Pension%' OR fact_name LIKE '%PostRetirement%'
       OR fact_name LIKE '%OperatingLease%') LIMIT 40;
```

### Step 11: Industry-Specific Metrics
After confirming the SIC code, discover unique metrics with LIKE patterns, then pivot-query those found.

**REIT / Real Estate (SIC 6500–6799)**:
`fact_name LIKE '%RealEstate%' OR '%FundsFrom%' OR '%Rental%' OR '%NumberOfReal%'`

**Oil & Gas / Mining (SIC 1000–1499, 1311, 2900)**:
`fact_name LIKE '%AssetRetirement%' OR '%Depletion%' OR '%Exploration%' OR '%Proved%'`

**Defense / Aerospace (SIC 3720–3812)**:
`fact_name LIKE '%RemainingPerformance%' OR '%ContractWith%' OR '%Unbilled%' OR '%CustomerAdvance%'`

**Pharmaceutical / Biotech (SIC 2830–2836)**:
`fact_name LIKE '%Research%' OR '%Development%' OR '%Collaboration%' OR '%Milestone%'`

**Financial Services / Banks (SIC 6000–6499)**:
`fact_name LIKE '%Interest%' OR '%Loan%' OR '%Deposit%' OR '%AllowanceFor%'`

**Software / SaaS (SIC 7370–7379)**: Focus on `ContractWithCustomerLiability` (deferred revenue),
remaining performance obligations, available-for-sale securities, and stock-based compensation.

**Industrial / Technology (SIC 3000–3999)**: Focus on R&D expense, PP&E, inventory, and acquisition goodwill.

### Step 12: Income Statement Structure (Cost, Expenses, SBC)
Compute gross margin and understand the full expense stack. Reveals operating leverage and investment intensity.

```sql
SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>'
  AND (fact_name LIKE '%CostOf%' OR fact_name LIKE '%SellingGeneral%'
       OR fact_name LIKE '%ResearchAndDevelop%' OR fact_name LIKE '%DepreciationDepletion%'
       OR fact_name LIKE '%AllocatedShareBased%' OR fact_name LIKE '%Restructuring%') LIMIT 30;

SELECT end_date,
  MAX(CASE WHEN fact_name = 'CostOfGoodsAndServicesSold' THEN fact_value END) AS COGS,
  MAX(CASE WHEN fact_name = 'SellingGeneralAndAdministrativeExpense' THEN fact_value END) AS SGA,
  MAX(CASE WHEN fact_name = 'ResearchAndDevelopmentExpense' THEN fact_value END) AS RD,
  MAX(CASE WHEN fact_name = 'DepreciationDepletionAndAmortization' THEN fact_value END) AS DDA,
  MAX(CASE WHEN fact_name = 'AllocatedShareBasedCompensationExpense' THEN fact_value END) AS SBC
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN (
    'CostOfGoodsAndServicesSold', 'SellingGeneralAndAdministrativeExpense',
    'ResearchAndDevelopmentExpense', 'DepreciationDepletionAndAmortization',
    'AllocatedShareBasedCompensationExpense'
  )
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

If `CostOfGoodsAndServicesSold` is null, try `CostOfGoodsSold` or search with `LIKE '%CostOf%Sold%'`.

### Step 13: Balance Sheet Detail (Retained Earnings, AOCI, PP&E, Comprehensive Income)
These dimensions reveal cumulative profit/payout history, unrealized FX/pension impacts, and whether
comprehensive income diverges materially from net income.

```sql
SELECT end_date,
  MAX(CASE WHEN fact_name = 'RetainedEarningsAccumulatedDeficit' THEN fact_value END) AS RetainedEarnings,
  MAX(CASE WHEN fact_name = 'AccumulatedOtherComprehensiveIncomeLossNetOfTax' THEN fact_value END) AS AOCI,
  MAX(CASE WHEN fact_name = 'ComprehensiveIncomeNetOfTax' THEN fact_value END) AS ComprehensiveIncome,
  MAX(CASE WHEN fact_name = 'PropertyPlantAndEquipmentNet' THEN fact_value END) AS PPE_Net
FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
  AND fact_name IN (
    'RetainedEarningsAccumulatedDeficit',
    'AccumulatedOtherComprehensiveIncomeLossNetOfTax',
    'ComprehensiveIncomeNetOfTax',
    'PropertyPlantAndEquipmentNet'
  )
GROUP BY end_date ORDER BY end_date DESC LIMIT 15;
```

---

## Common Pitfalls

| Mistake | Correct Approach |
|---------|-----------------|
| `SELECT DISTINCT tag FROM financial_facts` | Use `fact_name`, not `tag` |
| `GROUP BY form_type FROM filings` | `filings` uses `form`, not `form_type` |
| Assuming `Revenues` exists | Try multiple names; use LIKE fallback |
| Only fetching 3–5 year trends | Extend to 10–15 years — structural patterns require it |
| Skipping capital returns (buybacks, dividends) | Always run Step 9 — drives EPS trajectory |
| Skipping income statement structure | Always run Step 12 — COGS, SG&A, R&D reveal cost story |
| `Liabilities` null for all years | Compute as Assets − Equity, or query components |
| Skipping retained earnings and AOCI | Run Step 13 — retained earnings shows payout history |
| `fiscal_period = 'FY'` still mixes quarterly data | For December year-end, add `AND strftime('%m-%d', end_date) = '12-31'` |
| No LIMIT on SELECT * | Always add LIMIT to avoid huge results |

---

## Computed Ratios — Calculate Per Year Across the Full Trend

After collecting raw data, explicitly compute these derived metrics for each year and include them in insights
and the final report. These ratios are the connective tissue of strong analysis.

| Ratio | Formula | What It Reveals |
|-------|---------|-----------------|
| Operating margin | OperatingIncome / Revenue | Profitability leverage vs. growth |
| Gross margin | (Revenue − COGS) / Revenue | Pricing power and cost structure |
| OCF / Net Income | OperatingCF / NetIncome | Earnings quality; >1× = non-cash charges dominate |
| Current ratio | CurrentAssets / CurrentLiabilities | Near-term liquidity; <1× signals stress |
| Interest coverage | OperatingIncome / InterestExpense | Debt service safety margin |
| Debt / Equity | (LTDebt + CurrentDebt) / Equity | Leverage trend |
| Free cash flow | OperatingCF − Capex | Cash available after maintenance and growth investment |
| Capex / DD&A | Capex / DepreciationAmortization | >1× = net capacity expansion |
| SBC as % revenue | SBC / Revenue | Dilution cost for growth companies |
| Effective tax rate | TaxExpense / (NetIncome + TaxExpense) | Tax efficiency and one-time impacts |

Compute these ratios for **each available year**, not just the most recent. Trends in ratios (improving vs.
deteriorating) are more analytically valuable than point-in-time snapshots.

---

## Analytical Synthesis

Strong analysis connects data across dimensions — not just listing each metric in isolation. After gathering
data and computing ratios, identify and explain these linkages with specific dollar amounts and year ranges:

**Capital allocation narrative**: How did improving (or declining) operating cash flow change priorities over
time? E.g., debt-heavy growth → debt reduction → share repurchases → EPS expansion. Connect buyback amounts
to share count reduction to diluted EPS trajectory explicitly, year by year.

**Operating leverage**: Is revenue growing faster or slower than operating income? Report operating margin for
each year — expanding margins signal leverage; compressing margins signal cost pressure.

**Gross margin and cost structure**: Compute gross margin per year. Explain whether SG&A or R&D is growing
as a % of revenue (investment phase vs. harvest phase). Express SBC as % of revenue for growth companies.

**Earnings quality (OCF/NI ratio)**: Compute per year across the trend. Values consistently >1× indicate
non-cash charges dominate (depreciation, amortization, SBC). Values <1× signal working capital consumption
or aggressive accruals. Changes in this ratio often precede earnings quality issues.

**AOCI and comprehensive income**: If ComprehensiveIncome diverges materially from NetIncome, explain the
source — FX translation losses, pension remeasurement, or unrealized securities gains/losses. Persistent
negative AOCI signals cumulative foreign exposure or underfunded pension obligations.

**Debt and coverage**: Is debt growth supported by earnings and cash flow? Report interest coverage per year.
Note any debt spikes linked to acquisitions (cross-reference goodwill jumps in same year).

**Balance sheet composition**: What drives asset growth — organic PP&E, acquisitions (goodwill spike), or
financial assets? Note goodwill as % of total assets; flag if >40% as acquisition concentration risk.

**Working capital and liquidity**: Report current ratio per year. Deterioration below 1.0× signals near-term
stress. Connect to cash flow trends to explain whether it's structural or temporary.

**Retained earnings trajectory**: Growing retained earnings = earnings exceed distributions. Erosion or
negative retained earnings = aggressive buybacks/dividends exceeded cumulative earnings.

**Historical inflection points**: Identify years where metrics shifted sharply (acquisitions, divestitures,
downturns, regulatory changes). Long-term data (10+ years) surfaces these. State the year, the metric change,
and the likely cause. E.g.: "Goodwill jumped from $X to $Y in [year], consistent with the [acquisition]
announced in [year]; long-term debt rose simultaneously by $Z to fund the deal."

**Each insight should stand alone**: Include specific dollar amounts, year references, percentage changes, and
at least one cross-metric connection. Avoid observations that name only one metric without context.

---

## Output Structure

Always produce a comprehensive final report with a **"FINISH:"** prefix. Aim for 10–15 year trends.
Include specific dollar amounts, percentages, computed ratios, and multi-dimensional observations.

```
FINISH:

## Company Overview
- Name, CIK, Ticker (Exchange), SIC code and description
- Entity type, Filer category, State of incorporation
- Fiscal year end, Address, Phone, Website
- Former names (if any)

## Financial Performance (10–15 year trend)
- Revenue: [values by year with % YoY change]
- Gross Margin (%) per year [where COGS is available]
- Operating Income and margin (%) per year
- Net Income and Comprehensive Income (note divergence if material)
- Effective Tax Rate per year [TaxExpense / (NetIncome + TaxExpense)]
- EPS (Diluted, multi-year trend)

## Balance Sheet Composition (5–10 year trend)
- Total Assets vs. Liabilities vs. Stockholders' Equity
- Cash & Cash Equivalents
- Current Assets / Current Liabilities → current ratio per year
- Retained Earnings trajectory (positive growth or erosion?)
- AOCI trend (persistent negative = FX/pension exposure; explain source)
- Goodwill / Intangibles (% of total assets; spike years linked to acquisitions)
- PP&E Net (for capital-intensive sectors)
- Long-term Debt (with interest expense and coverage ratio per year)

## Income Statement Structure (expense stack)
- COGS and Gross Margin (%) per year
- SG&A expense (as % of revenue trend)
- R&D expense (as % of revenue trend, where applicable)
- D&A expense (non-cash weight vs. operating income)
- Stock-Based Compensation (% of revenue; dilution cost)
- Restructuring / impairment charges (if recurring or material)

## Cash Flow & Capital Allocation (5–10 year trend)
- Operating / Investing / Financing cash flows
- Earnings quality: OCF / Net Income ratio per year
- Free cash flow = OCF − Capex per year
- Capex / DD&A ratio (net capacity expansion indicator)
- Share repurchases (annual amounts and cumulative)
- Dividends per share (trend)
- Shares outstanding (trend — connect to EPS impact with $ amounts)

## Long-Term Obligations (where applicable)
- Environmental loss contingencies
- Asset retirement obligations
- Pension / post-retirement liabilities
- Operating lease liabilities

## Industry-Specific Metrics
[Sector-relevant metrics with historical trend, computed ratios, and interpretation]

## SEC Filing Activity
- Total filings, key form types and counts
- Most recent 10-K date

## Key Analytical Observations
- Capital allocation narrative: how strategy evolved (debt → buybacks → dividends) with dollar amounts
- Operating leverage trends (revenue vs. operating income growth rates)
- Gross margin trajectory and cost structure evolution
- Earnings quality (OCF/NI ratio trend — improving or deteriorating?)
- AOCI and comprehensive income divergence — source and magnitude
- Debt trajectory, leverage ratio, and interest coverage evolution
- Working capital / current ratio trend (tightening or improving?)
- Retained earnings trajectory (cumulative earnings vs. distributions)
- Shareholder returns mechanics: buyback $ → share count reduction → EPS amplification
- Historical inflection points with year, metric change, and cause
- Industry-specific strategic observations
```
