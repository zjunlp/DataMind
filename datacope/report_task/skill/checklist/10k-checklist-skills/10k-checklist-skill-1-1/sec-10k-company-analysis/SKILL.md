---
name: sec-10k-company-analysis
description: Analyze a company in an SEC 10-K SQLite database and produce high-quality evidence-grounded financial QA pairs. Use this whenever the user asks to analyze a company by CIK/ticker, inspect 10-K financial trends, generate finance QA datasets, or work with filings/financial_facts tables.
---

# SEC 10-K Company Analysis

Use this skill to analyze one company from a SQLite SEC filings database and produce distinct, data-grounded QA pairs.

## Inputs you need
- Company identifier: CIK preferred (or ticker/name if unavailable).
- Database connection or path.
- Target output count if specified; otherwise produce **12–20 distinct QA pairs**.

## Required workflow

### Step 1: Schema discovery
Always inspect tables first before querying. Confirm exact column names — never assume aliases.

Key schema facts:
- `filings` table: columns are `cik`, `form`, `filing_date`, `report_date`, `accession_number` (NOT `form_type`)
- `financial_facts` table: columns include `fact_name`, `fact_value`, `unit`, `fiscal_year`, `fiscal_period`, `end_date`, `accession_number`, `form_type`, `dimension_segment`, `dimension_geography`
- If a query fails with "no such column", inspect the table schema and correct immediately — do not retry the same failing query.

### Step 2: Company identity
```sql
SELECT * FROM companies WHERE cik = '<CIK>'
SELECT cik, ticker, exchange FROM company_tickers WHERE cik = '<CIK>'
```

### Step 3: Filing context
```sql
SELECT cik, form, filing_date, report_date, accession_number
FROM filings WHERE cik = '<CIK>' AND form = '10-K'
ORDER BY filing_date DESC LIMIT 10
```
Identify the 3–5 most recent annual 10-K accession numbers for trend queries.

### Step 4: Metric discovery (do this before bulk queries)
```sql
-- All available fact names for this company
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
ORDER BY fact_name LIMIT 300

-- Revenue alias search
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%'
     OR fact_name LIKE '%ContractWithCustomer%')
```
Revenue/income labels vary by company — discover actuals first, then use them.

### Step 5: Pull evidence in two rounds

**Round A — Core multi-year trends** (query with `form_type = '10-K'`, ordered by `end_date`):
- Revenue, net income, operating income, gross profit
- Total assets, liabilities, stockholders' equity
- Operating cash flow, investing cash flow, financing cash flow
- Long-term debt, shares outstanding, diluted EPS
- Dividends per share, interest expense, income tax expense

**Round B — Detail and niche metrics** (pull what's available; skip silently if absent):
- Comprehensive income, accumulated OCI
- Working capital components: accounts receivable, inventory, accounts payable
- Debt carrying amount, weighted average interest rate, debt fair value
- Operating lease right-of-use assets, operating lease income
- Depreciation and amortization (separate from D&A combined if available)
- Impairment charges, restructuring charges
- Share-based compensation, deferred revenue, deferred tax
- Segment or geography data (`dimension_segment`, `dimension_geography` filters)
- Industry-specific metrics: R&D expense (pharma/tech), benefits/claims expense (insurance), lease revenue (REITs), investment income (financial), capex intensity

### Step 6: Generate QA pairs from evidence

Submit a QA pair immediately when you have multi-datapoint support for a non-trivial conclusion. Keep exploring after each submission — aim for **12–20 distinct pairs** covering different angles.

## QA angle checklist

Work through as many distinct angles as the data supports:

1. Revenue growth drivers and volatility
2. Profitability trajectory (operating income, net income, margins)
3. Earnings quality: cash flow vs accounting income (OCF vs net income gap)
4. Capital allocation: dividends, buybacks, capex balance
5. Balance sheet evolution: leverage, equity growth, asset mix
6. Debt profile: level, interest rate, maturity, fair vs carrying value
7. Liquidity: cash position, working capital components (AR, inventory, AP)
8. Per-share trends: EPS, dividend per share, share count trajectory
9. Comprehensive income vs net income (OCI items, forex, hedging)
10. Cost structure shifts: COGS, SG&A, R&D as % of revenue
11. D&A and capex as signals of asset intensity and growth investment
12. Impairment and restructuring as transformation/risk signals
13. Tax rate dynamics: effective rate, deferred taxes, tax benefits
14. Segment or geographic concentration (if data present)
15. Industry-specific metrics (claims ratio, R&D intensity, lease income, etc.)
16. Lease obligations and right-of-use assets
17. Pension / post-retirement benefit obligations (if material)
18. Deferred revenue and contract liability trends

Do not repeat the same thesis with different wording. Each QA should occupy a **distinct analytical position**.

## QA style

**Question form**: "How has X evolved from Y to Z?" or "What does [metric trend] reveal about [business quality]?" Questions should be specific enough to be graded against retrieved data, but broad enough to require synthesis.

**Answer form**: 1–2 sentences. Lead with a concrete trend or comparison (include specific values and period references), then state the implication. Do not include more than 3–4 numbers per answer — prefer qualitative synthesis over numeric recaps.

Good example:
> q: How does AvalonBay's operating cash flow compare to its dividend obligations, and what does this indicate about sustainability?
> a: Operating cash flow of $1.61B in 2024 comfortably exceeds dividend payments of $969M (~1.65× coverage), and the pattern has held consistently from 2022–2024, indicating strong and sustainable dividend coverage.

Poor (too numeric, no synthesis):
> a: OCF was $1.61B in 2024, $1.52B in 2023, $1.42B in 2022. Dividends were $969M, $935M, $891M.

## Edge-case handling

- **Missing expected metrics**: search for alternate `fact_name` values; never invent absent fields.
- **Empty results**: relax one filter at a time (remove accession constraint, widen date range, try alternate tag names).
- **Mixed annual/quarterly facts**: keep 10-K trend analysis annual-focused; filter by `form_type = '10-K'` and use `fiscal_period` if needed to isolate FY facts.
- **Duplicate facts for same period**: prefer the latest accession number; document only stable comparisons.
- **Query errors**: read the error, correct schema usage, and continue — do not retry the identical failing query.

## Output format

For each QA pair:
- `q`: one analytical question with clear scope and period.
- `a`: concise answer grounded in retrieved facts (values, direction, period, implication).

Quality bar:
- Evidence-grounded, non-redundant, specific.
- No unsupported claims or speculation.
- Answers interpretable without extra context.
- Covers enough distinct angles that a reader gains a comprehensive financial picture of the company.
