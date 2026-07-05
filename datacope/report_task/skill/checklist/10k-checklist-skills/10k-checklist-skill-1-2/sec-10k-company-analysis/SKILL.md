---
name: sec-10k-company-analysis
description: Analyze a company in an SEC 10-K SQLite database and produce high-quality evidence-grounded financial QA pairs. Use this whenever the user asks to analyze a company by CIK/ticker, inspect 10-K financial trends, generate finance QA datasets, or work with filings/financial_facts tables.
---

# SEC 10-K Company Analysis

Use this skill to analyze one company from a SQLite SEC filings database and produce distinct, data-grounded QA pairs.

## Inputs you need
- Company identifier: CIK preferred (or ticker/name if unavailable).
- Database connection or path.
- Target output count if specified; otherwise produce **18–26 distinct QA pairs**.

## Required workflow

### Step 1: Schema discovery
Always inspect tables first before querying. Confirm exact column names — never assume aliases.

Key schema facts:
- `filings` table: columns are `cik`, `form`, `filing_date`, `report_date`, `accession_number` (NOT `form_type`)
- `financial_facts` table: columns include `fact_name`, `fact_value`, `unit`, `fiscal_year`, `fiscal_period`, `end_date`, `accession_number`, `form_type`, `dimension_segment`, `dimension_geography`
- If a query fails with "no such column", inspect the table schema and correct immediately — do not retry the same failing query.

### Step 2: Company identity and context
```sql
SELECT * FROM companies WHERE cik = '<CIK>'
SELECT cik, ticker, exchange FROM company_tickers WHERE cik = '<CIK>'
```
Note the SIC industry code — it governs which industry-specific metrics to prioritize in Steps 4–5.

### Step 3: Filing context — use the full available history
```sql
SELECT cik, form, filing_date, report_date, accession_number
FROM filings WHERE cik = '<CIK>' AND form = '10-K'
ORDER BY filing_date DESC LIMIT 15
```
Identify **all available 10-K filings**, not just the most recent 3–5. A longer time horizon enables richer comparisons (e.g., pre-crisis vs. post-crisis, pre-spinoff vs. post-spinoff). Use the full history in trend queries wherever data exists.

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
Revenue/income labels vary by company — discover actuals first, then use them. Also scan for industry-specific tags (e.g., `%Lease%`, `%Claims%`, `%Exploration%`, `%RemainingPerformanceObligation%`) based on the company's SIC code.

### Step 5: Pull evidence across two rounds

**Round A — Core multi-year trends** (query with `form_type = '10-K'`, ordered by `end_date`, spanning the full available history):
- Revenue, net income, operating income, gross profit
- Total assets, liabilities, stockholders' equity
- Operating cash flow, investing cash flow, financing cash flow
- Long-term debt, shares outstanding, diluted EPS
- Dividends per share, interest expense, income tax expense

**Round B — Detail and niche metrics** (pull what's available; skip silently if absent):
- Comprehensive income, accumulated OCI
- Working capital components: accounts receivable, inventory, accounts payable
- Debt carrying amount, weighted average interest rate, debt fair value
- Operating lease right-of-use assets, operating lease liabilities
- Depreciation and amortization (separate from combined D&A if available)
- Interest income (not just expense — relevant for cash-rich companies)
- Impairment charges, restructuring charges, goodwill and intangibles
- Share-based compensation, deferred revenue, deferred tax
- Segment or geography data (`dimension_segment`, `dimension_geography` filters)
- Industry-specific metrics: R&D expense (pharma/tech), benefits/claims expense (insurance), lease revenue (REITs), investment income (financial), DD&A and exploration expense (energy), capex intensity, remaining performance obligations (aerospace/defense/contract manufacturers), provisions for contract losses, asset retirement obligations (utilities/energy), environmental accruals

**Round C — Business context queries** (run these to understand the "why" behind metric changes):
```sql
-- Impairment and restructuring history — signals structural transformation
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Impairment%' OR fact_name LIKE '%Restructuring%'
     OR fact_name LIKE '%Goodwill%Impairment%')
ORDER BY end_date

-- Goodwill history — signals acquisition activity
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND fact_name LIKE '%Goodwill%'
ORDER BY end_date
```
Understanding structural events (acquisitions, divestitures, spinoffs, crises, regulatory changes) allows QA pairs to explain not just what changed but why — the most analytically valuable type of insight.

### Step 6: Generate QA pairs from evidence

**The core rule**: only submit a QA pair when the specific data points cited in the answer are present in your query results. Do not generalize beyond what was retrieved.

Submit a QA pair immediately when you have multi-datapoint support for a non-trivial conclusion. Keep exploring after each submission — aim for **18–26 distinct pairs** covering different angles.

**Before finalizing**, do a completeness sweep: review your query results and identify any significant findings (a trend, ratio shift, structural event, or comparison) that hasn't yet been captured in a QA pair. Each meaningful finding in your data deserves its own pair.

## QA angle checklist

Work through as many distinct angles as the data supports:

1. Revenue growth drivers, trajectory, and volatility
2. Profitability trajectory (operating income, net income, margins as % of revenue)
3. Earnings quality: operating cash flow vs. net income (OCF/net income ratio; divergence signals)
4. Capital allocation: dividends, buybacks, capex — what does the mix reveal about management priorities?
5. Balance sheet evolution: leverage, equity growth, asset mix
6. Debt profile: level, interest rate trajectory, maturity management, fair vs. carrying value
7. Liquidity: cash position, working capital components (AR, inventory, AP), current ratio
8. Per-share trends: EPS, dividend per share, share count (dilution or buyback)
9. Comprehensive income vs. net income (OCI items, forex exposure, pension adjustments)
10. Cost structure shifts: COGS, SG&A, R&D as % of revenue over time
11. D&A and capex as signals of asset intensity, growth investment, and capital cycle stage
12. Impairment and restructuring as transformation or risk signals
13. Tax dynamics: effective rate trend, deferred taxes, valuation allowances (signal of loss expectations)
14. Segment or geographic concentration (if data present)
15. Industry-specific metrics (claims ratio, R&D intensity, lease income, DD&A, exploration spending, RPO/backlog, contract loss provisions, etc.)
16. Lease obligations and right-of-use assets (both sides of the lease relationship)
17. Long-term obligations: pension/post-retirement benefits, AROs, environmental accruals
18. Deferred revenue and contract liability trends (signal of demand health or billing dynamics)
19. Goodwill and intangibles trajectory (signals acquisition history and impairment risk)
20. Historical anchoring: how does current performance compare to a prior peak, trough, or pre-event period (pre-crisis, pre-spinoff, pre-acquisition)?
21. Interest income and net interest position (especially for cash-rich companies)
22. Financing cash flow pattern: what does the composition (debt issuance, equity issuance, buybacks) reveal about financial flexibility?

Do not repeat the same thesis with different wording. Each QA should occupy a **distinct analytical position**.

## QA style

**Question form**: Prefer synthesis-oriented framing — "What does [metric trend] reveal about [business quality/risk/strategy/sustainability]?" over purely descriptive "How has X changed?" Both forms are acceptable, but synthesis questions produce richer answers and are harder to answer without the underlying evidence.

Good question examples:
- "What does EOG Resources' OCF-to-net-income ratio reveal about its earnings quality?"
- "How does ConocoPhillips' capex trajectory from 2020 to 2024 reflect its capital discipline strategy?"
- "What does Boeing's shift from positive to deeply negative free cash flow between 2018 and 2020 reveal about the operational and financial severity of the 737 MAX crisis?"

**Answer form**: 1–2 sentences. Lead with a concrete trend or comparison (include specific values and period references), then state the implication or business meaning. Limit to 3–4 numbers — prefer qualitative synthesis over numeric recaps.

Good example:
> q: How does AvalonBay's operating cash flow compare to its dividend obligations, and what does this indicate about sustainability?
> a: Operating cash flow of $1.61B in 2024 comfortably exceeds dividend payments of $969M (~1.65× coverage), and the pattern has held consistently from 2022–2024, indicating strong and sustainable dividend coverage.

Poor (too numeric, no synthesis):
> a: OCF was $1.61B in 2024, $1.52B in 2023, $1.42B in 2022. Dividends were $969M, $935M, $891M.

Poor (claim not in evidence — never submit without retrieved data):
> a: Operating margins improved from 15% to 22%, reflecting pricing power gains.  ← only submit this if you queried and retrieved those margin values.

## Edge-case handling

- **Missing expected metrics**: search for alternate `fact_name` values; never invent absent fields.
- **Empty results**: relax one filter at a time (remove accession constraint, widen date range, try alternate tag names).
- **Mixed annual/quarterly facts**: keep 10-K trend analysis annual-focused; filter by `form_type = '10-K'` and use `fiscal_period` if needed to isolate FY facts.
- **Duplicate facts for same period**: prefer the latest accession number; document only stable comparisons.
- **Query errors**: read the error, correct schema usage, and continue — do not retry the identical failing query.
- **Short filing history**: if fewer than 4 annual filings exist, note the limitation explicitly and focus QA on available periods.

## Output format

For each QA pair:
- `q`: one analytical question with clear scope and period.
- `a`: concise answer grounded in retrieved facts (values, direction, period, implication).

Quality bar:
- Evidence-grounded: every value cited was retrieved from the database in this session.
- Non-redundant: each pair occupies a distinct analytical angle.
- Specific: questions name the company, metric, and time period.
- Synthetic: answers explain what the data means, not just what it shows.
- Self-contained: answers are interpretable without additional context.
- Comprehensive: together, the pairs give a reader a full financial picture of the company across operational, balance sheet, cash flow, and strategic dimensions.
