---
name: sec-10k-company-analysis
description: Analyze a company in an SEC 10-K SQLite database and produce high-quality evidence-grounded financial QA pairs. Use this whenever the user asks to analyze a company by CIK/ticker, inspect 10-K financial trends, generate finance QA datasets, or work with filings/financial_facts tables.
---

# SEC 10-K Company Analysis

Use this skill to analyze one company from a SQLite SEC filings database and produce distinct, data-grounded QA pairs.

## Inputs you need
- Company identifier: preferably CIK (or ticker/name if CIK is unavailable).
- Database location (or the active DB MCP connection).
- Target output count if provided; otherwise produce 8-12 distinct QA pairs.

## Required workflow

1) Discover schema and confirm table/column names before deep queries.
- Always inspect available tables first.
- For `financial_facts`, confirm key columns such as:
  - `fact_name`, `fact_value`, `unit`, `fiscal_year`, `fiscal_period`, `end_date`, `accession_number`, `form_type`, `filed_date`, `dimension_segment`, `dimension_geography`.
- Never assume aliases like `tag`; use real column names.

2) Resolve the company identity.
- Query `companies` for base profile (name, SIC, description, fiscal year-end).
- If needed, query `company_tickers` for ticker mapping.

3) Build filing context.
- Query `filings` for recent records and isolate 10-K accession numbers.
- Keep a small set of recent annual filings (for trend analysis) plus the latest 10-K (for detail drills).

4) Discover available metrics before forcing templates.
- Enumerate candidate `fact_name` values for the company and latest 10-K.
- Revenue/profit labels vary by issuer; search for alternatives first, then lock in the best available tags.

5) Pull evidence in structured batches.
- Trend batch (multi-year): revenue, net income, assets, liabilities, equity, operating cash flow, financing/investing cash flow, debt, shares.
- Detail batch (latest filing): profitability, cost structure, liquidity, leverage, capital returns, leases, taxes, segment/geography if present.
- Use accession and date filters to avoid mixing inconsistent periods.

6) Draft QA pairs only from observed evidence.
- Each QA should cover a distinct analytical angle.
- Include concrete values/trends in the answer.
- Avoid speculation and avoid repeating the same thesis with different wording.

## Query strategy patterns

Use patterns like these and adapt to actual schema:

- Company lookup
  - `SELECT * FROM companies WHERE cik = '<CIK>'`

- Recent annual filings
  - `SELECT cik, form, filing_date, report_date, accession_number FROM filings WHERE cik = '<CIK>' AND form = '10-K' ORDER BY filing_date DESC LIMIT 10`

- Metric discovery
  - `SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K' ORDER BY fact_name LIMIT 200`

- Revenue alias discovery
  - `SELECT DISTINCT fact_name FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K' AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%' OR fact_name LIKE '%ContractWithCustomer%') ORDER BY fact_name`

- Multi-year core metrics
  - `SELECT fact_name, fact_value, unit, end_date, accession_number FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K' AND fact_name IN (...) ORDER BY accession_number, fact_name, end_date`

- Optional segment/geography coverage
  - `SELECT DISTINCT dimension_segment, fact_name, fact_value, unit, end_date FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K' AND dimension_segment IS NOT NULL AND dimension_segment != '' LIMIT 100`
  - `SELECT DISTINCT dimension_geography, fact_name, fact_value, unit, end_date FROM financial_facts WHERE cik = '<CIK>' AND form_type = '10-K' AND dimension_geography IS NOT NULL AND dimension_geography != '' LIMIT 100`

## Distinct QA angle checklist

Pick non-overlapping angles based on available evidence:
- Profitability trajectory and margin resilience
- Revenue growth/volatility drivers
- Cash flow quality vs accounting earnings
- Capital allocation (dividends, repurchases, capex)
- Balance sheet strength and debt servicing capacity
- Liquidity and working capital dynamics
- Cost structure shifts
- Share count / per-share implications
- Segment or geographic concentration (if present)
- Industry-specific risks visible in filings (e.g., lease burden, commodity cyclicality, impairment patterns)

## Edge-case handling

- Missing expected metrics:
  - Discover alternates by `fact_name` search; do not invent absent fields.
- Empty result sets:
  - Relax one filter at a time (date, accession, strict metric list), then re-check.
- Mixed quarterly and annual facts:
  - Keep 10-K trend analysis annual-focused; avoid blending 10-Q values unless explicitly requested.
- Duplicated facts for same period:
  - Prefer consistent unit/date/accession combinations and document only stable comparisons.
- Tool/query errors:
  - Correct query using schema feedback and continue; do not repeat the same failing query.

## Output format

For each QA pair:
- `q`: one analytical question with clear scope.
- `a`: concise answer grounded in retrieved facts (values, direction, period).

Quality bar:
- Evidence-grounded, specific, non-redundant.
- No unsupported claims.
- Answers should be interpretable without extra context.