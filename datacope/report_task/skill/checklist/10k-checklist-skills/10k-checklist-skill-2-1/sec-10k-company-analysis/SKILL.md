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
Identify **all available 10-K filings**. A longer time horizon enables richer comparisons (e.g., pre-crisis vs. post-crisis, pre-spinoff vs. post-spinoff). Use the full history in trend queries wherever data exists.

### Step 4: Metric discovery (do this before bulk queries)
```sql
-- All available fact names for this company (paginate with OFFSET if needed)
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
ORDER BY fact_name LIMIT 300

-- Revenue alias search
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Revenue%' OR fact_name LIKE '%Sales%'
     OR fact_name LIKE '%ContractWithCustomer%')
```
Revenue/income labels vary by company — discover actuals first, then use them. Also scan for industry-specific tags based on the company's SIC code.

**Annual data filter**: Add `AND fiscal_period = 'FY'` to evidence queries to isolate full-year facts and avoid quarterly contamination when pulling trend data.

### Step 5: Pull evidence across four rounds

**Round A — Core multi-year trends** (filter `form_type = '10-K'` and `fiscal_period = 'FY'`, order by `end_date`, spanning the full available history):
- Revenue, net income, operating income, gross profit
- Total assets, liabilities, stockholders' equity
- Operating cash flow, investing cash flow, financing cash flow
- Long-term debt, shares outstanding, diluted EPS
- Dividends per share, interest expense, income tax expense

**Round B — Detail and niche metrics** (pull what's available; skip silently if absent):
- Comprehensive income, accumulated OCI (`AccumulatedOtherComprehensiveIncomeLossNetOfTax`)
- Working capital components: accounts receivable, inventory, accounts payable, current assets, current liabilities
- **Working capital changes from OCF statement**: `IncreaseDecreaseInAccountsReceivable`, `IncreaseDecreaseInInventories`, `IncreaseDecreaseInAccountsPayable`, `IncreaseDecreaseInDeferredRevenue` — these reveal cash conversion dynamics beyond balance sheet levels
- Debt carrying amount vs. fair value: `DebtInstrumentCarryingAmount`, `LongTermDebtFairValue`, `DebtInstrumentFairValue`, `LongTermDebtWeightedAverageInterestRateAtPointInTime`
- Operating lease right-of-use assets, operating lease liabilities, finance lease assets and liabilities (query separately — both sides matter)
- Depreciation and amortization (separate from combined D&A if available)
- Interest income (relevant for cash-rich companies), deferred revenue, deferred tax
- Impairment charges, restructuring charges, goodwill and intangibles
- Share-based compensation, retained earnings
- Segment or geography data (`dimension_segment`, `dimension_geography` filters)
- Industry-specific: R&D expense (pharma/tech), benefits/claims expense (insurance), lease revenue (REITs), investment income (financial), DD&A and exploration expense (energy), capex intensity, remaining performance obligations (aerospace/defense/contract manufacturers), asset retirement obligations (utilities/energy), environmental accruals

**Round C — Business context and structural events**:
```sql
-- Impairment and restructuring history
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
AND (fact_name LIKE '%Impairment%' OR fact_name LIKE '%Restructuring%')
ORDER BY end_date

-- Goodwill history — signals acquisition activity
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND fact_name LIKE '%Goodwill%' ORDER BY end_date

-- Advertising and brand investment
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Advertising%' OR fact_name LIKE '%MarketingExpense%')
ORDER BY end_date

-- Segment structure changes
SELECT DISTINCT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND fact_name IN ('NumberOfOperatingSegments', 'NumberOfReportableSegments')
ORDER BY end_date
```

**Round D — Deep-dive niche metrics** (probe these to unlock hard-to-replicate QA angles):
```sql
-- Customer concentration risk
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Concentration%' OR fact_name LIKE '%MajorCustomer%')
ORDER BY end_date

-- Derivative and hedge activity
SELECT DISTINCT fact_name FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%Derivative%' OR fact_name LIKE '%Hedging%'
     OR fact_name LIKE '%HedgeGainLoss%')

-- Debt extinguishment
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%GainLossOnRepurchase%' OR fact_name LIKE '%ExtinguishmentOfDebt%'
     OR fact_name LIKE '%DebtIssuanceCosts%')
ORDER BY end_date

-- Equity method investment income
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND (fact_name LIKE '%EquityMethodInvestment%' OR fact_name LIKE '%IncomeLossFromEquityMethod%')
ORDER BY end_date

-- FX effects on cash
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND fact_name LIKE '%EffectOfExchangeRate%'
ORDER BY end_date

-- Nonoperating income and gains/losses from asset sales
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K' AND fiscal_period = 'FY'
AND (fact_name LIKE '%GainLossOnSale%' OR fact_name LIKE '%NonoperatingIncome%'
     OR fact_name LIKE '%OtherNonoperating%')
ORDER BY end_date

-- Allowance for doubtful accounts / credit risk
SELECT fact_name, fact_value, end_date FROM financial_facts
WHERE cik = '<CIK>' AND form_type = '10-K'
AND fact_name LIKE '%AllowanceForDoubtful%'
ORDER BY end_date
```

Understanding structural events (acquisitions, divestitures, spinoffs, crises, segment reorganizations, derivative programs) allows QA pairs to explain not just what changed but why.

### Step 6: Generate QA pairs from evidence

**The core rule**: only submit a QA pair when the specific data points cited in the answer are present in your query results. Do not generalize beyond what was retrieved.

**Before submitting any QA pair, verify:**
1. All cited values are exact retrieved figures — not approximations, ranges, or estimates. "approximately $11B" or "$7.4B–$8.3B" are not acceptable when exact figures are available. If a precise value was not retrieved, do not submit the pair.
2. The answer explains both WHAT happened (with exact data) AND WHY it matters (business interpretation). A pure list of numbers with no interpretation, or an interpretation with no grounding data, both fail this bar.
3. The angle has not already been covered by a submitted QA pair. Two questions about lease obligations from different phrasings still occupy the same analytical position — pick the stronger framing and discard the weaker.
4. The question and answer are specific to this company's situation — not generic statements that could apply to any firm in the sector.

Submit a QA pair immediately when you have multi-datapoint support for a non-trivial conclusion. Keep exploring after each submission — aim for **18–26 distinct pairs** covering different angles.

**Identify this company's defining story**: Before generating pairs, ask what makes this company's financial trajectory distinctive — a pandemic revenue cycle, a major acquisition, a regulatory restructuring, heavy infrastructure investment, a delevering campaign. Anchor several QA pairs to these defining narratives; they produce the most analytically valuable pairs and are impossible to generate without close reading of the data.

**Before finalizing**, do a completeness sweep: review your query results and identify any significant findings (a trend, ratio shift, structural event, or comparison) not yet captured. Each meaningful finding deserves its own pair. Niche metrics (AOCI, hedge activity, customer concentration, debt extinguishment, FX effects) often yield the most analytically distinctive pairs — do not skip them when data is available. Then review all submitted pairs for angle overlap and merge or drop any redundant ones.

## QA angle checklist

Work through as many distinct angles as the data supports. Each angle should occupy a **distinct analytical position** — do not generate multiple QA pairs on the same theme with different wording:

1. Revenue growth drivers, trajectory, and volatility
2. Profitability trajectory (operating income, net income, margins as % of revenue)
3. Earnings quality: operating cash flow vs. net income (OCF/net income ratio; divergence signals)
4. Capital allocation: dividends, buybacks, capex — what does the mix reveal about management priorities?
5. Balance sheet evolution: leverage, equity growth, asset mix
6. Debt profile: level, interest rate trajectory, maturity management, **fair vs. carrying value** divergence
7. Liquidity: cash position, working capital components (AR, inventory, AP), current ratio
8. Per-share trends: EPS, dividend per share, share count (dilution or buyback)
9. Comprehensive income vs. net income: OCI items, **AOCI composition** (forex, pension, hedges)
10. Cost structure shifts: COGS, SG&A, R&D as % of revenue over time
11. D&A and capex as signals of asset intensity, growth investment, and capital cycle stage
12. Impairment and restructuring as transformation or risk signals
13. Tax dynamics: effective rate trend, deferred taxes, valuation allowances
14. Segment or geographic concentration (if data present); **segment count changes** as reorganization signals
15. Industry-specific metrics (claims ratio, R&D intensity, lease income, DD&A, exploration spending, RPO/backlog, contract loss provisions, environmental accruals, etc.)
16. Lease obligations: operating AND finance lease profiles (both sides of the lease relationship)
17. Long-term obligations: pension/post-retirement benefits, AROs, environmental accruals
18. Deferred revenue and contract liability trends (signal of demand health or billing dynamics)
19. Goodwill and intangibles trajectory (signals acquisition history and impairment risk)
20. Historical anchoring: how does current performance compare to a prior peak, trough, or pre-event period?
21. Interest income and net interest position (especially for cash-rich companies)
22. Financing cash flow pattern: debt issuance, equity issuance, buybacks — what does composition reveal?
23. **Working capital changes from OCF** (IncreaseDecrease in AR/inventory/AP): reveals cash conversion vs. balance sheet levels
24. **Customer concentration risk**: revenue dependency on major customers
25. **Derivative and hedge activity**: commodity, interest rate, or FX risk management approach
26. **Debt extinguishment / refinancing**: early repayment gains/losses, cost-of-debt evolution
27. **Equity method investment income**: JV performance and strategic partnership contribution
28. **FX effects on cash**: foreign currency translation exposure for international operators
29. **Advertising / brand investment**: trend as % of revenue (relevant for consumer, pharma, technology)
30. **Nonoperating income and asset sale gains**: one-time vs. recurring contribution to reported earnings
31. **Allowance for doubtful accounts**: credit risk evolution and receivables quality
32. Retained earnings trajectory: cumulative profitability and capital return history

## QA style

**Question form**: Prefer synthesis-oriented framing — "What does [metric trend] reveal about [business quality/risk/strategy/sustainability]?" Questions should be specific to this company's period and situation.

Good question examples:
- "What does EOG Resources' OCF-to-net-income ratio reveal about its earnings quality?"
- "How does ConocoPhillips' capex trajectory from 2020 to 2024 reflect its capital discipline strategy?"
- "What does Kraft Heinz's derivative and hedge activity from 2022 to 2024 reveal about its risk management approach to commodity exposure?"

**Answer form**: 1–2 sentences. Lead with a concrete trend or comparison (include specific values and period references), then state the implication or business meaning. Limit to 3–4 numbers — prefer qualitative synthesis over numeric recaps.

**Answer structure**: `[Direction + magnitude with exact figures across named periods] → [What this means for the business — quality, risk, strategy, or sustainability]`. Both elements must be present in every answer.

Good example:
> q: How does AvalonBay's operating cash flow compare to its dividend obligations?
> a: Operating cash flow of $1.61B in 2024 comfortably exceeds dividend payments of $969M (~1.65× coverage), and the pattern has held consistently from 2022–2024, indicating strong and sustainable dividend coverage.

Poor — imprecise values (do not submit):
> a: Total assets grew from roughly $19B to approximately $21B, while stockholders' equity increased to around $11.8B (based on 2023 data), maintaining a healthy equity ratio.

Poor — no business interpretation:
> a: OCF was $1.61B in 2024, $1.52B in 2023, $1.42B in 2022. Dividends were $969M, $935M, $891M.

Poor — generic qualifier without quantitative support:
> a: The company demonstrated disciplined leverage management throughout the period, reflecting its commitment to financial stability.

Poor — claim not in evidence (never submit without retrieved data):
> a: Operating margins improved from 15% to 22%, reflecting pricing power gains. ← only submit if you queried and retrieved those margin values.

## Edge-case handling

- **Missing expected metrics**: search for alternate `fact_name` values; never invent absent fields.
- **Imprecise data**: if exact values cannot be retrieved (e.g., a metric only appears as a range in disclosure, or two overlapping accessions give different figures), note this explicitly in the answer or skip this data point. Never substitute a range or approximation when an exact figure was not retrieved.
- **Empty results**: relax one filter at a time (remove accession constraint, widen date range, try alternate tag names, drop `fiscal_period = 'FY'` if truly necessary).
- **Mixed annual/quarterly facts**: keep 10-K trend analysis annual-focused; use `fiscal_period = 'FY'` to isolate full-year facts when available.
- **Duplicate facts for same period**: prefer the latest accession number; document only stable comparisons.
- **Query errors**: read the error, correct schema usage, and continue — do not retry the identical failing query.
- **Short filing history**: if fewer than 4 annual filings exist, note the limitation explicitly and focus QA on available periods.
- **Taxonomy shifts**: some companies change XBRL tags across years (e.g., `NetIncomeLoss` → `ProfitLoss`). When this happens, query both tag names and note the discontinuity in the answer if it affects comparability.

## Output format

For each QA pair:
- `q`: one analytical question with clear scope and period.
- `a`: concise answer grounded in retrieved facts (values, direction, period, implication).

Quality bar:
- Evidence-grounded: every value cited was retrieved from the database in this session, and is exact — not approximated or estimated.
- Non-redundant: each pair occupies a distinct analytical angle; no two pairs address the same thesis from different phrasings.
- Specific: questions name the company, metric, and time period.
- Synthetic: answers explain what the data means (implication) in addition to what it shows (trend).
- Self-contained: answers are interpretable without additional context.
- Comprehensive: together, the pairs give a reader a full financial picture across operational, balance sheet, cash flow, strategic, and risk dimensions.
