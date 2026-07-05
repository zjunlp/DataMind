---
name: sec-10k-company-analysis
description: Analyze a company in an SEC 10-K SQLite database and produce high-quality evidence-grounded financial QA pairs. Use this whenever the user asks to analyze a company by CIK/ticker, inspect 10-K financial trends, generate finance QA datasets, or work with filings/financial_facts tables.
---

# SEC 10-K Company Analysis

Use this skill to analyze one company from a SQLite SEC filings database and produce distinct, data-grounded QA pairs.

## Inputs you need
- Company identifier: CIK preferred (or ticker/name if unavailable).
- Database connection or path.
- Target output count if specified; otherwise produce **20–32 distinct QA pairs**.

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

### Step 5.5: Pre-generation inventory (required before writing QA pairs)

Before writing any QA pair, scan all query results and create a brief mental inventory across these dimensions. This prevents coverage gaps and over-representation of one theme:

1. **Revenue/profitability** — key trend, inflection points, magnitude of change
2. **Cash flow quality** — OCF vs. NI relationship (convergence, divergence, ratio)
3. **Balance sheet / leverage** — direction and scale of debt/equity shift
4. **Capital returns** — dividends vs. buybacks vs. capex allocation and how the mix shifted
5. **Structural events** — acquisitions, divestitures, spinoffs, major industry shocks
6. **Industry-specific niche** — what unique metrics does this SIC code expose that other sectors don't?
7. **Niche financial items** — AOCI, derivatives, FX, debt fair value, deferred revenue, AROs

Then identify **2–3 defining themes** that characterize this company's financial story. If any single theme (e.g., pandemic revenue cycle, major acquisition, regulatory spinoff) appears to support more than 4–5 QA angles, keep only the 3–4 most analytically distinct angles on that theme and ensure remaining pairs cover other dimensions of the company's financials.

**Also plan 3–5 derived ratio QA pairs** — metrics computed from retrieved values that are not stored directly in the database. The strongest QA pairs often come from these cross-metric computations:
- **Net debt** = long-term debt + current debt − cash (and whether it shifted from net debt to net cash)
- **Interest coverage** = operating income ÷ interest expense (trend reveals financial flexibility)
- **OCF margin** = operating cash flow ÷ revenue (cash conversion efficiency)
- **Capex-to-DD&A ratio** = capex ÷ depreciation & amortization (>1.0× = expanding base; <1.0× = harvesting)
- **Effective interest rate** = interest expense ÷ average long-term debt (derived from two retrieved series)
- **Dividend coverage** = OCF ÷ total dividend payments

Write these derived computations down explicitly before proceeding to QA generation.

### Step 6: Generate QA pairs from evidence

**The core rule**: only submit a QA pair when the specific data points cited in the answer are present in your query results. Do not generalize beyond what was retrieved.

**Ratio synthesis rule**: when your answer cites two or more metrics on the same topic, compute and state their relationship as a ratio, percentage, or change magnitude. A list of raw values without synthesis fails the analytical bar. For example:
- Instead of: "OCF was $12,143M and net income was $6,403M"
- Write: "OCF of $12,143M was 1.9× net income ($6,403M)"
- Instead of: "capex was $5,353M; OCF was $12,143M"
- Write: "capex of $5,353M represented 44% of operating cash flow"
- Instead of: "debt fell from $6,990M to $4,640M"
- Write: "debt declined 34% from $6,990M in 2016 to $4,640M in 2024"
- When the question names a ratio (current ratio, interest coverage, OCF/NI, etc.): compute and state the actual ratio value with its numerator and denominator — do not merely describe the direction of change. Example: "current ratio of 2.1× ($11,230M/$5,354M)" rather than "the current ratio was healthy."

**Dominant theme cap**: if one event (pandemic, spinoff, acquisition, regulatory shift) supports more than 4–5 QA pairs, keep only the 3–4 most analytically distinct pairs on that theme. The remaining pairs must explore other financial dimensions — structural, operational, risk — that exist independent of that event.

**Before submitting any QA pair, verify:**
1. All cited values are exact retrieved figures — not approximations, ranges, or estimates. Phrases like "approximately $X", "roughly $X–Y range", or "around $X" are not acceptable. If a precise value was not retrieved, do not submit the pair.
2. The answer leads with a specific relationship (ratio, %, trend magnitude with named periods) and concludes with what that means for the business — quality, risk, strategy, or sustainability. A pure data list OR an interpretation with no grounding data both fail this bar.
3. Any computed ratios or percentages are mathematically consistent with the cited raw values (e.g., "$5,353M / $12,143M = 44%" must be verifiable). Verify the **direction of change** matches the actual figures: if a metric went from 35.6% to 30.8%, that is a decline — do not describe it as "expanded."
4. The angle has not already been covered by a submitted QA pair — two questions on the same metric from different phrasings occupy the same analytical position; pick the stronger framing.
5. The question and answer are specific to this company's situation, not generic statements applicable to any firm in the sector.
6. This is not a narrative summary of the company's overall trajectory — each pair must cite a specific metric or cross-metric relationship with exact figures. Questions that survey the whole company story ("What does the overall financial trajectory reveal about the success of restructuring?") are too broad and fail this bar.

Submit a QA pair immediately when you have multi-datapoint support for a non-trivial conclusion. Keep exploring after each submission — aim for **20–32 distinct pairs** covering different angles.

**Identify this company's defining story**: Before generating pairs, ask what makes this company's financial trajectory distinctive — a pandemic revenue cycle, a major acquisition, a regulatory restructuring, heavy infrastructure investment, a delevering campaign. Anchor several QA pairs to these defining narratives; they produce the most analytically valuable pairs and are impossible to generate without close reading of the data.

**Before finalizing**, do a completeness sweep: review your query results and identify any significant finding (trend, ratio shift, structural event, comparison) not yet captured. Niche metrics (AOCI, hedge activity, customer concentration, debt fair value, interest income vs. expense comparison) often yield the most analytically distinctive pairs — do not skip them when data is available. Then review all submitted pairs for angle overlap and merge or drop any redundant ones.

## QA angle checklist

Work through as many distinct angles as the data supports. Each angle should occupy a **distinct analytical position**:

**Core financial dimensions (cover these first):**
1. Revenue growth drivers, trajectory, and volatility
2. Profitability trajectory (operating income, net income, margins as % of revenue)
3. Earnings quality: operating cash flow vs. net income (OCF/NI ratio; divergence signals)
4. Capital allocation: dividends, buybacks, capex — what does the mix reveal about management priorities?
5. Balance sheet evolution: leverage, equity growth, asset mix
6. Debt profile: level, interest rate trajectory, maturity management, fair vs. carrying value divergence
7. Liquidity: cash position, working capital components (AR, inventory, AP), current ratio
8. Per-share trends: EPS, dividend per share, share count (dilution or buyback)

**Structural and risk dimensions:**
9. Comprehensive income vs. net income: OCI items, AOCI composition (forex, pension, hedges)
10. Cost structure shifts: COGS, SG&A, R&D as % of revenue over time
11. D&A and capex as signals of asset intensity, growth investment, and capital cycle stage
12. Impairment and restructuring as transformation or risk signals
13. Tax dynamics: effective rate trend, deferred taxes, valuation allowances
14. Segment or geographic concentration; segment count changes as reorganization signals
15. Industry-specific metrics (R&D intensity, DD&A, claims ratio, lease income, RPO/backlog, AROs, production cost per unit, etc.)
16. Lease obligations: operating AND finance lease profiles (both sides matter)
17. Long-term obligations: pension/post-retirement benefits, AROs, environmental accruals
18. Deferred revenue and contract liability trends (demand health or billing dynamics)
19. Goodwill and intangibles trajectory (acquisition history and impairment risk)
20. Historical anchoring: how does current performance compare to a prior peak, trough, or pre-event baseline?

**Advanced/niche dimensions (unlock distinctive QA pairs when data is available):**
21. Interest income and net interest position (especially for cash-rich companies — when interest income approaches or exceeds interest expense, that's a distinctive signal)
22. Financing cash flow pattern: debt issuance, equity issuance, buybacks — composition reveals strategic intent
23. Working capital changes from OCF (IncreaseDecrease in AR/inventory/AP): cash conversion vs. balance sheet levels
24. Customer concentration risk: revenue dependency on major customers
25. Derivative and hedge activity: commodity, interest rate, or FX risk management approach
26. Debt extinguishment / refinancing: early repayment gains/losses, cost-of-debt evolution
27. Equity method investment income: JV performance and strategic partnership contribution
28. FX effects on cash: foreign currency translation exposure for international operators
29. Advertising / brand investment: trend as % of revenue (consumer, pharma, technology)
30. Nonoperating income and asset sale gains: one-time vs. recurring contribution to reported earnings
31. Allowance for doubtful accounts: credit risk evolution and receivables quality
32. Retained earnings trajectory: cumulative profitability and capital return history
33. **Net debt position** (total debt − cash): direction, magnitude, and whether the company shifted from net debt to net cash
34. **Interest coverage ratio** (operating income ÷ interest expense): trend across years and what it signals about financial flexibility
35. **OCF margin** (operating cash flow ÷ revenue): efficiency of cash conversion across the commodity or business cycle
36. **Capex-to-DD&A ratio**: whether reinvestment rate exceeds or trails asset consumption; >1.0× signals expanding base

## QA style

**Question form**: Prefer synthesis-oriented framing — "What does [metric trend] reveal about [business quality/risk/strategy/sustainability]?" Questions should name the company, the metric, and the time period.

Good question examples:
- "What does EOG Resources' OCF-to-net-income ratio from 2021 to 2024 reveal about its earnings quality?"
- "How does Prologis's capex-to-OCF ratio from 2020 to 2024 reflect its real estate acquisition strategy?"
- "What does Kraft Heinz's derivative and hedge activity from 2022 to 2024 reveal about its commodity risk management approach?"

**Answer form**: 1–2 sentences. Lead with a specific relationship (ratio, %, or trend magnitude with exact figures and named periods), then state the business implication — quality, risk, strategy, or sustainability. Limit to 3–4 numbers.

**Answer structure**: `[Key relationship or trend with exact figures across named periods] → [What this means for quality/risk/strategy/sustainability — one clear signal]`. Both elements must be present in every answer.

Good examples:
> q: How does AvalonBay's operating cash flow compare to its dividend obligations from 2022 to 2024?
> a: Operating cash flow of $1.61B in 2024 comfortably exceeds dividend payments of $969M (1.65× coverage), and this pattern held from 2022–2024, indicating strong and sustainable dividend coverage with improving headroom.

> q: What does EOG Resources' capital expenditure trajectory from 2020 to 2024 reveal about its capital discipline?
> a: EOG cut capex from $6,152M in 2019 to $3,243M during the pandemic, then held it to 44% of operating cash flow ($5,353M/$12,143M) in 2024, demonstrating that recovery reinvestment was deliberately constrained to maximize free cash flow generation.

> q: What does EOG Resources' net debt position from 2020 to 2024 reveal about its balance sheet transformation?
> a: EOG shifted from a net debt position of $2,311M ($5,640M debt − $3,329M cash) in 2020 to a net cash position of $2,452M ($7,092M cash − $4,640M debt) by 2024, demonstrating a complete balance sheet transformation that eliminated net debt risk and created substantial financial optionality.

Poor — imprecise values (do not submit):
> a: Total assets grew from roughly $19B to approximately $21B, while stockholders' equity increased to around $11.8B.

Poor — approximate range (do not submit):
> a: Long-term debt declined to approximately $19–20B range through 2024, indicating disciplined deleveraging.

Poor — data dump with no ratio or implication:
> a: OCF was $1.61B in 2024, $1.52B in 2023, $1.42B in 2022. Dividends were $969M, $935M, $891M.

Poor — named ratio not computed (do not submit):
> a: The current ratio deteriorated as current liabilities grew faster than current assets, with current debt reaching $10.4B due to long-term debt reclassifications. ← names the ratio concept but never states the actual ratio value; always compute it: "current ratio of 0.85× ($42.0B/$49.5B)"

Poor — direction error (do not submit):
> a: Operating income margin expanded from 35.6% in 2020 to 30.8% in 2025 ← 35.6% → 30.8% is a decline; verify direction before writing "improved", "expanded", or "declined."

Poor — generic qualifier without quantitative support:
> a: The company demonstrated disciplined leverage management throughout the period, reflecting its commitment to financial stability.

Poor — narrative summary without specific metric (do not submit):
> q: What does [Company]'s overall financial trajectory from 2020 to 2024 reveal about the success of its post-merger restructuring strategy? ← too broad; name one specific metric and time period. Each pair must focus on a specific data relationship.

Poor — internally inconsistent ratios (do not submit if the math doesn't check out):
> a: COGS was $8.5B on $41.7B revenue (20%), rising to $30.8B on $81.3B revenue (38%), then spiking to $25.0B on $58.5B revenue (43%). ← verify each % before submitting.

## Edge-case handling

- **Missing expected metrics**: search for alternate `fact_name` values; never invent absent fields.
- **Imprecise data**: if exact values cannot be retrieved (range in disclosure, or two overlapping accessions give different figures), note explicitly in the answer or skip this data point. Never substitute a range or approximation when an exact figure was not retrieved.
- **Empty results**: relax one filter at a time (remove accession constraint, widen date range, try alternate tag names, drop `fiscal_period = 'FY'` if truly necessary).
- **Mixed annual/quarterly facts**: keep 10-K trend analysis annual-focused; use `fiscal_period = 'FY'` to isolate full-year facts when available.
- **Duplicate facts for same period**: prefer the latest accession number; document only stable comparisons.
- **Query errors**: read the error, correct schema usage, and continue — do not retry the identical failing query.
- **Short filing history**: if fewer than 4 annual filings exist, note the limitation explicitly and focus QA on available periods.
- **Taxonomy shifts**: some companies change XBRL tags across years (e.g., `NetIncomeLoss` → `ProfitLoss`). Query both tag names and note the discontinuity in the answer if it affects comparability.

## Output format

For each QA pair:
- `q`: one analytical question with clear scope and period.
- `a`: concise answer grounded in retrieved facts (relationship/ratio, direction, period, implication).

Quality bar:
- **Evidence-grounded**: every value cited was retrieved from the database in this session, and is exact.
- **Synthetic**: answers express a relationship (ratio, %, change magnitude) between metrics — not just a list of values.
- **Non-redundant**: each pair occupies a distinct analytical angle; no two pairs address the same thesis from different phrasings.
- **Specific**: questions name the company, metric, and time period.
- **Self-contained**: answers are interpretable without additional context.
- **Comprehensive**: together, the pairs give a reader a full financial picture across operational, balance sheet, cash flow, strategic, and risk dimensions.
