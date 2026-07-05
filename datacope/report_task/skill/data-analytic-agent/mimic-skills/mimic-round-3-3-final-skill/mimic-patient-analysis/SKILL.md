---
name: mimic-patient-analysis
description: Comprehensive patient analysis using the MIMIC-IV clinical database. Use this skill whenever asked to analyze, summarize, or investigate a patient's medical history, hospital admissions, diagnoses, medications, procedures, or clinical course from a MIMIC-IV SQLite database. Triggers on prompts like "Analyze patient [ID]", "summarize patient history", "what happened to patient X", or any request to explore patient-level EHR data from MIMIC-IV tables.
---

# MIMIC-IV Patient Analysis

Perform a comprehensive, systematic analysis of a patient's complete clinical record from the MIMIC-IV database by querying the SQLite database directly and efficiently.

## Database Structure

The database has 27 tables. Key tables and their primary columns:

**Core patient tables:**
- `hosp_patients` — demographics: `subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod`
- `hosp_admissions` — hospital stays: `subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag`

**Clinical data (per admission):**
- `hosp_diagnoses_icd` — ICD diagnoses: `subject_id, hadm_id, seq_num, icd_code, icd_version`
- `hosp_d_icd_diagnoses` — diagnosis dictionary: `icd_code, icd_version, long_title`
- `hosp_procedures_icd` — ICD procedures: `subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version`
- `hosp_d_icd_procedures` — procedure dictionary: `icd_code, icd_version, long_title`
- `hosp_drgcodes` — DRG billing: `subject_id, hadm_id, drg_type, drg_code, description, drg_severity, drg_mortality`
- `hosp_services` — clinical service: `subject_id, hadm_id, transfertime, prev_service, curr_service`
- `hosp_transfers` — unit movements: `subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime`

**Medications:**
- `hosp_prescriptions` — prescribed drugs: `subject_id, hadm_id, starttime, stoptime, drug, drug_type, dose_val_rx, dose_unit_rx, route`
- `hosp_emar` — administration record: `subject_id, hadm_id, emar_id, charttime, medication, event_txt, scheduletime`
- `hosp_pharmacy` — pharmacy fills: `subject_id, hadm_id, pharmacy_id, medication, starttime, stoptime` (**column is `medication`, NOT `drug`**)

**Diagnostics:**
- `hosp_microbiologyevents` — cultures: `subject_id, hadm_id, charttime, spec_type_desc, test_name, org_name, interpretation, comments`
- `hosp_omr` — vitals/anthropometrics: `subject_id, chartdate, seq_num, result_name, result_value`
- `hosp_hcpcsevents` — billing codes: `subject_id, hadm_id, chartdate, hcpcs_cd, short_description`
- `hosp_d_hcpcs` — HCPCS dictionary: `code, category, long_description, short_description`

**Orders:**
- `hosp_poe` — provider orders: `subject_id, hadm_id, poe_id, ordertime, order_type, order_subtype, transaction_type, order_status`

**ICU tables (only present if patient had ICU stay):**
- `icu_icustays` — ICU episodes: `subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los`
- `icu_inputevents` — IV fluids/medications: `stay_id, starttime, endtime, itemid, amount, amountuom, ordercategoryname`
- `icu_outputevents` — urine/drainage: `stay_id, charttime, itemid, value, valueuom`
- `icu_procedureevents` — ICU procedures: `stay_id, starttime, endtime, itemid, value, valueuom, ordercategoryname`
- `icu_d_items` — ICU item dictionary: `itemid, label, category`

## Critical Column Name Pitfalls

| Table | WRONG | CORRECT |
|-------|-------|---------|
| `hosp_transfers` | `transfertime` | `intime` |
| `hosp_poe` | `order_time` | `ordertime` |
| `hosp_omr` | `charttime` | `chartdate` |
| `hosp_pharmacy` | `drug` | `medication` |
| `hosp_hcpcsevents` JOIN `hosp_d_hcpcs` | `ON h.hcpcs_cd = d.hcpcs_cd` | `ON h.hcpcs_cd = d.code` |

## Analysis Workflow

Start with `get_database_info` to confirm table availability, then query directly — **do not call `describe_table` before each query**; use the column names listed above.

### Step 1 — Patient demographics
```sql
SELECT * FROM hosp_patients WHERE subject_id = <patient_id>
```
`anchor_age` is age in `anchor_year` (dates are shifted for privacy). If `dod` is not null, the patient died.

### Step 2 — All hospital admissions
```sql
SELECT * FROM hosp_admissions WHERE subject_id = <patient_id> ORDER BY admittime
```
For each `hadm_id`, note: admission/discharge times, type, source, destination, insurance, hospital_expire_flag.

**If `dod` is not null**, compute time from last discharge to death:
- `last_dischtime` to `dod` in days = how long the patient survived after final hospitalization.

**For multi-admission patients**, compute readmission intervals:
```sql
SELECT hadm_id, admittime, dischtime,
       CAST((julianday(admittime) - julianday(LAG(dischtime) OVER (ORDER BY admittime))) AS INTEGER) AS days_since_last_discharge
FROM hosp_admissions WHERE subject_id = <patient_id>
ORDER BY admittime
```
Track discharge destination progression (HOME → HOME HEALTH → SNF → LTACH → hospital death) as a functional decline signal.

### Step 3 — ICU stays
```sql
SELECT * FROM icu_icustays WHERE subject_id = <patient_id> ORDER BY intime
```
Empty result = no ICU. If ICU present, note care units and length of stay (`los`).

#### Step 3.5 — ICU deep dive (when ICU stays exist)

For patients with **≤3 ICU stays**, query all stay_ids. For **>3 stays**, prioritize by longest `los` and highest severity, covering at least the top 3.

For each `stay_id`, query in this order:

```sql
-- 1. Procedures (ventilation, dialysis, invasive lines)
SELECT pe.starttime, pe.endtime, d.label, d.category, pe.value, pe.valueuom
FROM icu_procedureevents pe
JOIN icu_d_items d ON pe.itemid = d.itemid
WHERE pe.stay_id = <stay_id>
ORDER BY pe.starttime

-- 2. Inputs (fluids, medications, blood products, nutrition)
SELECT ie.starttime, d.label, d.category, ie.amount, ie.amountuom, ie.ordercategoryname
FROM icu_inputevents ie
JOIN icu_d_items d ON ie.itemid = d.itemid
WHERE ie.stay_id = <stay_id>
ORDER BY ie.starttime LIMIT 50

-- 3. Outputs (urine, drainage)
SELECT oe.charttime, d.label, oe.value, oe.valueuom
FROM icu_outputevents oe
JOIN icu_d_items d ON oe.itemid = d.itemid
WHERE oe.stay_id = <stay_id>
ORDER BY oe.charttime LIMIT 30
```

**For each ICU stay, extract and summarize:**
- **Ventilation**: mechanical ventilation duration (label contains "Invasive Ventilation" or "Ventilation")
- **Vasopressors**: Norepinephrine, Epinephrine, Vasopressin, Phenylephrine, Dopamine (total amount)
- **Sedation/analgesia**: Propofol, Fentanyl, Midazolam, Dexmedetomidine
- **Blood products**: Packed RBCs, Platelets, FFP, Cryoprecipitate (`ordercategoryname = 'Blood Products'`)
- **Enteral nutrition**: formula names and total volumes (labels containing "Enteral", "Glucerna", "Promote", "Two Cal")
- **Fluid balance**: sum inputs (mL) minus sum outputs (mL) — positive = net accumulation
- **Vascular access**: arterial line, central line, Foley durations

Paginate inputs with OFFSET if >50 rows — blood products and nutrition often appear later.

### Step 4 — Diagnoses (with human-readable names)
```sql
SELECT d.icd_code, d.icd_version, d.long_title, diag.hadm_id, diag.seq_num
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id>
ORDER BY diag.hadm_id, diag.seq_num
```
`seq_num=1` is the primary diagnosis. Note total diagnosis count per admission (high count = high complexity).

**For patients with 4+ admissions**, identify recurring diagnoses:
```sql
SELECT d.long_title, COUNT(*) as admission_count
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id> AND diag.seq_num <= 5
GROUP BY d.long_title
ORDER BY admission_count DESC
LIMIT 20
```

**Always query for special ICD codes** — clinically critical status flags:
```sql
SELECT d.icd_code, d.long_title, diag.hadm_id
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id>
  AND (d.icd_code LIKE 'Z88%'   -- drug allergies
    OR d.icd_code = 'Z66'        -- do not resuscitate
    OR d.icd_code = 'Z515'       -- palliative care
    OR d.icd_code LIKE 'Z79%')   -- long-term medication use
ORDER BY diag.hadm_id
```

### Step 5 — Procedures
```sql
SELECT p.hadm_id, p.seq_num, p.chartdate, p.icd_code, proc.long_title
FROM hosp_procedures_icd p
JOIN hosp_d_icd_procedures proc ON p.icd_code = proc.icd_code AND p.icd_version = proc.icd_version
WHERE p.subject_id = <patient_id>
ORDER BY p.hadm_id, p.seq_num
```

### Step 6 — Medications prescribed
```sql
SELECT drug, COUNT(*) as prescription_count
FROM hosp_prescriptions
WHERE subject_id = <patient_id>
GROUP BY drug
ORDER BY prescription_count DESC
LIMIT 20
```

Then per-admission detail:
```sql
SELECT hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx, route
FROM hosp_prescriptions
WHERE subject_id = <patient_id>
ORDER BY hadm_id, starttime
```

Group medications by clinical class: anticoagulants/antiplatelets, cardiovascular, diuretics, analgesics/opioids, antibiotics, psychiatric agents, immunosuppressants. Note route transitions (IV → PO = improvement; PO/NG = nasogastric feeding from dysphagia). Multiple laxatives (Senna + Bisacodyl + Docusate) indicate immobility or opioid use.

#### Step 6.5 — Pharmacy fills (cross-validation)
```sql
SELECT hadm_id, medication, starttime, stoptime
FROM hosp_pharmacy
WHERE subject_id = <patient_id>
ORDER BY hadm_id, starttime
LIMIT 50
```
Column is `medication`, not `drug`. Paginate with OFFSET if needed.

### Step 7 — Physical measurements (BMI, weight, height, BP)

**Query all OMR data in a single consolidated query:**
```sql
SELECT chartdate, result_name, result_value
FROM hosp_omr
WHERE subject_id = <patient_id>
ORDER BY chartdate
```

**Synthesize trends, do not report individual data points.** For longitudinal patients with many OMR records, summarize the overall trajectory:
- Weight: report baseline, minimum, maximum, and final values with percent change. Flag ≥5% loss as clinically significant, ≥10% as cachexia risk.
- BP: report overall range (systolic and diastolic) and note any hypertensive spikes (>140/90) or hypotensive episodes (<90/60).
- BMI: report range and trend direction.
- Flag implausible values (e.g., weight of 1731 lbs is likely a typo for 173.1).

Do NOT issue multiple paginated queries stepping through OMR records month-by-month or year-by-year. One query with a single synthesized trend summary is sufficient and prevents wasting analysis effort on granular data recitation.

### Step 8 — Microbiology cultures

**Query all microbiology data in a single consolidated query:**
```sql
SELECT chartdate, spec_type_desc, test_name, org_name, interpretation, comments
FROM hosp_microbiologyevents
WHERE subject_id = <patient_id>
ORDER BY chartdate
```

Focus on clinically actionable findings:
- **Positive cultures**: Record organism name, specimen type, and antibiotic sensitivity (R/S/I).
- **MRSA colonization**: Note any MRSA screen positive results.
- **Negative cultures**: Summarize as "X cultures negative" rather than listing each one.
- Comments containing "< 10,000 CFU/mL" or "NO GROWTH" = negative.
- Comments containing "MIXED BACTERIAL FLORA" = likely contamination, not a true infection.

Paginate with OFFSET only if >50 results. Do NOT issue per-admission microbiology queries individually.

### Step 9 — Clinical service and transfers
```sql
SELECT * FROM hosp_services WHERE subject_id = <patient_id> ORDER BY transfertime

SELECT hadm_id, eventtype, careunit, intime, outtime
FROM hosp_transfers WHERE subject_id = <patient_id>
ORDER BY hadm_id, intime
LIMIT 50
```
For patients with many admissions, paginate transfers with LIMIT/OFFSET.

### Step 10 — DRG billing codes
```sql
SELECT * FROM hosp_drgcodes WHERE subject_id = <patient_id>
```
APR-DRG has severity (1-4) and mortality (1-4) scores. Severity 3-4 or mortality 3-4 = high-complexity/high-risk admission.

### Step 11 — HCPCS events
```sql
SELECT h.hadm_id, h.chartdate, h.hcpcs_cd, d.short_description
FROM hosp_hcpcsevents h
JOIN hosp_d_hcpcs d ON h.hcpcs_cd = d.code
WHERE h.subject_id = <patient_id>
ORDER BY h.chartdate
```
Zero results = no billed procedures (common). G0378 = observation-status admission.

### Step 12 — eMAR and Provider Orders
```sql
-- Medication administration record
SELECT charttime, medication, event_txt, scheduletime
FROM hosp_emar WHERE subject_id = <patient_id>
ORDER BY charttime LIMIT 50

-- Provider orders overview
SELECT order_type, COUNT(*) as order_count
FROM hosp_poe WHERE subject_id = <patient_id>
GROUP BY order_type
ORDER BY order_count DESC

-- Detailed orders for specific admission (when clinically relevant)
SELECT ordertime, order_type, order_subtype, transaction_type, order_status
FROM hosp_poe WHERE subject_id = <patient_id> AND hadm_id = <hadm_id>
ORDER BY ordertime LIMIT 30
```

The `event_txt` field distinguishes "Administered" from "Not Given" — this reveals medication compliance and route changes. Patterns of held critical drugs (immunosuppressants, anticoagulants) are adherence risk signals. "Not Given per Sliding Scale" for insulin is expected, not a compliance issue.

Order type distribution (Medications / Lab / Radiology / Nutrition / General Care) characterizes admission intensity. Rehabilitation consults (Speech/Swallowing, Occupational Therapy, Physical Therapy) signal functional impairment workup.

## Producing the Final Report

After gathering data, produce a **structured report with markdown headers and tables**. The report must be comprehensive enough to answer detailed clinical questions about any aspect of the patient's care.

### Required report sections:

1. **Demographics** — age, sex, race, insurance, vital status (alive/deceased + date); if deceased, compute days from last discharge to death; note insurance transitions (Private→Medicare = age 65)

2. **Admission Summary** — number of admissions, date range, types/sources; present as a **markdown table** for multi-admission patients with columns: hadm_id, admit date, discharge date, type, primary diagnosis, discharge location, LOS; computed readmission intervals highlighting any <30-day readmissions

3. **ICU Course** — whether ICU was needed, which units, LOS per stay; key interventions: ventilation duration, vasopressors (drug names + amounts), sedation agents, blood products (type + volume), enteral nutrition, fluid balance (total inputs − outputs mL)

4. **Primary Diagnoses by Admission** — primary condition per hadm_id; total diagnosis count per admission to indicate complexity; identify highest-complexity admissions

5. **Comorbidities** — significant secondary diagnoses; for multi-admission patients, note recurring conditions with frequency counts

6. **Procedures** — surgical and therapeutic interventions with dates; link to relevant admissions

7. **Medications** — organized by clinical class (anticoagulants, cardiovascular, diuretics, analgesics, antibiotics, immunosuppressants); for multi-admission patients, top prescriptions by frequency; route transitions and polypharmacy patterns; specific dose information for key drugs

8. **Diagnostics** — positive culture results (organism + specimen + sensitivity pattern); MRSA colonization status; physical measurement trends (summarized as ranges and percent changes, NOT individual data points); negative culture summary

9. **Clinical Service Trajectory** — services and care unit progression; transition patterns (e.g., Trauma SICU → Neuro SICU → Med/Surg = recovery)

10. **Key Clinical Insights** — clinically meaningful patterns with specific evidence:
    - Discharge to rehab/SNF → functional impairment
    - Multiple laxatives (Senna + Bisacodyl + Docusate) → immobility or opioid use
    - PO/NG drug routes → nasogastric feeding (dysphagia)
    - Sequential anticoagulant changes → treatment optimization (name the specific drug sequence and dates)
    - Z88x codes → drug allergies (list specific allergens)
    - Z66/Z515 → DNR/palliative care goals
    - Weight loss ≥10% → cachexia or disease progression (cite baseline and nadir weights)
    - Readmission <30 days → unstable underlying condition
    - Tacrolimus/Mycophenolate/steroids → transplant recipient (infection/rejection risk)
    - eMAR "Not Given" for critical drugs → adherence risk
    - Blood products + vasopressors + ventilation → critical illness severity
    - Time from last discharge to death <30 days → rapid terminal decline
    - Insurance transition Private→Medicare → crossed age 65
    - Discharge destinations HOME → HOME HEALTH → SNF → LTACH → death = functional decline trajectory

**Evidence anchors are required**: Every insight must cite specific ICD codes, exact dates, drug names with doses, organism names, DRG severity/mortality scores, or numeric values. Vague summaries without supporting data are not acceptable.

End the analysis with `FINISH:` followed by the full structured report.

## Query Efficiency Rules

These rules prevent the most common analysis failures — wasting queries on granular data instead of building clinical synthesis.

1. **One query per table, then synthesize.** Query each table with a consolidated SELECT that gets all needed columns at once. Use OFFSET to paginate, never reissue the same query with different columns.

2. **OMR: one query, one trend summary.** Never issue multiple OMR queries to step through measurements chronologically. One query → extract baseline, min, max, final values → report trend and percent changes.

3. **Microbiology: one query, separate positives from negatives.** Never issue per-admission microbiology queries. One query → report positive cultures with organisms and sensitivities → summarize negative cultures as a count.

4. **ICU events: paginate, don't re-query.** For `icu_inputevents`, use LIMIT 50 then OFFSET 50 to get blood products/nutrition that appear later. Never re-query the same stay_id with different filters.

5. **Diagnoses: use consolidated queries.** For multi-admission patients, use subject_id-level queries with GROUP BY for recurring diagnoses. Don't query each hadm_id individually unless you need specific seq_num detail for a particular admission.

6. **Skip ICU event tables when `icu_icustays` returns empty.** For non-ICU patients, use saved time for deeper eMAR and POE analysis.

7. **For >3 ICU stays**, focus on the top-severity stays (longest LOS or highest DRG severity), not all stays exhaustively.

8. **If a query fails with a column error**, correct it using the pitfalls table — do not call `describe_table`.
