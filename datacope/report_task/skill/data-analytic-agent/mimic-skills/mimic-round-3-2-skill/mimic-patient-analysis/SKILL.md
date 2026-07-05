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

Avoid these common errors that cause query failures:

| Table | WRONG | CORRECT |
|-------|-------|---------|
| `hosp_transfers` | `transfertime` | `intime` (sort by `intime`) |
| `hosp_poe` | `order_time` | `ordertime` |
| `hosp_omr` | `charttime` | `chartdate` |
| `hosp_pharmacy` | `drug` | `medication` |
| `hosp_hcpcsevents` JOIN `hosp_d_hcpcs` | `ON h.hcpcs_cd = d.hcpcs_cd` | `ON h.hcpcs_cd = d.code` |
| `hosp_procedures_icd` JOIN `hosp_d_icd_procedures` | alias mismatch | ensure alias used in JOIN matches the one defined |

## Analysis Workflow

Start with `get_database_info` to confirm table availability, then query directly — **do not call `describe_table` before each query**; use the column names listed above.

### Step 1 — Patient demographics
```sql
SELECT * FROM hosp_patients WHERE subject_id = <patient_id>
```
Note: `anchor_age` is age in `anchor_year` (dates are shifted for privacy). If `dod` is not null, the patient died.

### Step 2 — All hospital admissions
```sql
SELECT * FROM hosp_admissions WHERE subject_id = <patient_id> ORDER BY admittime
```
For each `hadm_id`, note: admission/discharge times, type, source, destination, insurance, hospital_expire_flag.

**If `dod` is not null**, compute time from last discharge to death:
- `last_dischtime` to `dod` in days = how long the patient survived after final hospitalization.

**For multi-admission patients**, compute readmission intervals explicitly:
```sql
SELECT hadm_id, admittime, dischtime,
       CAST((julianday(admittime) - julianday(LAG(dischtime) OVER (ORDER BY admittime))) AS INTEGER) AS days_since_last_discharge
FROM hosp_admissions WHERE subject_id = <patient_id>
ORDER BY admittime
```
Readmission within 30 days signals unstable underlying condition. Track discharge destination progression (HOME → HOME HEALTH → SNF → LTACH → hospital death) as a functional decline signal.

### Step 3 — ICU stays
```sql
SELECT * FROM icu_icustays WHERE subject_id = <patient_id> ORDER BY intime
```
Empty result = no ICU. If ICU present, note care units and length of stay (`los`).

#### Step 3.5 — ICU deep dive (when ICU stays exist)

For patients with **≤3 ICU stays**, query all stay_ids. For patients with **>3 stays**, prioritize by longest `los` and highest-severity admissions, covering at least the top 3.

For each `stay_id`, query in this order:

```sql
-- 1. Procedures (ventilation, dialysis, invasive lines — most discriminating)
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
- **Ventilation**: mechanical ventilation duration in minutes (from `icu_procedureevents`, label contains "Invasive Ventilation" or "Ventilation")
- **Vasopressors**: Norepinephrine, Epinephrine, Vasopressin, Phenylephrine, Dopamine (total amount administered)
- **Sedation/analgesia**: Propofol (mg), Fentanyl, Midazolam, Dexmedetomidine (identify from `icu_inputevents` category = "Medications")
- **Blood products**: Packed RBCs, Platelets, FFP, Cryoprecipitate (identify from `ordercategoryname = 'Blood Products'` or labels containing "RBC", "Platelet", "Plasma")
- **Enteral nutrition**: formula names and total volumes (labels containing "Enteral", "Glucerna", "Promote", "Two Cal")
- **Fluid balance**: sum all inputs (mL) minus sum all outputs (mL) — positive balance = net fluid accumulation
- **Vascular access durations**: arterial line, central line, Foley (from `icu_procedureevents`, value in minutes)

Paginate inputs with OFFSET if >50 rows to capture blood products and nutrition that may appear later in the record.

### Step 4 — Diagnoses (with human-readable names)
```sql
SELECT d.icd_code, d.icd_version, d.long_title, diag.hadm_id, diag.seq_num
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id>
ORDER BY diag.hadm_id, diag.seq_num
```
`seq_num=1` is the primary diagnosis. Note the total number of diagnoses per admission (high count = high complexity).

**For patients with 4+ admissions**, run an aggregate query to identify recurring diagnoses:
```sql
SELECT d.long_title, COUNT(*) as admission_count
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id> AND diag.seq_num <= 5
GROUP BY d.long_title
ORDER BY admission_count DESC
LIMIT 20
```

**Always query for special ICD codes** — these represent clinically critical status flags:
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

Then retrieve per-admission detail:
```sql
SELECT hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx, route
FROM hosp_prescriptions
WHERE subject_id = <patient_id>
ORDER BY hadm_id, starttime
```

Group medications by clinical class: anticoagulants/antiplatelets, cardiovascular, diuretics, analgesics/opioids, antibiotics, psychiatric agents, immunosuppressants. Note route transitions (IV → PO = clinical improvement; PO/NG = nasogastric feeding due to dysphagia). Presence of multiple laxatives (Senna + Bisacodyl + Docusate) indicates immobility or opioid use.

#### Step 6.5 — Pharmacy fills (cross-validation)
```sql
SELECT hadm_id, medication, starttime, stoptime
FROM hosp_pharmacy
WHERE subject_id = <patient_id>
ORDER BY hadm_id, starttime
LIMIT 50
```
Note: column is `medication`, not `drug`. Pharmacy fills confirm which prescribed drugs were actually dispensed. Paginate with OFFSET if needed.

### Step 7 — Physical measurements (BMI, weight, height, BP)
```sql
SELECT chartdate, result_name, result_value
FROM hosp_omr
WHERE subject_id = <patient_id>
ORDER BY chartdate
LIMIT 50
```

For longitudinal patients, track weight and blood pressure trends:
```sql
SELECT chartdate, result_value FROM hosp_omr
WHERE subject_id = <patient_id> AND result_name = 'Weight (Lbs)'
ORDER BY chartdate

SELECT chartdate, result_value FROM hosp_omr
WHERE subject_id = <patient_id> AND result_name = 'Blood Pressure'
ORDER BY chartdate
```

**Data quality**: Flag implausible OMR values (e.g., weight of 1731 lbs is a typo for 173.1 lbs). Use surrounding measurements to identify outliers before computing trends.

**Weight loss ≥5% from baseline is clinically significant**; ≥10% suggests disease-related cachexia or malnutrition. Compute percent change from earliest to minimum recorded weight.

### Step 8 — Microbiology cultures
```sql
SELECT chartdate, spec_type_desc, test_name, org_name, interpretation, comments
FROM hosp_microbiologyevents
WHERE subject_id = <patient_id>
ORDER BY chartdate
LIMIT 50
```
`org_name` null with comment like "< 10,000 CFU/mL" = negative culture. For positive cultures, record organism name, specimen type, and interpretation (R/S/I for antibiotic sensitivity when available). Note MRSA colonization specifically (MRSA screen positive = infection control significance). Paginate with OFFSET if > 50 results.

### Step 9 — Clinical service and transfers
```sql
-- Service
SELECT * FROM hosp_services WHERE subject_id = <patient_id> ORDER BY transfertime

-- Physical location movements
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
APR-DRG has severity (1-4) and mortality (1-4) scores. Severity 3-4 or mortality 3-4 indicates a high-complexity/high-risk admission.

### Step 11 — HCPCS events
```sql
SELECT h.hadm_id, h.chartdate, h.hcpcs_cd, d.short_description
FROM hosp_hcpcsevents h
JOIN hosp_d_hcpcs d ON h.hcpcs_cd = d.code
WHERE h.subject_id = <patient_id>
ORDER BY h.chartdate
```
Zero results = no billed procedures in this table (common). G0378 ("Hospital observation per hr") confirms observation-status admission. Other codes reveal additional diagnostic/therapeutic procedures not captured in ICD codes.

### Step 12 — eMAR and Provider Orders
```sql
-- Medication administration record (reveals compliance, route changes, held doses)
SELECT charttime, medication, event_txt, scheduletime
FROM hosp_emar WHERE subject_id = <patient_id>
ORDER BY charttime LIMIT 50

-- Provider orders overview (summary by type)
SELECT order_type, COUNT(*) as order_count
FROM hosp_poe WHERE subject_id = <patient_id>
GROUP BY order_type
ORDER BY order_count DESC

-- Detailed orders for specific admission (when clinically relevant)
SELECT ordertime, order_type, order_subtype, transaction_type, order_status
FROM hosp_poe WHERE subject_id = <patient_id> AND hadm_id = <hadm_id>
ORDER BY ordertime LIMIT 30
```

The `hosp_emar` `event_txt` field distinguishes "Administered" from "Not Given" — this reveals medication compliance and route changes. PO/NG route confirms nasogastric feeding (suggests dysphagia). Patterns of "Not Flushed" or interrupted IV access explain missed doses. "Not Given per Sliding Scale" for insulin is expected, not a compliance issue.

Provider order type distribution (Medications / Lab / Radiology / Nutrition / General Care / IV Access) characterizes admission intensity and care focus. Rehabilitation consults (Speech/Swallowing, Occupational Therapy, Physical Therapy) signal functional impairment workup.

## Synthesizing the Analysis

After gathering data, produce a structured report:

1. **Demographics** — age, sex, race, insurance, vital status (alive/deceased + date if known); if deceased, compute days from last discharge to death; note insurance transitions (Private→Medicare = age 65 crossed)
2. **Admission summary** — number of admissions, date range, types/sources, discharge destinations in a markdown table for multi-admission patients; computed readmission intervals highlighting any <30-day readmissions; admission type distribution (emergency vs observation)
3. **ICU course** — whether ICU was needed, which units, LOS per stay; key interventions summarized as: ventilation duration (minutes/hours), vasopressors used (drug names + amounts), sedation agents, blood products transfused (type + volume), enteral nutrition formulas, fluid balance per stay (total inputs mL − total outputs mL)
4. **Primary diagnoses by admission** — primary condition per hadm_id in table format; total diagnosis count per admission; note highest-complexity admissions
5. **Comorbidities** — significant secondary diagnoses; for multi-admission patients, note recurring conditions with how many admissions they appear in
6. **Procedures** — surgical and therapeutic interventions with dates
7. **Medications** — organized by clinical class (anticoagulants/antiplatelets, cardiovascular, diuretics, analgesics, antibiotics, immunosuppressants); for multi-admission patients, top prescriptions by frequency; route transitions and polypharmacy patterns
8. **Diagnostics** — positive culture results (organism + specimen + sensitivity/resistance pattern); MRSA colonization status; physical measurement trends (weight trajectory with percent change, BP range)
9. **Clinical service trajectory** — services and care unit progression; note transition patterns (e.g., Trauma SICU → Neuro SICU → Med/Surg = recovery course)
10. **Key clinical insights** — clinically meaningful patterns with explanations:
    - Discharge to rehab/SNF → functional impairment
    - Multiple laxatives (Senna + Bisacodyl + Docusate) → immobility or opioid use
    - PO/NG drug routes → nasogastric feeding (likely dysphagia)
    - Sequential anticoagulant changes → treatment optimization
    - Z88x codes → drug allergies (list specific allergens)
    - Z66/Z515 codes → DNR/palliative care goals
    - Weight loss ≥10% → cachexia or disease progression
    - Insurance transition Private→Medicare → age 65 crossed
    - Discharge destinations HOME → HOME HEALTH → SNF → LTACH → hospital death = functional decline
    - Readmission within 30 days → unstable underlying condition
    - Tacrolimus/Mycophenolate/steroids → transplant recipient (infection and rejection risk)
    - eMAR "Not Given" for critical drug (immunosuppressant, anticoagulant) → adherence risk
    - Blood products + vasopressors + mechanical ventilation in ICU → critical illness severity
    - Time from last discharge to death <30 days → rapid terminal decline

**Evidence anchors are required**: Every insight must cite specific ICD codes, exact dates, drug names with doses, organism names, DRG severity/mortality scores, or numeric values. Vague summaries without supporting data are not acceptable.

End the analysis with `FINISH:` followed by the full summary.

## Efficiency Tips

- Query all admissions first, then drill into individual `hadm_id` or `stay_id` for detail
- For patients with multiple admissions, use `subject_id`-level queries before `hadm_id`-level ones
- If a query fails with a column error, correct the column name immediately using the pitfalls table above — do not call `describe_table`
- **Pagination**: For large tables (eMAR, transfers, microbiologyevents, prescriptions, ICU inputs), use `LIMIT 50` first. If results are truncated, paginate with `LIMIT 50 OFFSET 50`, etc. For `icu_inputevents`, pagination is critical — blood products and nutrition often appear after the first 50 rows
- **Avoid redundant queries**: Each table should be queried with a consolidated SELECT that retrieves all needed columns at once. If initial results were incomplete due to LIMIT, use OFFSET — never reissue the same query with slightly different columns
- For ICU patients with many stays (>3), focus the deep dive on the top-severity stays (longest LOS or highest APR-DRG severity), not all stays exhaustively
- Skip ICU event tables entirely when `icu_icustays` returns empty
- For non-ICU patients (no ICU stays), spend the saved time on deeper eMAR and POE analysis to characterize ward-level care intensity
