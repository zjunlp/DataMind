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
- `hosp_pharmacy` — pharmacy fills: `subject_id, hadm_id, pharmacy_id, drug, starttime, stoptime`

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

**Track discharge destination progression across admissions** (HOME → HOME HEALTH CARE → SNF → LTACH → died in hospital) as it signals functional decline trajectory.

### Step 3 — ICU stays
```sql
SELECT * FROM icu_icustays WHERE subject_id = <patient_id> ORDER BY intime
```
Empty result = no ICU. If ICU present, note care units and length of stay (`los`).

#### Step 3.5 — ICU deep dive (when ICU stays exist)

For each `stay_id`, query ICU event tables for critical clinical detail:

```sql
-- Procedures (ventilation, dialysis, arterial lines, etc.)
SELECT pe.starttime, pe.endtime, d.label, d.category, pe.value, pe.valueuom
FROM icu_procedureevents pe
JOIN icu_d_items d ON pe.itemid = d.itemid
WHERE pe.stay_id = <stay_id>
ORDER BY pe.starttime

-- Key inputs (fluids, vasopressors, medications)
SELECT ie.starttime, d.label, d.category, ie.amount, ie.amountuom, ie.ordercategoryname
FROM icu_inputevents ie
JOIN icu_d_items d ON ie.itemid = d.itemid
WHERE ie.stay_id = <stay_id>
ORDER BY ie.starttime LIMIT 50

-- Outputs (urine, drainage)
SELECT oe.charttime, d.label, oe.value, oe.valueuom
FROM icu_outputevents oe
JOIN icu_d_items d ON oe.itemid = d.itemid
WHERE oe.stay_id = <stay_id>
ORDER BY oe.charttime LIMIT 30
```

From ICU events, capture: mechanical ventilation duration, vasopressor use, fluid balance (total inputs vs outputs), dialysis/CRRT, invasive monitoring (arterial line, central line).

### Step 4 — Diagnoses (with human-readable names)
```sql
SELECT d.icd_code, d.icd_version, d.long_title, diag.hadm_id, diag.seq_num
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id>
ORDER BY diag.hadm_id, diag.seq_num
```
`seq_num=1` is the primary diagnosis. Note special ICD codes:
- **Z88x** = drug allergy documentation (e.g., Z880 = penicillin allergy)
- **Z66** = do not resuscitate (DNR) order
- **Z515** = encounter for palliative care
- **Z79x** = long-term medication use (e.g., Z7901 = anticoagulants)

**For patients with 4+ admissions**, also run an aggregate query to identify recurring diagnoses:
```sql
SELECT d.long_title, COUNT(*) as admission_count
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id> AND diag.seq_num <= 5
GROUP BY d.long_title
ORDER BY admission_count DESC
LIMIT 20
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
SELECT hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx, route
FROM hosp_prescriptions
WHERE subject_id = <patient_id>
ORDER BY hadm_id, starttime
```

**For patients with 4+ admissions**, rank medications by frequency:
```sql
SELECT drug, COUNT(*) as prescription_count
FROM hosp_prescriptions
WHERE subject_id = <patient_id>
GROUP BY drug
ORDER BY prescription_count DESC
LIMIT 20
```

### Step 7 — Physical measurements (BMI, weight, height, BP)
```sql
SELECT chartdate, result_name, result_value
FROM hosp_omr
WHERE subject_id = <patient_id>
ORDER BY chartdate
```

For longitudinal patients, separately track weight and blood pressure trends:
```sql
SELECT chartdate, result_value FROM hosp_omr
WHERE subject_id = <patient_id> AND result_name = 'Weight (Lbs)'
ORDER BY chartdate

SELECT chartdate, result_value FROM hosp_omr
WHERE subject_id = <patient_id> AND result_name = 'Blood Pressure'
ORDER BY chartdate
```

**Weight loss ≥5% from baseline is clinically significant**; ≥10% suggests disease-related cachexia or malnutrition.

### Step 8 — Microbiology cultures
```sql
SELECT chartdate, spec_type_desc, test_name, org_name, interpretation, comments
FROM hosp_microbiologyevents
WHERE subject_id = <patient_id>
ORDER BY chartdate
```
`org_name` null with a comment like "< 10,000 CFU/mL" = negative culture. For positive cultures, record the organism name, specimen type, and interpretation.

### Step 9 — Clinical service and transfers
```sql
-- Service
SELECT * FROM hosp_services WHERE subject_id = <patient_id> ORDER BY transfertime

-- Physical location movements (per hadm_id)
SELECT * FROM hosp_transfers WHERE hadm_id = <hadm_id> ORDER BY intime
```

### Step 10 — DRG billing codes
```sql
SELECT * FROM hosp_drgcodes WHERE subject_id = <patient_id>
```
APR-DRG has severity (1-4) and mortality (1-4) scores. Severity 3-4 or mortality 3-4 indicates a high-complexity/high-risk admission.

### Step 11 — Provider orders and eMAR (targeted)
```sql
-- Provider orders overview
SELECT ordertime, order_type, order_subtype, transaction_type, order_status
FROM hosp_poe WHERE subject_id = <patient_id> AND hadm_id = <hadm_id>
ORDER BY ordertime LIMIT 30

-- Medication administration (check "Not Given" vs "Administered")
SELECT charttime, medication, event_txt, scheduletime
FROM hosp_emar WHERE subject_id = <patient_id> AND hadm_id = <hadm_id>
ORDER BY charttime LIMIT 50
```

The `hosp_emar` `event_txt` field distinguishes "Administered" from "Not Given" — this reveals medication compliance and route changes (PO/NG route = nasogastric feeding, suggesting dysphagia).

## Synthesizing the Analysis

After gathering data, produce a structured summary covering:

1. **Demographics** — age, sex, race, insurance, vital status (alive/deceased + date if known)
2. **Admission summary** — number of admissions, date range, types, sources, discharge destinations; explicitly note trajectory (e.g., progressive shift to institutional discharge = functional decline)
3. **ICU course** — whether ICU was needed, which units, total duration, key interventions (ventilation, vasopressors, fluid balance if queried)
4. **Primary diagnoses by admission** — primary condition per hadm_id; use a table format for multi-admission patients
5. **Comorbidities** — significant secondary diagnoses across admissions; for multi-admission patients, note which conditions appear across how many admissions
6. **Procedures** — surgical and therapeutic interventions with dates
7. **Medications** — key drug classes, notable transitions or polypharmacy; for multi-admission patients, list top prescriptions by frequency
8. **Diagnostics** — positive culture results (organism + specimen), physical measurement trends (weight trajectory, BP range)
9. **Clinical service trajectory** — services and care unit progression across admissions
10. **Key clinical insights** — clinically meaningful patterns with explanations:
    - Discharge to rehab/SNF → functional impairment
    - Multiple laxatives (Senna + Bisacodyl + Docusate) → immobility or opioid use
    - PO/NG drug routes → nasogastric feeding (likely dysphagia)
    - Sequential anticoagulant changes → treatment optimization
    - Z88x codes → drug allergies
    - Z66/Z515 codes → DNR/palliative care goals
    - Weight loss ≥10% → cachexia or disease progression
    - Insurance transition Private→Medicare → age 65 crossed during observation period
    - Discharge destinations: HOME → HOME HEALTH → SNF → LTACH → hospital death = functional decline

**Include evidence anchors**: specific ICD codes, exact dates, drug names with doses, organism names, DRG severity/mortality scores, and weight values. These factual anchors make the analysis verifiable and clinically useful.

End the analysis with `FINISH:` followed by the full summary.

## Efficiency Tips

- Query all admissions first, then drill into individual `hadm_id` values for detailed data
- For patients with multiple admissions, use `subject_id`-level queries before `hadm_id`-level ones
- If a query fails with a column error, correct the column name immediately using the pitfalls table above — do not call `describe_table`
- Use LIMIT when exploring eMAR/POE (large tables); paginate with OFFSET if needed
- For ICU patients, prioritize `icu_procedureevents` (procedures are most clinically discriminating) over exhaustive input/output enumeration
- Skip ICU event tables entirely when `icu_icustays` returns empty
