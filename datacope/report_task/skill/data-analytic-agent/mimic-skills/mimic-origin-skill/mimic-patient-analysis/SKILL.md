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
- `icu_inputevents`, `icu_outputevents`, `icu_procedureevents` — ICU events by `stay_id`

## Critical Column Name Pitfalls

Avoid these common errors that cause query failures:

| Table | WRONG | CORRECT |
|-------|-------|---------|
| `hosp_transfers` | `transfertime` | `intime` (sort by `intime`) |
| `hosp_poe` | `order_time` | `ordertime` |
| `hosp_omr` | `charttime` | `chartdate` |
| `hosp_hcpcsevents` JOIN `hosp_d_hcpcs` | `ON h.hcpcs_cd = d.hcpcs_cd` | `ON h.hcpcs_cd = d.code` |

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

### Step 3 — ICU stays
```sql
SELECT * FROM icu_icustays WHERE subject_id = <patient_id> ORDER BY intime
```
Empty result = no ICU. If ICU present, note care units and length of stay (los).

### Step 4 — Diagnoses (with human-readable names)
```sql
SELECT d.icd_code, d.icd_version, d.long_title, diag.hadm_id, diag.seq_num
FROM hosp_diagnoses_icd diag
JOIN hosp_d_icd_diagnoses d ON diag.icd_code = d.icd_code AND diag.icd_version = d.icd_version
WHERE diag.subject_id = <patient_id>
ORDER BY diag.hadm_id, diag.seq_num
```
seq_num=1 is the primary diagnosis. Group by hadm_id to see diagnoses per admission.

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

### Step 7 — Physical measurements (BMI, weight, height, BP)
```sql
SELECT chartdate, result_name, result_value
FROM hosp_omr
WHERE subject_id = <patient_id>
ORDER BY chartdate
```

### Step 8 — Microbiology cultures
```sql
SELECT chartdate, spec_type_desc, test_name, org_name, interpretation, comments
FROM hosp_microbiologyevents
WHERE subject_id = <patient_id>
ORDER BY chartdate
```
`org_name` null with a comment like "< 10,000 CFU/mL" = negative culture.

### Step 9 — Clinical service and transfers
```sql
-- Service
SELECT * FROM hosp_services WHERE subject_id = <patient_id> ORDER BY transfertime

-- Physical location movements
SELECT * FROM hosp_transfers WHERE hadm_id = <hadm_id> ORDER BY intime
```

### Step 10 — DRG billing codes
```sql
SELECT * FROM hosp_drgcodes WHERE subject_id = <patient_id>
```
APR-DRG has severity (1-4) and mortality (1-4) scores.

### Optional — Provider orders and eMAR
Query `hosp_poe` (by `hadm_id`, ORDER BY `ordertime`) to see what was ordered and when. Query `hosp_emar` (by `hadm_id`, ORDER BY `charttime`) to see which medications were actually administered vs "Not Given".

## Synthesizing the Analysis

After gathering data, produce a structured summary covering:

1. **Demographics** — age, sex, race, insurance, vital status (alive/deceased + date if known)
2. **Admission summary** — number of admissions, dates, types, sources, discharge destinations
3. **ICU course** — whether ICU was needed, which units, duration
4. **Primary diagnosis and DRG** — main condition(s), billing classification, severity
5. **Comorbidities** — significant secondary diagnoses across admissions
6. **Procedures** — surgical and therapeutic interventions
7. **Medications** — key drug classes, notable transitions (e.g., anticoagulation changes), route
8. **Diagnostics** — culture results, physical measurements/trends
9. **Clinical trajectory** — how the patient's condition evolved across admissions
10. **Key clinical insights** — clinically meaningful patterns (e.g., discharge to rehab suggesting functional impairment, multiple laxatives suggesting immobility, sequential anticoagulants suggesting treatment adjustment)

End the analysis with `FINISH:` followed by the full summary.

## Efficiency Tips

- Query all admissions first, then drill into individual `hadm_id` values for detailed data
- For patients with multiple admissions, query diagnoses and medications across all `hadm_id`s at once using `subject_id`
- If a query returns an error mentioning available columns, correct the column name immediately — do not call `describe_table`; instead consult the column reference above
- Use LIMIT when exploring eMAR/POE (large tables); paginate with OFFSET if needed
- The `hosp_emar` `event_txt` field distinguishes "Administered" from "Not Given" — this reveals medication compliance
