---
name: mimic-iv-patient-analysis
description: >
  Comprehensive strategy for analyzing individual patient records in MIMIC-IV EHR database
  and generating high-quality, diverse QA pairs. Use this skill whenever the task involves
  analyzing a specific patient's clinical data from MIMIC-IV (or similar EHR databases),
  querying across hospital and ICU tables, and submitting QA pairs that cover the patient's
  full clinical story — diagnoses, procedures, medications, care trajectory, and outcomes.
  Trigger when you see tasks like "Analyze patient <ID>", "Generate QA pairs for patient",
  or any patient-centric EHR exploration task.
---

# MIMIC-IV Patient Analysis: Comprehensive QA Generation

## Goal

Systematically explore a patient's complete clinical record and submit diverse, high-quality QA pairs covering all meaningful clinical domains. Successful analyses generate 12–24 QA pairs spanning multiple dimensions of care.

## Database Overview

The database has 27 tables with two prefixes:
- **`hosp_`** — hospital-level data (diagnoses, procedures, medications, admissions, labs)
- **`icu_`** — ICU-specific data (stays, inputs/outputs, procedures, events)

Three special metadata tables: `table_comments`, `column_comments`, `column_documentation`

Start with `get_database_info` to confirm available tables, then proceed without re-describing every table.

## Key Column Names (Common Pitfalls)

Incorrect column names are the #1 cause of failed queries. Memorize these:

| Table | Use This | NOT This |
|---|---|---|
| `hosp_d_icd_diagnoses` | `long_title` | `description`, `title` |
| `hosp_d_icd_procedures` | `long_title` | `description` |
| `icu_icustays` | `los` | `length`, `length_of_stay` |
| `hosp_omr` | `subject_id`, `chartdate`, `result_name`, `result_value` | `hadm_id`, `charttime`, `result_unit` |
| `hosp_drgcodes` | `description` (own column, no JOIN needed) | joining a separate dictionary |
| `hosp_d_hcpcs` | `code` (join key), `short_description` | `hcpcs_cd` as join key |
| `hosp_hcpcsevents` | `hcpcs_cd`, `short_description` | joining `hosp_d_hcpcs` on `hcpcs_cd` |
| `hosp_pharmacy` | `medication`, `route`, `frequency` | `drug` |
| `hosp_emar` | `medication`, `event_txt`, `charttime` | `route`, `dose_val_rx` |
| `hosp_prescriptions` | `starttime`, `doses_per_24_hrs` | `start_date`, `frequency` |
| `hosp_poe` | `order_type`, `order_subtype`, `ordertime` | `order_name` |
| `hosp_transfers` | `careunit`, `intime`, `outtime`, `eventtype` | `unit`, `transfer_type` |
| `hosp_services` | `transfertime`, `curr_service`, `prev_service` | `starttime` |
| `hosp_microbiologyevents` | `spec_type_desc`, `org_name`, `ab_name`, `interpretation` | `specimen_type`, `organism_name` |

**Critical**: `hosp_labevents` does NOT exist. For outpatient measurements use `hosp_omr`. For ICU lab-like data use `icu_d_items` + `icu_inputevents`/`icu_outputevents`.

## JOIN Patterns

Many tables store `hadm_id` but not `subject_id`. To filter by patient:
```sql
-- Pattern for tables with hadm_id only
SELECT ... FROM hosp_services s
JOIN hosp_admissions ha ON s.hadm_id = ha.hadm_id
WHERE ha.subject_id = <subject_id>

-- hosp_omr only has subject_id — query directly
SELECT chartdate, result_name, result_value
FROM hosp_omr WHERE subject_id = <subject_id>
ORDER BY chartdate

-- ICD diagnosis with readable title
SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, dt.long_title
FROM hosp_diagnoses_icd d
JOIN hosp_d_icd_diagnoses dt ON d.icd_code = dt.icd_code AND d.icd_version = dt.icd_version
WHERE d.subject_id = <subject_id>

-- DRG codes (description is already in hosp_drgcodes)
SELECT drg_type, drg_code, description, drg_severity, drg_mortality
FROM hosp_drgcodes WHERE subject_id = <subject_id>

-- ICU stays for a patient
SELECT ic.* FROM icu_icustays ic
JOIN hosp_admissions ha ON ic.hadm_id = ha.hadm_id
WHERE ha.subject_id = <subject_id>
```

When a query fails with "no such column", check `column_comments` for the correct name:
```sql
SELECT column_name, comment FROM column_comments WHERE table_name = '<table>'
```

## Systematic Exploration Order

Work through domains in this order, querying and drafting QA pairs as you go:

### Phase 1 — Foundation (always do first)
1. **Patient demographics**: `hosp_patients` → age, gender, date of death
2. **Admissions overview**: `hosp_admissions` → count, dates, admission types, insurance, discharge locations, in-hospital deaths
3. **Diagnoses**: `hosp_diagnoses_icd` JOIN `hosp_d_icd_diagnoses` → primary and comorbid conditions
4. **Procedures**: `hosp_procedures_icd` JOIN `hosp_d_icd_procedures` → surgical and clinical interventions
5. **Clinical services**: `hosp_services` → service transitions per admission

### Phase 2 — Medications & Care (do for all patients)
6. **Prescriptions**: `hosp_prescriptions` → drugs, routes, dosing
7. **Pharmacy orders**: `hosp_pharmacy` → `medication`, `route`, `frequency`
8. **Transfers & care units**: `hosp_transfers` → intra-hospital movements, care unit progression
9. **DRG classifications**: `hosp_drgcodes` → billing severity and mortality risk

### Phase 3 — Detailed Clinical Data (pursue based on what you find)
10. **ICU stays**: `icu_icustays` → if ICU admissions exist, explore `icu_inputevents`, `icu_outputevents`, `icu_procedureevents` using `stay_id`
11. **Microbiology**: `hosp_microbiologyevents` → infections, cultures, antibiotic sensitivities
12. **Outpatient measurements**: `hosp_omr` → weight, BMI, blood pressure trends over time
13. **eMAR**: `hosp_emar` → actual medication administrations (vs. just orders)
14. **HCPCS events**: `hosp_hcpcsevents` → billed procedures/services

### Phase 4 — Synthesis
15. Look for longitudinal trends: disease progression, care escalation, discharge destination changes
16. Identify clinically interesting patterns: unusual comorbidity combinations, high-severity DRGs, recurrent infections

## QA Generation Strategy

### Coverage Targets
Generate QA pairs across these domains (not all may be relevant for every patient):

| Domain | Example question angles |
|---|---|
| Primary diagnosis & chief complaint | What condition drove this admission? What intervention was performed? |
| Comorbid conditions | What chronic diseases complicate this patient's care? |
| Surgical/procedural interventions | What procedures were performed? What was the clinical indication? |
| Medication regimen | What drug classes were prescribed? Why? (anticoagulants, immunosuppressants, etc.) |
| Care trajectory & hospitalization pattern | How many admissions? What was the progression over time? |
| Clinical service assignments | Which services managed this patient and when did they transition? |
| ICU care | What critical care interventions were used? How long was ICU stay? |
| Infectious complications | What organisms were cultured? What was the treatment pattern? |
| DRG severity & billing | What DRG classifications reflect the complexity of care? |
| Discharge & outcomes | Where was the patient discharged? Did they die in-hospital or post-discharge? |
| Longitudinal trends | How did weight, vitals, or disease burden change over time? |
| Transfer & care unit patterns | How did the patient move through the hospital? |

### QA Quality Standards

**Write QA pairs that are:**
- Self-contained: the question and answer together tell a complete clinical story
- Specific: include concrete values (dates, quantities, drug names, diagnoses), not vague generalities
- Clinically meaningful: focus on facts that matter for understanding the patient's care
- Diverse: each QA pair should cover a different clinical domain or aspect

**Avoid:**
- Redundant QA pairs covering the same information in slightly different words
- Questions answerable without querying the data (too obvious)
- Questions about columns or database structure (not clinical)
- Submitting before you've verified the data from a query

### Submission Pattern

Submit QA pairs after each thematic cluster of queries — don't wait until the end. Aim to interleave: query → verify data → submit 1-2 QA pairs → continue exploring. This ensures you don't lose work if the session ends early.

A good target is 12–20+ QA pairs for a patient with multiple admissions, fewer (8–12) for simple single-admission cases.

## Handling Query Failures

When a query fails:
1. Read the error — it shows the available columns for that table
2. Correct the column name immediately and retry
3. If a table doesn't exist (e.g., `hosp_labevents`), use the alternative listed above

Do not spend more than 2 retries on any single query — move on if data isn't available.
