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

Systematically explore a patient's complete clinical record and submit diverse, high-quality QA pairs covering all meaningful clinical domains. Target 20–30 QA pairs for complex patients with multiple admissions; 10–15 for simple cases.

## Database Overview

27 tables with two prefixes:
- **`hosp_`** — hospital-level data (diagnoses, procedures, medications, admissions, labs)
- **`icu_`** — ICU-specific data (stays, inputs/outputs, procedures, events)

Three metadata tables: `table_comments`, `column_comments`, `column_documentation`

Start with `get_database_info` to confirm available tables.

## Key Column Names (Common Pitfalls)

Incorrect column names are the #1 cause of failed queries.

| Table | Use This | NOT This |
|---|---|---|
| `hosp_d_icd_diagnoses` | `long_title` | `description`, `title` |
| `hosp_d_icd_procedures` | `long_title` | `description` |
| `icu_icustays` | `los` | `length`, `length_of_stay` |
| `hosp_omr` | `subject_id`, `chartdate`, `result_name`, `result_value` | `hadm_id`, `charttime`, `result_unit` |
| `hosp_drgcodes` | `description` (own column, no JOIN needed) | joining a separate dictionary |
| `hosp_hcpcsevents` | `hcpcs_cd`, `short_description` | joining `hosp_d_hcpcs` on `hcpcs_cd` |
| `hosp_pharmacy` | `medication`, `route`, `frequency` | `drug` |
| `hosp_emar` | `medication`, `event_txt`, `charttime` | `route`, `dose_val_rx` |
| `hosp_prescriptions` | `starttime`, `doses_per_24_hrs` | `start_date`, `frequency` |
| `hosp_poe` | `order_type`, `order_subtype`, `ordertime` | `order_name` |
| `hosp_transfers` | `careunit`, `intime`, `outtime`, `eventtype` | `unit`, `transfer_type` |
| `hosp_services` | `transfertime`, `curr_service`, `prev_service` | `starttime` |
| `hosp_microbiologyevents` | `spec_type_desc`, `org_name`, `ab_name`, `interpretation` | `specimen_type`, `organism_name` |

**Critical**: `hosp_labevents` does NOT exist. Use `hosp_omr` for outpatient measurements. Use `icu_inputevents`/`icu_outputevents` for ICU lab-like data.

## Core JOIN Patterns

```sql
-- Tables with hadm_id only (no subject_id): JOIN through hosp_admissions
SELECT ... FROM hosp_services s
JOIN hosp_admissions ha ON s.hadm_id = ha.hadm_id
WHERE ha.subject_id = <subject_id>

-- hosp_omr: query directly by subject_id
SELECT chartdate, result_name, result_value
FROM hosp_omr WHERE subject_id = <subject_id> ORDER BY chartdate

-- ICD diagnosis with readable title
SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, dt.long_title
FROM hosp_diagnoses_icd d
JOIN hosp_d_icd_diagnoses dt ON d.icd_code = dt.icd_code AND d.icd_version = dt.icd_version
WHERE d.subject_id = <subject_id>

-- ICU stays: JOIN through hosp_admissions (icu_icustays lacks subject_id)
SELECT ic.* FROM icu_icustays ic
JOIN hosp_admissions ha ON ic.hadm_id = ha.hadm_id
WHERE ha.subject_id = <subject_id>

-- ICU inputs (medications/fluids): query by stay_id
SELECT ie.starttime, di.label, ie.amount, ie.amountuom
FROM icu_inputevents ie JOIN icu_d_items di ON ie.itemid = di.itemid
WHERE ie.stay_id = <stay_id> ORDER BY ie.starttime

-- ICU inputs aggregated (total per medication)
SELECT di.label, SUM(ie.amount) as total, ie.amountuom
FROM icu_inputevents ie JOIN icu_d_items di ON ie.itemid = di.itemid
WHERE ie.stay_id = <stay_id> GROUP BY di.label, ie.amountuom ORDER BY total DESC

-- ICU outputs (urine, drainage): use oe.value + oe.valueuom
SELECT di.label, SUM(oe.value) as total, oe.valueuom
FROM icu_outputevents oe JOIN icu_d_items di ON oe.itemid = di.itemid
WHERE oe.stay_id = <stay_id> GROUP BY di.label, oe.valueuom

-- ICU procedures (ventilation, dialysis)
SELECT pe.starttime, pe.endtime, di.label, pe.value, pe.valueuom
FROM icu_procedureevents pe JOIN icu_d_items di ON pe.itemid = di.itemid
WHERE pe.stay_id = <stay_id>
```

When a query fails with "no such column", check `column_comments`:
```sql
SELECT column_name, comment FROM column_comments WHERE table_name = '<table>'
```

## Systematic Exploration Order

### Phase 1 — Foundation (always first)
1. **Patient demographics**: `hosp_patients` → age, gender, date of death
2. **Admissions overview**: `hosp_admissions` → count, dates, admission types, insurance, discharge locations, in-hospital deaths. For many admissions, query total count first, then fetch in batches.
3. **Diagnoses**: `hosp_diagnoses_icd` JOIN `hosp_d_icd_diagnoses` → primary and comorbid conditions. Results are capped at 100 rows; use OFFSET to paginate if needed.
4. **Procedures**: `hosp_procedures_icd` JOIN `hosp_d_icd_procedures` → surgical and clinical interventions
5. **ICU stays**: `icu_icustays` (JOIN through `hosp_admissions`) → LOS, care units, timing

### Phase 2 — Care Context
6. **Clinical services**: `hosp_services` → service transitions per admission
7. **Prescriptions**: `hosp_prescriptions` → drugs, routes, dosing. Get distinct drugs or frequency counts to avoid drowning in detail.
8. **DRG classifications**: `hosp_drgcodes` → billing severity and mortality risk (description is inline, no JOIN needed)
9. **Transfers**: `hosp_transfers` → intra-hospital care unit movements
10. **Microbiology**: `hosp_microbiologyevents` → organisms, antibiotic sensitivities (include `ab_name` and `interpretation` columns)

### Phase 3 — Clinical Depth (pursue based on what you find)
11. **ICU inputs/outputs**: For each ICU stay, query `icu_inputevents` and `icu_outputevents` (by `stay_id`) → continuous infusions, vasopressors, urine output, drainage. Use aggregated queries (`GROUP BY di.label, SUM(amount)`) to efficiently identify the most significant medications and fluid balance.
12. **ICU procedures**: `icu_procedureevents` → ventilation details, duration, dialysis
13. **Outpatient measurements**: `hosp_omr` → weight, BMI, blood pressure trends
14. **eMAR**: `hosp_emar` → actual medication administrations (vs. just orders)
15. **Provider orders**: `hosp_poe` → order type distribution via `COUNT(*) GROUP BY order_type`
16. **HCPCS events**: `hosp_hcpcsevents` → billed services/procedures

### Phase 4 — Synthesis
17. Look for longitudinal trends: disease progression, care escalation, discharge destination changes
18. Identify clinically interesting patterns: unusual comorbidity combinations, high-severity DRGs, recurrent infections, resistance patterns, ICU readmissions

**Aggregation tip**: When a table returns truncated results, use `COUNT(*)` to know the total scope, `GROUP BY` to summarize by category, and `OFFSET` to paginate. Prefer compact aggregate queries over many sequential offset queries.

## QA Generation Strategy

### Coverage Targets

Generate QA pairs across these domains. Not all apply to every patient — focus on what's clinically rich in this patient's data.

| Domain | Example question angles |
|---|---|
| Primary diagnosis & admission drivers | What condition drove each admission? What was the sequence of complications? |
| Comorbid conditions | What chronic diseases appear consistently across admissions? |
| Surgical/procedural interventions | What procedures were performed and when? What was the clinical indication? |
| Medication regimen | What drug classes were prescribed? What were the dosing details for critical medications? |
| Care trajectory | How did admission frequency, sources, and discharge destinations change over time? |
| Clinical service assignments | Which services managed the patient and when did they transition? |
| ICU care | What continuous infusions (vasopressors, sedation, antibiotics) were used? What were total dosages? What was fluid balance? |
| Infectious complications | What organisms were cultured? What was the antibiotic resistance/sensitivity pattern? |
| DRG severity | What DRG classifications reflect care complexity? How did severity scores change? |
| Discharge & outcomes | Where was the patient discharged across admissions? Any in-hospital deaths? DNR/advance directive documentation? |
| Longitudinal trends | How did weight, BMI, blood pressure change over the observation period? |
| Transfer & care unit patterns | What was the intra-hospital care unit progression during complex admissions? |
| Medication frequency analysis | What were the most prescribed medications across all admissions? |
| Provider order patterns | What order types dominated this patient's care? |

### QA Quality Standards

**Write QA pairs that are:**
- **Specific**: Include concrete values — exact dates, drug names with doses, organism names with resistance patterns, procedure names, LOS values. Vague answers like "multiple medications were given" are low value.
- **Self-contained**: The question and answer together tell a complete, verifiable clinical story without needing to look up the data.
- **Clinically meaningful**: Focus on facts that matter for understanding the patient's care — not database structure questions.
- **Diverse**: Each QA pair should cover a different aspect. Avoid multiple pairs that essentially say the same thing.
- **Cross-cutting when appropriate**: Some of the best QA pairs synthesize across multiple admissions or data sources (e.g., "How did the patient's care escalate over 5 years?").

**Avoid:**
- Generic demographic-only pairs ("The patient is a 45-year-old male on Medicare") — embed demographics into richer clinical context
- Questions answerable from a single trivial fact
- Redundant pairs covering the same information in slightly different wording
- Questions about database schema or column names

### High-Value QA Types

These question patterns tend to produce rich, specific QA pairs:

1. **ICU medication details**: "What vasopressors/sedatives were used during [ICU stay] and in what amounts?" (requires icu_inputevents aggregation)
2. **Antibiotic resistance patterns**: "What organisms were identified and what was the resistance/sensitivity pattern?" (requires hosp_microbiologyevents with ab_name + interpretation)
3. **Longitudinal trajectory**: "How did the patient's discharge destinations change over [N] years?" or "What was the pattern of disease escalation?"
4. **Specific procedural technique**: "What specific approach was used for [procedure] and what was the clinical indication?"
5. **Medication regimen across admissions**: "What were the most frequently prescribed medications across all admissions?"
6. **DRG severity evolution**: "How did DRG severity and mortality scores change over successive admissions?"
7. **Care unit progression during complex admission**: "What care units did the patient transit through during their longest hospitalization and in what order?"
8. **Fluid balance during ICU**: "What were the total inputs and outputs during the [ICU stay]?" (requires icu_inputevents + icu_outputevents)
9. **Advance care planning**: "When was DNR status first documented and how consistently was it recorded?"
10. **Admission pattern analysis**: "What was the distribution of admission types, sources, and frequency over the observation period?"

### Submission Pattern

Submit QA pairs after each thematic cluster — don't wait until the end. Interleave: query → verify data → submit 1-3 QA pairs → continue exploring. This ensures progress is saved if the session ends early.

## Handling Query Failures

When a query fails:
1. Read the error — it often shows available columns for that table
2. Correct the column name and retry once
3. If still failing, check `column_comments` for the correct schema
4. If a table doesn't exist (e.g., `hosp_labevents`), use the alternatives listed above

Do not spend more than 2 retries on any single query — move on if data isn't available.
