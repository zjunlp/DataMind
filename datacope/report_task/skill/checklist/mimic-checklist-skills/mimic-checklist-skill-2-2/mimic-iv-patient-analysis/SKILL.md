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

Systematically explore a patient's complete clinical record and submit diverse, high-quality QA pairs covering all meaningful clinical domains. Target **25–40 QA pairs** for complex patients (multiple admissions, ICU stays, rich medication history); **15–20** for simpler cases. More QA pairs of focused scope are better than fewer compound ones.

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
| `hosp_prescriptions` | `drug`, `starttime`, `doses_per_24_hrs` | `medication`, `start_date`, `frequency` |
| `hosp_pharmacy` | `medication`, `route`, `frequency` | `drug` |
| `hosp_emar` | `medication`, `event_txt`, `charttime` | `route`, `dose_val_rx` |
| `hosp_d_icd_diagnoses` | `long_title` | `description`, `title` |
| `hosp_d_icd_procedures` | `long_title` | `description` |
| `icu_icustays` | `los` | `length`, `length_of_stay` |
| `hosp_omr` | `subject_id`, `chartdate`, `result_name`, `result_value` | `hadm_id`, `charttime`, `result_unit` |
| `hosp_drgcodes` | `description` (own column, no JOIN needed) | joining a separate dictionary |
| `hosp_hcpcsevents` | `hcpcs_cd`, `short_description` | joining `hosp_d_hcpcs` on `hcpcs_cd` |
| `hosp_poe` | `order_type`, `order_subtype`, `ordertime` | `order_name` |
| `hosp_transfers` | `careunit`, `intime`, `outtime`, `eventtype` | `unit`, `transfer_type` |
| `hosp_services` | `transfertime`, `curr_service`, `prev_service` | `starttime` |
| `hosp_microbiologyevents` | `spec_type_desc`, `org_name`, `ab_name`, `interpretation` | `specimen_type`, `organism_name` |

**Critical drug column distinction**: `hosp_prescriptions` uses `drug` (orders). `hosp_pharmacy` and `hosp_emar` use `medication` (dispensed/administered). Using `medication` in `hosp_prescriptions` will always fail.

**Critical**: `hosp_labevents` does NOT exist. Use `hosp_omr` for outpatient measurements. Use `icu_inputevents`/`icu_outputevents` for ICU lab-like data.

## Core JOIN Patterns

```sql
-- hosp_prescriptions: drug frequency across all admissions
SELECT drug, COUNT(*) as cnt FROM hosp_prescriptions
WHERE hadm_id IN (SELECT hadm_id FROM hosp_admissions WHERE subject_id = <sid>)
GROUP BY drug ORDER BY cnt DESC LIMIT 20

-- hosp_pharmacy: dispensed medications (uses medication, not drug; no subject_id column)
SELECT medication, route, frequency, COUNT(*) as cnt
FROM hosp_pharmacy
WHERE hadm_id IN (SELECT hadm_id FROM hosp_admissions WHERE subject_id = <sid>)
GROUP BY medication ORDER BY cnt DESC LIMIT 20

-- hosp_omr: query directly by subject_id (outpatient measurements)
SELECT chartdate, result_name, result_value
FROM hosp_omr WHERE subject_id = <sid> ORDER BY chartdate

-- ICD diagnosis with readable title
SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, dt.long_title
FROM hosp_diagnoses_icd d
JOIN hosp_d_icd_diagnoses dt ON d.icd_code = dt.icd_code AND d.icd_version = dt.icd_version
WHERE d.subject_id = <sid>

-- Tables with hadm_id only (no subject_id): JOIN through hosp_admissions
SELECT ... FROM hosp_services s
JOIN hosp_admissions ha ON s.hadm_id = ha.hadm_id
WHERE ha.subject_id = <sid>

-- ICU stays: JOIN through hosp_admissions
SELECT ic.stay_id, ic.hadm_id, ic.intime, ic.outtime, ic.los, ic.first_careunit
FROM icu_icustays ic
JOIN hosp_admissions ha ON ic.hadm_id = ha.hadm_id
WHERE ha.subject_id = <sid>

-- ICU inputs aggregated (total per medication)
SELECT di.label, SUM(ie.amount) as total, ie.amountuom
FROM icu_inputevents ie JOIN icu_d_items di ON ie.itemid = di.itemid
WHERE ie.stay_id = <stay_id> GROUP BY di.label, ie.amountuom ORDER BY total DESC

-- ICU outputs (urine, drainage)
SELECT di.label, SUM(oe.value) as total, oe.valueuom
FROM icu_outputevents oe JOIN icu_d_items di ON oe.itemid = di.itemid
WHERE oe.stay_id = <stay_id> GROUP BY di.label, oe.valueuom

-- ICU procedures (ventilation, dialysis): duration in minutes
SELECT di.label, SUM(pe.value) as total_minutes, pe.valueuom
FROM icu_procedureevents pe JOIN icu_d_items di ON pe.itemid = di.itemid
WHERE pe.stay_id = <stay_id> GROUP BY di.label, pe.valueuom
```

When a query fails with "no such column", check `column_comments`:
```sql
SELECT column_name, comment FROM column_comments WHERE table_name = '<table>'
```

## Systematic Exploration Order

### Phase 1 — Foundation (always first)
1. **Patient demographics**: `hosp_patients` → age, gender, date of death
2. **Admissions overview**: `hosp_admissions` → count, dates, admission types, insurance, discharge locations, in-hospital deaths. For many admissions, query total count first, then fetch in batches.
3. **Diagnoses**: `hosp_diagnoses_icd` JOIN `hosp_d_icd_diagnoses` → primary and comorbid conditions. Use `OFFSET` to paginate if results are capped.
4. **Procedures**: `hosp_procedures_icd` JOIN `hosp_d_icd_procedures` → surgical and clinical interventions
5. **ICU stays**: `icu_icustays` (JOIN through `hosp_admissions`) → LOS, care units, timing

### Phase 2 — Care Context
6. **Clinical services**: `hosp_services` → service transitions per admission
7. **Prescriptions (ordered)**: `hosp_prescriptions` → `GROUP BY drug ORDER BY COUNT(*) DESC` for most-ordered drugs. Use `drug` column, not `medication`.
8. **Pharmacy (dispensed)**: `hosp_pharmacy` → `GROUP BY medication ORDER BY COUNT(*) DESC` for most-dispensed drugs with route/frequency detail. This complements prescriptions and is often more clinically specific.
9. **DRG classifications**: `hosp_drgcodes` → billing severity and mortality risk (description is inline, no JOIN needed)
10. **Transfers**: `hosp_transfers` → intra-hospital care unit movement sequences
11. **Microbiology**: `hosp_microbiologyevents` → organisms, antibiotic sensitivities (always include `ab_name` and `interpretation` columns for resistance patterns)

### Phase 3 — Clinical Depth (when ICU stays exist, do steps 12–13; otherwise pursue as relevant)
12. **ICU inputs/outputs**: For each ICU stay, query `icu_inputevents` and `icu_outputevents` by `stay_id` → aggregate (`GROUP BY di.label, SUM(amount)`) to identify key medications, vasopressors, fluid totals, and urine output. For extended stays (>5 days), also check `icu_ingredientevents` for nutritional formula totals.
13. **ICU procedures**: `icu_procedureevents` → ventilation duration (sum of minutes), dialysis, vascular access lines and their durations
14. **Outpatient measurements**: `hosp_omr` → weight, BMI, blood pressure trends over time
15. **eMAR**: `hosp_emar` → actual medication administrations with `GROUP BY medication, event_txt ORDER BY COUNT(*) DESC`
16. **Provider orders**: `hosp_poe` → `COUNT(*) GROUP BY order_type` for order distribution
17. **HCPCS events**: `hosp_hcpcsevents` → billed services/procedures; identify observation vs. inpatient billing patterns

### Phase 4 — Synthesis
18. Identify clinically interesting patterns: readmission intervals (days between discharge and next admission), disease progression, care escalation, discharge destination evolution, per-admission diagnosis complexity (diagnoses count per hadm_id)
19. Look for cross-cutting themes: recurrent infections with same/different organisms, resistance evolution, DRG severity trajectory, ICU readmissions, pivotal admissions that marked a turning point
20. Check for allergy/implant documentation: search ICD codes Z88x (drug allergies) and Z95-Z96 (implanted devices) for additional clinically meaningful facts

**Aggregation tip**: When a table returns truncated results, use `COUNT(*)` first, then `GROUP BY` for summary, and `OFFSET` to paginate. Prefer compact aggregate queries over many sequential offset queries.

## QA Generation Strategy

### Coverage Targets

Generate QA pairs across these domains — focus on what's clinically rich for this patient.

| Domain | Example question angles |
|---|---|
| Primary diagnoses & admission drivers | What condition drove each admission? Sequence of complications? |
| Comorbid conditions | Which chronic diseases appear across all/most admissions? |
| Surgical/procedural interventions | What procedures were performed, when, and for what indication? |
| Medication regimen | Most prescribed drugs across all admissions? Dosing details for critical medications? |
| Pharmacy dispensing | Most frequently dispensed medications with route/frequency details? |
| Care trajectory | How did admission frequency, sources, and discharge destinations change over time? |
| Clinical service assignments | Which services managed the patient and when did they transition? |
| ICU care — overview | What ICU stays occurred, in which units, and for how long? |
| ICU care — vasopressors | What vasopressors were used and in what total amounts? |
| ICU care — sedation/analgesia | What sedatives and analgesics were administered during critical care? |
| ICU care — anticoagulation | What anticoagulation was used during ICU stays and in what volumes? |
| ICU care — fluid balance | What were total inputs and outputs during the ICU stay? |
| ICU care — vascular access | What lines/catheters were placed and for what durations? |
| Infectious complications | What organisms were cultured? Full resistance/sensitivity per organism? |
| DRG severity | How did DRG classifications and severity scores change over admissions? |
| Discharge & outcomes | Where was the patient discharged across admissions? In-hospital deaths? |
| Longitudinal trends | How did weight, BMI, blood pressure change over the observation period? |
| Transfer & care unit patterns | What was the intra-hospital care unit sequence during complex admissions? |
| Readmission patterns | What were the intervals between discharge and readmission? |
| Admission complexity | How many diagnoses per admission? Which admissions were most diagnostically complex? |
| Nutritional support | What enteral/parenteral nutrition was provided during prolonged ICU stays? |
| Advance care planning | When was DNR status first documented and how consistently maintained? |
| Allergy & implant documentation | What drug allergies or implanted devices are documented? |
| Blood products | What blood product transfusions were administered and during which stays? |
| Observation billing | Which admissions were classified as observation vs. inpatient by HCPCS billing? |

### QA Decomposition Rules

**The single most impactful way to increase QA quantity and quality is decomposing compound topics into separate focused questions.** Apply these rules:

1. **Multiple organisms → one question per organism's resistance pattern** (when ≥2 organisms with susceptibility data exist). Additionally generate a cross-cutting "which admission had the most intensive infectious workup?" question.

2. **ICU stays with rich intervention data → separate questions per medication category**:
   - Vasopressors (norepinephrine, vasopressin, phenylephrine, dopamine)
   - Sedatives/analgesics (propofol, fentanyl, midazolam, dexmedetomidine)
   - Anticoagulation (heparin, argatroban)
   - Fluid balance (total inputs vs outputs)
   - Vascular access devices and line durations
   - Nutritional support (TPN, enteral feeds)

3. **Long medication list → group by drug class for separate QAs** when clinically distinct classes are present:
   - Anticoagulants/antiplatelets
   - Bowel/GI medications (if extensive immobility or opioid use)
   - Cardiac drugs (antiarrhythmics, rate control, vasodilators)
   - Psychiatric/neurological medications
   - Immunosuppressants (post-transplant patients)
   - Pain management regimen

4. **Pivotal admissions → generate admission-specific QA** for the most complex or clinically significant hospitalization (e.g., "What were the diagnoses and care unit progression during the patient's most severe admission [hadm_id]?")

5. **Long observation period with functional changes → split trajectory into sub-questions**: e.g., "How did discharge destinations change over time?" AND "What was the timeline from last discharge to death?" as separate QAs.

### QA Quality Standards

**Strong QA pairs include:**
- **Concrete values**: Exact dates, drug names with doses/totals (e.g., "Heparin 69,193 units"), organism names with full resistance patterns, LOS in days, procedure names with laterality
- **Clinical context**: Not just the fact but why it matters (e.g., "discharged to rehab, indicating functional impairment")
- **Completeness**: Full enumeration when there are only a few items; summaries with top items when there are many
- **Focused scope**: Each QA addresses one specific clinical question — if an answer requires listing medications AND organisms AND procedures, split it into three separate QAs

**Anti-patterns to avoid:**
- Bundling multiple clinical domains into one question: "What was the patient's ICU care including stays, interventions, and medication/fluid administration?" → split into 4–6 focused questions
- Vague counts without specifics: "19 prescription orders were placed" → name the top drugs with counts
- Redundant pairs covering the same information in slightly different wording
- Schema questions ("What columns does this table have?")

**Example: compound → decomposed**
- Compound: "What ICU care did this patient receive, including stays, interventions, and medication/fluid administration?"
- Decomposed:
  1. "What ICU stays did this patient have and what were their durations and care units?"
  2. "What vasopressors were administered during ICU stays and in what total amounts?"
  3. "What sedation and analgesia were used during critical care?"
  4. "What was the fluid balance (inputs vs. outputs) during the longest ICU stay?"
  5. "What vascular access lines were placed and maintained?"

### High-Value QA Types

These patterns tend to produce rich, specific QA pairs:

1. **ICU medication by category**: Separate questions for vasopressors, sedatives, anticoagulation, and nutrition — each requires `icu_inputevents` aggregation with total amounts
2. **Antibiotic resistance per organism**: One question per clinically significant organism with full sensitivity/resistance pattern from `hosp_microbiologyevents`
3. **Longitudinal trajectory**: "How did discharge destinations change over N years?" or "What was the pattern of care escalation?"
4. **Readmission intervals**: "What were the shortest intervals between discharge and subsequent readmission, and what were the associated conditions?"
5. **Specific procedural detail**: "What specific approach was used for [procedure] and what was the clinical indication?"
6. **Drug frequency across all admissions**: "What were the top 10 most prescribed/dispensed medications?" (requires both `hosp_prescriptions` + `hosp_pharmacy`)
7. **DRG severity evolution**: "How did DRG severity and mortality scores change over successive admissions?"
8. **Care unit progression during complex admission**: "What care units did the patient transit through during their longest hospitalization and in what order?"
9. **Fluid balance during ICU**: "What were total inputs and outputs during [ICU stay]?" (requires `icu_inputevents` + `icu_outputevents`)
10. **Advance care planning**: "When was DNR status first documented and how consistently was it recorded?"
11. **Admission pattern analysis**: "What was the distribution of admission types, sources, and frequency over the observation period?"
12. **Per-admission complexity**: "Which admissions had the most diagnoses and what conditions drove their complexity?"
13. **Blood product transfusions**: "What blood products were administered and during which admissions?" (search `icu_inputevents` for PRBCs, FFP, platelets)
14. **Vascular access devices**: "What invasive lines were placed during ICU stays and for what durations?" (from `icu_procedureevents`)
15. **Observation vs. inpatient billing**: "Which admissions were billed as observation stays based on HCPCS records?"
16. **Drug class-specific analysis**: "What anticoagulant medications were prescribed and what was the clinical context?" (when multiple anticoagulants are used)

### Submission Pattern

Submit QA pairs in thematic batches after completing each exploration phase — don't submit one at a time. Interleave: explore a domain → verify data quality → submit 3–6 related QA pairs → continue. This ensures progress is saved and helps maintain thematic coherence.

After completing Phase 3, review your collected QA pairs and ask: "Are there compound QAs I can split into two focused ones? Are there domains I explored but didn't generate QAs for?" Then generate any missing decomposed pairs before final synthesis.

## Handling Query Failures

When a query fails:
1. Read the error — it often lists the available columns for that table
2. Correct the column name using the table above and retry once
3. If still failing, check `column_comments` for the correct schema
4. If a table doesn't exist, use the alternatives listed above

Do not spend more than 2 retries on any single query — move on if data isn't available.
