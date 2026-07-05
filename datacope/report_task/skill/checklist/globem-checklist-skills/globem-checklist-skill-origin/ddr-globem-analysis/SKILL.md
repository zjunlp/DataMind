---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (identified by `pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **10+ high-quality, distinct QA pairs** covering every available data modality.

## Dataset Structure

All CSV files share the columns `pid` (participant ID) and `date`. Filter every file by the target `pid`.

**Sensor files** (92 days per participant, many NaN rows are normal):
- `activity_allday_raw.csv` — daily step count, active/sedentary bout durations
- `sleep_allday_raw.csv` — sleep duration (minutes), efficiency, bedtime/wake-time
- `communication_allday_raw.csv` — incoming/outgoing/missed call counts and durations
- `location_allday_raw.csv` — distance traveled, radius of gyration, home time, number of significant places
- `phone_usage_allday_raw.csv` — unlock episode count and total duration
- `connectivity_allday_raw.csv` — Bluetooth scan count, unique devices

**Assessment files** (one row per observation per participant):
- `ema.csv` — `negative_affect_EMA` score, timestamped across the study period
- `dep_weekly.csv` — weekly `feel_anxious`, `feel_depressed`, `BDI2` (endterm only), `dep` (binary), `dep_weekly_subscale`, `anx_weekly_subscale`
- `pre.csv` / `post.csv` — baseline vs. end-of-study psychological scales (loneliness, perceived stress, anxiety, depression, social support, emotion regulation, resilience, mindfulness, coping)
- `dep_endterm.csv` — final depression label and BDI2 score

**Schema tip**: When column names are unclear, use `get_field_description(data_file="<filename>")` — it works for the six sensor CSV files. For `ema.csv`, `dep_weekly.csv`, `pre.csv`, `post.csv`, infer meaning from column names directly.

## Analysis Workflow

### 1. Orient to the participant

```python
import pandas as pd

# Confirm which files have data for this participant
pid = "INS-W_011"  # replace with target
files = ["ema.csv", "dep_weekly.csv", "activity_allday_raw.csv",
         "sleep_allday_raw.csv", "communication_allday_raw.csv",
         "location_allday_raw.csv", "phone_usage_allday_raw.csv",
         "connectivity_allday_raw.csv"]
for f in files:
    df = pd.read_csv(f)
    sub = df[df['pid'] == pid]
    n_valid = sub.select_dtypes('number').notna().any(axis=1).sum()
    print(f"{f}: {len(sub)} rows, {n_valid} with any numeric data")
```

### 2. Temporal segmentation for sensor data

Split the participant's non-NaN rows into early and late halves by date. This is the standard comparison unit for sensor modalities.

```python
def early_late(df, pid, value_col):
    sub = df[df['pid'] == pid].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date').dropna(subset=[value_col])
    mid = len(sub) // 2
    early = sub.iloc[:mid][value_col].mean()
    late = sub.iloc[mid:][value_col].mean()
    return early, late
```

### 3. Data to extract per modality

Collect these for every modality that has non-NaN data:

| Modality | Key columns | QA angle |
|---|---|---|
| EMA | `negative_affect_EMA` | Trend and magnitude of change from early to late |
| Weekly depression | `feel_depressed`, `feel_anxious`, `dep`, `BDI2` | Persistent depression, symptom trajectory, final severity |
| Pre/Post surveys | All `_PRE` / `_POST` pairs | Change in loneliness, stress, anxiety, depression, social support, emotion regulation, resilience, coping, mindfulness |
| Activity | `intraday_rapids_sumsteps` | Steps increase/decrease early vs. late |
| Sleep | `summary_rapids_sumdurationasleepmain` (primary), `summary_rapids_avgefficiencymain` | Duration and quality changes |
| Communication | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count` | Social engagement signals |
| Location | `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` | Mobility and home-time changes |
| Phone usage | `rapids_countepisodeunlock`, `rapids_sumdurationunlock` | Engagement/withdrawal from digital device |
| Connectivity | `rapids_countscans`, `rapids_uniquedevices` | Environmental exposure changes |

### 4. Go beyond simple comparisons — look for:

**Temporal anomalies**: Aggregate sensor data by ISO week. If any week's value is >2× the participant's median, flag it as a potential travel/event spike.

```python
sub['week'] = pd.to_datetime(sub['date']).dt.isocalendar().week
weekly = sub.groupby('week')['barnett_disttravelled'].mean()
anomaly_weeks = weekly[weekly > weekly.median() * 2]
```

**Cross-modal correlations**: When two behavioral streams both have ≥10 valid observations, compute Pearson correlation to surface joint patterns (e.g., does high mobility coincide with more incoming calls?).

```python
merged = pd.merge(df_comm[['date','rapids_incoming_count']],
                  df_loc[['date','barnett_disttravelled']], on='date')
r = merged[['rapids_incoming_count','barnett_disttravelled']].corr().iloc[0,1]
```

**Self-report vs. behavioral discrepancies**: Look for cases where the survey direction contradicts the behavioral signal (e.g., social support improves in surveys but communication calls drop), then surface that tension explicitly.

### 5. Formulate and submit QA pairs

Each QA pair must:
- Ask about a **specific behavioral or psychological dimension** with a clear time reference ("between early and late periods" / "from pre to post assessment")
- Include **concrete numbers** in the answer (mean values, direction of change, scale names)
- Describe the **direction and magnitude** of change using natural language ("increased substantially", "remained stable", "decreased modestly")
- Remain factual — do not over-interpret causation

Submit with: `submit_qa_pair(q="...", a="...")`

**Good QA pair examples** (structure to emulate):
- Q: "How did the user's negative affect change over the observation period?"  
  A: "It showed a moderate increase, rising from 8.44 in the early period to 10.56 in the later period."
- Q: "How did outgoing call behavior change between early and late periods?"  
  A: "Outgoing calls increased notably from 9.25 calls/day to 13.92 calls/day while incoming calls remained stable."
- Q: "Was there any exceptional mobility event during the study?"  
  A: "Week 24 showed exceptional mobility with distance spiking to 176,804 m — approximately 5× the participant's median — followed by a sharp decline, suggesting a temporary travel event."

### 6. QA coverage checklist

Aim to cover all of these (skip only if data is entirely NaN):
- [ ] EMA negative affect trajectory
- [ ] Weekly depression status and severity (including BDI2 endterm)
- [ ] Pre-post psychological state (depression, anxiety/stress)
- [ ] Pre-post social factors (loneliness, social support, social fit)
- [ ] Pre-post emotion regulation / coping / resilience
- [ ] Physical activity (steps) early vs. late
- [ ] Sleep duration and quality early vs. late
- [ ] Communication patterns (calls) early vs. late
- [ ] Mobility patterns early vs. late
- [ ] Phone usage behavior early vs. late
- [ ] Connectivity / environmental exposure early vs. late
- [ ] Cross-modal correlation (if data permits)
- [ ] Temporal anomaly (if detected)
- [ ] Self-report vs. behavioral discrepancy (if present)

## Common Pitfalls

- **Don't split raw 92-row arrays** — many rows will be NaN. Always `dropna()` on the target column before computing early/late statistics.
- **`summary_rapids_*` columns are often identical repeated values** (they summarize the whole period) — use `intraday_rapids_*` for within-day variation and daily totals.
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — these have no JSON field files. Infer column meanings from names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score, not a weekly measure.
- **Pre/post survey scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better regulation (good), higher PSS = more stress (bad). Interpret in context.
- **Submit QA pairs incrementally** as you finish each modality — don't batch them all at the end.
