---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (identified by `pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **20+ high-quality, distinct QA pairs** covering every available data modality and multiple sub-dimensions within each modality.

## Dataset Structure

All CSV files share the columns `pid` (participant ID) and `date`. Filter every file by the target `pid`.

**Sensor files** (92 days per participant, many NaN rows are normal):
- `activity_allday_raw.csv` — daily step count, active/sedentary bout counts and durations
- `sleep_allday_raw.csv` — sleep duration (minutes), efficiency, bedtime/wake-time
- `communication_allday_raw.csv` — incoming/outgoing/missed call counts, durations, distinct contacts
- `location_allday_raw.csv` — distance traveled, radius of gyration, home time, significant places, circadian routine, location entropy, location transitions
- `phone_usage_allday_raw.csv` — unlock episode count, total duration, average duration per episode; also contains columns for phone usage at different location contexts (home, study, etc.)
- `connectivity_allday_raw.csv` — Bluetooth scan count, unique devices

**Assessment files** (one row per observation per participant):
- `ema.csv` — `negative_affect_EMA` score, timestamped across the study period
- `dep_weekly.csv` — weekly `feel_anxious`, `feel_depressed`, `BDI2` (endterm only), `dep` (binary), `dep_weekly_subscale`, `anx_weekly_subscale`
- `pre.csv` / `post.csv` — baseline vs. end-of-study psychological scales (loneliness, perceived stress, anxiety, depression, social support, emotion regulation, resilience, mindfulness, coping)
- `dep_endterm.csv` — final depression label and BDI2 score

**Schema tip**: Use `get_field_description(data_file="<filename>")` for the six sensor CSV files. For `ema.csv`, `dep_weekly.csv`, `pre.csv`, `post.csv`, infer column meanings from names.

**Known column pitfalls**:
- `platform.csv` uses column `platform`, not `os`
- Home time: use `barnett_hometime` (not `barnett_homelabel`, which does not exist)
- `summary_rapids_*` columns repeat the same period-wide value on every row — use `intraday_rapids_*` for daily variation
- Phone usage location columns: look for columns with `home`, `study`, or location-context suffixes in `phone_usage_allday_raw.csv`

## Analysis Workflow

### 1. Orient to the participant

```python
import pandas as pd

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

**Early/late split** — the standard comparison unit for sensor modalities. Always `dropna()` on the target column first.

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

**Thirds segmentation** — use for modalities where a non-linear trajectory is plausible (location entropy, negative affect, mobility, phone usage). Reveals U-shaped or progressive patterns that early/late splits miss.

```python
def thirds(df, pid, value_col):
    sub = df[df['pid'] == pid].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date').dropna(subset=[value_col])
    n = len(sub)
    t1 = sub.iloc[:n//3][value_col].mean()
    t2 = sub.iloc[n//3:2*n//3][value_col].mean()
    t3 = sub.iloc[2*n//3:][value_col].mean()
    return t1, t2, t3
```

**Weekday vs weekend** — compute for every modality with ≥10 weekday and ≥5 weekend valid days. Report if difference is >15%.

```python
sub['dayofweek'] = pd.to_datetime(sub['date']).dt.dayofweek
weekday = sub[sub['dayofweek'] < 5][value_col].mean()
weekend = sub[sub['dayofweek'] >= 5][value_col].mean()
```

### 3. Data to extract per modality

| Modality | Primary columns | Secondary columns (also extract) | QA angles |
|---|---|---|---|
| EMA | `negative_affect_EMA` | — | Early/late trend; thirds trajectory; spike dates |
| Weekly depression | `feel_depressed`, `feel_anxious`, `dep`, `BDI2` | `dep_weekly_subscale`, `anx_weekly_subscale` | Persistent depression; symptom trajectory; final severity |
| Pre/Post — psychological | `CESD*`, `STAI*`, `PSS*`, `UCLA*` | — | Separate QA: change in depression, anxiety, stress, loneliness |
| Pre/Post — social | `2waySSS_*` (all 4 dimensions), `SocialFit*` | — | Separate QA: receiving/giving emotional & instrumental support; social fit |
| Pre/Post — regulation | `ERQ_*`, `BRS*`, `MAAS*`, `CHIPS*` | — | Separate QA: emotion regulation strategy, resilience, mindfulness, coping |
| Activity | `intraday_rapids_sumsteps` | `intraday_rapids_sumdurationactivebout`, `intraday_rapids_sumdurationsedentarybout`, bout counts | Steps change; sedentary/active bout ratio change |
| Sleep | `summary_rapids_sumdurationasleepmain` (primary), `summary_rapids_avgefficiencymain` | `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain` | Duration, quality, timing changes |
| Communication | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count` | `rapids_outgoing_sumduration`, `rapids_outgoing_distinctcontacts`, `rapids_incoming_sumduration` | Count vs duration dissociation; proactivity ratio change; network diversity (distinct contacts) |
| Location | `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` | `barnett_circdnrtn` (circadian routine 0–1), `doryab_locationentropy` or `barnett_siglocentropy`, `doryab_numberlocationtransitions`, `barnett_siglocsvisited` | Mobility; home-time; spatial diversity; routine consistency; location transitions |
| Phone usage | `rapids_countepisodeunlock`, `rapids_sumdurationunlock` | `rapids_avgdurationunlock`, location-context columns (home/study) | Count vs duration dissociation; home vs non-home phone usage |
| Connectivity | `rapids_countscans`, `rapids_uniquedevices` | — | Environmental exposure diversity |

### 4. Extract richer sub-dimension insights

**Weekday vs weekend**: Report if difference >15% for at least one modality. Often highest contrast for sleep duration, weekend distance traveled, home time, and incoming calls.

**Active/sedentary ratio**: Compute active bout duration / sedentary bout duration for early vs late. Even small ratio change is QA-worthy.

**Communication count vs duration dissociation**: If outgoing call count increases but total duration decreases (or vice versa), name this pattern explicitly (more frequent but shorter calls, or fewer but longer calls).

**Communication proactivity ratio**: Compute outgoing/incoming ratio in early vs late periods. A shift in this ratio (e.g., 3.0→8.4) is a meaningful QA pair indicating change in who initiates contact.

**Communication network diversity**: Extract `rapids_outgoing_distinctcontacts` for early vs late. An increase signals broader social reach; a decrease signals more concentrated engagement.

**Phone count vs duration dissociation**: If unlock count drops but total duration rises (or vice versa), report this as a meaningful pattern.

**Phone usage by location context**: If columns for home vs study (or other location) phone usage exist, compare unlock count and duration at home vs non-home contexts between early and late periods.

**Location diversity**: Extract `doryab_locationentropy` (or `barnett_siglocentropy`) and `doryab_numberlocationtransitions` for early vs late and thirds. An increase signals more spatially diverse behavior.

**Circadian routine score** (`barnett_circdnrtn`, 0=chaotic, 1=perfectly consistent): Changes indicate shifts in daily routine regularity. Particularly useful when mobility changes but routine may not follow.

**Temporal anomalies with specific dates**: After weekly aggregation, identify anomaly weeks (>2× median distance). Drill into those weeks for peak dates and magnitudes.

```python
sub['week'] = pd.to_datetime(sub['date']).dt.isocalendar().week
weekly = sub.groupby('week')['barnett_disttravelled'].mean()
anomaly_weeks = weekly[weekly > weekly.median() * 2]
for wk in anomaly_weeks.index:
    wk_days = sub[sub['week'] == wk].nlargest(3, 'barnett_disttravelled')
    print(wk_days[['date', 'barnett_disttravelled']])
```

**EMA spike analysis**: Identify dates with highest negative affect. Compare behavioral signals on those dates vs participant average (travel, sleep, home time). Report specific dates and values.

**Event-day behavioral comparison**: On anomaly travel days, compare sleep, phone usage, communication, and other metrics vs participant overall average. This produces high-value cross-modal QA pairs.

**Cross-modal correlations**: When two streams have ≥10 shared observations, compute Pearson correlation.

```python
merged = pd.merge(df_comm[['date','rapids_outgoing_count']],
                  df_loc[['date','barnett_disttravelled']], on='date')
r = merged[['rapids_outgoing_count','barnett_disttravelled']].corr().iloc[0,1]
```

**Self-report vs behavioral discrepancies**: When survey direction contradicts behavioral signal (e.g., social support worsens but outgoing calls increase), surface that tension explicitly.

**Behavioral vs psychological trajectory dissociation**: If behavioral metrics (calls, mobility, phone use) increase while psychological metrics worsen (or vice versa), name this meta-pattern as a dedicated QA pair.

### 5. Formulate and submit QA pairs

Each QA pair must:
- Ask about a **specific behavioral or psychological dimension** with a clear time reference
- Include **concrete numbers** in the answer (mean values, direction and magnitude of change, scale names)
- Describe the **direction and magnitude** using natural language ("increased substantially", "remained stable", "decreased modestly")
- Remain factual — do not over-interpret causation

Submit with: `submit_qa_pair(q="...", a="...")`

**Good QA pair examples** (structure to emulate):

- Q: "How did the user's negative affect change over the observation period?"  
  A: "It showed a moderate increase, rising from 8.44 in the early period to 10.56 in the later period."

- Q: "How did the user's physical activity levels and sedentary behavior change between early and late periods?"  
  A: "Physical activity decreased modestly (steps: 10,884→9,249/day; -15%). Sedentary bout duration increased slightly while active bout duration fell, shifting the active-to-sedentary ratio from 0.19 to 0.16."

- Q: "How did outgoing call frequency and duration change between early and late periods?"  
  A: "Outgoing calls increased from 1.24 to 3.67/day (+196%), but total duration increased more dramatically (from 58s to 305s mean/call), suggesting fewer but much longer conversations in the later period."

- Q: "How did the user's mobility and phone usage differ between weekdays and weekends?"  
  A: "Weekend distance traveled averaged 162 km vs 23 km on weekdays, with home time decreasing by 2.2 hours. Phone usage also increased modestly on weekends (+4 unlocks/day)."

- Q: "Was there any exceptional mobility event during the study?"  
  A: "Week 24 showed exceptional mobility with distance spiking to 176,804 m — approximately 5× the participant's median. Peak travel days occurred on May 24 (4.4M m) and May 27 (3.98M m), suggesting major long-distance travel."

- Q: "Did the user's peak negative affect episodes coincide with any specific behavioral events?"  
  A: "Yes, the two highest negative affect days (May 20: 5.0, May 24: 8.0) coincided with the largest travel days (641 km and 4,428 km), suggesting travel-related stress during the high-mobility period."

- Q: "How did the user's spatial behavior and routine patterns change between early and late periods?"  
  A: "Location entropy decreased from 0.28 to 0.21 nats suggesting less diverse location usage, circadian routine weakened from 0.62 to 0.50 indicating less consistent daily patterns, while location transitions increased from 2.34 to 5.00/day."

- Q: "Is there a discrepancy between self-reported social support and behavioral communication patterns?"  
  A: "Yes: self-reported emotional social support decreased (giving: 14→8, receiving: 20→14), yet outgoing calls increased from 9.25 to 13.92/day, suggesting behavioral social engagement increased despite perceived support decline."

- Q: "How did the user's proactive vs reactive communication patterns change over the study period?"  
  A: "The user became substantially more proactive: outgoing calls increased 250% (0.92→3.23/day) while incoming calls only rose 25% (0.31→0.38/day). The outgoing/incoming ratio shifted from 3.0 to 8.4, indicating a strong shift toward initiating rather than receiving calls."

- Q: "How did the user's spatial diversity (location entropy) change across the study period?"  
  A: "Location entropy showed a declining trend across thirds: 0.449 nats (first third) → 0.485 nats (middle third) → 0.238 nats (last third). The sharp drop in the final third indicates substantially reduced spatial diversity toward the end of the study."

- Q: "How did the user's emotion regulation, resilience, and mindfulness change from pre-study to post-study?"  
  A: "Resilience increased substantially (BRS: 3.00→3.67), and mindfulness improved (MAAS: 4.0→5.0). Emotion regulation showed mixed changes: reappraisal slightly decreased (5.17→5.00) while suppression increased (4.50→5.00). Perceived stress decreased notably (PSS: 21→17)."

### 6. QA coverage checklist

Aim to cover all of these (skip only if data is entirely NaN):
- [ ] EMA negative affect trajectory (early/late) + thirds if non-linear pattern
- [ ] EMA spike analysis — specific high-affect dates and coinciding behaviors
- [ ] Weekly depression status and severity (including BDI2 endterm)
- [ ] Weekly depression/anxiety subscale trajectory
- [ ] **Pre/post: psychological state** (depression, anxiety/stress, loneliness) — one QA pair
- [ ] **Pre/post: social factors** (all 4 social support dimensions, social fit) — one QA pair
- [ ] **Pre/post: emotion regulation / coping / resilience / mindfulness** — one QA pair
- [ ] Physical activity (steps) early vs. late
- [ ] Activity/sedentary bout ratio early vs. late
- [ ] Sleep duration, quality, and timing (bedtime/wake-time) early vs. late
- [ ] Weekday vs weekend differences (at least one modality where difference >15%)
- [ ] Communication: outgoing count AND duration, early vs late; count/duration dissociation
- [ ] Communication: incoming count trend; missed call trend
- [ ] **Communication proactivity ratio** (outgoing/incoming ratio change over time)
- [ ] **Communication network diversity** (distinct contacts) early vs late
- [ ] Mobility patterns (distance, RoG, home time) early vs. late
- [ ] Location diversity (entropy, circadian routine score, significant places, location transitions) early vs. late
- [ ] **Location entropy thirds trajectory** (if progressive or U-shaped pattern present)
- [ ] Phone usage (unlock count AND duration) early vs late; count/duration dissociation
- [ ] **Phone usage by location context** (home vs non-home) if columns exist
- [ ] Connectivity / environmental exposure early vs. late
- [ ] Temporal anomaly (if detected) — with specific dates, magnitudes, cross-modal coincidences
- [ ] Cross-modal correlation (if data permits, ≥10 shared observations)
- [ ] Self-report vs. behavioral discrepancy (if present)
- [ ] Behavioral trajectory vs. psychological trajectory dissociation (if trends diverge)

## Common Pitfalls

- **Never split raw 92-row arrays** — many rows will be NaN. Always `dropna()` on the target column before computing early/late or thirds statistics.
- **`summary_rapids_*` columns are period-wide summaries** — they repeat the same value across all rows. Use `intraday_rapids_*` for genuine daily variation.
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — infer column meanings from names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score, not a weekly measure.
- **Pre/post survey scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better regulation (good), higher PSS = more stress (bad), higher BRS = better resilience (good).
- **`barnett_homelabel` does not exist** — use `barnett_hometime` for Barnett-algorithm home time.
- **`platform.csv` column is `platform`, not `os`** — KeyError on `os` is a common bug.
- **Submit QA pairs incrementally** as you finish each modality — don't batch them all at the end.
- **Communication data is often sparse** (<30% valid days for some participants) — note this limitation but still extract the available patterns.
- **Distance outliers**: filter values >10× median before computing means for location data to avoid skew from extreme travel days inflating averages.
- **Pre/post QA pairs should be split into three separate pairs**: one for depression/anxiety/stress/loneliness, one for social support dimensions, and one for emotion regulation/coping/resilience/mindfulness. This ensures adequate coverage and generates more distinct QA pairs.
- **Communication proactivity**: always compute and report the outgoing/incoming ratio in both early and late periods — the ratio change often reveals a meaningful social behavior shift even when absolute counts change modestly.
