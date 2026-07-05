---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (identified by `pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **25–30 high-quality, distinct QA pairs** covering every available data modality and multiple sub-dimensions within each modality.

## Dataset Structure

All CSV files share the columns `pid` (participant ID) and `date`. Filter every file by the target `pid`.

**Sensor files** (92 days per participant, many NaN rows are normal):
- `activity_allday_raw.csv` — daily step count, active/sedentary bout counts and durations
- `sleep_allday_raw.csv` — sleep duration (minutes), efficiency, bedtime/wake-time
- `communication_allday_raw.csv` — incoming/outgoing/missed call counts, durations, distinct contacts
- `location_allday_raw.csv` — distance traveled, radius of gyration, home time, significant places, circadian routine, location entropy, location transitions
- `phone_usage_allday_raw.csv` — unlock episode count, total duration, average duration per episode; also contains columns for phone usage at different location contexts (home, study, greens, exercise)
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
- Phone usage location columns: look for columns with `home`, `living`, `study`, `greens`, or `exercise` suffixes in `phone_usage_allday_raw.csv`

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

### 2. Temporal segmentation — apply to ALL modalities

Apply all three segmentations to every modality with sufficient valid data. Always `dropna()` on the target column first.

**Early/late split** — primary comparison unit.
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

**Thirds segmentation** — apply to every modality, not just ones expected to be non-linear. Thirds reveal patterns (U-shaped, progressive, peak-in-middle, inverted-U) that early/late splits hide. When reporting thirds results, **always name the pattern explicitly**: progressive decline/increase, U-shaped, inverted-U, peak-in-middle, stable-then-drop, etc.
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

**Weekday vs weekend** — compute for every modality with ≥10 weekday and ≥5 weekend valid days. Report if difference >15%.
```python
sub['dayofweek'] = pd.to_datetime(sub['date']).dt.dayofweek
weekday = sub[sub['dayofweek'] < 5][value_col].mean()
weekend = sub[sub['dayofweek'] >= 5][value_col].mean()
```

### 3. Data to extract per modality

| Modality | Primary columns | Secondary columns (also extract) | QA angles |
|---|---|---|---|
| EMA | `negative_affect_EMA` | — | Early/late + thirds pattern; spike dates + behavioral coincidence |
| Weekly depression | `feel_depressed`, `feel_anxious`, `dep`, `BDI2` | `dep_weekly_subscale`, `anx_weekly_subscale` | Persistent depression; symptom trajectory; final severity |
| Pre/Post — psychological | `CESD*`, `STAI*`, `PSS*`, `UCLA*` | — | Change in depression, anxiety, stress, loneliness |
| Pre/Post — social | `2waySSS_*` (all 4), `SocialFit*` | — | Each support dimension; social fit change |
| Pre/Post — regulation | `ERQ_*`, `BRS*`, `MAAS*`, `CHIPS*` | — | Emotion regulation strategy, resilience, mindfulness, coping |
| Activity | `intraday_rapids_sumsteps` | `intraday_rapids_sumdurationactivebout`, `intraday_rapids_sumdurationsedentarybout`, bout counts, avg bout duration | Steps change; sedentary/active ratio; avg bout duration shift |
| Sleep | `summary_rapids_sumdurationasleepmain` (primary), `summary_rapids_avgefficiencymain` | `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain` | Duration, quality, timing (bedtime/wake phase shift) |
| Communication | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count` | `rapids_outgoing_sumduration`, `rapids_outgoing_distinctcontacts`, `rapids_incoming_sumduration`, `rapids_outgoing_timefirstcall`, `rapids_outgoing_timelastcall` | Count vs duration dissociation; proactivity ratio shift; network diversity; call timing window shift |
| Location | `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` | `barnett_circdnrtn`, `doryab_locationentropy` or `barnett_siglocentropy`, `doryab_numberlocationtransitions`, `barnett_siglocsvisited` | Mobility; home-time; spatial diversity; routine consistency; transitions |
| Phone usage | `rapids_countepisodeunlock`, `rapids_sumdurationunlock` | `rapids_avgdurationunlock`, `rapids_firstuseafter00unlock`, location-context columns | Count vs duration dissociation; session length shift; home vs non-home; first-use timing |
| Connectivity | `rapids_countscans`, `rapids_uniquedevices` | — | Scans vs unique device divergence; weekday-weekend contrast |

### 4. Extract richer sub-dimension insights

**Pattern naming for thirds results**: Always state the pattern explicitly when reporting thirds data: "U-shaped" (low-high-low), "inverted U" (high-low-high), "progressive decline/increase" (monotone), "peak-then-recovery", "stable-then-drop", etc.

**Sleep timing phase shift**: Convert bedtime/wake-time from minutes-from-midnight to HH:MM. Report the shift direction and magnitude (e.g., "bedtime delayed by 90 minutes"). This often generates a distinct QA pair separate from duration.

**Active/sedentary ratio**: Compute active bout duration / sedentary bout duration for early vs late. Also check if avg sedentary bout duration changed (fewer but longer sedentary periods is a meaningful pattern).

**Communication count vs duration dissociation**: If outgoing call count increases but total/mean duration decreases (or vice versa), name this pattern explicitly as "count/duration dissociation."

**Communication proactivity ratio**: Compute outgoing_count / incoming_count in early vs late periods separately. A shifted ratio (e.g., 3.0→8.4) is QA-worthy. Avoid dividing by near-zero values — use outgoing_count / max(incoming_count, 0.1).

**Communication network diversity**: Extract `rapids_outgoing_distinctcontacts` for early vs late. An increase signals broader social reach; a decrease signals concentration.

**Communication timing window**: If `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` are available, report whether the calling window shifted (earlier/later first call, extended/contracted window).

**Phone count vs duration dissociation**: If unlock count drops but total duration rises (or vice versa), report as count/duration dissociation with the changed avg duration per session.

**Phone usage by location context**: Compare unlock count and duration at home vs study vs other contexts between early and late. A shift (e.g., home usage down, study usage up) is a meaningful QA pair.

**Phone first-use timing** (`rapids_firstuseafter00unlock`): Report if this shifted between early and late periods — earlier first phone use can indicate changed sleep/wake habits.

**Location diversity**: Extract `doryab_locationentropy` (or `barnett_siglocentropy`) and `doryab_numberlocationtransitions` for early/late and thirds. Report the pattern name.

**Circadian routine score** (`barnett_circdnrtn`, 0–1): Report change direction. Include thirds trajectory if non-trivial.

**Temporal anomalies with specific dates**: After weekly aggregation, identify anomaly weeks (>2× median distance). Drill into peak dates.

```python
sub['week'] = pd.to_datetime(sub['date']).dt.isocalendar().week
weekly = sub.groupby('week')['barnett_disttravelled'].mean()
anomaly_weeks = weekly[weekly > weekly.median() * 2]
for wk in anomaly_weeks.index:
    wk_days = sub[sub['week'] == wk].nlargest(3, 'barnett_disttravelled')
    print(wk_days[['date', 'barnett_disttravelled']])
```

**EMA spike analysis — always include specific dates**: Identify the top 3 highest negative-affect days. For each, report the specific date and value, then check distance traveled, home time, sleep duration, and phone usage on that day vs participant average. Describe whether the spike coincides with travel, isolation, disrupted sleep, or other patterns.

**Cross-modal correlations**: When two streams have ≥10 shared observations, compute Pearson correlation. Report correlations |r| > 0.3 as QA pairs.

**Self-report vs behavioral discrepancy**: When survey direction contradicts behavioral signal (e.g., social support worsens but outgoing calls increase), surface that tension explicitly as a dedicated QA pair.

**Behavioral vs psychological trajectory dissociation**: If behavioral metrics (calls, mobility, phone use) and psychological metrics trend in opposite directions, name this meta-pattern as a dedicated QA pair.

### 5. Formulate and submit QA pairs

Each QA pair must:
- Ask about a **specific behavioral or psychological dimension** with a clear time reference
- Include **concrete numbers** (mean values, direction and magnitude of change)
- State the **pattern name** when using thirds data (U-shaped, progressive, etc.)
- Describe magnitude using natural language ("increased substantially", "remained stable", "decreased modestly")
- Remain factual — do not over-interpret causation

Submit incrementally as you finish each modality: `submit_qa_pair(q="...", a="...")`

**Good QA pair examples**:

- Q: "How did the user's negative affect change over the observation period?"  
  A: "It showed a moderate increase, rising from 8.44 (early) to 10.56 (late). The thirds trajectory was progressive: 7.9 (T1) → 9.1 (T2) → 11.2 (T3)."

- Q: "How did outgoing call frequency and duration change, and was there a count/duration dissociation?"  
  A: "Outgoing calls increased from 1.24 to 3.67/day (+196%), but mean duration per call fell sharply from 305s to 58s (-81%), a classic count/duration dissociation — more frequent but much shorter conversations in the late period."

- Q: "How did the user's sleep timing change between early and late periods?"  
  A: "Bedtime shifted later by ~90 minutes (from 23:30 to 01:00), and wake time delayed by ~60 minutes (from 08:00 to 09:00), indicating a consistent sleep phase delay in the later period despite stable sleep duration."

- Q: "How did the user's spatial diversity (location entropy) change across the study period?"  
  A: "Location entropy showed a U-shaped pattern: 0.449 (T1) → 0.485 (T2) → 0.238 nats (T3). The sharp 51% drop in the final third indicates substantially reduced spatial diversity toward study end."

- Q: "Did the user's peak negative affect episodes coincide with specific behavioral events?"  
  A: "The two highest EMA days (May 20: 5.0, May 24: 8.0) coincided with the two largest travel days (641 km and 4,428 km), suggesting travel-related stress. EMA-distance correlation was moderately positive (r=0.31)."

- Q: "How did the user's proactive vs reactive communication patterns change over the study period?"  
  A: "The proactivity ratio (outgoing/incoming) shifted from 3.0 (early) to 8.4 (late): outgoing calls increased 250% (0.92→3.23/day) while incoming calls rose only 25% (0.31→0.38/day), indicating a strong shift toward initiating contact."

- Q: "Is there a discrepancy between self-reported social support and behavioral communication patterns?"  
  A: "Yes: receiving emotional support decreased (25→21, -16%) yet outgoing calls increased from 1.71 to 2.83/day (+65%), suggesting behavioral social engagement rose despite perceived support decline."

### 6. QA coverage checklist

Aim to cover all of these (skip only if data is entirely NaN):
- [ ] EMA negative affect trajectory: early/late change + named thirds pattern
- [ ] EMA spike analysis — top 3 specific high-affect dates with coinciding behavioral signals
- [ ] Weekly depression status, symptom trajectory, final BDI2
- [ ] Weekly depression/anxiety subscale trajectory
- [ ] **Pre/post: psychological state** (depression, anxiety/stress, loneliness)
- [ ] **Pre/post: social factors** (all 4 support dimensions + social fit)
- [ ] **Pre/post: emotion regulation / coping / resilience / mindfulness**
- [ ] Physical activity: steps early vs. late + thirds pattern
- [ ] Activity bout ratio (active/sedentary) + avg sedentary bout duration shift
- [ ] Sleep duration + efficiency early vs. late + thirds pattern
- [ ] **Sleep timing phase shift** (bedtime/wake-time hours, early vs late)
- [ ] Weekday vs weekend differences (any modality with >15% difference)
- [ ] Communication: outgoing + incoming count AND duration, early vs late
- [ ] **Count/duration dissociation** for outgoing and/or incoming calls if present
- [ ] Communication: missed call trend
- [ ] **Communication proactivity ratio** (outgoing/incoming) early vs. late
- [ ] **Communication network diversity** (distinct contacts) early vs late
- [ ] Communication timing window shift (first/last call times) if data available
- [ ] Mobility: distance, RoG, home time early vs. late
- [ ] Location diversity: entropy + circadian routine + transitions early vs. late + named thirds pattern
- [ ] Temporal anomaly (if detected) — with specific peak dates and magnitudes
- [ ] Phone usage: unlock count AND duration early vs late; count/duration dissociation if present
- [ ] **Phone first-use timing shift** (`rapids_firstuseafter00unlock`) if available
- [ ] **Phone usage by location context** (home vs study vs other) early vs late
- [ ] Connectivity: scans vs unique devices (look for divergence); weekday-weekend contrast
- [ ] Connectivity thirds pattern
- [ ] Cross-modal correlation (|r| > 0.3 with ≥10 shared observations)
- [ ] Self-report vs. behavioral discrepancy (if present)
- [ ] Behavioral trajectory vs. psychological trajectory dissociation (if trends diverge)

## Common Pitfalls

- **Never split raw 92-row arrays** — always `dropna()` on the target column before computing early/late or thirds.
- **`summary_rapids_*` columns are period-wide summaries** — use `intraday_rapids_*` for daily variation.
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — infer from names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score.
- **Pre/post scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better (good), higher PSS = more stress (bad), higher BRS = better resilience (good).
- **`barnett_homelabel` does not exist** — use `barnett_hometime`.
- **`platform.csv` column is `platform`, not `os`**.
- **Proactivity ratio with near-zero incoming calls**: use `max(incoming, 0.1)` to avoid unrealistic ratios. A ratio >10 is likely a near-zero denominator artifact — report the actual counts instead.
- **Distance outliers**: filter values >10× median before computing means.
- **Communication data is often sparse** (<30% valid days for some participants) — note this but still extract available patterns.
- **Submit QA pairs incrementally** as you finish each modality.
- **Pre/post QA pairs should be split into three separate pairs**: psychological state, social support, and emotion regulation/coping.
- **Thirds pattern naming is mandatory** — don't just list T1/T2/T3 numbers; always state whether the pattern is progressive, U-shaped, inverted-U, peak-in-middle, stable-then-drop, etc.
