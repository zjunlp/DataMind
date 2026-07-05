---
name: globem-user-analysis
description: >
  Comprehensive individual-user analysis on the GLOBEM dataset — a longitudinal
  passive-sensing + mental-health study of college students. Use this skill
  whenever a task involves analyzing a specific participant (e.g. "Analyze user
  INS-W_002") from the GLOBEM dataset, exploring behavioral patterns from
  smartphone sensors, correlating behavioral signals with mental health outcomes,
  or producing a comprehensive user profile from multimodal sensing data.
---

# GLOBEM Individual User Analysis

## Dataset Overview

The GLOBEM dataset tracks college students over a 92-day period (~April–July)
using passive smartphone sensing. Each participant has:

**Sensor CSVs** (daily rows, columns: `Unnamed: 0`, `pid`, `date`, + features):
| File | Modality | Key Columns |
|---|---|---|
| `activity_allday_raw.csv` | Steps, sedentary/active bouts | `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout` |
| `sleep_allday_raw.csv` | Duration, efficiency, timing | `summary_rapids_avgdurationasleepmain`, `summary_rapids_avgefficiencymain`, `summary_rapids_avgdurationtofallasleepmain` |
| `communication_allday_raw.csv` | Call counts, duration, contacts | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_outgoing_distinctcontacts` |
| `connectivity_allday_raw.csv` | Bluetooth scans, unique devices | `rapids_countscans`, `rapids_uniquedevices` |
| `location_allday_raw.csv` | Mobility, home time, entropy | `barnett_avgflightdur`, `barnett_avgflightlen`, `barnett_homelabel`, `barnett_circdnrtn`, `barnett_rog`, `barnett_siglocsvisited`, `barnett_disttravelled` |
| `phone_usage_allday_raw.csv` | Unlock frequency, usage duration | `rapids_countepisodeunlock`, `rapids_sumdurationunlock`, `rapids_sumdurationunlockhome` |

**Mental-health / survey files** (read directly via `execute_code`; `get_field_description` will fail on these):
| File | Contents |
|---|---|
| `dep_weekly.csv` | Columns: `pid`, `date`, `feel_anxious`, `feel_depressed`, `BDI2`, `dep`, `dep_weekly_subscale`, `anx_weekly_subscale` |
| `dep_endterm.csv` | End-of-study BDI2 score and `dep` flag |
| `ema.csv` | Daily `negative_affect_EMA` scores |
| `pre.csv` | Pre-study surveys (see exact columns below) |
| `post.csv` | Post-study surveys (see exact columns below) |
| `platform.csv` | iOS vs Android |

### Exact Pre/Post Survey Column Names

**pre.csv** (suffix `_PRE`):
`UCLA_10items_PRE`, `SocialFit_PRE`, `2waySSS_receiving_emotional_PRE`, `2waySSS_giving_emotional_PRE`, `2waySSS_giving_instrumental_PRE`, `2waySSS_receiving_instrumental_PRE`, `ERQ_reappraisal_PRE`, `ERQ_suppression_PRE`, `BRS_PRE`, `CHIPS_PRE`, `PSS_10items_PRE`, `STAIS_PRE`, `MAAS_7items_PRE`, `CESD_9items_PRE`, `CESD_10items_PRE`, `BFI10_extroversion_PRE`, `BFI10_agreeableness_PRE`, `BFI10_conscientiousness_PRE`, `BFI10_neuroticism_PRE`, `BFI10_openness_PRE`

**post.csv** (suffix `_POST`; BFI10 not in POST):
`UCLA_10items_POST`, `SocialFit_POST`, `2waySSS_receiving_emotional_POST`, `2waySSS_giving_emotional_POST`, `2waySSS_giving_instrumental_POST`, `2waySSS_receiving_instrumental_POST`, `ERQ_reappraisal_POST`, `ERQ_suppression_POST`, `BRS_POST`, `CHIPS_POST`, `PSS_10items_POST`, `STAIS_POST`, `MAAS_7items_POST`, `CESD_9items_POST`, `CESD_10items_POST`

**Scale meanings** (higher = more unless noted):
- `STAIS` — state anxiety (higher = more anxious)
- `PSS_10items` — perceived stress (higher = more stressed)
- `CESD_9items` / `CESD_10items` — depression symptoms (higher = worse)
- `UCLA_10items` — loneliness (higher = more lonely)
- `BRS` — resilience (higher = more resilient, good)
- `ERQ_reappraisal` — cognitive reappraisal use (higher = more adaptive coping)
- `ERQ_suppression` — emotional suppression (higher = more suppression)
- `CHIPS` — health stressors count (higher = more stressed)
- `MAAS_7items` — mindfulness (higher = more mindful, good)
- `SocialFit` — social fit (higher = better fit)
- `2waySSS_receiving/giving_emotional` — emotional social support exchange
- `2waySSS_receiving/giving_instrumental` — instrumental social support exchange
- `BFI10_*` — Big Five personality (extroversion, agreeableness, conscientiousness, neuroticism, openness)

## Analysis Pipeline

### Phase 1 — Orientation (1–2 calls)
```python
# 1. list_files to confirm available files
# 2. get_field_description on 2-3 sensor files to learn any extra column names needed
# Do NOT call get_field_description for dep_weekly, ema, pre, post, dep_endterm, platform
```

### Phase 2 — Per-modality stats (one call per modality)

Filter all sensor DFs by `pid == '<user_id>'`. For each modality compute:
- Mean ± std, min/max over valid (non-NaN) rows
- Count of valid days (report as n/92)
- **Weekday vs. weekend difference** (add `df['date'].dt.dayofweek` → 0–4 weekday, 5–6 weekend)
- **Temporal trend**: split by study midpoint → first-half vs. second-half means

```python
user_df['date'] = pd.to_datetime(user_df['date'])
mid = user_df['date'].min() + (user_df['date'].max() - user_df['date'].min()) / 2
first_half = user_df[user_df['date'] < mid]
second_half = user_df[user_df['date'] >= mid]
```

**Activity** — `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout`

**Sleep** — `summary_rapids_avgdurationasleepmain` (minutes), `summary_rapids_avgefficiencymain` (**already 0–100**, NOT decimal — never multiply by 100), `summary_rapids_avgdurationtofallasleepmain`. Convert bedtime/wake minutes-since-midnight to HH:MM for readability.

**Communication** — `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_outgoing_distinctcontacts`. Compute outgoing/incoming ratio (>1 = proactive).

**Location** — `barnett_disttravelled`, `barnett_rog`, `barnett_homelabel` (minutes/day at home), `barnett_circdnrtn` (0–1 circadian consistency), `barnett_siglocsvisited`. Filter GPS outliers: drop values > median × 10 for distance and rog before averaging.

**Phone Usage** — `rapids_countepisodeunlock`, `rapids_sumdurationunlock` (minutes), `rapids_sumdurationunlockhome`. Report home-use fraction.

**Connectivity** — `rapids_countscans`, `rapids_uniquedevices`.

### Phase 3 — Mental health profile (1–2 calls)

```python
dep_weekly = pd.read_csv('dep_weekly.csv')
dep_endterm = pd.read_csv('dep_endterm.csv')
ema = pd.read_csv('ema.csv')
pre = pd.read_csv('pre.csv')
post = pd.read_csv('post.csv')
platform = pd.read_csv('platform.csv')

uid = '<user_id>'
user_dep = dep_weekly[dep_weekly['pid'] == uid].copy()
user_dep['date'] = pd.to_datetime(user_dep['date'])
user_endterm = dep_endterm[dep_endterm['pid'] == uid]
user_ema = ema[ema['pid'] == uid].copy()
user_ema['date'] = pd.to_datetime(user_ema['date'])
user_pre = pre[pre['pid'] == uid]
user_post = post[post['pid'] == uid]
user_platform = platform[platform['pid'] == uid]
```

Extract and report:
- Platform (iOS/Android)
- Depression: weekly flag rate, end-term BDI2 + dep status
- EMA: mean, std, min/max, first-half vs. second-half trend
- **Pre→Post survey changes** for ALL key scales using exact column names above (report Pre value, Post value, and change with ↑↓ arrows). Interpret direction: improved or worsened.
- Personality (BFI10 pre only)

### Phase 4 — Cross-modal correlation & synthesis (1–2 calls)

This phase drives the most analytically rich insights. Complete all sub-analyses:

**1. EMA ↔ Behavioral correlations** (Pearson r):
Merge EMA with each sensor modality on `pid` + `date`, then correlate `negative_affect_EMA` with:
- `intraday_rapids_sumsteps` (activity)
- `summary_rapids_avgdurationasleepmain` (sleep duration)
- `barnett_homelabel` (home time)
- `rapids_sumdurationunlock` (screen time)
- `intraday_rapids_countepisodesedentarybout` (sedentary behavior)

**2. High vs. low EMA day comparisons**:
Split days by EMA median → report behavioral metric means for each group:
```python
ema_median = user_ema['negative_affect_EMA'].median()
high_ema_dates = user_ema[user_ema['negative_affect_EMA'] > ema_median]['date']
low_ema_dates = user_ema[user_ema['negative_affect_EMA'] <= ema_median]['date']
# Compare steps, screen time, home time, etc. on high vs. low EMA days
```

**3. Depression-flagged week behavior comparison**:
Aggregate daily sensor data to weekly means (group by week). Join on dep_weekly dates (±7 days), then compare depressed vs. non-depressed week means for steps, sleep duration, phone unlocks, home time.

**4. Temporal trends across modalities**:
Summarize first-half → second-half changes for all key metrics in a consolidated table (steps, sleep, phone unlocks, distance, EMA).

### Phase 5 — Data quality check (integrate into output)
For each modality, report valid days as n/92. Flag modalities with <20% coverage as "critically sparse — interpret with caution."

## Synthesis Template

```
## Comprehensive Analysis of User <pid>

### Study Context
- Platform, study period (date range), data completeness per modality (n/92 days)

### Physical Activity
- Steps (mean ± std, min, max, valid days), sedentary/active balance, weekday vs. weekend
- Temporal trend: first-half → second-half step change

### Sleep
- Duration (hours), efficiency (%), timing (bedtime HH:MM, wake HH:MM), variability
- Temporal trend: sleep duration first-half → second-half

### Communication
- Call frequency, outgoing/incoming ratio, social diversity (distinct contacts)
- Temporal trend: outgoing call count first-half → second-half

### Location & Mobility
- Daily distance, home time (hours/day, %), circadian routine score, radius of gyration
- Temporal trend: distance or home time first-half → second-half

### Phone Usage
- Unlock count, screen time (hours), home vs. elsewhere split
- Temporal trend: unlock count or duration first-half → second-half

### Social Proximity (Connectivity)
- BT scan rate, unique devices per day

### Mental Health
- Depression trajectory: weekly flag rate (n/total weeks), end-term BDI2 + dep status
- EMA negative affect: mean ± std, trend (first-half → second-half mean)
- Pre→Post survey changes (all key scales with ↑↓ changes and interpretation)

### Cross-Modal Patterns
- EMA correlations with behavioral signals (list r values)
- Behavioral differences on high vs. low negative affect days
- Behavioral differences in depressed vs. non-depressed weeks
- Notable first-half → second-half behavioral shifts

### User Profile
- 3–5 sentence synthesis connecting behavioral patterns, temporal trends, and mental health
```

## Common Pitfalls

1. **Sleep efficiency**: `summary_rapids_avgefficiencymain` is already percentage (e.g., 93.5). Never multiply by 100.

2. **Survey columns must use exact names with _PRE/_POST suffix**. Column discovery via `df.columns.tolist()` recommended on first access; `get_field_description` does NOT work on survey files.

3. **Minute encoding**: Bedtime/wake times are often minutes-since-midnight. Convert: `f"{int(m//60):02d}:{int(m%60):02d}"`. Values like 1500 → 25:00 means 1:00 AM next day.

4. **Sparse data**: Some users have very few valid rows (<14/92 days for some modalities). Always check `df[col].notna().sum()` before computing stats and note effective n.

5. **Location GPS outliers**: `barnett_disttravelled` and `barnett_rog` can have extreme values from GPS errors. Use `values[values < values.median() * 10]` filtering before averaging.

6. **Weekly vs. daily merge**: `dep_weekly` is weekly; sensor data is daily. To compare, aggregate daily data into 7-day windows aligned with each `dep_weekly` date row.

7. **EMA correlation requires merging on date**: User's EMA may not overlap with all sensor dates. Use inner join and report n for each correlation.

8. **2waySSS and ERQ are often missed but frequently asked about**: Always extract emotion regulation (ERQ_reappraisal/suppression) and social support exchange (2waySSS giving/receiving) from pre/post surveys.
