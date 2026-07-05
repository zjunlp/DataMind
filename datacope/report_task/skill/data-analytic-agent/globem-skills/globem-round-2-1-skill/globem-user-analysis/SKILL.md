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
| `sleep_allday_raw.csv` | Duration, efficiency, timing | `summary_rapids_avgdurationasleepmain`, `summary_rapids_avgefficiencymain`, `summary_rapids_avgdurationtofallasleepmain`, `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain` |
| `communication_allday_raw.csv` | Call counts, duration, contacts, timing | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_incoming_meanduration`, `rapids_outgoing_distinctcontacts`, `rapids_outgoing_timefirstcall`, `rapids_outgoing_timelastcall` |
| `connectivity_allday_raw.csv` | Bluetooth scans, unique devices | `rapids_countscans`, `rapids_uniquedevices` |
| `location_allday_raw.csv` | Mobility, home time, entropy | `barnett_disttravelled`, `barnett_rog`, `barnett_hometime`, `barnett_circdnrtn`, `barnett_siglocsvisited`, `barnett_siglocentropy`, `barnett_avgflightdur`, `barnett_stdflightdur`, `doryab_numberlocationtransitions`, `doryab_avgspeed`, `doryab_timeattop1location`, `doryab_timeattop2location`, `doryab_timeattop3location` |
| `phone_usage_allday_raw.csv` | Unlock frequency, usage duration, context | `rapids_countepisodeunlock`, `rapids_sumdurationunlock`, `rapids_avgdurationunlock`, `rapids_stddurationunlock`, `rapids_firstuseafter00unlock`, `rapids_sumdurationunlock_locmap_home`, `rapids_countepisodeunlock_locmap_home`, `rapids_countepisodeunlock_locmap_study` |

**Mental-health / survey files** (read directly via `execute_code`; `get_field_description` will fail on these):
| File | Contents |
|---|---|
| `dep_weekly.csv` | `pid`, `date`, `feel_anxious`, `feel_depressed`, `BDI2`, `dep`, `dep_weekly_subscale`, `anx_weekly_subscale` |
| `dep_endterm.csv` | End-of-study BDI2 score and `dep` flag |
| `ema.csv` | Daily `negative_affect_EMA` scores |
| `pre.csv` | Pre-study surveys (suffix `_PRE`) |
| `post.csv` | Post-study surveys (suffix `_POST`) |
| `platform.csv` | iOS vs Android |

### Exact Pre/Post Survey Column Names

**pre.csv** (suffix `_PRE`):
`UCLA_10items_PRE`, `SocialFit_PRE`, `2waySSS_receiving_emotional_PRE`, `2waySSS_giving_emotional_PRE`, `2waySSS_giving_instrumental_PRE`, `2waySSS_receiving_instrumental_PRE`, `ERQ_reappraisal_PRE`, `ERQ_suppression_PRE`, `BRS_PRE`, `CHIPS_PRE`, `PSS_10items_PRE`, `STAIS_PRE`, `MAAS_7items_PRE`, `CESD_9items_PRE`, `CESD_10items_PRE`, `BFI10_extroversion_PRE`, `BFI10_agreeableness_PRE`, `BFI10_conscientiousness_PRE`, `BFI10_neuroticism_PRE`, `BFI10_openness_PRE`

**post.csv** (suffix `_POST`; BFI10 not in POST):
`UCLA_10items_POST`, `SocialFit_POST`, `2waySSS_receiving_emotional_POST`, `2waySSS_giving_emotional_POST`, `2waySSS_giving_instrumental_POST`, `2waySSS_receiving_instrumental_POST`, `ERQ_reappraisal_POST`, `ERQ_suppression_POST`, `BRS_POST`, `CHIPS_POST`, `PSS_10items_POST`, `STAIS_POST`, `MAAS_7items_POST`, `CESD_9items_POST`, `CESD_10items_POST`

**Scale meanings** (higher = more unless noted):
- `STAIS` — state anxiety; `PSS_10items` — perceived stress; `CESD_9/10items` — depression symptoms (higher = worse)
- `UCLA_10items` — loneliness; `BRS` — resilience (higher = good); `ERQ_reappraisal` — adaptive coping (higher = good)
- `ERQ_suppression` — emotional suppression; `CHIPS` — health stressors; `MAAS_7items` — mindfulness (higher = good)
- `SocialFit` — social fit; `2waySSS_*` — social support exchange; `BFI10_*` — Big Five personality

## Analysis Pipeline

### Phase 1 — Orientation (1–2 calls)
```python
# 1. list_files to confirm available files
# 2. get_field_description on 2-3 sensor files to learn extra column names
# Do NOT call get_field_description for dep_weekly, ema, pre, post, dep_endterm, platform
```

### Phase 2 — Per-modality stats (one call per modality)

Filter all sensor DFs by `pid == '<user_id>'`. For each modality compute:
- Mean ± std, min/max over valid (non-NaN) rows
- Count of valid days (report as n/92)
- **Weekday vs. weekend difference** (`df['date'].dt.dayofweek` → 0–4 weekday, 5–6 weekend)
- **Three-period temporal trend**: split into thirds T1/T2/T3 and report each mean, plus early (T1) vs. late (T3) comparison and % change

```python
user_df['date'] = pd.to_datetime(user_df['date'])
d_min, d_max = user_df['date'].min(), user_df['date'].max()
span = (d_max - d_min) / 3
t1 = user_df[user_df['date'] < d_min + span]
t2 = user_df[(user_df['date'] >= d_min + span) & (user_df['date'] < d_min + 2*span)]
t3 = user_df[user_df['date'] >= d_min + 2*span]
# Report T1_mean, T2_mean, T3_mean and classify trajectory:
# progressive increase/decline: monotonic T1→T2→T3
# inverted-U: T2 > T1 and T2 > T3; U-shaped: T2 < T1 and T2 < T3
# mixed/stable: otherwise
```

**Activity** — `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout`. Also compute active/sedentary ratio and avg sedentary bout duration.

**Sleep** — `summary_rapids_avgdurationasleepmain` (minutes), `summary_rapids_avgefficiencymain` (**already 0–100**, never multiply by 100), `summary_rapids_avgdurationtofallasleepmain`, `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain`. Convert bedtime/wake minutes-since-midnight to HH:MM.

**Communication** — `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_incoming_meanduration`, `rapids_outgoing_distinctcontacts`. Compute:
- Outgoing/incoming ratio (>1 = proactive)
- **Count/duration dissociation**: note when count and duration trend in opposite directions
- **Call timing window**: `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → convert to HH:MM); report first-half vs. second-half shift

**Location** — `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` (minutes/day at home), `barnett_circdnrtn` (0–1 circadian consistency), `barnett_siglocsvisited`, `barnett_siglocentropy` (entropy in nats), `barnett_avgflightdur` ± `barnett_stdflightdur` (seconds), `doryab_avgspeed` (km/hr), `doryab_numberlocationtransitions`, `doryab_timeattop1location` / `doryab_timeattop2location` / `doryab_timeattop3location` (minutes). Filter GPS outliers: drop values > median × 10 for `barnett_disttravelled` and `barnett_rog` before averaging.

**Phone Usage** — `rapids_countepisodeunlock`, `rapids_sumdurationunlock` (minutes), `rapids_avgdurationunlock` or compute as sum/count, `rapids_stddurationunlock` (session variability), `rapids_firstuseafter00unlock` (minutes from midnight to first unlock → convert to HH:MM), `rapids_sumdurationunlock_locmap_home`, `rapids_countepisodeunlock_locmap_home`, `rapids_countepisodeunlock_locmap_study`. Report home-use fraction and **count/duration dissociation** if present.

**Connectivity** — `rapids_countscans`, `rapids_uniquedevices`. Also compute **scan efficiency** = countscans / uniquedevices (higher = fewer distinct devices, more repeat scanning).

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
- Depression: weekly flag rate (n/total weeks), end-term BDI2 + dep status; also extract `feel_depressed`, `feel_anxious`, `dep_weekly_subscale`, `anx_weekly_subscale` with T1/T2/T3 temporal trends
- EMA: mean, std, min/max, T1/T2/T3 trend with trajectory pattern classification
- **Pre→Post survey changes** for ALL key scales (report Pre value, Post value, % change, ↑↓ arrows, and interpret direction: improved or worsened)
- Personality (BFI10 pre only)

### Phase 4 — Cross-modal correlation & synthesis (1–2 calls)

This phase drives the most analytically rich insights. Complete all sub-analyses:

**1. EMA ↔ Behavioral correlations** (Pearson r with p-value and n):
Merge EMA with each sensor modality on `pid` + `date` (inner join), then correlate `negative_affect_EMA` with:
- `intraday_rapids_sumsteps` (activity)
- `summary_rapids_avgdurationasleepmain` (sleep duration)
- `barnett_hometime` (home time)
- `rapids_sumdurationunlock` (screen time)
- `intraday_rapids_countepisodesedentarybout` (sedentary behavior)
- `barnett_siglocentropy` (location entropy)

**2. Peak EMA event analysis**:
Find top 3 highest EMA days and for each, compare behavioral metrics to their respective modality averages with % deviation:
```python
user_ema_sorted = user_ema.sort_values('negative_affect_EMA', ascending=False).head(3)
for _, row in user_ema_sorted.iterrows():
    peak_date = row['date']
    # Compare barnett_disttravelled, barnett_hometime, rapids_sumdurationunlock, etc. on that day vs. mean
    # Report: "Date X (EMA=Y): distance=Z vs avg=W (+/-N%)"
```

**3. High vs. low EMA day comparisons**:
Split days by EMA median → report behavioral metric means for each group:
```python
ema_median = user_ema['negative_affect_EMA'].median()
high_ema_dates = user_ema[user_ema['negative_affect_EMA'] > ema_median]['date']
low_ema_dates = user_ema[user_ema['negative_affect_EMA'] <= ema_median]['date']
# Compare steps, screen time, home time, etc. on high vs. low EMA days
```

**4. Depression-flagged week behavior comparison**:
Aggregate daily sensor data to weekly means (group by week). Join on dep_weekly dates (±7 days), then compare depressed vs. non-depressed week means for steps, sleep duration, phone unlocks, home time.

**5. Temporal trends across modalities**:
Summarize T1 → T3 changes for all key metrics in a consolidated table (steps, sleep, phone unlocks, distance, EMA). Classify each as progressive increase/decline, inverted-U, U-shaped, or mixed/stable.

### Phase 5 — Data quality check (integrate into output)
For each modality, report valid days as n/92. Flag modalities with <20% coverage as "critically sparse — interpret with caution."

## Synthesis Template

```
## Comprehensive Analysis of User <pid>

### Study Context
- Platform, study period (date range), data completeness per modality (n/92 days)

### Physical Activity
- Steps (mean ± std, min/max, valid days), sedentary/active balance and ratio, weekday vs. weekend
- Temporal trend: T1/T2/T3 step means and trajectory pattern

### Sleep
- Duration (hours), efficiency (%), timing (bedtime HH:MM, wake HH:MM), variability
- Temporal trend: T1/T2/T3 sleep duration and trajectory pattern

### Communication
- Call frequency (outgoing/incoming/missed), proactivity ratio, social diversity (distinct contacts)
- Call timing window: first-call HH:MM and last-call HH:MM, early vs. late shift
- Temporal trend: T1/T2/T3 outgoing count; note any count/duration dissociation

### Location & Mobility
- Daily distance, home time (hours/day, %), circadian routine score, radius of gyration
- Location entropy (nats), transitions/day, significant places visited
- Top-3 location time distribution (minutes)
- Avg flight duration ± std (seconds), avg speed (km/hr)
- Temporal trend: T1/T2/T3 distance and home time

### Phone Usage
- Unlock count, screen time (hours), avg session duration ± std (minutes), first-use HH:MM
- Home vs. study unlock count and duration split
- Temporal trend: T1/T2/T3 unlock count and duration; count/duration dissociation if present

### Social Proximity (Connectivity)
- BT scan rate, unique devices per day, scan efficiency (scans/device)
- Temporal trend: T1/T2/T3

### Mental Health
- Depression trajectory: weekly flag rate (n/total weeks), feel_depressed/feel_anxious means, T1/T2/T3 trends
- End-term BDI2 + dep status
- EMA negative affect: mean ± std, T1/T2/T3 trend and trajectory pattern
- Pre→Post survey changes (all key scales with ↑↓ changes and interpretation)

### Cross-Modal Patterns
- EMA correlations with behavioral signals (list r, p, n for each)
- Peak EMA days: top 3 dates with behavioral context (deviations from mean)
- Behavioral differences on high vs. low negative affect days
- Behavioral differences in depressed vs. non-depressed weeks
- Consolidated temporal shift table (T1→T3 for all key metrics with trajectory pattern)

### User Profile
- 3–5 sentence synthesis connecting behavioral patterns, temporal trends, and mental health
```

## Common Pitfalls

1. **Home time column**: Use `barnett_hometime` (minutes/day). `barnett_homelabel` and `doryab_homelabel` are cluster labels, not durations.

2. **Sleep efficiency**: `summary_rapids_avgefficiencymain` is already a percentage (e.g., 93.5). Never multiply by 100.

3. **Minute encoding**: Bedtime/wake/call times are minutes-since-midnight. Convert: `f"{int(m//60):02d}:{int(m%60):02d}"`. Values ≥ 1440 span next day (e.g., 1500 → 01:00 next day).

4. **Survey columns must use exact names with _PRE/_POST suffix**. Use `df.columns.tolist()` on first access; `get_field_description` does NOT work on survey files.

5. **Three-period analysis**: Split by thirds of the study period (not calendar months). Always report T1, T2, T3 means individually and classify the pattern.

6. **Sparse data**: Always check `df[col].notna().sum()` before computing stats. Some users have <14/92 days for some modalities — correlations require >5 overlapping days.

7. **Location GPS outliers**: `barnett_disttravelled` and `barnett_rog` can have extreme GPS errors. Use `values[values < values.median() * 10]` before averaging.

8. **Weekly vs. daily merge**: `dep_weekly` is weekly; sensor data is daily. Aggregate daily data into 7-day windows aligned with each `dep_weekly` date row.

9. **EMA correlation requires inner merge on date**: Report n for each correlation; p-values are unreliable when n < 10.

10. **2waySSS, ERQ, BRS are frequently asked about**: Always extract emotion regulation (ERQ_reappraisal/suppression), social support (2waySSS all four dimensions), resilience (BRS), and mindfulness (MAAS) from pre/post surveys.

11. **Scan efficiency** = `rapids_countscans / rapids_uniquedevices`. Rising efficiency with declining unique devices suggests narrowing social environment.

12. **Trajectory pattern classification**: Label each metric's temporal trend as: *progressive increase*, *progressive decline*, *inverted-U* (peak in T2), *U-shaped* (trough in T2), or *mixed/stable* (no clear pattern).
