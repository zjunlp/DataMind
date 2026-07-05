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
| `activity_allday_raw.csv` | Steps, sedentary/active bouts | `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout`, `intraday_rapids_avgdurationsedentarybout` |
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
| `platform.csv` | Column: `platform` (NOT `os`) — values: `ios` / `android` |

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

Filter all sensor DFs by `pid == '<user_id>'`. Always convert `date` to datetime first:
```python
user_df['date'] = pd.to_datetime(user_df['date'])
```

For each modality compute:
- Mean ± std, min/max over valid (non-NaN) rows
- Count of valid days (report as n/92)
- **Weekday vs. weekend difference** (`df['date'].dt.dayofweek` → 0–4 weekday, 5–6 weekend)
- **Three-period temporal trend** (T1/T2/T3) AND **early/late comparison** (first half vs. second half):

```python
# T1/T2/T3: split into thirds of the study date range
d_min, d_max = user_df['date'].min(), user_df['date'].max()
span = (d_max - d_min) / 3
t1 = user_df[user_df['date'] < d_min + span]
t2 = user_df[(user_df['date'] >= d_min + span) & (user_df['date'] < d_min + 2*span)]
t3 = user_df[user_df['date'] >= d_min + 2*span]

# Early/Late: split at midpoint
mid = d_min + (d_max - d_min) / 2
early = user_df[user_df['date'] < mid]
late = user_df[user_df['date'] >= mid]
```

Report T1_mean, T2_mean, T3_mean, early_mean, late_mean, early→late % change, and trajectory label:
- **progressive increase**: monotonic T1→T2→T3 increase
- **progressive decline**: monotonic T1→T2→T3 decrease
- **inverted-U**: T2 > T1 and T2 > T3
- **U-shaped**: T2 < T1 and T2 < T3
- **mixed/stable**: no clear pattern

**Activity** — Primary: `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout`. Compute and trend:
- Active/sedentary bout ratio and `intraday_rapids_avgdurationsedentarybout` (avg sedentary bout duration)
- T1/T2/T3 and early/late for: steps, active/sedentary ratio, avg sedentary bout duration, active bout count, sedentary bout count

**Sleep** — `summary_rapids_avgdurationasleepmain` (minutes), `summary_rapids_avgefficiencymain` (**already 0–100**, never multiply by 100), `summary_rapids_avgdurationtofallasleepmain`, `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain`. Convert bedtime/wake minutes-since-midnight to HH:MM. Compute and trend:
- T1/T2/T3 and early/late for: duration, efficiency, bedtime, wake time, time-to-fall-asleep

**Communication** — Primary: `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_incoming_meanduration`, `rapids_outgoing_distinctcontacts`. Compute and trend:
- Outgoing/incoming ratio (>1 = proactive)
- **Count/duration dissociation**: explicitly note when count and duration trend in opposite directions
- **Call timing window**: `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → HH:MM); report shift between early and late periods
- T1/T2/T3 and early/late for: outgoing count, incoming count, missed count, outgoing/incoming ratio, distinct contacts, mean outgoing duration, call timing window

**Location** — Primary: `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` (minutes/day), `barnett_circdnrtn` (0–1), `barnett_siglocsvisited`, `barnett_siglocentropy` (nats), `barnett_avgflightdur` ± `barnett_stdflightdur` (seconds), `doryab_avgspeed` (km/hr), `doryab_numberlocationtransitions`, `doryab_timeattop1location` / `doryab_timeattop2location` / `doryab_timeattop3location` (minutes). Filter GPS outliers: drop values > median × 10 for `barnett_disttravelled` and `barnett_rog` before averaging. Compute and trend:
- T1/T2/T3 and early/late for: distance, radius of gyration, home time, circadian routine, location entropy, location transitions, significant places, avg flight duration, std flight duration, avg speed, top-1/top-2/top-3 location time

**Phone Usage** — Primary: `rapids_countepisodeunlock`, `rapids_sumdurationunlock` (minutes), `rapids_avgdurationunlock`, `rapids_stddurationunlock`, `rapids_firstuseafter00unlock` (minutes → HH:MM), `rapids_sumdurationunlock_locmap_home`, `rapids_countepisodeunlock_locmap_home`, `rapids_countepisodeunlock_locmap_study`. Compute and trend:
- Home-use fraction, **count/duration dissociation** if present (unlock count vs. avg session duration trending opposite directions)
- T1/T2/T3 and early/late for: unlock count, total duration, avg session duration, session std, first-use time, home unlocks, study unlocks

**Connectivity** — `rapids_countscans`, `rapids_uniquedevices`. **Scan efficiency** = countscans / uniquedevices. Compute and trend:
- T1/T2/T3 and early/late for: scan count, unique devices, scan efficiency

### Phase 3 — Mental health profile (1–2 calls)

```python
dep_weekly = pd.read_csv('dep_weekly.csv')
dep_endterm = pd.read_csv('dep_endterm.csv')
ema = pd.read_csv('ema.csv')
pre = pd.read_csv('pre.csv')
post = pd.read_csv('post.csv')
platform = pd.read_csv('platform.csv')  # column is 'platform', NOT 'os'

uid = '<user_id>'
user_dep = dep_weekly[dep_weekly['pid'] == uid].copy()
user_dep['date'] = pd.to_datetime(user_dep['date'])
user_endterm = dep_endterm[dep_endterm['pid'] == uid]
user_ema = ema[ema['pid'] == uid].copy()
user_ema['date'] = pd.to_datetime(user_ema['date'])
user_pre = pre[pre['pid'] == uid]
user_post = post[post['pid'] == uid]
user_platform = platform[platform['pid'] == uid]
print(f"Platform: {user_platform['platform'].values[0]}")  # use 'platform' not 'os'
```

Extract and report:
- Platform (use `user_platform['platform'].values[0]`)
- Depression: weekly flag rate (n/total weeks), end-term BDI2 + dep status; extract `feel_depressed`, `feel_anxious`, `dep_weekly_subscale`, `anx_weekly_subscale` with T1/T2/T3 and early/late trends
- EMA: mean, std, min/max, T1/T2/T3 and early/late trend with trajectory classification
- **Pre→Post survey changes** for ALL key scales: report Pre value, Post value, % change, ↑↓ arrows, and direction (improved/worsened). Include ALL of: UCLA, SocialFit, 2waySSS (all 4), ERQ reappraisal/suppression, BRS, CHIPS, PSS, STAIS, MAAS, CESD-9, CESD-10
- Personality (BFI10 pre only): extroversion, agreeableness, conscientiousness, neuroticism, openness

### Phase 4 — Cross-modal correlation & synthesis (1–2 calls)

This phase drives the most analytically rich insights. Complete all sub-analyses:

**1. EMA ↔ Behavioral correlations** (Pearson r with p-value and n):
Merge EMA with each sensor modality on `pid` + `date` (inner join). Report n for each; skip if n < 5:
- `negative_affect_EMA` vs `intraday_rapids_sumsteps`
- `negative_affect_EMA` vs `summary_rapids_avgdurationasleepmain`
- `negative_affect_EMA` vs `barnett_hometime`
- `negative_affect_EMA` vs `rapids_sumdurationunlock`
- `negative_affect_EMA` vs `intraday_rapids_countepisodesedentarybout`
- `negative_affect_EMA` vs `barnett_siglocentropy`

If EMA is constant (all same value), explicitly note: "EMA has zero variance — Pearson correlation is undefined; report n and note the limitation."

**2. Cross-behavioral correlations** (Pearson r with p-value and n):
Compute pairwise correlations between behavioral metrics on days where both are valid:
- Home time ↔ phone unlock count
- Distance traveled ↔ phone unlock count
- Outgoing call count ↔ phone unlock count (if communication data available)
- Outgoing call count ↔ location entropy (if available)
- Incoming call count ↔ distance traveled (if available)
- Location entropy ↔ phone unlock count

**3. Peak EMA event analysis**:
Find top 3 highest EMA days and for each, compare behavioral metrics to their respective modality averages with % deviation:
```python
user_ema_sorted = user_ema.sort_values('negative_affect_EMA', ascending=False).head(3)
for _, row in user_ema_sorted.iterrows():
    peak_date = row['date']
    # Compare barnett_disttravelled, barnett_hometime, rapids_sumdurationunlock, etc. on that day vs. mean
```

**4. High vs. low EMA day comparisons**:
```python
ema_median = user_ema['negative_affect_EMA'].median()
high_ema_dates = user_ema[user_ema['negative_affect_EMA'] > ema_median]['date']
low_ema_dates = user_ema[user_ema['negative_affect_EMA'] <= ema_median]['date']
# Compare steps, screen time, home time, distance on high vs. low EMA days
```
If EMA is constant → all days are "low" → note no split possible; use the cross-behavioral correlations for insight instead.

**5. Depression-flagged week behavior comparison**:
Aggregate daily sensor data to weekly means (group by week). Join on dep_weekly dates (±7 days), then compare depressed vs. non-depressed week means for steps, sleep duration, phone unlocks, home time.
- If ALL weeks are flagged, compare high-symptom vs. low-symptom weeks using `feel_depressed` or `dep_weekly_subscale` (split at median).

**6. Consolidated temporal trends across modalities**:
Summarize T1 → T3 changes for ALL key metrics in a table. Classify each trajectory pattern.

### Phase 5 — Data quality (integrate into output)
For each modality, report valid days as n/92. Flag modalities with <20% coverage as "CRITICALLY SPARSE — interpret with caution."

## Synthesis Template

```
## Comprehensive Analysis of User <pid>

### Study Context
- Platform, study period (date range), data completeness per modality (n/92 days)

### Physical Activity
- Steps (mean ± std, min/max, valid days), sedentary/active balance and ratio, weekday vs. weekend
- Avg sedentary bout duration; active/sedentary ratio temporal trend
- Temporal trend: T1/T2/T3 step means, early/late means, % change, trajectory pattern

### Sleep
- Duration (hours), efficiency (%), timing (bedtime HH:MM, wake HH:MM), variability, time-to-fall-asleep
- Weekday vs. weekend; temporal trend T1/T2/T3 with early/late comparison

### Communication
- Call frequency (outgoing/incoming/missed), proactivity ratio, distinct contacts
- Call timing window: first-call HH:MM and last-call HH:MM; early vs. late shift in window
- Count/duration dissociation: explicitly flag if call count and duration trend opposite
- T1/T2/T3 + early/late for: outgoing count, incoming count, proactivity ratio, distinct contacts, duration

### Location & Mobility
- Daily distance (mean ± std, early/late %), home time (hours/day, %)
- Circadian routine score, location entropy (nats), transitions/day, significant places
- Top-3 location time distribution (minutes, T1/T2/T3)
- Avg flight duration ± std (seconds), avg speed (km/hr)
- Temporal trend T1/T2/T3 for distance, radius of gyration, home time, circadian, entropy, transitions, significant places, flight duration, speed

### Phone Usage
- Unlock count, screen time (hours), avg session duration ± std (minutes), first-use HH:MM
- Home vs. study unlock count and duration split; home-use fraction
- Count/duration dissociation: flag if unlock count and avg session duration trend opposite
- Temporal trend T1/T2/T3 + early/late for unlock count, total duration, avg session, first-use time, home/study unlocks

### Social Proximity (Connectivity)
- BT scan rate, unique devices per day, scan efficiency (scans/device)
- Temporal trend T1/T2/T3 for scan count, unique devices, scan efficiency

### Mental Health
- Depression trajectory: weekly flag rate (n/total weeks), feel_depressed/feel_anxious means, T1/T2/T3 trends
- Depression and anxiety subscale T1/T2/T3 trends; end-term BDI2 + dep status
- EMA negative affect: mean ± std, T1/T2/T3, early/late means, % change, trajectory pattern
- Pre→Post changes for ALL scales (UCLA, SocialFit, 2waySSS×4, ERQ×2, BRS, CHIPS, PSS, STAIS, MAAS, CESD-9, CESD-10) with ↑↓ and improved/worsened labels

### Cross-Modal Patterns
- EMA correlations with behavioral signals (list r, p, n for each; note if EMA is constant)
- Cross-behavioral correlations (home time vs. unlocks, distance vs. unlocks, calls vs. entropy, etc.)
- Peak EMA days: top 3 dates with behavioral context (deviations from mean)
- Behavioral differences on high vs. low EMA days (or high vs. low symptom if EMA constant)
- Behavioral differences in depressed vs. non-depressed weeks (or high vs. low symptom if all weeks flagged)
- Consolidated temporal shift table (T1→T3 and early→late for all key metrics with trajectory pattern)

### User Profile
- 4–6 sentence synthesis explicitly connecting behavioral patterns, temporal trends, and mental health
- Highlight discrepancies (e.g., low self-reported depression but worsening stress/social support)
- Identify dominant behavioral signals (which metrics most distinguish this user's mental state)
- Note any behavioral-mental health paradoxes (e.g., improving clinical scores but worsening behavioral markers)
```

## Common Pitfalls

1. **Platform column**: Use `user_platform['platform'].values[0]` — the column is `platform`, NOT `os`. Accessing `['os']` raises a KeyError.

2. **Always convert date to datetime before using `.dt`**: `user_df['date'] = pd.to_datetime(user_df['date'])` — omitting this causes `AttributeError: Can only use .dt accessor with datetimelike values`.

3. **Home time column**: Use `barnett_hometime` (minutes/day). `barnett_homelabel` and `doryab_homelabel` are cluster labels, not durations.

4. **Sleep efficiency**: `summary_rapids_avgefficiencymain` is already a percentage (e.g., 93.5). Never multiply by 100.

5. **Minute encoding**: Bedtime/wake/call times are minutes-since-midnight. Convert: `f"{int(m//60):02d}:{int(m%60):02d}"`. Values ≥ 1440 span next day (e.g., 1500 → 01:00 next day).

6. **Survey columns must use exact names with _PRE/_POST suffix**. Use `df.columns.tolist()` on first access; `get_field_description` does NOT work on survey files.

7. **Three-period analysis**: Split by thirds of the study date range (not calendar months). Always report T1, T2, T3 means individually and classify the pattern.

8. **Sparse data**: Always check `df[col].notna().sum()` before computing stats. Some users have <14/92 days for some modalities — correlations require >5 overlapping days.

9. **Location GPS outliers**: `barnett_disttravelled` and `barnett_rog` can have extreme GPS errors. Use `values[values < values.median() * 10]` before averaging.

10. **Weekly vs. daily merge**: `dep_weekly` is weekly; sensor data is daily. Aggregate daily data into 7-day windows aligned with each `dep_weekly` date row.

11. **EMA correlation requires inner merge on date**: Report n for each correlation; p-values are unreliable when n < 10. If EMA values are constant (all same value), Pearson r is undefined — note this explicitly and rely on cross-behavioral correlations instead.

12. **All weeks flagged for depression**: If dep flag rate = 100%, perform high-symptom vs. low-symptom comparison using `feel_depressed` or `dep_weekly_subscale` (split at median).

13. **Count/duration dissociation** is analytically important for both communication and phone usage. Always check whether count (frequency) and duration per session are trending in opposite directions, as this reveals behavioral quality shifts beyond simple quantity changes.

14. **Scan efficiency** = `rapids_countscans / rapids_uniquedevices`. Rising efficiency with declining unique devices suggests narrowing social environment.

15. **Trajectory pattern classification**: Label each metric's temporal trend as: *progressive increase*, *progressive decline*, *inverted-U* (peak in T2), *U-shaped* (trough in T2), or *mixed/stable*.

16. **Insight quality**: Each generated insight should include specific numeric values, % changes, T1/T2/T3 values (or early/late), trajectory pattern label, and a brief behavioral interpretation connecting to mental health context. Avoid purely descriptive statements without interpretation.
