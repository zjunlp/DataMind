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

## Recommended Execution Plan

Execute in this sequence (~14–18 `execute_code` calls total). Each call is dedicated — avoid combining multiple modalities into one call, as this degrades insight depth:

1. `list_files` → verify all files present
2. `get_field_description` on 2–3 sensor files (activity, sleep, location) to discover extra columns
3. Activity analysis (steps, bouts, weekday/weekend, T1/T2/T3, active/sedentary ratio)
4. Sleep analysis (duration, efficiency, timing, weekday/weekend, T1/T2/T3)
5. Communication analysis (counts, duration, proactivity ratio, call timing window, count/duration dissociation, T1/T2/T3)
6. Location analysis (distance, home time, entropy, circadian, flight duration, top locations, T1/T2/T3)
7. Phone usage analysis (unlock count, duration, session stats, home/study split, count/duration dissociation, T1/T2/T3)
8. Connectivity analysis (scan count, unique devices, scan efficiency, T1/T2/T3)
9. Mental health — weekly surveys (dep_weekly, ema, dep_endterm with T1/T2/T3 for feel_depressed, feel_anxious, subscales)
10. Mental health — pre/post surveys and personality (all scale changes with ↑↓ arrows)
11. **Phase 4a** — EMA ↔ behavioral correlations (all 6 pairs with r, p, n)
12. **Phase 4b** — Cross-behavioral correlations (all 6 pairs) + peak EMA event analysis (top 3 days)
13. **Phase 4c** — High vs. low EMA day comparisons + depression-flagged week comparisons
14. Consolidated temporal trends table (all key metrics: T1/T2/T3, early/late, % change, trajectory label)
15. Data quality summary + weekday/weekend verification for all modalities
16. Final synthesis (user profile with paradox identification)

Steps 11–14 are the most analytically rich and must all be completed.

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

**CRITICAL — T1/T2/T3 and Early/Late date boundaries:**

Always compute d_min/d_max from the **full user dataframe including NaN rows**. Never filter to valid rows first, or the date boundaries shift and all T1/T2/T3 values become wrong.

```python
# CORRECT: use all 92 rows to anchor the date range
d_min, d_max = user_df['date'].min(), user_df['date'].max()
span = (d_max - d_min) / 3
t1 = user_df[user_df['date'] < d_min + span]          # NaN rows included; pandas .mean() skips them
t2 = user_df[(user_df['date'] >= d_min + span) & (user_df['date'] < d_min + 2*span)]
t3 = user_df[user_df['date'] >= d_min + 2*span]

mid = d_min + (d_max - d_min) / 2
early = user_df[user_df['date'] < mid]
late  = user_df[user_df['date'] >= mid]

# Report T1/T2/T3 and early/late means (pandas skips NaN by default)
t1_mean = t1[col].mean()
early_mean = early[col].mean()
late_mean  = late[col].mean()
pct_change = (late_mean - early_mean) / abs(early_mean) * 100  # early→late %
```

After computing T1/T2/T3, label the trajectory pattern:
- **progressive increase**: T1 < T2 < T3 (monotonic)
- **progressive decline**: T1 > T2 > T3 (monotonic)
- **inverted-U**: T2 > T1 and T2 > T3
- **U-shaped**: T2 < T1 and T2 < T3
- **mixed/stable**: no clear pattern

**Verify sign consistency**: The sign of the early→late % change must be consistent with the trajectory label (decline → negative %; increase → positive %). If they disagree, recheck the computation.

For each modality compute:
- Mean ± std, min/max over valid (non-NaN) rows
- Count of valid days (report as n/92)
- **Weekday vs. weekend difference** (`df['date'].dt.dayofweek` → 0–4 weekday, 5–6 weekend)
- T1/T2/T3 and early/late for all key metrics

**Activity** — Primary: `intraday_rapids_sumsteps`, `intraday_rapids_countepisodesedentarybout`, `intraday_rapids_countepisodeactivebout`. Compute and trend:
- Active/sedentary bout ratio and `intraday_rapids_avgdurationsedentarybout` (avg sedentary bout duration)
- T1/T2/T3 and early/late for: steps, active/sedentary ratio, avg sedentary bout duration, active bout count, sedentary bout count
- Weekday vs. weekend for steps (weekday steps often higher due to campus routine)

**Sleep** — `summary_rapids_avgdurationasleepmain` (minutes), `summary_rapids_avgefficiencymain` (**already 0–100**, never multiply by 100), `summary_rapids_avgdurationtofallasleepmain`, `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain`. Convert bedtime/wake minutes-since-midnight to HH:MM. Compute and trend:
- T1/T2/T3 and early/late for: duration, efficiency, bedtime, wake time, time-to-fall-asleep
- Weekday vs. weekend for sleep duration and timing

**Communication** — Primary: `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`, `rapids_outgoing_meanduration`, `rapids_incoming_meanduration`, `rapids_outgoing_distinctcontacts`. Compute and trend:
- Outgoing/incoming ratio (>1 = proactive)
- **Count/duration dissociation**: explicitly note when count and duration trend in opposite directions
- **Call timing window**: `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → HH:MM); report shift between early and late periods
- T1/T2/T3 and early/late for: outgoing count, incoming count, missed count, outgoing/incoming ratio, distinct contacts, mean outgoing duration, call timing window
- Weekday vs. weekend for outgoing call count

**Location** — Primary: `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` (minutes/day), `barnett_circdnrtn` (0–1), `barnett_siglocsvisited`, `barnett_siglocentropy` (nats), `barnett_avgflightdur` ± `barnett_stdflightdur` (seconds), `doryab_avgspeed` (km/hr), `doryab_numberlocationtransitions`, `doryab_timeattop1location` / `doryab_timeattop2location` / `doryab_timeattop3location` (minutes). Filter GPS outliers: drop values > median × 10 for `barnett_disttravelled` and `barnett_rog` before averaging. Compute and trend:
- T1/T2/T3 and early/late for: distance, radius of gyration, home time, circadian routine, location entropy, location transitions, significant places, avg flight duration, **std flight duration**, avg speed, top-1/top-2/top-3 location time
- Weekday vs. weekend for home time and distance
- Note: for students with remote/online classes, weekday home time can be GREATER than weekend home time — report both values and direction without assuming a fixed pattern

**Phone Usage** — Primary: `rapids_countepisodeunlock`, `rapids_sumdurationunlock` (minutes), `rapids_avgdurationunlock`, `rapids_stddurationunlock`, `rapids_firstuseafter00unlock` (minutes → HH:MM), `rapids_sumdurationunlock_locmap_home`, `rapids_countepisodeunlock_locmap_home`, `rapids_countepisodeunlock_locmap_study`. Compute and trend:
- Home-use fraction, **count/duration dissociation** if present (unlock count vs. avg session duration trending opposite directions)
- T1/T2/T3 and early/late for: unlock count, total duration, avg session duration, session std, first-use time, home unlocks, study unlocks
- Weekday vs. weekend for unlock count

**Connectivity** — `rapids_countscans`, `rapids_uniquedevices`. **Scan efficiency** = countscans / uniquedevices. Compute and trend:
- T1/T2/T3 and early/late for: scan count, unique devices, scan efficiency
- Weekday vs. weekend for scan count

### Phase 3 — Mental health profile (2 calls)

**Call 1 — Weekly surveys and EMA:**
```python
dep_weekly = pd.read_csv('dep_weekly.csv')
dep_endterm = pd.read_csv('dep_endterm.csv')
ema = pd.read_csv('ema.csv')
platform = pd.read_csv('platform.csv')  # column is 'platform', NOT 'os'

uid = '<user_id>'
user_dep = dep_weekly[dep_weekly['pid'] == uid].copy()
user_dep['date'] = pd.to_datetime(user_dep['date'])
user_endterm = dep_endterm[dep_endterm['pid'] == uid]
user_ema = ema[ema['pid'] == uid].copy()
user_ema['date'] = pd.to_datetime(user_ema['date'])
user_platform = platform[platform['pid'] == uid]
print(f"Platform: {user_platform['platform'].values[0]}")  # use 'platform' not 'os'
```

Extract and report:
- Platform, weekly depression flag rate (n/total weeks), end-term BDI2 + dep status
- `feel_depressed`, `feel_anxious`, `dep_weekly_subscale`, `anx_weekly_subscale` with T1/T2/T3 and early/late trends
- EMA: mean, std, min/max, T1/T2/T3 and early/late trend with trajectory classification

**Call 2 — Pre/post surveys and personality:**
- **Pre→Post survey changes** for ALL key scales: report Pre value, Post value, % change, ↑↓ arrows, and direction (improved/worsened). Include ALL of: UCLA, SocialFit, 2waySSS (all 4), ERQ reappraisal/suppression, BRS, CHIPS, PSS, STAIS, MAAS, CESD-9, CESD-10
- Personality (BFI10 pre only): extroversion, agreeableness, conscientiousness, neuroticism, openness

### Phase 4 — Cross-modal correlation & synthesis (3–4 calls)

These calls must all be completed. They generate the most analytically valuable insights.

**Phase 4a call — EMA ↔ Behavioral correlations** (Pearson r with p-value and n):
Merge EMA with each sensor modality on `pid` + `date` (inner join). After merge, `dropna()` on both columns before computing. Report n for each; skip if n < 5:
- `negative_affect_EMA` vs `intraday_rapids_sumsteps`
- `negative_affect_EMA` vs `summary_rapids_avgdurationasleepmain`
- `negative_affect_EMA` vs `barnett_hometime`
- `negative_affect_EMA` vs `rapids_sumdurationunlock`
- `negative_affect_EMA` vs `intraday_rapids_countepisodesedentarybout`
- `negative_affect_EMA` vs `barnett_siglocentropy`

If EMA is constant (all same value), note: "EMA has zero variance — Pearson correlation is undefined; report n and rely on cross-behavioral correlations for insight."

Also check: `negative_affect_EMA` vs `rapids_uniquedevices` (BT unique devices) if n ≥ 5; negative EMA↔unique-devices correlations signal social proximity effects on mood.

**Phase 4b call — Cross-behavioral correlations + Peak EMA events:**
Compute all 6 pairwise correlations (report r, p, n; skip if n < 5 after merge and dropna):
1. Home time ↔ phone unlock count
2. Distance traveled ↔ phone unlock count (apply GPS outlier filter first)
3. Outgoing call count ↔ phone unlock count
4. Outgoing call count ↔ location entropy
5. Incoming call count ↔ distance traveled
6. Location entropy ↔ phone unlock count

All 6 pairs must be reported (even if non-significant); omit only when data is unavailable.

Then compute **Peak EMA event analysis** — top 3 highest EMA days:
```python
user_ema_sorted = user_ema.sort_values('negative_affect_EMA', ascending=False).head(3)
for _, row in user_ema_sorted.iterrows():
    peak_date = row['date']
    # Compare barnett_disttravelled, barnett_hometime, rapids_sumdurationunlock,
    # intraday_rapids_sumsteps on that day vs. modality means; report % deviation
```

**Phase 4c call — High/low EMA day comparisons + Depression-flagged week behavior:**
```python
ema_median = user_ema['negative_affect_EMA'].median()
high_ema_dates = user_ema[user_ema['negative_affect_EMA'] > ema_median]['date']
low_ema_dates = user_ema[user_ema['negative_affect_EMA'] <= ema_median]['date']
# Compare steps, screen time, home time, distance on high vs. low EMA days
```
If EMA is constant → note no split possible; analyze high vs. low phone usage or other behavioral proxy.

**Depression-flagged week comparison:**
Aggregate daily sensor data to weekly means (group by week). Join on dep_weekly dates (±7 days), then compare depressed vs. non-depressed week means for steps, sleep duration, phone unlocks, home time.
- If ALL weeks are flagged: compare high-symptom vs. low-symptom weeks using `feel_depressed` or `dep_weekly_subscale` (split at median).

**Phase 4d call — Consolidated temporal trends table:**
Summarize T1/T2/T3 and early/late changes for ALL key metrics. Use values already computed in per-modality calls (don't recompute from scratch to avoid errors).

Required metrics to include — every row must have T1, T2, T3, Early, Late, %Change, Trajectory:
```
| Metric | T1 | T2 | T3 | Early | Late | % Change | Trajectory |
```
Include ALL of: steps, active bout count, sedentary bout count, active/sed ratio, avg sed bout duration; sleep duration, sleep efficiency, bedtime, wake time; outgoing calls, incoming calls, proactivity ratio, distinct contacts, outgoing call duration; distance, radius of gyration, home time, circadian routine, location entropy, transitions, significant places, **avg flight duration**, **std flight duration**, avg speed, top-1 location time, top-2 location time, top-3 location time; unlock count, total phone duration, avg session duration, first-use time, home unlocks, study unlocks; BT scans, unique devices, scan efficiency; EMA negative affect, feel_depressed, feel_anxious, dep_weekly_subscale, anx_weekly_subscale.

**Table accuracy rule**: After filling the table, verify each row: the sign of % change must match the trajectory direction (progressive decline → negative %; progressive increase → positive %). Any mismatch signals a computation error — recheck and correct.

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
- **Avg flight duration ± std flight duration** (seconds, T1/T2/T3 and early/late)
- Avg speed (km/hr); temporal trend T1/T2/T3 for all metrics

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
- Cross-behavioral correlations (all 6 pairs: home time vs. unlocks, distance vs. unlocks, outgoing calls vs. unlocks, outgoing calls vs. entropy, incoming calls vs. distance, entropy vs. unlocks)
- Peak EMA days: top 3 dates with behavioral context (deviations from mean)
- Behavioral differences on high vs. low EMA days (or high vs. low symptom if EMA constant)
- Behavioral differences in depressed vs. non-depressed weeks (or high vs. low symptom if all weeks flagged)
- Consolidated temporal shift table (T1→T3 and early→late for all key metrics with trajectory pattern)

### User Profile
- 4–6 sentence synthesis explicitly connecting behavioral patterns, temporal trends, and mental health
- **Behavioral-mental health paradoxes**: explicitly note any case where clinical survey scores improve but behavioral markers worsen (or vice versa). These paradoxes are among the most analytically valuable findings.
- **SocialFit discrepancy**: compare SocialFit_PRE→POST change with BT unique devices trend and outgoing call trend — flag divergence between perceived social fit and behavioral social engagement.
- Highlight discrepancies between self-reported mental health and passive behavioral signals
- Identify dominant behavioral signals (which metrics most distinguish this user's mental state)
```

## Common Pitfalls

1. **Platform column**: Use `user_platform['platform'].values[0]` — the column is `platform`, NOT `os`. Accessing `['os']` raises a KeyError.

2. **Always convert date to datetime before using `.dt`**: `user_df['date'] = pd.to_datetime(user_df['date'])` — omitting this causes `AttributeError: Can only use .dt accessor with datetimelike values`.

3. **T1/T2/T3 boundary must use the FULL user dataframe**: Always compute `d_min, d_max = user_df['date'].min(), user_df['date'].max()` on all 92 rows BEFORE any NaN filtering. If you compute d_min/d_max after `dropna()`, the boundaries shift and all T1/T2/T3 values become systematically wrong, producing incorrect trajectory patterns.

4. **Home time column**: Use `barnett_hometime` (minutes/day). `barnett_homelabel` and `doryab_homelabel` are cluster labels, not durations.

5. **Sleep efficiency**: `summary_rapids_avgefficiencymain` is already a percentage (e.g., 93.5). Never multiply by 100.

6. **Minute encoding**: Bedtime/wake/call times are minutes-since-midnight. Convert: `f"{int(m//60):02d}:{int(m%60):02d}"`. Values ≥ 1440 span next day (e.g., 1500 → 01:00 next day).

7. **Survey columns must use exact names with _PRE/_POST suffix**. Use `df.columns.tolist()` on first access; `get_field_description` does NOT work on survey files.

8. **Weekday vs. weekend home time**: For students with remote/online classes, weekday home time is often GREATER than weekend (they go out on weekends). Report both values explicitly and note which is higher.

9. **Sparse data**: Always check `df[col].notna().sum()` before computing stats. Some users have <14/92 days for some modalities — correlations require >5 overlapping days after inner merge and dropna.

10. **Location GPS outliers**: `barnett_disttravelled` and `barnett_rog` can have extreme GPS errors. Use `values[values < values.median() * 10]` before averaging; also apply this filter before computing correlations involving distance.

11. **Weekly vs. daily merge**: `dep_weekly` is weekly; sensor data is daily. Aggregate daily data into 7-day windows aligned with each `dep_weekly` date row.

12. **EMA correlation requires inner merge on date + dropna**: After merge, `dropna()` on both columns before `pearsonr`. Report n for each correlation; p-values are unreliable when n < 10. If EMA values are constant (std = 0), note explicitly and skip to cross-behavioral correlations.

13. **All weeks flagged for depression**: If dep flag rate = 100%, perform high-symptom vs. low-symptom comparison using `feel_depressed` or `dep_weekly_subscale` (split at median).

14. **Count/duration dissociation** is analytically important for both communication and phone usage. Always check whether count (frequency) and duration per session are trending in opposite directions, as this reveals behavioral quality shifts beyond simple quantity changes.

15. **Scan efficiency** = `rapids_countscans / rapids_uniquedevices`. Rising efficiency with declining unique devices suggests narrowing social environment.

16. **Consolidated table accuracy**: Every % change must be verified against the trajectory label. A "progressive decline" row must have a negative % change; a "progressive increase" must be positive. If they disagree, there is a computation error. Recompute using the formula: `(late_mean - early_mean) / abs(early_mean) * 100` where early/late are the actual half-study means, NOT T1/T3 proxies.

17. **Flight duration analysis**: Report both `barnett_avgflightdur` (avg) and `barnett_stdflightdur` (variability) with T1/T2/T3 and early/late trends. These measure movement episode durations (not air travel) and their variability is informative about routine consistency.

18. **Insight quality**: Every generated insight must include specific numeric values, % changes, T1/T2/T3 values (or early/late), trajectory pattern label, and a behavioral interpretation connecting to mental health context. Purely descriptive statements without interpretation or numbers are insufficient. Paradoxes between clinical self-reports and passive behavioral signals are the highest-value findings.
