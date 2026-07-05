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
| File | Modality | # Features |
|---|---|---|
| `activity_allday_raw.csv` | Steps, sedentary/active bouts | 22 |
| `sleep_allday_raw.csv` | Duration, efficiency, timing | 34 |
| `communication_allday_raw.csv` | Call counts, duration, contacts | 29 |
| `connectivity_allday_raw.csv` | Bluetooth scans, unique devices | 36 |
| `location_allday_raw.csv` | Mobility, home time, entropy | 41 |
| `phone_usage_allday_raw.csv` | Unlock frequency, usage duration | 42 |

**Mental-health / survey files** (no `_fields.json` — only readable via `execute_code`):
| File | Contents |
|---|---|
| `dep_weekly.csv` | Weekly depression/anxiety flags, feel_anxious, feel_depressed, dep, BDI2 |
| `dep_endterm.csv` | End-of-study BDI2 score and dep flag |
| `ema.csv` | Daily negative_affect_EMA scores |
| `pre.csv` | Pre-study surveys: STAIS, PSS, CESD_9/10, BFI10, UCLA, BRS, CHIPS, MAAS, ERQ, SocialFit, 2waySSS |
| `post.csv` | Post-study surveys: same scales + BDI2_POST |
| `platform.csv` | iOS vs Android flag |
| `dataset_summary.json` | Metadata overview |

## Tool Usage Rules

`get_field_description` **only works** for the six sensor CSVs above.
**Do NOT call it** for dep_weekly, dep_endterm, ema, pre, post, platform — those
have no `_fields.json` and will always fail. Read them directly with
`execute_code`.

## Recommended Analysis Pipeline

### Phase 1 — Orientation (2–3 calls)
```python
# 1. list_files to confirm available files
# 2. get_field_description on 2-3 sensor files to learn key column names
# 3. Optional: read dataset_summary.json once for participant count & date range
```

### Phase 2 — Per-modality stats for the target user (6 calls, one per modality)
Filter all sensor DFs by `pid == '<user_id>'` and compute:

**Activity** — key columns:
- `intraday_rapids_sumsteps` — daily step count
- `intraday_rapids_countepisodesedentarybout` / `activebout` — sedentary vs. active episodes
- Compute: mean ± std, min/max, weekday vs. weekend difference

**Sleep** — key columns:
- `summary_rapids_avgdurationasleepmain` — sleep duration (minutes)
- `summary_rapids_avgefficiencymain` — sleep efficiency (**already 0–100**, NOT decimal)
- `summary_rapids_avgdurationtofallasleepmain` — sleep onset latency
- Time of going to sleep / waking up can be inferred from related columns

**Communication** — key columns:
- `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count`
- `rapids_outgoing_meanduration`, `rapids_outgoing_distinctcontacts`
- Compute: outgoing/incoming ratio (>1 = proactive communicator)

**Location** — key columns:
- `barnett_avgflightdur`, `barnett_avgflightlen` — mobility metrics
- `barnett_homelabel` — minutes at home
- `barnett_circdnrtn` — circadian routine consistency (0–1)
- `barnett_rog` — radius of gyration (meters)
- `barnett_siglocsvisited` — significant locations visited

**Phone Usage** — key columns:
- `rapids_countepisodeunlock` — unlock frequency
- `rapids_sumdurationunlock` — total screen time (minutes)
- `rapids_sumdurationunlockhome` — home screen time

**Connectivity** — key columns:
- `rapids_countscans` — total BT scans/day
- `rapids_uniquedevices` — unique BT devices detected
- `rapids_countscanshighestpowerlevel` — social proximity signal

### Phase 3 — Mental health profile (1–2 calls)
```python
dep_weekly = pd.read_csv('dep_weekly.csv')
dep_endterm = pd.read_csv('dep_endterm.csv')
ema = pd.read_csv('ema.csv')
pre = pd.read_csv('pre.csv')
post = pd.read_csv('post.csv')
platform = pd.read_csv('platform.csv')

user_dep = dep_weekly[dep_weekly['pid'] == uid]
user_endterm = dep_endterm[dep_endterm['pid'] == uid]
user_ema = ema[ema['pid'] == uid]
user_pre = pre[pre['pid'] == uid]
user_post = post[post['pid'] == uid]
user_platform = platform[platform['pid'] == uid]
```

Extract and report:
- Platform (iOS/Android)
- Pre/post survey changes: STAIS (anxiety), PSS (stress), CESD (depression), BFI10 personality, UCLA (loneliness), BRS (resilience)
- Weekly depression flag rate (dep=True weeks / total weeks)
- EMA negative affect: mean, std, trend over time
- End-term BDI2 score and dep status

### Phase 4 — Cross-modal correlation & synthesis (1–2 calls)

Merge sensor dataframes on `pid` + `date`, then compute:
- Pearson correlations: steps ↔ phone usage, steps ↔ sleep duration, EMA ↔ activity/sleep
- Compare behavioral metrics between depression-flagged vs non-flagged weeks
- Temporal trend: compare first-half vs second-half metrics for activity, sleep, phone usage

### Phase 5 — Data quality check (optional, 1 call)
Count non-NaN rows per modality. If a modality has fewer valid rows than the 92-day
maximum, note this in the summary (e.g., activity data available only 14/92 days).

## Synthesis Template

Structure the final output as:

```
## Comprehensive Analysis of User <pid>

### Study Context
- Platform, study period, data completeness per modality

### Physical Activity
- Daily steps (mean ± std, min, max), sedentary/active balance, weekday vs. weekend

### Sleep
- Duration (hours), efficiency (%), timing (bedtime, wake time), variability

### Communication
- Call frequency, outgoing/incoming ratio, social diversity

### Location & Mobility
- Daily distance, home time (%), circadian routine score, radius of gyration

### Phone Usage
- Unlock count, screen time, home vs. elsewhere split

### Social Proximity (Connectivity)
- BT scan rate, unique devices per day

### Mental Health
- Depression trajectory (weekly flag rate, BDI2 endpoint)
- EMA negative affect trend
- Pre→post survey changes (highlight ↑↓ on key scales)

### Cross-Modal Patterns
- Notable correlations between behavioral and mental health signals
- Behavioral differences in depressed vs. non-depressed weeks

### User Profile
- 3–5 sentence synthesis connecting lifestyle, behavior, and mental health
```

## Common Pitfalls

1. **Sleep efficiency**: `summary_rapids_avgefficiencymain` is already in percentage
   (e.g., 93.5, not 0.935). Never multiply by 100 — values of 9000%+ are a sign
   that you double-counted.

2. **Missing field descriptions**: `get_field_description` fails for dep_weekly,
   ema, pre, post, dep_endterm, platform. Read their structure directly by printing
   `df.columns.tolist()` and `df.head(2)` inside `execute_code`.

3. **Sparse data**: Some participants have very few valid rows for activity or sleep
   (e.g., 14/92 days). Always check `df.notna().sum()` before computing averages
   and note the effective sample size.

4. **Merging sensor + weekly data**: Sensor data is daily; dep_weekly is weekly.
   When comparing, aggregate sensor data to weekly means before joining on `pid`
   and approximate date window.

5. **Avoid re-running dataset_summary.json**: It gives population-level metadata.
   One read is enough — don't repeat it across multiple turns.

6. **Timezone / minute encoding**: Bedtime and wake time are often stored as
   minutes-since-midnight. Convert to HH:MM for readability (e.g., 150 min →
   2:30 AM).
