---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (`pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **28–32 high-quality, distinct QA pairs**. Each QA pair covers ONE specific dimension — never bundle multiple modalities or metrics into a single pair.

## Dataset Structure

All CSV files share columns `pid` and `date`. Filter every file by the target `pid`.

**Sensor files** (92 days per participant, many NaN rows are normal):
- `activity_allday_raw.csv` — daily step count, active/sedentary bout counts and durations
- `sleep_allday_raw.csv` — sleep duration (minutes), efficiency, bedtime/wake-time
- `communication_allday_raw.csv` — incoming/outgoing/missed call counts, durations, distinct contacts
- `location_allday_raw.csv` — distance traveled, radius of gyration, home time, significant places, circadian routine, location entropy, location transitions, top-location times, flight characteristics, average speed
- `phone_usage_allday_raw.csv` — unlock episode count, total/avg/std duration; location-context columns
- `connectivity_allday_raw.csv` — Bluetooth scan count, unique devices

**Assessment files** (one row per observation per participant):
- `ema.csv` — `negative_affect_EMA` score, timestamped
- `dep_weekly.csv` — weekly `feel_anxious`, `feel_depressed`, `BDI2` (endterm only), `dep`, `dep_weekly_subscale`, `anx_weekly_subscale`
- `pre.csv` / `post.csv` — baseline vs. end-of-study psychological scales
- `dep_endterm.csv` — final depression label and BDI2 score

**Schema tip**: Use `get_field_description(data_file="<filename>")` for the six sensor CSV files.

**Known column pitfalls**:
- `platform.csv` uses column `platform`, not `os`
- Home time: use `barnett_hometime` (not `barnett_homelabel`)
- `summary_rapids_*` columns repeat period-wide values — use `intraday_rapids_*` for daily variation
- Phone usage location columns: suffixes `_locmap_home`, `_locmap_living`, `_locmap_study`, `_locmap_greens`, `_locmap_exercise`
- Proactivity ratio with near-zero incoming: use `max(incoming, 0.1)` to avoid artifacts; a ratio >10 is likely a near-zero denominator — report raw counts instead

## Analysis Workflow

### 1. Orient to the participant

Check data availability across all files first:
```python
import pandas as pd
pid = "INS-W_011"
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

Always `dropna()` on the target column before splitting.

**Early/late split** (primary comparison unit):
```python
def early_late(df, pid, value_col):
    sub = df[df['pid'] == pid].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date').dropna(subset=[value_col])
    mid = len(sub) // 2
    return sub.iloc[:mid][value_col].mean(), sub.iloc[mid:][value_col].mean()
```

**Thirds segmentation** — apply to EVERY modality. Thirds reveal non-linear patterns (U-shaped, inverted-U, progressive, peak-in-middle, stable-then-drop) that early/late splits hide. Always name the pattern explicitly.

**Weekday vs weekend** — compute for every modality with ≥10 weekday and ≥5 weekend valid days. Report if difference >15%.

### 3. Data to extract per modality

| Modality | Primary columns | Secondary columns (also extract) |
|---|---|---|
| EMA | `negative_affect_EMA` | spike dates + behavioral coincidence |
| Weekly depression | `feel_depressed`, `feel_anxious`, `dep`, `BDI2` | `dep_weekly_subscale`, `anx_weekly_subscale` |
| Pre/Post — psychological | `CESD*`, `STAI*`, `PSS*`, `UCLA*` | — |
| Pre/Post — social | `2waySSS_*` (all 4), `SocialFit*` | — |
| Pre/Post — regulation | `ERQ_*`, `BRS*`, `MAAS*`, `CHIPS*` | — |
| Activity | `intraday_rapids_sumsteps` | `intraday_rapids_sumdurationactivebout`, `intraday_rapids_sumdurationsedentarybout`, bout counts, avg bout durations |
| Sleep | `summary_rapids_sumdurationasleepmain` | `summary_rapids_avgefficiencymain`, `summary_rapids_firstbedtimemain`, `summary_rapids_lastwaketimemain` |
| Communication | `rapids_outgoing_count`, `rapids_incoming_count`, `rapids_missed_count` | `rapids_outgoing_sumduration`, `rapids_incoming_sumduration`, `rapids_outgoing_distinctcontacts`, `rapids_outgoing_timefirstcall`, `rapids_outgoing_timelastcall`, `rapids_incoming_meanduration` |
| Location | `barnett_disttravelled`, `barnett_rog`, `barnett_hometime` | `barnett_circdnrtn`, `doryab_locationentropy`, `doryab_numberlocationtransitions`, `doryab_avgspeed`, `barnett_avgflightdur`, `barnett_stdflightdur`, `doryab_timeattop1location`, `doryab_timeattop2location`, `doryab_timeattop3location`, `doryab_numberofsignificantplaces` |
| Phone usage | `rapids_countepisodeunlock`, `rapids_sumdurationunlock` | `rapids_avgdurationunlock`, `rapids_stddurationunlock`, `rapids_maxdurationunlock`, `rapids_firstuseafter00unlock`, location-context columns |
| Connectivity | `rapids_countscans`, `rapids_uniquedevices` | scans-per-device ratio |

### 4. Extract sub-dimension insights

**Named thirds pattern**: Always state the pattern type: "U-shaped" (low-high-low), "inverted-U" (high-low-high), "progressive decline/increase" (monotone), "peak-then-drop", "stable-then-spike", etc.

**Sleep timing phase shift**: Convert bedtime/wake-time from minutes-from-midnight to HH:MM. Report shift direction and magnitude (e.g., "bedtime delayed by 90 minutes").

**Active/sedentary ratio**: Compute `active_duration / sedentary_duration` for early vs late. Also check if avg sedentary bout duration changed.

**Communication count vs duration dissociation**: If outgoing call count increases but mean duration decreases (or vice versa), name this explicitly. Compute mean duration per call = `rapids_outgoing_sumduration / rapids_outgoing_count`. Do the same for incoming calls.

**Proactivity ratio**: `outgoing_count / max(incoming_count, 0.1)` early vs late. Report the actual counts if ratio >10.

**Communication timing window**: From `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → HH:MM), report whether the calling window shifted or expanded.

**Phone session variability**: Compare `rapids_stddurationunlock` and `rapids_maxdurationunlock` early vs late. Increasing variability (std, max) indicates more extreme phone sessions.

**Phone first-use timing**: Report if `rapids_firstuseafter00unlock` shifted (earlier = more disrupted sleep/wake).

**Phone by location context**: Compare home vs study vs other contexts between early and late.

**Location diversity**: Extract `doryab_locationentropy` and `doryab_numberlocationtransitions` early/late + thirds.

**Top-location time distribution**: Extract `doryab_timeattop1location`, `doryab_timeattop2location`, `doryab_timeattop3location` early vs late. A shift (e.g., top-1 decreasing, top-2 increasing) indicates changed location usage pattern.

**Movement characteristics**: Extract `barnett_avgflightdur`, `barnett_stdflightdur` (flight = movement episode, in seconds) early vs late. Shorter avg flight duration can indicate more frequent short trips vs longer point-to-point travel.

**Average speed**: `doryab_avgspeed` (km/hr during movement periods). Changes signal shifts in transport mode or travel pattern.

**Circadian routine**: `barnett_circdnrtn` (0–1) early/late + thirds.

**EMA spike analysis**: Identify top 3 highest negative-affect days. For each, report the date and value, then check distance, home time, sleep, and phone usage vs participant average. State whether isolation, travel, or disrupted sleep coincided.

```python
# Anomaly weeks
sub['week'] = pd.to_datetime(sub['date']).dt.isocalendar().week
weekly = sub.groupby('week')[col].mean()
anomaly_weeks = weekly[weekly > weekly.median() * 2]
```

**Bluetooth scan efficiency**: `scans / unique_devices` early vs late. A rising ratio means fewer, more-repeatedly-detected devices.

**Cross-modal correlations**: Compute Pearson r for all pairs with ≥10 shared valid days. Report |r| > 0.3.

**Self-report vs behavioral discrepancy**: When survey direction contradicts behavioral signal (e.g., perceived support falls but outgoing calls rise), name this explicitly.

**Behavioral vs psychological dissociation**: If behavioral metrics trend opposite to psychological metrics, surface as a dedicated QA pair.

### 5. Formulate and submit QA pairs

**One dimension per QA pair** — never combine two modalities or two metrics in one question. Each item in the checklist below should produce its own QA pair.

Valid QA pair types:
- **Early/late change**: "How did X change between early and late periods?"
- **Trajectory-only**: "What was the trajectory pattern of X across the three study periods?" (thirds T1→T2→T3 + named pattern)
- **Sub-dimension**: "Was there a count/duration dissociation in X?"
- **Cross-modal**: "What is the relationship between X and Y?"
- **Meta-pattern**: "Is there a discrepancy between self-reported X and behavioral Y?"

Each answer must include:
- Concrete numbers (mean values, % change, direction)
- Pattern name when using thirds data
- Magnitude description ("increased substantially" ≈ >20%, "modestly" ≈ 5–20%, "remained stable" ≈ <5%)

Submit incrementally: `submit_qa_pair(q="...", a="...")`

**Good QA pair examples**:

Early/late change:
> Q: "How did outgoing call frequency and duration change, and was there a count/duration dissociation?"  
> A: "Outgoing calls increased from 1.24 to 3.67/day (+196%), but mean duration per call fell sharply from 305s to 58s (-81%) — a classic count/duration dissociation: more frequent but much shorter conversations in the late period."

Trajectory-only:
> Q: "What was the trajectory pattern of the user's location entropy across the three study periods?"  
> A: "Location entropy showed a U-shaped pattern: 0.449 (T1) → 0.485 (T2) → 0.238 nats (T3). The sharp 51% drop in the final third indicates substantially reduced spatial diversity toward study end."

EMA spike analysis:
> Q: "Did the user's peak negative affect episodes coincide with specific behavioral events?"  
> A: "The two highest EMA days (May 20: 5.0, May 24: 8.0) coincided with the two largest travel days (641 km and 4,428 km), suggesting travel-related stress. EMA-distance correlation was moderately positive (r=0.31)."

### 6. QA coverage checklist

Aim to cover all of these (skip only if data is entirely NaN):

**Psychological — assessment**
- [ ] EMA negative affect: early/late change + named thirds pattern
- [ ] EMA spike analysis: top 3 specific high-affect dates with coinciding behavioral signals
- [ ] Weekly depression: status, symptom trajectory (`feel_depressed`, `feel_anxious`), final BDI2
- [ ] Weekly depression/anxiety subscale trajectory (separate thirds analysis)
- [ ] Pre/post: psychological state (depression, anxiety/stress, loneliness)
- [ ] Pre/post: social support (all 4 dimensions + social fit)
- [ ] Pre/post: emotion regulation / coping / resilience / mindfulness

**Activity**
- [ ] Steps: early/late + named thirds pattern
- [ ] Active/sedentary duration ratio + avg sedentary bout duration shift
- [ ] Active/sedentary bout count trajectory

**Sleep**
- [ ] Sleep duration: early/late + named thirds pattern
- [ ] Sleep efficiency: early/late + named thirds pattern (separate from duration)
- [ ] Sleep timing phase shift (bedtime + wake-time in HH:MM)

**Communication**
- [ ] Outgoing call count + duration: early/late, note dissociation if present
- [ ] Incoming call count + mean duration per call: early/late, note dissociation if present
- [ ] Missed call trend (early/late + thirds pattern)
- [ ] Proactivity ratio (outgoing/incoming) early vs late
- [ ] Network diversity (distinct contacts) early vs late
- [ ] Communication timing window shift (first/last call times) if available

**Location**
- [ ] Mobility: distance traveled early/late + named thirds pattern
- [ ] Radius of gyration trajectory
- [ ] Home time: early/late + named thirds pattern
- [ ] Circadian routine trajectory
- [ ] Location entropy + transitions: early/late + named thirds pattern
- [ ] Top-location time distribution (top-1, top-2, top-3) shift
- [ ] Movement characteristics: avg/std flight duration early vs late
- [ ] Average speed change
- [ ] Temporal anomaly (if detected) — specific peak dates + magnitudes

**Phone usage**
- [ ] Unlock count + total duration: early/late, note dissociation if present
- [ ] Session length shift (avg session duration) + variability (std, max session duration)
- [ ] First-use timing shift
- [ ] Phone by location context (home vs study vs other) early vs late

**Connectivity**
- [ ] Scan count + unique devices: early/late + named thirds pattern
- [ ] Bluetooth scan efficiency (scans-per-device ratio) trajectory
- [ ] Weekday vs weekend contrast (if >15% difference)

**Cross-modal & meta**
- [ ] Cross-modal correlation (|r| > 0.3, ≥10 shared days)
- [ ] Self-report vs behavioral discrepancy (if present)
- [ ] Behavioral trajectory vs psychological trajectory dissociation (if trends diverge)
- [ ] Weekday vs weekend analysis for any modality showing >15% difference

## Common Pitfalls

- **Never split raw 92-row arrays** — always `dropna()` on the target column before early/late or thirds.
- **`summary_rapids_*` are period-wide summaries** — use `intraday_rapids_*` for daily variation.
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — infer from column names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score.
- **Pre/post scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better (good), higher PSS = more stress (bad), higher BRS = better resilience (good).
- **`barnett_homelabel` does not exist** — use `barnett_hometime`.
- **Distance outliers**: filter values >10× median before computing means.
- **Communication data is often sparse** (<30% valid days for some participants) — note this but still extract available patterns.
- **Pre/post QA pairs must be split into three separate pairs**: psychological state, social support, and emotion regulation/coping.
- **Thirds pattern naming is mandatory** — always state whether progressive, U-shaped, inverted-U, peak-in-middle, stable-then-drop, etc. Do not just list T1/T2/T3 numbers without naming the pattern.
- **One QA pair per dimension** — do not bundle "activity and Bluetooth" or "mobility and home time" in one question. Each checklist item deserves its own pair.
