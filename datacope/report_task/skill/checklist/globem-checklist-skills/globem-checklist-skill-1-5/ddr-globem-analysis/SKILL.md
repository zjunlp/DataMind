---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (`pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **28–37 high-quality, distinct QA pairs**. Each QA pair covers ONE specific dimension — never bundle multiple modalities or metrics into a single pair.

**Critical workflow principle**: Submit each QA pair *immediately* after computing the data for that dimension. Do not batch all submissions at the end — incremental submission ensures complete coverage and prevents missed sub-dimensions.

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
- `pre.csv` / `post.csv` — baseline vs. end-of-study psychological scales (columns use `_PRE` / `_POST` suffixes, e.g., `CESD_9items_PRE`, `CESD_9items_POST`)
- `dep_endterm.csv` — final depression label and BDI2 score

**Schema tip**: Use `get_field_description(data_file="<filename>")` for the six sensor CSV files.

**Known column pitfalls**:
- `platform.csv` uses column `platform`, not `os`
- Home time: use `barnett_hometime` (not `barnett_homelabel`)
- Sleep duration: use `summary_rapids_sumdurationasleepmain` for daily sleep duration; bedtime/waketime use `summary_rapids_firstbedtimemain` / `summary_rapids_lastwaketimemain`
- `summary_rapids_*` columns repeat period-wide values — use `intraday_rapids_*` for daily activity variation
- Phone usage location columns: suffixes `_locmap_home`, `_locmap_living`, `_locmap_study`, `_locmap_greens`, `_locmap_exercise`
- Proactivity ratio with near-zero incoming: use `max(incoming, 0.1)` to avoid artifacts; a ratio >10 is likely a near-zero denominator — report raw counts instead
- Pre/post columns use exact names like `CESD_9items_PRE`, `UCLA_10items_PRE`, `2waySSS_receiving_emotional_PRE`, `BRS_PRE`, `CHIPS_PRE` (not `CESD_sum` etc.)

## Starter Code — Define Once, Reuse Throughout

Copy this helper at the top of your first code block and reuse for all modalities:

```python
import pandas as pd
import numpy as np

pid = "INS-W_XXX"  # replace with actual pid

def early_late_thirds(df, pid, value_col):
    sub = df[df['pid'] == pid].copy()
    sub = sub.sort_values('date').dropna(subset=[value_col])
    n = len(sub)
    if n < 2:
        return None, None, None, None, None
    mid = n // 2
    early = sub.iloc[:mid][value_col].mean()
    late = sub.iloc[mid:][value_col].mean()
    third = n // 3
    if third > 0:
        t1 = sub.iloc[:third][value_col].mean()
        t2 = sub.iloc[third:2*third][value_col].mean()
        t3 = sub.iloc[2*third:][value_col].mean()
        if t1 < t2 > t3: pattern = "inverted-U"
        elif t1 > t2 < t3: pattern = "U-shaped"
        elif t1 < t2 < t3: pattern = "progressive increase"
        elif t1 > t2 > t3: pattern = "progressive decline"
        else: pattern = "mixed/stable"
    else:
        t1 = t2 = t3 = pattern = None
    pct = (late - early) / early * 100 if early and early != 0 else 0
    return early, late, pct, (t1, t2, t3), pattern

def mins_to_hhmm(mins):
    if pd.isna(mins): return "N/A"
    h = int(mins // 60) % 24
    m = int(mins % 60)
    return f"{h:02d}:{m:02d}"
```

## Analysis Workflow

### 1. Orient to the participant

Check data availability across all files first:
```python
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

Sparse coverage is normal for some modalities (communication can be <30% valid). Always extract what data exists rather than skipping modalities. If valid days <5, note sparsity in the QA answer but still report available trends.

### 2. Temporal segmentation — apply to ALL modalities

Always `dropna()` on the target column before splitting. The `early_late_thirds()` function handles this automatically.

**Early/late split** (primary comparison unit): first vs second half of valid observations.

**Thirds segmentation**: apply to EVERY modality. Thirds reveal non-linear patterns (U-shaped, inverted-U, progressive, peak-in-middle, stable-then-drop) that early/late splits hide. Always name the pattern explicitly.

**Weekday vs weekend** — compute for every modality with ≥10 weekday and ≥5 weekend valid days. Report if difference >15%.

### 3. Compute sub-dimension insights per modality

**Sleep timing phase shift**: Convert bedtime/wake-time from minutes-from-midnight to HH:MM using `mins_to_hhmm()`. Report shift direction and magnitude.

**Active/sedentary ratio**: Compute `active_duration / sedentary_duration` for early vs late. Also check if avg sedentary bout duration changed.

**Communication count vs duration dissociation**: If outgoing call count increases but mean duration decreases (or vice versa), name this explicitly. Compute `mean_duration_per_call = sumduration / count`. Do same for incoming calls.

**Proactivity ratio**: `outgoing_count / max(incoming_count, 0.1)` early vs late. Report raw counts if ratio >10.

**Communication timing window**: From `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → HH:MM), report whether the calling window shifted or expanded.

**Phone session variability**: Compare `rapids_stddurationunlock` and `rapids_maxdurationunlock` early vs late. Increasing variability (std, max) indicates more extreme phone sessions.

**Phone first-use timing**: Report if `rapids_firstuseafter00unlock` shifted (earlier = more disrupted sleep/wake).

**Phone by location context**: Compare home vs study vs other contexts between early and late using `_locmap_home`, `_locmap_study` columns.

**Top-location time distribution**: Extract `doryab_timeattop1location`, `doryab_timeattop2location`, `doryab_timeattop3location` early vs late. A shift indicates changed location usage pattern.

**Movement characteristics**: Extract `barnett_avgflightdur`, `barnett_stdflightdur` (seconds) early vs late. Shorter avg flight = more frequent short trips vs longer point-to-point travel.

**Average speed**: `doryab_avgspeed` (km/hr during movement). Changes signal shifts in transport mode.

**EMA spike analysis**: Identify top 3 highest negative-affect days. For each, report the date and value, then check distance, home time, sleep duration, and phone unlock count vs participant average. State whether isolation, travel, or disrupted sleep coincided.

**Bluetooth scan efficiency**: `scans / unique_devices` early vs late. Rising ratio = fewer devices, each scanned more repeatedly.

**Cross-modal correlations**: Merge EMA with each behavioral modality on date (inner join), then dropna on both columns together, compute Pearson r. Require ≥10 shared valid days. Report |r| > 0.3.

```python
# Correct cross-modal correlation pattern (merge first, then dropna together)
merged = ema_sub[['date','negative_affect_EMA']].merge(
    sensor_sub[['date', col]], on='date', how='inner')
merged = merged.dropna()  # dropna AFTER merge, on both columns
if len(merged) >= 10:
    from scipy import stats
    r, p = stats.pearsonr(merged['negative_affect_EMA'], merged[col])
    if abs(r) > 0.3:
        print(f"r={r:.2f}, n={len(merged)}")
```

**Self-report vs behavioral discrepancy**: When survey direction contradicts behavioral signal (e.g., perceived support falls but outgoing calls rise), name this explicitly.

### 4. Formulate and submit QA pairs

**Submit immediately after computing each modality** — don't wait until all analysis is done.

**One dimension per QA pair** — never combine two modalities or two metrics in one question.

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

Submit: `submit_qa_pair(q="...", a="...")`

**Good QA pair examples**:

Early/late change with count/duration dissociation:
> Q: "How did outgoing call frequency and duration change, and was there a count/duration dissociation?"
> A: "Outgoing calls increased from 1.24 to 3.67/day (+196%), but mean duration per call fell sharply from 305s to 58s (-81%) — a count/duration dissociation: more frequent but much shorter conversations in the late period."

Trajectory-only:
> Q: "What was the trajectory pattern of the user's location entropy across the three study periods?"
> A: "Location entropy showed a U-shaped pattern: 0.449 (T1) → 0.485 (T2) → 0.238 nats (T3). The sharp 51% drop in the final third indicates substantially reduced spatial diversity toward study end."

EMA spike analysis:
> Q: "Did the user's peak negative affect episodes coincide with specific behavioral events?"
> A: "The two highest EMA days (May 20: 5.0, May 24: 8.0) coincided with the two largest travel days (641 km and 4,428 km), suggesting travel-related stress. EMA-distance correlation was moderately positive (r=0.31)."

### 5. QA coverage checklist

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
- [ ] Number of significant places visited
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

- **Never split raw arrays without dropna** — always `dropna()` on the target column before early/late or thirds.
- **`summary_rapids_*` are period-wide summaries** — use `intraday_rapids_*` for daily activity variation; `summary_rapids_sumdurationasleepmain` is the exception (valid daily sleep duration).
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — infer from column names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score.
- **Pre/post column names use suffixes** `_PRE` / `_POST`, not generic names: `CESD_9items_PRE`, `UCLA_10items_PRE`, `2waySSS_receiving_emotional_PRE`, `BRS_PRE`, `CHIPS_PRE`, `STAIS_PRE`, `MAAS_7items_PRE`.
- **Pre/post scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better (good), higher PSS = more stress (bad), higher BRS = better resilience (good), lower CHIPS = lower impulsivity (good).
- **`barnett_homelabel` does not exist** — use `barnett_hometime`.
- **Distance outliers**: filter values >10× median before computing means.
- **Communication data is often sparse** (<30% valid days for some participants) — note sparsity in QA answer but still extract available patterns.
- **Cross-modal correlation**: merge on date first (inner join), then dropna together — never dropna each series independently before pearsonr.
- **Pre/post QA pairs must be split into three separate pairs**: psychological state, social support, and emotion regulation/coping.
- **Thirds pattern naming is mandatory** — always state whether progressive, U-shaped, inverted-U, peak-in-middle, stable-then-drop, etc.
- **One QA pair per dimension** — do not bundle "activity and Bluetooth" or "mobility and home time" in one question. Each checklist item deserves its own pair.
- **platform.csv uses `platform` column**, not `os`.
