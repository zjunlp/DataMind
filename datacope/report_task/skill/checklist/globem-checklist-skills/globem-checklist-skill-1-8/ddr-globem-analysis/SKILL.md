---
name: ddr-globem-analysis
description: >
  Analyze a specific participant's longitudinal passive-sensing and psychological data in the GLOBEM digital depression research dataset. Use this skill whenever the task involves: analyzing a user's mental health or behavioral data from wearables/smartphones, generating QA pairs about behavioral/psychological changes over time, working with EMA, depression scores, activity, sleep, communication, location, or phone-usage data, or any user-profile analysis in the DDR/GLOBEM context.
---

# DDR GLOBEM Participant Analysis

## Task Overview

Analyze all available data for a specified participant (`pid`, e.g., `INS-W_011`) and submit QA pairs covering their behavioral and psychological changes across the observation period. Aim for **28–37 high-quality, distinct QA pairs**. Each QA pair covers ONE specific dimension — never bundle multiple modalities or metrics into a single pair.

**Critical workflow rule**: Always compute all numbers BEFORE calling `submit_qa_pair()`. Never submit a QA pair with a placeholder, partial, or incomplete answer — doing so produces an unfixable record. Compute first, construct the full answer string, then submit.

**Submit incrementally**: Submit each QA pair immediately after computing that modality's data. Do not batch all submissions at the end.

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
- `dep_weekly.csv` — weekly `feel_anxious`, `feel_depressed`, `dep_weekly_subscale`, `anx_weekly_subscale`, `BDI2` (endterm only), `dep`
- `pre.csv` / `post.csv` — baseline vs. end-of-study psychological scales (columns use `_PRE` / `_POST` suffixes)
- `dep_endterm.csv` — final depression label and BDI2 score

**Schema tip**: Use `get_field_description(data_file="<filename>")` for the six sensor CSV files.

**Known column pitfalls**:
- `platform.csv` uses column `platform`, not `os`
- Home time: use `barnett_hometime` (not `barnett_homelabel`)
- Sleep duration: use `summary_rapids_sumdurationasleepmain` for daily sleep duration; bedtime/waketime use `summary_rapids_firstbedtimemain` / `summary_rapids_lastwaketimemain`
- `summary_rapids_*` columns repeat period-wide values — use `intraday_rapids_*` for daily activity variation
- Phone usage location columns: suffixes `_locmap_home`, `_locmap_living`, `_locmap_study`, `_locmap_greens`, `_locmap_exercise`
- Proactivity ratio with near-zero incoming: use `max(incoming, 0.1)` to avoid artifacts; a ratio >10 is likely a near-zero denominator — report raw counts instead
- Pre/post columns use exact names like `CESD_9items_PRE`, `UCLA_10items_PRE`, `2waySSS_receiving_emotional_PRE`, `BRS_PRE`, `CHIPS_PRE`, `STAIS_PRE`, `MAAS_7items_PRE`, `2waySSS_social_fit_PRE`

## Starter Code — Define Once, Reuse Throughout

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

### 2. Temporal segmentation — apply to ALL modalities including derived metrics

Always `dropna()` on the target column before splitting. The `early_late_thirds()` function handles this automatically.

**Early/late split** (primary comparison unit): first vs second half of valid observations.

**Thirds segmentation — mandatory for every quantitative column**, including derived ones:
- Ratios: active/sedentary ratio, proactivity ratio, scan efficiency (scans-per-device)
- Timing: first-use timing, avg flight duration, avg speed
- Counts: active bouts, sedentary bouts, network diversity, significant places
- Scores: sleep efficiency, circadian routine

Compute T1/T2/T3 and name the pattern for every such metric. Thirds reveal non-linear patterns (U-shaped, inverted-U, progressive, stable) that early/late splits hide.

**Thirds NOT required for** (early/late only is sufficient):
- EMA spike analysis (per-date comparisons, not aggregated thirds)
- Sleep timing in HH:MM (shift direction is the key insight)
- Communication and phone first-use timing windows (window shift, not thirds)
- Cross-modal correlations (no temporal split)
- Weekday vs. weekend pairs (no temporal split)

**Weekday vs weekend** — compute for every modality with ≥10 weekday and ≥5 weekend valid days. Check at minimum: steps, home time, distance, phone unlocks, sleep duration, Bluetooth scan count, active duration. When difference >15%, create a **dedicated QA pair for that specific modality** (e.g., "How did the user's step count differ between weekdays and weekends?"). Do NOT bundle all modalities into one catch-all QA pair.

### 3. Compute sub-dimension insights per modality

**Sleep timing phase shift**: Convert bedtime/wake-time from minutes-from-midnight to HH:MM using `mins_to_hhmm()`. Report shift direction and magnitude in minutes.

**Active/sedentary ratio**: Compute `active_duration / sedentary_duration` for early vs late, plus T1/T2/T3 and named pattern. Also report avg sedentary bout duration with T1/T2/T3.

**Communication count vs duration dissociation**: If outgoing call count increases but mean duration decreases (or vice versa), name this explicitly. Compute `mean_duration_per_call = sumduration / count`. Apply same check to incoming calls.

**Proactivity ratio**: `outgoing_count / max(incoming_count, 0.1)` early vs late + T1/T2/T3 + named pattern. Report raw counts if ratio >10.

**Communication timing window**: From `rapids_outgoing_timefirstcall` and `rapids_outgoing_timelastcall` (minutes from midnight → HH:MM), report: (1) first call time shift, (2) last call time shift, (3) window duration = last - first, and whether the window expanded or contracted. This is early/late only, no thirds needed.

**Phone session variability**: Compare `rapids_stddurationunlock` and `rapids_maxdurationunlock` early vs late. Increasing variability indicates more extreme phone sessions.

**Phone first-use timing**: Compute `rapids_firstuseafter00unlock` early/late plus T1/T2/T3 with named pattern. Earlier values = more disrupted sleep/wake.

**Phone by location context**: Compare home vs study vs other contexts early vs late using `_locmap_home`, `_locmap_study` columns.

**Top-location time distribution**: Extract `doryab_timeattop1location`, `doryab_timeattop2location`, `doryab_timeattop3location` early vs late. Report % change and concentration shift.

**Movement characteristics**: Extract `barnett_avgflightdur` (seconds) early vs late plus T1/T2/T3. Also compute `barnett_stdflightdur` for variability. Shorter avg flight = shift to local movement.

**Average speed**: `doryab_avgspeed` (km/hr) early vs late plus T1/T2/T3 and named pattern. Report as transport mode shift when applicable.

**EMA spike analysis** — compute participant means for ALL 4 behavioral metrics FIRST, then identify spike dates:
```python
# Pre-compute participant means BEFORE writing the QA answer
ema_sub = pd.read_csv("ema.csv"); ema_sub = ema_sub[ema_sub['pid'] == pid]
loc_sub = pd.read_csv("location_allday_raw.csv"); loc_sub = loc_sub[loc_sub['pid'] == pid]
phone_sub = pd.read_csv("phone_usage_allday_raw.csv"); phone_sub = phone_sub[phone_sub['pid'] == pid]
sleep_sub = pd.read_csv("sleep_allday_raw.csv"); sleep_sub = sleep_sub[sleep_sub['pid'] == pid]

mean_dist = loc_sub['barnett_totalpath'].dropna().mean()  # participant avg distance
mean_home = loc_sub['barnett_hometime'].dropna().mean()   # participant avg home time
mean_sleep = sleep_sub['summary_rapids_sumdurationasleepmain'].dropna().mean()  # avg sleep
mean_unlocks = phone_sub['rapids_countuniqueunlocksepisodes'].dropna().mean()   # avg unlocks

top3 = ema_sub.nlargest(3, 'negative_affect_EMA')
for _, row in top3.iterrows():
    date = row['date']
    # Look up each metric for this date, compare to participant mean
```

For each spike date, include in the answer: `actual_value vs participant_avg, deviation%` for ALL 4 metrics. Conclude with the common behavioral pattern across all 3 spikes (e.g., "Social isolation strongly correlates with peak negative affect episodes").

**Bluetooth scan efficiency**: `scans / unique_devices` early vs late plus T1/T2/T3 with named pattern. Rising ratio = fewer devices each scanned more repeatedly.

**Cross-modal correlations**: Merge EMA with each behavioral modality on date (inner join), then dropna on both columns together, compute Pearson r. Require ≥10 shared valid days. For every |r| > 0.3 found, create a **dedicated QA pair**: "What is the relationship between the user's [EMA metric] and [behavioral metric]?" If two closely related columns yield similar correlations (e.g., phone unlock count and phone total duration both ~r=-0.5), report only the one with higher |r| — avoid near-duplicate cross-modal pairs.

```python
# Correct cross-modal correlation pattern
merged = ema_sub[['date','negative_affect_EMA']].merge(
    sensor_sub[['date', col]], on='date', how='inner')
merged = merged.dropna()
if len(merged) >= 10:
    from scipy import stats
    r, p = stats.pearsonr(merged['negative_affect_EMA'], merged[col])
    if abs(r) > 0.3:
        print(f"r={r:.2f}, n={len(merged)}")
```

**Self-report vs behavioral discrepancy**: When survey scores contradict behavioral signals (e.g., perceived support falls but outgoing calls rise, or weekly depression remains low while mobility declines substantially), name this explicitly as a separate QA pair.

### 4. Formulate and submit QA pairs

**Mandatory answer format** — every answer must include ALL of the following that apply:
1. **Early and late values** with units (e.g., "9.3 (early) to 10.4 (late)")
2. **% change and direction** (e.g., "+12.0%")
3. **T1/T2/T3 values** when applicable (all quantitative modality QAs)
4. **Named pattern** when thirds are computed (inverted-U, U-shaped, progressive increase, progressive decline, mixed/stable)
5. **Brief interpretation** — what the numbers suggest behaviorally or clinically

**Preferred compact answer format**: `"X.XX (early) to X.XX (late), +Y.Y%. T1=a, T2=b, T3=c — pattern_name. [Interpretation sentence.]"` Keep answers information-dense. Avoid restating the question or padding with phrases like "This indicates that the trajectory showed..."; instead state conclusions directly.

If you realize you are missing a needed number while writing the answer, compute it before calling `submit_qa_pair()`.

**One dimension per QA pair** — never combine two modalities or two metrics in one question:
- ✗ "How did the user's home time AND location entropy change?" → split
- ✗ "How did the user's proactivity ratio AND network diversity change?" → split
- ✗ "Were there weekday vs weekend differences?" (all modalities at once) → one dedicated pair per modality with >15% difference

**Prioritize checklist coverage** — aim to generate QA pairs for every checklist item before adding novel sub-metrics. If approaching 37 pairs, skip niche additions rather than skip a checklist item.

**Avoid metric duplication** — when barnett and doryab cover the same behavioral concept (e.g., radius of gyration, home time, distance), use the barnett or checklist-specified column. Do not generate separate QA pairs for doryab-specific equivalents (e.g., `doryab_totaltime`, `doryab_radiusofgyration`, `barnett_maxdiam`) when a checklist column already covers that concept.

Valid QA pair types:
- **Early/late change**: "How did X change between early and late periods?"
- **Trajectory-only**: "What was the trajectory pattern of X across the three study periods?" (when the thirds pattern is more informative than early/late)
- **Sub-dimension**: "Was there a count/duration dissociation in X?"
- **Cross-modal**: "What is the relationship between the user's X and Y?"
- **Meta-pattern**: "Is there a discrepancy between self-reported X and behavioral Y?"

**Good QA pair examples**:

Early/late change with dissociation:
> Q: "How did outgoing call frequency and duration change, and was there a count/duration dissociation?"
> A: "Outgoing calls increased from 1.24 to 3.67/day (+196%), with a U-shaped pattern: T1=1.57, T2=1.43, T3=4.13. Mean duration per call fell sharply from 305s to 58s (-81%) — count/duration dissociation: more frequent but much shorter conversations in the late period."

Trajectory with thirds (mandatory named pattern):
> Q: "What was the trajectory pattern of the user's location entropy across the three study periods?"
> A: "Location entropy showed a U-shaped pattern: 0.449 (T1) → 0.485 (T2) → 0.238 nats (T3). Early mean 0.467, late mean 0.362 (-22.5%). The sharp 51% drop in the final third indicates substantially reduced spatial diversity toward study end."

EMA spike analysis (include absolute value, participant avg, % deviation for ALL metrics):
> Q: "Did the user's peak negative affect episodes coincide with specific behavioral events?"
> A: "Highest EMA day (June 10: 9.0) showed minimal travel (1.12 km vs 39.59 km avg, -97%), elevated home time (1424 min vs 1029 avg, +38%), above-average sleep (406 min vs 425 avg, -5%), reduced phone use (20 vs 54 avg, -63%). Second peak (April 22: 8.0)... Third peak (April 29: 8.0)... Social isolation appears most strongly linked to peak negative affect."

Weekday vs weekend (separate pair per modality):
> Q: "How did the user's step count differ between weekdays and weekends?"
> A: "Step count showed a substantial weekday/weekend difference: weekday average 9,899 steps/day vs weekend 5,803 steps/day (-41.4%), indicating significantly lower physical activity on weekends."

Cross-modal:
> Q: "What is the relationship between the user's daily step count and time spent at home?"
> A: "Steps and home time were strongly negatively correlated (r=-0.56, p<0.001, n=60 days). Higher step counts coincided with less home time, indicating more active days involved greater time away from home."

Submit: `submit_qa_pair(q="...", a="...")`

### 5. QA coverage checklist

Aim to cover all of these (skip only if data is entirely NaN):

**Psychological — assessment**
- [ ] EMA negative affect: early/late change + named thirds pattern
- [ ] EMA spike analysis: top 3 specific high-affect dates with multi-dimensional behavioral comparison (distance, home time, sleep, phone use) vs participant average — include `actual_value vs participant_avg, deviation%` for each metric; conclude with common pattern across spikes
- [ ] Weekly depression: `feel_depressed` trajectory (early/late + named thirds) + final BDI2 score
- [ ] Weekly anxiety: `feel_anxious` trajectory (early/late + named thirds) — separate pair from depression
- [ ] Weekly depression subscale: `dep_weekly_subscale` early/late + named thirds — report as continuous **mean value**, not as binary endorsement count or percentage; separate pair
- [ ] Weekly anxiety subscale: `anx_weekly_subscale` early/late + named thirds — report as continuous mean value; separate pair
- [ ] Pre/post: psychological state (depression CESD, anxiety STAIS, stress PSS, loneliness UCLA — all in one pair)
- [ ] Pre/post: social support (all 4 dimensions + social fit `2waySSS_social_fit` — all in one pair)
- [ ] Pre/post: emotion regulation / coping / resilience / mindfulness (ERQ reappraisal, ERQ suppression, BRS, CHIPS, MAAS — all in one pair)

**Activity**
- [ ] Steps: early/late + named thirds pattern
- [ ] Active/sedentary duration ratio (early/late + T1/T2/T3 + named pattern) + avg sedentary bout duration (early/late + T1/T2/T3 + named pattern) — combine in one pair
- [ ] Active bout count: early/late + named thirds — separate pair
- [ ] Sedentary bout count: early/late + named thirds — separate pair

**Sleep**
- [ ] Sleep duration: early/late + named thirds pattern
- [ ] Sleep efficiency: early/late + named thirds — separate from duration
- [ ] Sleep timing phase shift: bedtime and wake-time in HH:MM, shift direction and magnitude in minutes

**Communication**
- [ ] Outgoing call count + mean duration: early/late, note dissociation if count/duration diverge
- [ ] Incoming call count + mean duration per call: early/late, note dissociation
- [ ] Missed call trend: early/late + named thirds
- [ ] Proactivity ratio: early/late + T1/T2/T3 + named pattern (use raw counts if ratio >10)
- [ ] Network diversity (distinct contacts): early/late + T1/T2/T3 + named pattern
- [ ] Communication timing window: first/last outgoing call in HH:MM + window duration (last − first); report shift direction and whether window expanded or contracted

**Location**
- [ ] Mobility: distance traveled early/late + named thirds
- [ ] Radius of gyration: early/late + named thirds
- [ ] Home time: early/late + named thirds
- [ ] Circadian routine: early/late + named thirds
- [ ] Location entropy: early/late + named thirds
- [ ] Location transitions: early/late + named thirds — separate pair from entropy
- [ ] Top-location time distribution (top-1, top-2, top-3) shift
- [ ] Movement characteristics: avg flight duration (early/late + T1/T2/T3) + std flight duration
- [ ] Average speed: early/late + T1/T2/T3 + named pattern
- [ ] Significant places visited: early/late + named thirds

**Phone usage**
- [ ] Unlock count + total duration: early/late + named thirds, note dissociation if count/duration diverge
- [ ] Session length shift (avg duration) + variability (std, max): early/late
- [ ] First-use timing: early/late + T1/T2/T3 + named pattern
- [ ] Phone by location context (home vs study): early vs late

**Connectivity**
- [ ] Scan count + unique devices: early/late + named thirds (combine in one pair)
- [ ] Bluetooth scan efficiency (scans-per-device ratio): early/late + T1/T2/T3 + named pattern

**Cross-modal & meta**
- [ ] Cross-modal correlation: one dedicated QA pair for EACH distinct |r| > 0.3 found (≥10 shared days); skip near-duplicate correlations between closely related columns
- [ ] Weekday vs weekend: one dedicated QA pair per modality with >15% difference (check at minimum: steps, home time, distance, phone unlocks, sleep duration, Bluetooth, active duration)
- [ ] Self-report vs behavioral discrepancy (if survey direction contradicts behavioral signal)

## Common Pitfalls

- **Submit only complete answers** — never call `submit_qa_pair()` with "let me compute" or a partial placeholder. Compute all values, form the full answer, then submit.
- **dep_weekly_subscale and anx_weekly_subscale are reported as continuous means** — these may be 0/1 binary flags, but always report their mean value (e.g., "0.12") with T1/T2/T3. Never report as endorsement counts (e.g., "1/7 = 14.3%").
- **Thirds for derived metrics are mandatory** — ratios (active/sedentary, proactivity, scan efficiency), timing (first-use, avg flight duration, avg speed), and counts (active bouts, network diversity, significant places) all require T1/T2/T3 and named pattern, not just early/late.
- **Named pattern is always required** when thirds are computed — state inverted-U, U-shaped, progressive increase, progressive decline, or mixed/stable explicitly.
- **Never split raw arrays without dropna** — always `dropna()` on the target column before early/late or thirds.
- **EMA spike analysis requires participant means** — compute the participant's personal mean for distance, home time, sleep, and phone unlocks BEFORE describing each spike date. Report as `"actual_value vs participant_avg, deviation%"` for each metric.
- **`summary_rapids_*` are period-wide summaries** — use `intraday_rapids_*` for daily activity variation; `summary_rapids_sumdurationasleepmain` is the exception (valid daily sleep duration).
- **`get_field_description` won't work for `ema.csv` or `dep_weekly.csv`** — infer from column names.
- **BDI2 appears in `dep_weekly.csv` only in the final row** — it's the endterm score.
- **Pre/post column names use suffixes** `_PRE` / `_POST`: `CESD_9items_PRE`, `UCLA_10items_PRE`, `2waySSS_receiving_emotional_PRE`, `BRS_PRE`, `CHIPS_PRE`, `STAIS_PRE`, `MAAS_7items_PRE`, `2waySSS_social_fit_PRE`.
- **Pre/post scale directions vary**: higher UCLA = more loneliness (bad), higher ERQ_reappraisal = better (good), higher PSS = more stress (bad), higher BRS = better resilience (good), lower CHIPS = lower impulsivity (good).
- **`barnett_homelabel` does not exist** — use `barnett_hometime`.
- **Distance outliers**: filter values >10× median before computing means.
- **Communication data is often sparse** — note sparsity in QA answer but still extract available patterns.
- **Cross-modal correlation**: merge on date first (inner join), then dropna together — never dropna each series independently before pearsonr.
- **No doryab duplicates** — do not generate separate QA pairs for `doryab_totaltime`, `doryab_radiusofgyration`, or `barnett_maxdiam` when barnett or intraday equivalents are already in the checklist. Checklist columns take priority.
- **platform.csv uses `platform` column**, not `os`.
- **Pre/post QA pairs must be split into exactly three pairs**: psychological state, social support, and emotion regulation/coping.
- **Communication timing window answer must include window duration** — compute `window = last_call_time - first_call_time` and report whether it expanded or contracted, in addition to the absolute HH:MM times.
