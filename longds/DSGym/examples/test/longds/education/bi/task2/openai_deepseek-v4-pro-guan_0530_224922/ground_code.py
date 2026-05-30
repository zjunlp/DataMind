###### Task 1:
# Context: Throughout this analysis, compute derived quantities such as sums, gaps, ratios, means, or correlations using unrounded values, and report all decimal-valued final results rounded to 4 decimal places. Before any cleaning, inspect the raw data to quantify the problems that downstream analyses must handle: - missing values in score columns, - inconsistent casing / spelling of categorical values.
# Question: Load the raw data and, for the raw untouched data, report (a) the shape, (b) the count of missing values per column, and (c) the full set of unique values observed for each of the four categorical columns: gender, country, residence, and prevEducation. Return the categorical uniques sorted alphabetically so the output is deterministic.

# T1: load raw data and record baseline inspection. No cleaning applied yet.
raw = pd.read_csv(f'{BASE_PATH}/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv',
                  encoding='latin1')

missing_per_column = {k: int(v) for k, v in raw.isnull().sum().items()}
uniques_before_clean = {
    col: sorted(raw[col].dropna().astype(str).unique().tolist())
    for col in ['gender', 'country', 'residence', 'prevEducation']
}

result = {
    "answer_type": "raw_inspection",
    "shape": list(raw.shape),
    "missing_per_column": missing_per_column,
    "unique_values_before_clean": uniques_before_clean,
}
print_result(result)


###### Task 2:
# Context: Now build the canonical category mappings that the rest of the benchmark will rely on. These standardization rules are FIXED and any later task that references "standardized categories" uses the mappings you establish here. Rules: - gender: map to 'Female' and 'Male'. - country: Standardized country names. - residence: collapse BI-Residence,BIResidence,BI_Residence to BI Residence;  - prevEducation: map to High School, Bachelor, Diploma, Masters, Doctorate
# Question: After applying the standardization rules above, report the set of distinct values and their counts for each of the four standardized categorical columns (gender, country, residence, prevEducation). Also report the total number of distinct values per column.

# T2: apply canonical category mappings. Produces `cleaned` (still un-imputed).
cleaned = raw.copy()

# gender: lowercase + strip, then map
cleaned['gender'] = cleaned['gender'].str.strip().str.lower().replace(
    {'f': 'Female', 'female': 'Female', 'm': 'Male', 'male': 'Male'})

# country: title-case + strip, then collapse
cleaned['country'] = cleaned['country'].str.strip().str.title().replace(
    {'Norge': 'Norway', 'Rsa': 'RSA', 'South Africa': 'RSA', 'Uk': 'UK'})

# residence: collapse BI Residence variants
cleaned['residence'] = cleaned['residence'].replace(
    {'BI-Residence': 'BI Residence',
     'BIResidence': 'BI Residence',
     'BI_Residence': 'BI Residence'})

# prevEducation: collapse education variants
cleaned['prevEducation'] = cleaned['prevEducation'].replace(
    {'HighSchool': 'High School',
     'Bachelors': 'Bachelor', 'Barrrchelors': 'Bachelor',
     'diploma': 'Diploma', 'DIPLOMA': 'Diploma', 'Diplomaaa': 'Diploma'})

cat_counts = {
    col: {k: int(v) for k, v in cleaned[col].value_counts().sort_index().items()}
    for col in ['gender', 'country', 'residence', 'prevEducation']
}
n_distinct = {col: int(cleaned[col].nunique()) for col in cat_counts}

result = {
    "answer_type": "standardized_counts",
    "counts_per_column": cat_counts,
    "n_distinct_per_column": n_distinct,
}
print_result(result)


###### Task 3:
# Context: The only remaining data-quality issue after standardization is the missing values in the Python score column. Impute them with the column mean computed on the standardized data. The resulting data becomes the baseline cleaned sample for every the analysis.
# Question: Fill the missing Python scores with the mean of the non-null Python scores. Report the mean value used for imputation, the number of rows that were imputed, and the final row count of the baseline cleaned sample.

# T3: create `baseline_sample` by imputing Python missing values with column mean.
baseline_sample = cleaned.copy()

python_mean_for_impute = float(baseline_sample['Python'].mean())
rows_imputed = int(baseline_sample['Python'].isnull().sum())
baseline_sample['Python'] = baseline_sample['Python'].fillna(python_mean_for_impute)

result = {
    "answer_type": "imputation_report",
    "mean_used_for_imputation": python_mean_for_impute,
    "rows_imputed": rows_imputed,
    "baseline_sample_size": int(len(baseline_sample)),
}
print_result(result)


###### Task 4:
# Question: For the numeric columns Age, studyHOURS, Python, and DB within the baseline cleaned sample, compute mean, median, population standard deviation, min, max, q25, and q75. Report all four columns' stats side by side.

# T4: distribution summary on baseline_sample (from T3).
NUM_COLS = ['Age', 'studyHOURS', 'Python', 'DB']

def _summary(s):
    return {
        "mean":   float(s.mean()),
        "median": float(s.median()),
        "std":    float(s.std(ddof=0)),
        "min":    float(s.min()),
        "max":    float(s.max()),
        "q25":    float(s.quantile(0.25)),
        "q75":    float(s.quantile(0.75)),
    }

stats_per_column = {c: _summary(baseline_sample[c]) for c in NUM_COLS}

result = {
    "answer_type": "distribution_summary",
    "columns": NUM_COLS,
    "stats_per_column": stats_per_column,
}
print_result(result)


###### Task 5:
# Question: Compute the Pearson correlation matrix over Age, studyHOURS, Python, and DB. Report the full 4x4 matrix and identify both the strongest and weakest off-diagonal pairs by absolute correlation.

# T5: correlation matrix on baseline_sample.
corr_matrix = baseline_sample[NUM_COLS].corr()

pairs = pd.DataFrame([
    {"a": a, "b": b, "r": float(corr_matrix.loc[a, b])}
    for i, a in enumerate(NUM_COLS)
    for b in NUM_COLS[i + 1:]
])
pairs['abs_r'] = pairs['r'].abs()
pairs = pairs.sort_values('abs_r', ascending=False).reset_index(drop=True)

result = {
    "answer_type": "correlation_matrix",
    "matrix": {c: {c2: float(corr_matrix.loc[c, c2]) for c2 in NUM_COLS}
               for c in NUM_COLS},
    "strongest_pair": {
        "a": pairs.iloc[0]['a'], "b": pairs.iloc[0]['b'],
        "r": float(pairs.iloc[0]['r']),
    },
    "weakest_pair": {
        "a": pairs.iloc[-1]['a'], "b": pairs.iloc[-1]['b'],
        "r": float(pairs.iloc[-1]['r']),
    },
}
print_result(result)


###### Task 6:
# Question: Calculate the Pearson correlation coefficient between study hours and Python scores within the current sample. Report the value.

# T6: single-point anchor — correlation of studyHOURS and Python.
corr_studyhours_python = float(
    baseline_sample['studyHOURS'].corr(baseline_sample['Python']))

result = {
    "answer_type": "correlation",
    "pair": ["studyHOURS", "Python"],
    "value": corr_studyhours_python,
}
print_result(result)


###### Task 7:
# Question: Compute the mean Python score within each previous-education level. Return the per-level means sorted in descending order, and identify the education level with the highest mean Python score.

# T7: mean Python by prevEducation; remember the leading level as `edu_leader`.
mean_python_by_edu = (baseline_sample.groupby('prevEducation')['Python']
                      .mean().sort_values(ascending=False))
edu_leader = str(mean_python_by_edu.idxmax())
edu_leader_value = float(mean_python_by_edu.max())

result = {
    "answer_type": "ranking",
    "primary_metric": "mean_python",
    "records": [{"name": k, "value": float(v)}
                for k, v in mean_python_by_edu.items()],
    "leader": edu_leader,
    "leader_value": edu_leader_value,
}
print_result(result)


###### Task 8:
# Question: Compare the two genders on Python scores by both mean and median. Return both statistics per gender, the absolute gap in means, and the gender with the higher median.

# T8: gender-level means and medians; store `gender_leader_median`.
_gender_group = baseline_sample.groupby('gender')['Python']
gender_mean   = _gender_group.mean()
gender_median = _gender_group.median()

gender_leader_median = str(gender_median.idxmax())

mean_gap = abs(float(gender_mean.iloc[0]) - float(gender_mean.iloc[1]))

result = {
    "answer_type": "gender_comparison",
    "mean":   {k: float(v) for k, v in gender_mean.items()},
    "median": {k: float(v) for k, v in gender_median.items()},
    "mean_gap_absolute": mean_gap,
    "median_leader": gender_leader_median,
    "median_leader_value": float(gender_median.max()),
}
print_result(result)


###### Task 9:
# Question: For each residence category, compute the descriptive statistics of the entry exam score (count, mean, sample standard deviation, min, 25th percentile, median, 75th percentile, and max). Return the full table and identify the residence category with the highest median entry exam score.

# T9: entryEXAM descriptive per residence on baseline_sample.
entry_exam_desc = baseline_sample.groupby('residence')['entryEXAM'].describe()
entry_exam_medians = baseline_sample.groupby('residence')['entryEXAM'].median()
residence_median_leader = str(entry_exam_medians.idxmax())

records = []
for res, row in entry_exam_desc.iterrows():
    record = {"residence": res}
    for k, v in row.items():
        record[k] = int(v) if k == "count" else float(v)
    records.append(record)

result = {
    "answer_type": "group_descriptive",
    "metric": "entryEXAM",
    "records": records,
    "median_leader": residence_median_leader,
    "median_leader_value": float(entry_exam_medians.max()),
}
print_result(result)


###### Task 10:
# Question: Restrict attention to learners whose previous-education level is the one with the highest mean Python score identified earlier, AND whose gender is the one with the higher median Python score identified earlier. For this doubly-filtered subset, report the count of learners and their mean Python score.

# T10: [multi-hop DD] depends on:
#   - baseline_sample (T3)
#   - edu_leader                (T7 — highest mean Python education)
#   - gender_leader_median      (T8 — gender with higher median Python)
subset_t10 = baseline_sample[
    (baseline_sample['prevEducation'] == edu_leader) &
    (baseline_sample['gender']        == gender_leader_median)
]
subset_t10_count       = int(len(subset_t10))
subset_t10_mean_python = (float(subset_t10['Python'].mean())
                          if subset_t10_count > 0 else None)

result = {
    "answer_type": "conditional_subset",
    "condition": {"prevEducation": edu_leader, "gender": gender_leader_median},
    "count": subset_t10_count,
    "mean_python": subset_t10_mean_python,
}
print_result(result)


###### Task 11:
# Context: Introduce the first age-binning scheme. Treat this scheme as the active age grouping from this task onward, unless a subsequent task explicitly redefines it. Introduce a five-tier, right-inclusive age-grouping scheme—18–24, 25–34, 35–44, 45–54, and 55+—that becomes the active grouping from this task onward unless explicitly redefined later.
# Question: Create the age-group column per the binning scheme above, then compute mean Python score within each age group. Return the per-group means and identify the age group with the highest mean Python score.

# T11: define age_group_v1 on baseline_sample. This scheme stays active
# until T23 explicitly replaces it.
AGE_BINS_V1   = [17, 24, 34, 44, 54, 100]
AGE_LABELS_V1 = ['18-24', '25-34', '35-44', '45-54', '55+']
baseline_sample['age_group_v1'] = pd.cut(
    baseline_sample['Age'], bins=AGE_BINS_V1, labels=AGE_LABELS_V1)

mean_python_by_age_v1 = baseline_sample.groupby('age_group_v1', observed=True)['Python'].mean()
age_v1_python_leader = str(mean_python_by_age_v1.idxmax())

result = {
    "answer_type": "group_stat",
    "grouping": "age_group_v1",
    "metric": "mean_python",
    "records": [{"name": str(k), "value": float(v)}
                for k, v in mean_python_by_age_v1.items()],
    "leader": age_v1_python_leader,
    "leader_value": float(mean_python_by_age_v1.max()),
}
print_result(result)


###### Task 12:
# Context: Introduce a composite performance score defined as the simple average of each learner's Python score and DB score. Treat this equal-weight composite as the currently active composite score from this task onward.
# Question: Using the simple average of the two course scores as the composite, rank learners by this composite within the baseline cleaned sample and return the top 5 with first name, last name, Python score, DB score, and the composite score. If learners tie on the composite score, break ties by their original row order in the baseline cleaned sample.

# T12: introduce final_score_v1 (active composite for the next several tasks).
baseline_sample['final_score_v1'] = (baseline_sample['Python']
                                     + baseline_sample['DB']) / 2

top5_v1 = (baseline_sample
           .sort_values('final_score_v1', ascending=False, kind='mergesort')
           .head(5).copy())

top5_v1_records_df = top5_v1[['fNAME', 'lNAME', 'Python', 'DB', 'final_score_v1']].copy()
top5_v1_records_df[['Python', 'DB', 'final_score_v1']] = top5_v1_records_df[
    ['Python', 'DB', 'final_score_v1']].astype(float)
top5_v1_records = top5_v1_records_df.to_dict(orient='records')

result = {
    "answer_type": "ranking",
    "formula": "final_score_v1 = (Python + DB) / 2",
    "top5": top5_v1_records,
}
print_result(result)


###### Task 13:
# Question: Compute the Pearson correlation of the currently active composite score with each of four factors: study hours, the entry exam score, age, and the natural logarithm of one plus each learner's age. Report all four correlations, plus the factor with the largest absolute correlation.

# T13: correlations of final_score_v1 (still active) with four factors.
# log_age is computed here and kept on baseline_sample so later tasks may reuse it.
baseline_sample['log_age'] = np.log(baseline_sample['Age'] + 1)

corrs_v1 = {
    'studyHOURS': float(baseline_sample['final_score_v1'].corr(baseline_sample['studyHOURS'])),
    'entryEXAM':  float(baseline_sample['final_score_v1'].corr(baseline_sample['entryEXAM'])),
    'Age':        float(baseline_sample['final_score_v1'].corr(baseline_sample['Age'])),
    'log_age':    float(baseline_sample['final_score_v1'].corr(baseline_sample['log_age'])),
}
strongest_factor_v1 = max(corrs_v1, key=lambda k: abs(corrs_v1[k]))

result = {
    "answer_type": "correlations",
    "target": "final_score_v1",
    "values": corrs_v1,
    "strongest_factor": strongest_factor_v1,
}
print_result(result)


###### Task 14:
# Question: For the current age grouping, compute the mean of the active composite score within each age bin. Report the per-bin means and the leading age bin.

# T14: [multi-hop DD] depends on:
#   - age_group_v1   (T11 — active age binning)
#   - final_score_v1 (T12 — active composite)
mean_fs_v1_by_age_v1 = (baseline_sample
                        .groupby('age_group_v1', observed=True)['final_score_v1']
                        .mean())
age_v1_fs_leader = str(mean_fs_v1_by_age_v1.idxmax())

result = {
    "answer_type": "group_stat",
    "grouping": "age_group_v1",
    "metric": "mean_final_score_v1",
    "records": [{"name": str(k), "value": float(v)}
                for k, v in mean_fs_v1_by_age_v1.items()],
    "leader": age_v1_fs_leader,
    "leader_value": float(mean_fs_v1_by_age_v1.max()),
}
print_result(result)


###### Task 15:
# Question: Compute the mean active composite score by previous-education level. Return the ranking and identify the education level with the highest mean composite score.

# T15: group active composite by prevEducation (still the v1 formula here).
mean_fs_v1_by_edu = (baseline_sample.groupby('prevEducation')['final_score_v1']
                     .mean().sort_values(ascending=False))
edu_fs_v1_leader = str(mean_fs_v1_by_edu.idxmax())

result = {
    "answer_type": "ranking",
    "primary_metric": "mean_final_score_v1",
    "records": [{"name": k, "value": float(v)}
                for k, v in mean_fs_v1_by_edu.items()],
    "leader": edu_fs_v1_leader,
    "leader_value": float(mean_fs_v1_by_edu.max()),
}
print_result(result)


###### Task 16:
# Context: Introduce an engineered feature defined as the product of a learner's age and their study hours. This feature remains available for later questions that reference it without restating the formula.
# Question: Add the age-study interaction feature as a new column on the baseline cleaned sample, then report its Pearson correlation with the currently active composite score.

# T16: define age_study_interaction (persistent feature).
baseline_sample['age_study_interaction'] = (baseline_sample['Age']
                                            * baseline_sample['studyHOURS'])

corr_interaction_fs_v1 = float(
    baseline_sample['age_study_interaction'].corr(baseline_sample['final_score_v1']))

result = {
    "answer_type": "correlation",
    "pair": ["age_study_interaction", "final_score_v1"],
    "value": corr_interaction_fs_v1,
}
print_result(result)


###### Task 17:
# Question: For the learners who form the top-5 ranking by the active composite score, report the mean of the age-study interaction feature. Compare it with the overall mean of the same feature across the full baseline sample, and report the absolute difference.

# T17: [multi-hop DD] depends on:
#   - top5_v1                   (T12 — top 5 by final_score_v1)
#   - age_study_interaction     (T16)
# Reuses the stored DataFrame `top5_v1`; its index maps back to baseline_sample rows.
avg_interaction_top5 = float(
    baseline_sample.loc[top5_v1.index, 'age_study_interaction'].mean())
avg_interaction_full = float(
    baseline_sample['age_study_interaction'].mean())

difference_interaction = abs(avg_interaction_top5 - avg_interaction_full)

result = {
    "answer_type": "comparison",
    "metric": "age_study_interaction",
    "top5_mean": avg_interaction_top5,
    "full_sample_mean": avg_interaction_full,
    "difference_absolute": difference_interaction,
    "top5_higher": bool(avg_interaction_top5 > avg_interaction_full),
}
print_result(result)


###### Task 18:
# Context: From this task onward, the composite score changes to a weighted combination of the Python and DB scores, with the Python score weighted 0.6 and the DB score weighted 0.4. The earlier equal-weight version stays available, but every subsequent question that mentions "the currently active composite score" should use this new weighted form.
# Question: Apply the new weighting to every learner in the baseline cleaned sample and recompute the top-5 ranking under the new active composite. Return first name, last name, Python score, DB score, and the new composite score for the new top 5.

# T18: UPDATE — switch active composite to weighted form.
# `final_score_v1` is preserved on baseline_sample for future explicit backtracks.
baseline_sample['final_score_v2'] = (0.6 * baseline_sample['Python']
                                     + 0.4 * baseline_sample['DB'])

top5_v2 = (baseline_sample
           .sort_values('final_score_v2', ascending=False, kind='mergesort')
           .head(5).copy())

top5_v2_records_df = top5_v2[['fNAME', 'lNAME', 'Python', 'DB', 'final_score_v2']].copy()
top5_v2_records_df[['Python', 'DB', 'final_score_v2']] = top5_v2_records_df[
    ['Python', 'DB', 'final_score_v2']].astype(float)
top5_v2_records = top5_v2_records_df.to_dict(orient='records')

result = {
    "answer_type": "ranking",
    "formula": "final_score_v2 = 0.6 * Python + 0.4 * DB",
    "top5": top5_v2_records,
}
print_result(result)


###### Task 19:
# Question: Using the currently active composite score, recompute its Pearson correlations with study hours, the entry exam score, age, and the natural logarithm of one plus each learner's age. Return all four correlations and identify which factor's correlation with the composite score shifted the most in absolute terms compared with the earlier correlation analysis done under the equal-weight formulation.

# T19: [multi-hop] depends on:
#   - corrs_v1         (T13 — correlations under v1)
#   - final_score_v2   (T18 — new active composite)
corrs_v2 = {
    'studyHOURS': float(baseline_sample['final_score_v2'].corr(baseline_sample['studyHOURS'])),
    'entryEXAM':  float(baseline_sample['final_score_v2'].corr(baseline_sample['entryEXAM'])),
    'Age':        float(baseline_sample['final_score_v2'].corr(baseline_sample['Age'])),
    'log_age':    float(baseline_sample['final_score_v2'].corr(baseline_sample['log_age'])),
}
shifts = {k: abs(corrs_v2[k] - corrs_v1[k]) for k in corrs_v2}
largest_shift_factor = max(shifts, key=lambda k: shifts[k])

result = {
    "answer_type": "correlations_with_comparison",
    "target_now": "final_score_v2",
    "values_now": corrs_v2,
    "absolute_shift_vs_v1": shifts,
    "largest_shift_factor": largest_shift_factor,
    "largest_shift_value":  shifts[largest_shift_factor],
}
print_result(result)


###### Task 20:
# Question: Compare the earlier top-5 ranking with the current top-5 ranking. Report the number of learners appearing in both lists, the set of learners who entered the new top 5, and the set of learners who dropped out.

# T20: [PERTURBATION compare] depends on:
#   - top5_v1  (T12 — top 5 under final_score_v1)
#   - top5_v2  (T18 — top 5 under final_score_v2)
names_v1 = {(r['fNAME'], r['lNAME']) for r in top5_v1[['fNAME','lNAME']].to_dict(orient='records')}
names_v2 = {(r['fNAME'], r['lNAME']) for r in top5_v2[['fNAME','lNAME']].to_dict(orient='records')}

overlap = names_v1 & names_v2
entered = names_v2 - names_v1
exited  = names_v1 - names_v2

def _fmt(pairs):
    return [f"{fn} {ln}" for fn, ln in sorted(pairs)]

result = {
    "answer_type": "top5_comparison",
    "overlap_count": len(overlap),
    "overlap_members": _fmt(overlap),
    "entered_new_top5": _fmt(entered),
    "exited_from_top5": _fmt(exited),
}
print_result(result)


###### Task 21:
# Question: Using the active weighted composite and the currently active age-binning scheme, compute the mean composite score within each age band. Return the per-band means and identify whether the leading age band is the same as the one that led when the equal-weight composite was used.

# T21: [multi-hop] depends on:
#   - age_group_v1           (T11 — still the active binning)
#   - final_score_v2         (T18 — active composite)
#   - age_v1_fs_leader       (T14 — previous leading band under v1)
mean_fs_v2_by_age_v1 = (baseline_sample
                        .groupby('age_group_v1', observed=True)['final_score_v2']
                        .mean())
current_leader = str(mean_fs_v2_by_age_v1.idxmax())

result = {
    "answer_type": "group_stat_with_comparison",
    "grouping": "age_group_v1",
    "metric": "mean_final_score_v2",
    "records": [{"name": str(k), "value": float(v)}
                for k, v in mean_fs_v2_by_age_v1.items()],
    "current_leader": current_leader,
    "previous_leader_under_v1": age_v1_fs_leader,
    "leader_unchanged": bool(current_leader == age_v1_fs_leader),
}
print_result(result)


###### Task 22:
# Question: Probe the robustness of the top-1 learner against the choice of composite weights. Besides the active weighted formula, evaluate two alternative schemes: 70% Python with 30% DB, and 50% Python with 50% DB. For each of the three schemes report the top-1 learner's full name and their composite score, plus a boolean indicating whether the same learner tops the ranking under every scheme.

# T22: [PERTURBATION robustness] depends on:
#   - top5_v2  (T18) — first row = top-1 under 0.6/0.4
SCHEMES = {
    "0.6*Python + 0.4*DB (active)": (0.6, 0.4),
    "0.7*Python + 0.3*DB":          (0.7, 0.3),
    "0.5*Python + 0.5*DB":          (0.5, 0.5),
}

top1_per_scheme = {}
for label, (w_py, w_db) in SCHEMES.items():
    series = w_py * baseline_sample['Python'] + w_db * baseline_sample['DB']
    idx_top = series.idxmax()
    top_row = baseline_sample.loc[idx_top]
    top1_per_scheme[label] = {
        "name":  f"{top_row['fNAME']} {top_row['lNAME']}",
        "value": float(series.loc[idx_top]),
    }

unique_names = {v["name"] for v in top1_per_scheme.values()}
same_top1 = len(unique_names) == 1

result = {
    "answer_type": "weight_perturbation",
    "top1_per_scheme": top1_per_scheme,
    "same_top1_across_schemes": bool(same_top1),
}
print_result(result)


###### Task 23:
# Context: From this task onward, the age-binning scheme switches to a new four-band version: The age grouping is consolidated from five tiers into four. The youngest band extends its upper boundary by one year, and the next band starts immediately after it and runs through age 35. The two former middle-age bands are merged into a single 36–50 bracket, Treat this four-band binning as the currently active age grouping from this point forward.
# Question: Apply the four-band age binning to every learner in the baseline cleaned sample. Using the currently active composite score, compute the mean composite score within each of the new bands. Return per-band means and the leading band.

# T23: UPDATE — redefine active age binning as `age_group_v2`.
AGE_BINS_V2   = [17, 25, 35, 50, 100]
AGE_LABELS_V2 = ['<=25', '26-35', '36-50', '51+']
baseline_sample['age_group_v2'] = pd.cut(
    baseline_sample['Age'], bins=AGE_BINS_V2, labels=AGE_LABELS_V2)

mean_fs_v2_by_age_v2 = (baseline_sample
                        .groupby('age_group_v2', observed=True)['final_score_v2']
                        .mean())
age_v2_fs_leader = str(mean_fs_v2_by_age_v2.idxmax())

result = {
    "answer_type": "group_stat",
    "grouping": "age_group_v2",
    "metric": "mean_final_score_v2",
    "records": [{"name": str(k), "value": float(v)}
                for k, v in mean_fs_v2_by_age_v2.items()],
    "leader": age_v2_fs_leader,
    "leader_value": float(mean_fs_v2_by_age_v2.max()),
}
print_result(result)


###### Task 24:
# Question: Compute the population standard deviation of the active composite score within each age band. Report all bands' std values and the band with the highest variability.

# T24: [multi-hop] depends on:
#   - age_group_v2     (T23 — active binning)
#   - final_score_v2   (T18 — active composite)
std_fs_v2_by_age_v2 = (baseline_sample
                       .groupby('age_group_v2', observed=True)['final_score_v2']
                       .std(ddof=0))
high_std_band = str(std_fs_v2_by_age_v2.idxmax())

result = {
    "answer_type": "group_stat",
    "grouping": "age_group_v2",
    "metric": "std_final_score_v2",
    "records": [{"name": str(k), "value": float(v)}
                for k, v in std_fs_v2_by_age_v2.items()],
    "high_std_band": high_std_band,
    "high_std_value": float(std_fs_v2_by_age_v2.max()),
}
print_result(result)


###### Task 25:
# Question: Assess the sensitivity of the age-band ranking to the choice of central tendency. Redo the per-band aggregation using the median of the active composite score (instead of the mean), and compare the induced ranking of bands with the mean-based ranking. Report both rankings, identify whether the leading bin changes, and return a boolean indicating whether the two orderings are identical.

# T25: [PERTURBATION aggregation-swap] depends on:
#   - age_group_v2         (T23)
#   - final_score_v2       (T18)
#   - mean_fs_v2_by_age_v2 (T23 — mean per bin)
median_fs_v2_by_age_v2 = (baseline_sample
                          .groupby('age_group_v2', observed=True)['final_score_v2']
                          .median())

mean_order   = list(mean_fs_v2_by_age_v2.sort_values(ascending=False).index)
median_order = list(median_fs_v2_by_age_v2.sort_values(ascending=False).index)

mean_leader_band   = str(mean_order[0])
median_leader_band = str(median_order[0])
orderings_identical = [str(x) for x in mean_order] == [str(x) for x in median_order]

result = {
    "answer_type": "aggregation_perturbation",
    "mean_ranking":   [str(x) for x in mean_order],
    "median_ranking": [str(x) for x in median_order],
    "mean_leader":   mean_leader_band,
    "median_leader": median_leader_band,
    "leader_unchanged":    bool(mean_leader_band == median_leader_band),
    "orderings_identical": bool(orderings_identical),
    "median_values": {str(k): float(v)
                      for k, v in median_fs_v2_by_age_v2.items()},
}
print_result(result)


###### Task 26:
# Question: Within the age band that has the highest variability in the active composite score, break the learners down by gender. Report the count of learners and the mean active composite score per gender in that band.

# T26: [multi-hop DD] depends on:
#   - high_std_band      (T24 — band with highest std)
#   - baseline_sample    (T3, now carrying age_group_v2 + final_score_v2)
subset_high_std = baseline_sample[baseline_sample['age_group_v2'] == high_std_band]

gender_breakdown = (subset_high_std.groupby('gender')
                    .agg(count=('fNAME', 'count'),
                         mean_final_score_v2=('final_score_v2', 'mean')))

records = [{"gender": g,
            "count": int(row['count']),
            "mean_final_score_v2": float(row['mean_final_score_v2'])}
           for g, row in gender_breakdown.iterrows()]

result = {
    "answer_type": "gender_breakdown_within_band",
    "band": high_std_band,
    "records": records,
    "band_size": int(len(subset_high_std)),
}
print_result(result)


###### Task 27:
# Context: The sample base is restricted to learners whose study hours are at least 140.
# Question: Form the restricted sample by filtering learners with at least 140 study hours and report (a) the sample size of this restricted sample and (b) the mean of the currently active composite score within the restricted sample.

# T27: UPDATE — sample base becomes sample_v2 (studyHOURS >= 140).
sample_v2 = baseline_sample[baseline_sample['studyHOURS'] >= 140].copy()

sample_v2_size = int(len(sample_v2))
sample_v2_mean_fs = float(sample_v2['final_score_v2'].mean())

result = {
    "answer_type": "sample_update",
    "rule": "studyHOURS >= 140",
    "sample_v2_size": sample_v2_size,
    "sample_v2_mean_final_score_v2": sample_v2_mean_fs,
}
print_result(result)


###### Task 28:
# Question: Within the currently active restricted sample, compute the Pearson correlation between the currently active composite score and each of four factors: study hours, the entry exam score, age, and the age-study interaction feature introduced earlier. Report the four correlations and rank them by absolute value.

# T28: [multi-hop] depends on:
#   - sample_v2                (T27)
#   - final_score_v2           (T18, present on baseline_sample and carried to sample_v2)
#   - age_study_interaction    (T16)
corrs_sv2 = {
    'studyHOURS':            float(sample_v2['final_score_v2'].corr(sample_v2['studyHOURS'])),
    'entryEXAM':             float(sample_v2['final_score_v2'].corr(sample_v2['entryEXAM'])),
    'Age':                   float(sample_v2['final_score_v2'].corr(sample_v2['Age'])),
    'age_study_interaction': float(sample_v2['final_score_v2'].corr(sample_v2['age_study_interaction'])),
}
ranked_factors = sorted(corrs_sv2.items(), key=lambda kv: abs(kv[1]), reverse=True)

result = {
    "answer_type": "factor_ranking_within_sample_v2",
    "correlations": corrs_sv2,
    "ranking_by_abs_r": [{"factor": f, "value": v} for f, v in ranked_factors],
    "strongest_factor": ranked_factors[0][0],
}
print_result(result)


###### Task 29:
# Question: Within the currently active restricted sample and using the currently active age binning, compute the mean of the currently active composite score in each band. Compare the leading band observed here against the leading band obtained earlier on the full cleaned sample under the same active definitions. Return all per-band means on the restricted sample, both leading bands, and a boolean indicating whether they coincide.

# T29: [multi-hop DD] depends on:
#   - sample_v2         (T27)
#   - age_group_v2      (T23, present on baseline_sample and carried to sample_v2)
#   - final_score_v2    (T18)
#   - age_v2_fs_leader  (T23 — leading band on full baseline_sample under v2 defs)
mean_fs_v2_age_v2_on_sv2 = (sample_v2
                            .groupby('age_group_v2', observed=True)['final_score_v2']
                            .mean())
sv2_leader = str(mean_fs_v2_age_v2_on_sv2.idxmax())

result = {
    "answer_type": "leader_comparison",
    "metric": "mean_final_score_v2",
    "grouping": "age_group_v2",
    "records_on_sample_v2": [{"name": str(k), "value": float(v)}
                             for k, v in mean_fs_v2_age_v2_on_sv2.items()],
    "leader_on_sample_v2": sv2_leader,
    "leader_on_full_baseline": age_v2_fs_leader,
    "leaders_match": bool(sv2_leader == age_v2_fs_leader),
}
print_result(result)


###### Task 30:
# Context: Apply the following rules here: - the composite score uses the equal-weight form introduced earlier, not the currently active weighted form; - the age grouping uses the earlier five-band scheme, not the currently active four-band scheme; - analyses use the full cleaned sample, ignoring the study-hours restriction imposed earlier.
# Question: Equal-weight composite, five-band age binning, full cleaned sample, compute the mean composite score within each age band. Return the per-band means and the leading band. Finally, compare this leading band against the leading band under the currently active definitions on the restricted sample and report whether they are the same band name.

# T30: [EXPLICIT BACKTRACK + multi-hop] depends on:
#   - baseline_sample     (T3 — full cleaned sample, ignore sample_v2)
#   - age_group_v1        (T11 — rolled back binning)
#   - final_score_v1      (T12 — rolled back composite, preserved on baseline_sample)
#   - sv2_leader          (T29 — currently active leader on sample_v2 under v2 defs)
rollback_means = (baseline_sample
                  .groupby('age_group_v1', observed=True)['final_score_v1']
                  .mean())
rollback_leader = str(rollback_means.idxmax())

result = {
    "answer_type": "explicit_backtrack_comparison",
    "rollback_definitions": {
        "composite": "final_score_v1 = (Python + DB) / 2",
        "age_binning": "age_group_v1 (5 bands)",
        "sample_base": "full baseline_sample",
    },
    "rollback_records": [{"name": str(k), "value": float(v)}
                         for k, v in rollback_means.items()],
    "rollback_leader": rollback_leader,
    "current_active_leader_on_sample_v2": sv2_leader,
    "same_leader_name": bool(rollback_leader == sv2_leader),
}
print_result(result)


