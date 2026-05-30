###### Task 1:
# Context: This dataset contains records of BI (Business Intelligence) program students. Throughout this analysis, compute derived quantities such as sums, gaps, ratios, means, or correlations using unrounded values, and report all decimal-valued final results rounded to 4 decimal places
# Question: Load the student dataset and report the total number of rows and columns, the name and data type of each column, and the number of missing values per column. How many total missing values are there across the entire dataset?

df = pd.read_csv(f"{BASE_PATH}/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv", encoding='latin1')

shape_info = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
nulls_info = {col: int(v) for col, v in df.isnull().sum().items()}

result = {
    "answer_type": "dataset_overview",
    "shape": shape_info,
    "column_dtypes": dtypes_info,
    "missing_values": nulls_info,
    "total_missing": int(df.isnull().sum().sum())
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 2:
# Context: The dataset has missing values in the Python score column. These should be imputed with the column mean so that the overall distribution shape is preserved. We also need to verify there are no duplicate student records before proceeding with further analysis.
# Question: Fill all missing values in the Python score column with the column mean. Then check whether any duplicate rows exist. Report the mean value used for imputation, the number of duplicates found, and confirm the total missing values remaining across all columns.

python_mean = float(df['Python'].mean())
df['Python'] = df['Python'].fillna(python_mean)
num_duplicates = int(df.duplicated().sum())

result = {
    "answer_type": "scalar",
    "python_fill_mean": round(python_mean, 4),
    "num_duplicates": num_duplicates,
    "missing_values_after": {col: int(v) for col, v in df.isnull().sum().items()},
    "total_missing_after": int(df.isnull().sum().sum())
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 3:
# Question: Calculate the summary statistics (count, mean, standard deviation, min, 25th percentile, median, 75th percentile, max) for all numeric columns in the dataset after imputation. Report the full statistical summary for each numeric column.

desc = df.describe()
stats = {}
for col in desc.columns:
    stats[col] = {k: round(float(v), 4) for k, v in desc[col].items()}

result = {"answer_type": "distribution_summary", "statistics": stats}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 4:
# Question: Standardize all column names to PascalCase. Report the complete mapping from each original column name to its cleaned version.

original_cols = df.columns.tolist()

def clean_column(name):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', name)
    return ''.join([w.capitalize() for w in words])

df.columns = [clean_column(col) for col in df.columns]
cleaned_cols = df.columns.tolist()

mapping = [{"original": o, "cleaned": c} for o, c in zip(original_cols, cleaned_cols)]
print(json_dumps_rounded({"answer_type": "column_mapping", "mapping": mapping}, ensure_ascii=False, indent=2))

###### Task 5:
# Question: Standardize the country names Report the number of unique countries before and after cleaning, and the student count per country after cleaning sorted in descending order. Which are the top 4 countries by student count?

unique_before = int(df['Country'].nunique())

country_mapping = {
    'Rsa': 'South Africa',
    'Norge': 'Norway',
    'norway': 'Norway',
    'UK': 'United Kingdom',
    'Somali': 'Somalia'
}
df['Country'] = df['Country'].replace(country_mapping)

counts = df['Country'].value_counts().reset_index()
counts.columns = ['country', 'student_count']

result = {
    "answer_type": "ranking",
    "unique_before": unique_before,
    "unique_after": int(df['Country'].nunique()),
    "top4": counts.head(4).to_dict(orient='records'),
    "all_countries": counts.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 6:
# Question: Standardize the education level values. Report the number of unique levels before and after cleaning, and the student count per education level after cleaning, sorted in descending order.

unique_before = int(df['PrevEducation'].nunique())

education_mapping = {
    'HighSchool': 'High School',
    'Barrrchelors': 'Bachelors',
    'diploma': 'Diploma',
    'DIPLOMA': 'Diploma',
    'Diplomaaa': 'Diploma'
}
df['PrevEducation'] = df['PrevEducation'].replace(education_mapping)

counts = df['PrevEducation'].value_counts().reset_index()
counts.columns = ['education_level', 'student_count']

result = {
    "answer_type": "ranking",
    "unique_before": unique_before,
    "unique_after": int(df['PrevEducation'].nunique()),
    "education_counts": counts.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 7:
# Question: Standardize all gender values. Report the count of students for each gender after cleaning.

gender_mapping = {
    'F': 'Female',
    'female': 'Female',
    'M': 'Male',
    'male': 'Male'
}
df['Gender'] = df['Gender'].replace(gender_mapping)

counts = df['Gender'].value_counts().reset_index()
counts.columns = ['gender', 'student_count']

result = {
    "answer_type": "group_stat",
    "gender_counts": counts.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 8:
# Question: Standardize all residence values. Report the student count for each residence type after cleaning, sorted in descending order.

residence_mapping = {
    'BI-Residence': 'BI Residence',
    'BIResidence': 'BI Residence',
    'BI_Residence': 'BI Residence'
}
df['Residence'] = df['Residence'].replace(residence_mapping)

counts = df['Residence'].value_counts().reset_index()
counts.columns = ['residence', 'student_count']

result = {
    "answer_type": "group_stat",
    "residence_counts": counts.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 9:
# Question: Combine the first name and last name into a single full name column named `Name`, place it as the first column in the dataset, and remove the original separate name columns. Report the first 5 full names created and the complete list of column names in the final dataset.

df.insert(0, 'Name', df['FName'] + ' ' + df['LName'])
df = df.drop(columns=['FName', 'LName'])

result = {
    "answer_type": "scalar",
    "first_5_names": df['Name'].head(5).tolist(),
    "final_columns": df.columns.tolist(),
    "total_columns": len(df.columns)
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 10:
# Context: Outliers in the Age column should be detected using the Interquartile Range (IQR) method.
# Question: Apply the IQR method to detect outliers in student ages. Report Q1, Q3, IQR, the lower and upper bounds, the number of outliers found, and the outlier values with corresponding student names. Then replace the outliers with the column mean and report the age distribution statistics (mean, median, population standard deviation, min, max, Q1, Q3, IQR) after replacement.

Q1_age = float(df['Age'].quantile(0.25))
Q3_age = float(df['Age'].quantile(0.75))
IQR_age = Q3_age - Q1_age
lower_bound = Q1_age - 1.5 * IQR_age
upper_bound = Q3_age + 1.5 * IQR_age

outlier_mask = (df['Age'] < lower_bound) | (df['Age'] > upper_bound)
outliers_df = df[outlier_mask][['Name', 'Age']].copy()
outlier_records = []
for _, row in outliers_df.iterrows():
    outlier_records.append({"name": str(row['Name']), "age": int(row['Age'])})

df['Age'] = df['Age'].astype(float)
mean_age = float(df['Age'].mean())
df.loc[outlier_mask, 'Age'] = mean_age

s = df['Age']
result = {
    "answer_type": "outlier_detection",
    "Q1": round(Q1_age, 4),
    "Q3": round(Q3_age, 4),
    "IQR": round(IQR_age, 4),
    "lower_bound": round(lower_bound, 4),
    "upper_bound": round(upper_bound, 4),
    "num_outliers": int(outlier_mask.sum()),
    "outliers": outlier_records,
    "mean_used_for_replacement": round(mean_age, 4),
    "post_replacement_stats": {
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std(ddof=0)), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "q25": round(float(s.quantile(0.25)), 4),
        "q75": round(float(s.quantile(0.75)), 4),
        "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4)
    }
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 11:
# Question: Calculate the distribution statistics for the four academic performance variables: study hours, entry exam score, Python course score, and database course score. For each variable, report the mean, median, population standard deviation, min, max, 25th percentile, 75th percentile, and interquartile range.

result = {"answer_type": "distribution_summary", "metrics": {}}
for col in ['StudyHours', 'EntryExam', 'Python', 'Db']:
    s = df[col].dropna()
    result["metrics"][col] = {
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std(ddof=0)), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "q25": round(float(s.quantile(0.25)), 4),
        "q75": round(float(s.quantile(0.75)), 4),
        "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4)
    }
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 12:
# Question: For each country with at least 5 students, calculate the average age and the average entry exam score. Which country has the highest average age and which has the highest average entry exam score? Report all qualifying countries with both metrics, sorted by average entry exam score in descending order.

# --- re-derive: country student counts (from country standardization) ---
country_counts = df['Country'].value_counts()
qualifying_countries = country_counts[country_counts >= 5].index.tolist()

# --- re-derive: age is already outlier-treated in df (from outlier replacement) ---

# --- novel computation: country-level averages for qualifying countries ---
subset = df[df['Country'].isin(qualifying_countries)]
country_stats = (subset.groupby('Country')
                 .agg(avg_age=('Age', 'mean'),
                      avg_entry_exam=('EntryExam', 'mean'),
                      student_count=('Name', 'count'))
                 .round(4)
                 .sort_values('avg_entry_exam', ascending=False)
                 .reset_index())

highest_age_row = country_stats.loc[country_stats['avg_age'].idxmax()]
highest_exam_row = country_stats.iloc[0]

result = {
    "answer_type": "group_stat",
    "qualifying_countries": country_stats.to_dict(orient='records'),
    "highest_avg_age": {"country": str(highest_age_row['Country']),
                        "avg_age": round(float(highest_age_row['avg_age']), 4)},
    "highest_avg_entry_exam": {"country": str(highest_exam_row['Country']),
                               "avg_entry_exam": round(float(highest_exam_row['avg_entry_exam']), 4)}
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 13:
# Question: Compare entry exam scores between male and female students. Calculate the mean, median, and standard deviation for each gender. Which gender has a higher average entry exam score, and what is the absolute difference between the two averages?

gender_stats = (df.groupby('Gender')['EntryExam']
               .agg(['mean', 'median', 'std'])
               .round(4)
               .reset_index())
gender_stats.columns = ['gender', 'mean', 'median', 'std']

means = df.groupby('Gender')['EntryExam'].mean()
higher = str(means.idxmax())
diff = float(abs(means.iloc[0] - means.iloc[1]))

result = {
    "answer_type": "comparison",
    "gender_stats": gender_stats.to_dict(orient='records'),
    "higher_gender": higher,
    "diff_absolute": round(diff, 4)
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 14:
# Question: Calculate the average Python course score and average database course score for each country. Which are the top 3 and bottom 3 countries by average Python score? Report all countries with both average scores, sorted by average Python score in descending order.

country_scores = (df.groupby('Country')
                  .agg(avg_python=('Python', 'mean'),
                       avg_db=('Db', 'mean'))
                  .round(4)
                  .sort_values('avg_python', ascending=False)
                  .reset_index())

result = {
    "answer_type": "bidirectional_ranking",
    "primary_metric": "avg_python",
    "top3": country_scores.head(3).to_dict(orient='records'),
    "bottom3": country_scores.tail(3).to_dict(orient='records'),
    "all_countries": country_scores.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 15:
# Question: Among students belonging to the gender group with the higher average entry exam score, identify the top 6 countries by student count within that group. For each of these countries, report the number of students of that gender, their average Python score, and their average database score.

# --- re-derive: which gender has higher avg EntryExam (from gender comparison) ---
gender_means = df.groupby('Gender')['EntryExam'].mean()
higher_gender = str(gender_means.idxmax())

# --- re-derive: country distribution (from country standardization) ---
gender_subset = df[df['Gender'] == higher_gender]

# --- novel computation: top countries within higher-scoring gender ---
top_countries = gender_subset['Country'].value_counts().head(6).index.tolist()

details = (gender_subset[gender_subset['Country'].isin(top_countries)]
           .groupby('Country')
           .agg(student_count=('Name', 'count'),
                avg_python=('Python', 'mean'),
                avg_db=('Db', 'mean'))
           .round(4)
           .sort_values('student_count', ascending=False)
           .reset_index())

result = {
    "answer_type": "ranking",
    "higher_gender": higher_gender,
    "top_countries_details": details.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 16:
# Question: Calculate the average database course score for each education level. Which education level has the highest average score and which has the lowest? Report all education levels with their average scores, sorted in descending order.

edu_db = (df.groupby('PrevEducation')['Db']
          .mean()
          .round(4)
          .sort_values(ascending=False)
          .reset_index()
          .rename(columns={'PrevEducation': 'education_level', 'Db': 'avg_db_score'}))

result = {
    "answer_type": "ranking",
    "primary_metric": "avg_db_score",
    "highest": edu_db.iloc[0].to_dict(),
    "lowest": edu_db.iloc[-1].to_dict(),
    "all_levels": edu_db.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 17:
# Question: Create a cross-tabulation of education level and gender showing the count of students in each combination. Which education level has the most balanced gender ratio, and which has the most imbalanced? Report the female percentage for each education level.

ct = pd.crosstab(df['PrevEducation'], df['Gender'])
ct_records = ct.reset_index().to_dict(orient='records')

ct_analysis = ct.copy()
ct_analysis['total'] = ct_analysis.sum(axis=1)
ct_analysis['female_pct'] = (ct_analysis['Female'] / ct_analysis['total'] * 100).round(4)
ct_analysis['balance_score'] = (50 - (ct_analysis['female_pct'] - 50).abs()).round(4)

most_balanced = str(ct_analysis['balance_score'].idxmax())
most_imbalanced = str(ct_analysis['balance_score'].idxmin())

result = {
    "answer_type": "group_stat",
    "crosstab": ct_records,
    "most_balanced": {
        "education_level": most_balanced,
        "female_pct": float(ct_analysis.loc[most_balanced, 'female_pct'])
    },
    "most_imbalanced": {
        "education_level": most_imbalanced,
        "female_pct": float(ct_analysis.loc[most_imbalanced, 'female_pct'])
    }
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 18:
# Question: Among students whose age falls within the interquartile range and whose weekly study hours exceed the overall median, calculate the average entry exam score for each education level. How do these filtered averages compare to the unfiltered education-level averages? Report the difference for each education level and identify which level benefits most from this filtering.

# --- re-derive: IQR bounds for Age (from outlier detection) ---
age_q1 = df['Age'].quantile(0.25)
age_q3 = df['Age'].quantile(0.75)

# --- re-derive: StudyHours median (from distribution analysis) ---
study_median = df['StudyHours'].median()

# --- re-derive: unfiltered avg EntryExam by education (from education analysis) ---
unfiltered_avg = df.groupby('PrevEducation')['EntryExam'].mean()

# --- novel computation: multi-condition filtered comparison ---
filtered = df[(df['Age'] >= age_q1) & (df['Age'] <= age_q3) & (df['StudyHours'] > study_median)]
filtered_avg = filtered.groupby('PrevEducation')['EntryExam'].mean()
filtered_count = filtered.groupby('PrevEducation').size()

comparison = pd.DataFrame({
    'unfiltered_avg': unfiltered_avg,
    'filtered_avg': filtered_avg,
    'filtered_count': filtered_count
}).reindex(unfiltered_avg.index)
comparison['filtered_count'] = comparison['filtered_count'].fillna(0).astype(int)
comparison['difference'] = comparison['filtered_avg'] - comparison['unfiltered_avg']

comparison = comparison.round(4).reset_index()
comparison = comparison.rename(columns={'PrevEducation': 'education_level'})
comparison = comparison.sort_values(
    ['difference', 'education_level'],
    ascending=[False, True],
    na_position='last'
).reset_index(drop=True)

benefit_candidates = comparison.dropna(subset=['difference'])
most_benefit = None if benefit_candidates.empty else benefit_candidates.iloc[0].to_dict()

result = {
    "answer_type": "comparison",
    "filter_criteria": {
        "age_q1": round(float(age_q1), 4),
        "age_q3": round(float(age_q3), 4),
        "study_hours_median": round(float(study_median), 4)
    },
    "total_filtered_students": int(len(filtered)),
    "comparison": json.loads(comparison.to_json(orient='records')),
    "most_benefit": None if most_benefit is None else json.loads(pd.Series(most_benefit).to_json())
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))


###### Task 19:
# Context: We now move to multivariate analysis to understand the relationships between all numeric features. A Pearson correlation matrix reveals the linear associations between age, entry exam score, study hours, Python score, and database score.
# Question: Compute the Pearson correlation matrix for all five numeric features. What are the 3 strongest correlated pairs and the 3 weakest correlated pairs by absolute value? Report the full matrix and the ranked pairs.

numeric_cols = ['Age', 'EntryExam', 'StudyHours', 'Python', 'Db']
corr_matrix = df[numeric_cols].corr().round(4)

pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        pairs.append({
            "col_a": numeric_cols[i],
            "col_b": numeric_cols[j],
            "correlation": round(float(corr_matrix.iloc[i, j]), 4)
        })

pairs_desc = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)
pairs_asc = sorted(pairs, key=lambda x: abs(x['correlation']))

result = {
    "answer_type": "correlation_matrix",
    "matrix": {col: {c: round(float(corr_matrix.loc[col, c]), 4)
                     for c in numeric_cols} for col in numeric_cols},
    "top3_strongest": pairs_desc[:3],
    "top3_weakest": pairs_asc[:3]
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))


###### Task 20:
# Question: For countries with a student count above the median country size, compute the Pearson correlation between Python and database scores within that subset. Compare this to the full-dataset correlation between the same two variables. Is the relationship stronger or weaker among students from larger-population countries? Report the qualifying countries and both correlation values.

# --- re-derive: country student counts (from country standardization) ---
country_counts = df['Country'].value_counts()
median_count = float(country_counts.median())
large_countries = country_counts[country_counts > median_count].index.tolist()

# --- re-derive: full Python-Db correlation (from correlation matrix) ---
full_corr = float(df['Python'].corr(df['Db']))

# --- novel computation: subset correlation for large countries ---
subset = df[df['Country'].isin(large_countries)]
subset_corr = float(subset['Python'].corr(subset['Db']))

result = {
    "answer_type": "correlation",
    "median_country_size": round(median_count, 4),
    "large_countries": large_countries,
    "num_students_in_subset": int(len(subset)),
    "full_dataset_correlation": round(full_corr, 4),
    "subset_correlation": round(subset_corr, 4),
    "difference": round(subset_corr - full_corr, 4),
    "stronger_in_subset": bool(abs(subset_corr) > abs(full_corr))
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 21:
# Question: Using the linear relationship between study hours and entry exam scores, predict the expected entry exam score for the average study hours of each education level. Compare these predicted scores to the actual average entry exam scores per education level. Which education level outperforms its study-hours-based prediction the most (largest positive residual)? Report the regression slope, intercept, and the full comparison table.

# --- re-derive: StudyHours-EntryExam relationship (from correlation analysis) ---
valid = df[['StudyHours', 'EntryExam']].dropna()
coeffs = np.polyfit(valid['StudyHours'].values, valid['EntryExam'].values, 1)

# --- re-derive: education-level grouping (from education analysis) ---
edu_actual = df.groupby('PrevEducation').agg(
    avg_study_hours=('StudyHours', 'mean'),
    actual_avg_exam=('EntryExam', 'mean')
).reset_index()

# --- novel computation: predicted vs actual by education level ---
edu_actual['predicted_exam'] = np.poly1d(coeffs)(edu_actual['avg_study_hours'])
edu_actual['residual'] = edu_actual['actual_avg_exam'] - edu_actual['predicted_exam']
edu_actual = edu_actual.sort_values('residual', ascending=False).round(4)

best = edu_actual.iloc[0]

result = {
    "answer_type": "regression",
    "model": "linear (StudyHours -> EntryExam)",
    "slope": round(float(coeffs[0]), 4),
    "intercept": round(float(coeffs[1]), 4),
    "education_comparison": edu_actual.rename(
        columns={'PrevEducation': 'education_level'}).to_dict(orient='records'),
    "largest_positive_residual": {
        "education_level": str(best['PrevEducation']),
        "residual": round(float(best['residual']), 4)
    }
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 22:
# Question: For each residence type, calculate the average entry exam score separately for male and female students, then compute the gender gap (male average minus female average). Which residence type shows the largest absolute gender gap in entry exam performance? Report the full breakdown for all residence types.

# --- re-derive: residence types (from residence standardization) ---
# --- re-derive: gender-based comparison approach (from gender analysis) ---

# --- novel computation: gender gap by residence ---
pivot = (df.groupby(['Residence', 'Gender'])['EntryExam']
         .mean()
         .unstack(fill_value=0)
         .round(4))
pivot['gender_gap'] = (pivot['Male'] - pivot['Female']).round(4)
pivot['abs_gap'] = pivot['gender_gap'].abs()
pivot = pivot.sort_values('abs_gap', ascending=False).reset_index()

largest = pivot.iloc[0]

result = {
    "answer_type": "comparison",
    "residence_gender_gaps": pivot[['Residence', 'Female', 'Male', 'gender_gap']].to_dict(orient='records'),
    "largest_absolute_gap": {
        "residence": str(largest['Residence']),
        "male_avg": round(float(largest['Male']), 4),
        "female_avg": round(float(largest['Female']), 4),
        "gap": round(float(largest['gender_gap']), 4)
    }
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 23:
# Question: Define "academically strong" countries as those where both the average entry exam score and the average Python score exceed their respective overall country-level medians. How many countries qualify as academically strong? List them in country-name ascending order with their average entry exam, Python, and database scores. What is the total number of students from these countries, and what is their gender distribution?

# --- re-derive: country-level averages (from country score ranking) ---
country_avgs = df.groupby('Country').agg(
    avg_exam=('EntryExam', 'mean'),
    avg_python=('Python', 'mean'),
    avg_db=('Db', 'mean')
)

# --- re-derive: gender analysis approach (from gender comparison) ---

# --- novel computation: multi-threshold country classification ---
median_exam = float(country_avgs['avg_exam'].median())
median_python = float(country_avgs['avg_python'].median())

strong = country_avgs[
    (country_avgs['avg_exam'] > median_exam) &
    (country_avgs['avg_python'] > median_python)
].round(4).reset_index()

strong_countries = strong['Country'].tolist()
students_in_strong = df[df['Country'].isin(strong_countries)]
gender_split = students_in_strong['Gender'].value_counts().to_dict()

result = {
    "answer_type": "ranking",
    "thresholds": {
        "median_exam": round(median_exam, 4),
        "median_python": round(median_python, 4)
    },
    "num_qualifying_countries": len(strong),
    "qualifying_countries": strong.to_dict(orient='records'),
    "total_students": int(len(students_in_strong)),
    "gender_split": {str(k): int(v) for k, v in gender_split.items()}
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 24:
# Context: For predictive modeling, we use the cleaned dataset to predict entry exam scores. All categorical features (excluding the student name) are label-encoded to convert them to numeric form. The data is split into 80% training and 20% test sets using random_state=42. Five regression models are trained and compared: Linear Regression, Random Forest, Gradient Boosting, SVR, and XGBoost.
# Question: Train the five regression models described above to predict entry exam scores. Report each model's R-squared percentage, MAE, and RMSE on the test set in the same order the models are listed in the context. Which model achieves the best R-squared?

df_model = df.copy()
cat_cols_model = df_model.select_dtypes(exclude=['number']).columns.tolist()
num_cols_model = df_model.select_dtypes(include=['number']).columns.tolist()

df_model[num_cols_model] = df_model[num_cols_model].fillna(df_model[num_cols_model].median())
for col in cat_cols_model:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

le = LabelEncoder()
for col in cat_cols_model:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop(columns=['Name', 'EntryExam'])
y = df_model['EntryExam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(random_state=42)
}

results_list = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred) * 100)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    results_list.append({
        "model": name,
        "r2_pct": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4)
    })

best = max(results_list, key=lambda x: x['r2_pct'])

result = {
    "answer_type": "ranking",
    "primary_metric": "r2_pct",
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "num_features": int(X.shape[1]),
    "feature_names": X.columns.tolist(),
    "model_results": results_list,
    "best_model": best
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 25:
# Question: Select only the top 3 features most strongly correlated with entry exam scores (by absolute Pearson correlation from the encoded feature set) and retrain the best-performing model using just those features. Compare the R-squared, MAE, and RMSE to the full-feature model. Does reducing to the top 3 correlated features significantly impact predictive performance?

# --- re-derive: feature correlations with EntryExam (from correlation analysis) ---
feature_cols = X.columns.tolist()
exam_corrs = {col: abs(float(df_model[col].corr(df_model['EntryExam'])))
              for col in feature_cols}
top3_features = sorted(exam_corrs, key=exam_corrs.get, reverse=True)[:3]

# --- re-derive: best model identity (from model comparison) ---
model_r2 = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    model_r2[name] = float(r2_score(y_test, y_pred))
best_model_name = max(model_r2, key=model_r2.get)

# --- novel computation: retrain with top 3 features ---
X_train_top3 = X_train[top3_features]
X_test_top3 = X_test[top3_features]

model_class = models[best_model_name].__class__
if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    reduced_model = model_class(random_state=42)
else:
    reduced_model = model_class()

reduced_model.fit(X_train_top3, y_train)
y_pred_reduced = reduced_model.predict(X_test_top3)

full_r2 = round(model_r2[best_model_name] * 100, 4)
reduced_r2 = round(float(r2_score(y_test, y_pred_reduced)) * 100, 4)
reduced_mae = round(float(mean_absolute_error(y_test, y_pred_reduced)), 4)
reduced_rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred_reduced))), 4)

result = {
    "answer_type": "comparison",
    "best_model": best_model_name,
    "top3_features": [{"feature": f, "abs_correlation": round(exam_corrs[f], 4)}
                      for f in top3_features],
    "full_model_r2_pct": full_r2,
    "reduced_model_r2_pct": reduced_r2,
    "reduced_model_mae": reduced_mae,
    "reduced_model_rmse": reduced_rmse,
    "r2_change_pct": round(reduced_r2 - full_r2, 4),
    "significant_impact": abs(reduced_r2 - full_r2) > 5
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 26:
# Question: Compute a composite academic z-score for each student by standardizing (zero mean, unit variance) the entry exam, Python, and database scores, then averaging the three z-scores per student. Who are the top 5 students by this composite score? Report their names, countries, education levels, individual z-scores, and composite scores.

# --- re-derive: score distributions (from descriptive statistics) ---
scaler = StandardScaler()
z_scores = pd.DataFrame(
    scaler.fit_transform(df[['EntryExam', 'Python', 'Db']]),
    columns=['z_EntryExam', 'z_Python', 'z_Db'],
    index=df.index
)

# --- novel computation: composite z-score ranking ---
z_scores['composite'] = z_scores[['z_EntryExam', 'z_Python', 'z_Db']].mean(axis=1)
df_ranked = pd.concat([df[['Name', 'Country', 'PrevEducation']], z_scores], axis=1)
top5 = (df_ranked.sort_values('composite', ascending=False)
        .head(5)
        [['Name', 'Country', 'PrevEducation', 'z_EntryExam', 'z_Python', 'z_Db', 'composite']]
        .round(4))

result = {
    "answer_type": "ranking",
    "primary_metric": "composite_z_score",
    "top5": top5.to_dict(orient='records')
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

###### Task 27:
# Question: Compare the feature importance rankings from the best-performing tree-based model with the univariate absolute Pearson correlation rankings between each feature and entry exam scores. For each feature, report its model-based importance score and its absolute correlation with entry exam scores. Do the two ranking methods agree on the most important feature? Calculate Spearman's rank correlation between the two rankings to quantify agreement.

# --- re-derive: Pearson correlations with EntryExam (from correlation analysis) ---
feature_cols = X.columns.tolist()
pearson_corrs = {col: round(abs(float(df_model[col].corr(df_model['EntryExam']))), 4)
                 for col in feature_cols}

# --- re-derive: best tree-based model (from model comparison) ---
tree_models = {name: model for name, model in models.items()
               if hasattr(model, 'feature_importances_')}
best_tree_name = max(tree_models,
                     key=lambda n: float(r2_score(y_test, tree_models[n].predict(X_test))))
best_tree = tree_models[best_tree_name]

# --- novel computation: importance vs correlation comparison ---
importances = dict(zip(feature_cols,
                       [round(float(v), 4) for v in best_tree.feature_importances_]))

comp_df = pd.DataFrame({
    'feature': feature_cols,
    'model_importance': [importances[f] for f in feature_cols],
    'abs_pearson_corr': [pearson_corrs[f] for f in feature_cols]
})
comp_df['importance_rank'] = comp_df['model_importance'].rank(ascending=False).astype(int)
comp_df['correlation_rank'] = comp_df['abs_pearson_corr'].rank(ascending=False).astype(int)
comp_df = comp_df.sort_values('model_importance', ascending=False)

spearman_val, p_val = spearmanr(
    comp_df['importance_rank'], comp_df['correlation_rank'])

top_imp_feature = comp_df.iloc[0]['feature']
top_corr_feature = comp_df.sort_values('abs_pearson_corr', ascending=False).iloc[0]['feature']

result = {
    "answer_type": "comparison",
    "best_tree_model": best_tree_name,
    "feature_comparison": comp_df.round(4).to_dict(orient='records'),
    "top_by_importance": str(top_imp_feature),
    "top_by_correlation": str(top_corr_feature),
    "rankings_agree_on_top": str(top_imp_feature) == str(top_corr_feature),
    "spearman_rank_correlation": round(float(spearman_val), 4),
    "spearman_p_value": round(float(p_val), 4)
}
print(json_dumps_rounded(result, ensure_ascii=False, indent=2))

