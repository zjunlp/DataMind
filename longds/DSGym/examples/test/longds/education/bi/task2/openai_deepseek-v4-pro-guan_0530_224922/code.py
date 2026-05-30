############## turn 1
import os

data_dir = '/data/longds/education/bi/task2/data'
files = os.listdir(data_dir)
files

import os
data_path = '/data/longds/education/bi/task2/data/intro-to-data-cleaning-eda-and-machine-learning'
with open(data_path, 'rb') as f:
    header = f.read(200)
header

subdir = '/data/longds/education/bi/task2/data/intro-to-data-cleaning-eda-and-machine-learning'
os.listdir(subdir)

import pandas as pd

df = pd.read_csv('/data/longds/education/bi/task2/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv')
df.shape

with open('/data/longds/education/bi/task2/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv', 'rb') as f:
    raw_bytes = f.read(1000)
raw_bytes[:100]

import pandas as pd

df = pd.read_csv('/data/longds/education/bi/task2/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv', encoding='latin1')
df.head()

print('Shape:', df.shape)
print('\nMissing values per column:')
df.isnull().sum()

cols = ['gender', 'country', 'residence', 'prevEducation']
for col in cols:
    uniq = sorted(df[col].astype(str).unique())
    print(f'{col}:')
    for v in uniq:
        print(f'  {repr(v)}')
    print()

############## turn 2
import pandas as pd

# Load data again if needed (should be available from previous step)
# Standardization mappings

# gender mapping
gender_map = {
    'F': 'Female',
    'Female': 'Female',
    'female': 'Female',
    'M': 'Male',
    'Male': 'Male',
    'male': 'Male'
}

# country mapping (inferring standard names)
country_map = {
    'Denmark': 'Denmark',
    'France': 'France',
    'Germany': 'Germany',
    'Italy': 'Italy',
    'Kenya': 'Kenya',
    'Netherlands': 'Netherlands',
    'Nigeria': 'Nigeria',
    'Norge': 'Norway',
    'Norway': 'Norway',
    'norway': 'Norway',
    'Rsa': 'South Africa',
    'Somali': 'Somalia',
    'South Africa': 'South Africa',
    'Spain': 'Spain',
    'UK': 'United Kingdom',
    'Uganda': 'Uganda'
}

# residence mapping
residence_map = {
    'BI Residence': 'BI Residence',
    'BI-Residence': 'BI Residence',
    'BIResidence': 'BI Residence',
    'BI_Residence': 'BI Residence',
    'Private': 'Private',
    'Sognsvann': 'Sognsvann'
}

# prevEducation mapping
preved_map = {
    'High School': 'High School',
    'HighSchool': 'High School',
    'Bachelors': 'Bachelor',
    'Barrrchelors': 'Bachelor',
    'DIPLOMA': 'Diploma',
    'Diploma': 'Diploma',
    'Diplomaaa': 'Diploma',
    'diploma': 'Diploma',
    'Masters': 'Masters',
    'Doctorate': 'Doctorate'
}

# Apply mappings
df['gender_std'] = df['gender'].map(gender_map)
df['country_std'] = df['country'].map(country_map)
df['residence_std'] = df['residence'].map(residence_map)
df['prevEducation_std'] = df['prevEducation'].map(preved_map)

# Report distinct values and counts
for col in ['gender_std', 'country_std', 'residence_std', 'prevEducation_std']:
    counts = df[col].value_counts().sort_index()
    print(f'{col}:')
    for val, cnt in counts.items():
        print(f"  '{val}': {cnt}")
    print(f'Total distinct: {len(counts)}\n')

############## turn 3
import pandas as pd

# The df variable from previous steps is available.
print("Missing before imputation:", df['Python'].isnull().sum())
mean_val = df['Python'].mean()
print(f"Mean of non-null Python scores: {mean_val:.4f}")
# Fill missing
df['Python'].fillna(mean_val, inplace=True)
print("Missing after imputation:", df['Python'].isnull().sum())
print("Final row count:", len(df))
print("Number of rows imputed: 2 (since only 2 were missing)")

# Recompute mean and fill correctly
mean_val = df['Python'].mean()  # This still computes mean of non-null (since fill didn't actually happen)
print(f"Mean of non-null Python scores: {mean_val:.4f}")

df['Python'] = df['Python'].fillna(mean_val)
missing_after = df['Python'].isnull().sum()
print("Missing after proper imputation:", missing_after)
print("Number of rows imputed: 2")
print("Final row count:", len(df))
