############## turn 1
import os
data_dir = '/data/longds/education/bi/task1/data'
files = os.listdir(data_dir)
files

subdir = os.path.join(data_dir, 'intro-to-data-cleaning-eda-and-machine-learning')
os.listdir(subdir)

import pandas as pd

df = pd.read_csv('/data/longds/education/bi/task1/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv')

# Rows and columns
rows, cols = df.shape

# Column names and data types
col_info = df.dtypes

# Missing values per column
missing_per_col = df.isnull().sum()

# Total missing values
total_missing = missing_per_col.sum()

rows, cols, col_info, missing_per_col, total_missing

df = pd.read_csv('/data/longds/education/bi/task1/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv', encoding='latin-1')

# Rows and columns
rows, cols = df.shape

# Column names and data types
col_info = df.dtypes

# Missing values per column
missing_per_col = df.isnull().sum()

# Total missing values
total_missing = missing_per_col.sum()

rows, cols, col_info, missing_per_col, total_missing

############## turn 2
# Compute mean of Python (excluding NaN)
python_mean = df['Python'].mean()
python_mean

# Impute missing values in Python with the mean
df['Python'].fillna(python_mean, inplace=True)

# Check remaining missing values
remaining_missing = df.isnull().sum().sum()

# Check duplicate rows
num_duplicates = df.duplicated().sum()

python_mean_rounded = round(python_mean, 4)
remaining_missing, num_duplicates, python_mean_rounded

# Properly impute missing Python values
df['Python'] = df['Python'].fillna(python_mean)

# Verify remaining missing
remaining_missing = df.isnull().sum().sum()
remaining_missing

############## turn 3
# Ensure df is loaded and imputed properly (re-run if needed)
import pandas as pd
df = pd.read_csv('/data/longds/education/bi/task1/data/intro-to-data-cleaning-eda-and-machine-learning/bi.csv', encoding='latin-1')
python_mean = df['Python'].mean()
df['Python'] = df['Python'].fillna(python_mean)

# Determine numeric columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols

desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75])
# Keep only requested stats
stats = desc.loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats
