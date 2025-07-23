mkdir -p data
cd data

####################### Download QRData data
mkdir -p QRData
cd QRData
# Download data.zip
if [ ! -f "data.zip" ]; then
    wget https://github.com/xxxiaol/QRData/raw/refs/heads/main/benchmark/data.zip
else
    echo "data.zip already exists. Skipping download."
fi

# Unzip only if not already extracted
if [ ! -d "data" ]; then
    unzip data.zip
    rm data.zip
else
    echo "Data already extracted. Skipping unzip."
fi
wget -nc https://raw.githubusercontent.com/xxxiaol/QRData/refs/heads/main/benchmark/QRData.json
cd ..



####################### Download DiscoveryBench data
mkdir -p DiscoveryBench
cd DiscoveryBench
wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/eval/answer_key_real.csv
mkdir -p archaeology introduction_pathways_non-native_plants meta_regression
mkdir -p meta_regression_raw nls_incarceration nls_raw nls_ses
mkdir -p requirements_engineering_for_ML_enabled_systems worldbank_education_gdp worldbank_education_gdp_indicators
cd archaeology
files=("capital.csv"
    "metadata_0.json"
    "metadata_1.json"
    "metadata_10.json"
    "metadata_11.json"
    "metadata_12.json"
    "metadata_13.json"
    "metadata_14.json"
    "metadata_15.json"
    "metadata_16.json"
    "metadata_17.json"
    "metadata_18.json"
    "metadata_19.json"
    "metadata_2.json"
    "metadata_20.json"
    "metadata_21.json"
    "metadata_22.json"
    "metadata_23.json"
    "metadata_24.json"
    "metadata_25.json"
    "metadata_26.json"
    "metadata_27.json"
    "metadata_28.json"
    "metadata_29.json"
    "metadata_3.json"
    "metadata_30.json"
    "metadata_31.json"
    "metadata_32.json"
    "metadata_33.json"
    "metadata_34.json"
    "metadata_35.json"
    "metadata_36.json"
    "metadata_37.json"
    "metadata_4.json"
    "metadata_5.json"
    "metadata_6.json"
    "metadata_7.json"
    "metadata_8.json"
    "metadata_9.json"
    "pollen_openness_score_Belau_Woserin_Feeser_et_al_2019.csv"
    "time_series_data.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/archaeology/${file}; done
cd ..
cd introduction_pathways_non-native_plants
files=("invaded_niche_pathways.csv"
       "invasion_success_pathways.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "phylogenetic_tree.txt"
       "temporal_trends_contingency_table.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/introduction_pathways_non-native_plants/${file}; done
cd ..
cd meta_regression
files=("meta-regression_joined_data_heterogeneity_in_replication_projects.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_16.json"
       "metadata_17.json"
       "metadata_18.json"
       "metadata_19.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/meta_regression/${file}; done
cd ..
cd meta_regression_raw
files=("meta-regression_replication_success_data_heterogeneity_in_replication_projects.csv"
       "meta-regression_study_data_heterogeneity_in_replication_projects.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_16.json"
       "metadata_17.json"
       "metadata_18.json"
       "metadata_19.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/meta_regression_raw/${file}; done
cd ..
cd nls_incarceration
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "nls_incarceration_processed.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_incarceration/${file}; done
cd ..
cd nls_raw
files=("metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "nls_raw.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_raw/${file}; done
cd ..
cd nls_ses
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "nls_ses_processed.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_ses/${file}; done
cd ..
cd requirements_engineering_for_ML_enabled_systems
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "requirements_engineering_for_ML-enabled_systems.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/requirements_engineering_for_ML_enabled_systems/${file}; done
cd ..
cd worldbank_education_gdp
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "worldbank_education_gdp.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/worldbank_education_gdp/${file}; done
cd ..
cd worldbank_education_gdp_indicators
files=("Adjusted_savings_education_expenditure_percentage_of_GNI.csv"
       "Exports_of_goods_and_services_annual_percentage_growth.csv"
       "GNI_per_capita_constant_2015_USdollar.csv"
       "Labor_force_participation_rate_total_percentage_of_total_population_ages_15+_modeled_ILO_estimate.csv"
       "School_enrollment_primary_percentage_gross.csv"
       "School_enrollment_secondary_percentage_gross.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/worldbank_education_gdp_indicators/${file}; done
cd ..


target_dir="$(pwd)"
tables_dir="$target_dir/tables"

# Create the "tables" folder, move all the required tables to it uniformly
mkdir -p "$tables_dir"

find "$target_dir" -type f \( -name "*.csv" -o -name "*.txt" \) -print0 | while IFS= read -r -d $'\0' file; do
    filename=$(basename -- "$file")
    dest="$tables_dir/$filename"
    cp -v -- "$file" "$dest"
done

cd tables
rm -rf answer_key_real.csv