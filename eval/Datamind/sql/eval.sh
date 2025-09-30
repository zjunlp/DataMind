PORT=19008
export OPENAI_BASE_URL=http://0.0.0.0:${PORT}/v1
export OPENAI_API_KEY=placeholder_key

python eval_bird.py \
    --model datamind \
    --temperature 0.7 \
    --top_p 0.95 \
    --bs 5 \
    --test_bench bird \
    --test_file bird/test_file/bird_dev.parquet \
    --csv_or_db_folder bird/dev_sqlite_files \
    --gold_csv_results_dir bird/bird_dev_csv_results \
    --db_schema_data_path bird/bird_dev_omni_ddl.json