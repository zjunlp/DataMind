PORT=19007
export OPENAI_BASE_URL=http://0.0.0.0:${PORT}/v1
export OPENAI_API_KEY=placeholder_key

python eval_python.py \
    --model datamind \
    --temperature 0.7 \
    --top_p 0.95 \
    --bs 5 \
    --test_bench dabench \
    --test_file test_file/daeval_test.parquet \
    --csv_or_db_folder da-dev-tables \