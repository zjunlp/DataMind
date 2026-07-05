# Report Task

The code for this project is the report task (DDR-Bench) from the [DataCOPE](https://arxiv.org/pdf/2606.06416) paper. This release includes two main components:

- `DataCOPE_DDR/`: the DataCOPE pipeline for generating skills.
- `DDR_Bench/`: the agent runner, data tools, and evaluation scripts adapted for Deep Data Research-style, open-ended report-generation data analysis tasks.

## Environment

We recommend Python 3.10 or later. Install the required Python packages with:

```bash
pip install -r requirements.txt
```

If you plan to generate DataCOPE skills, install Claude Code and complete the interactive login:

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude
```

## Data Preparation

Prepare the benchmark data before running the agent.

- **10-K**
  - Checklist and metadata: [thinkwee/DDR_Bench](https://github.com/thinkwee/DDR_Bench)
  - Database: [Hugging Face DDRBench collection](https://huggingface.co/collections/thinkwee/ddrbench)
- **MIMIC**
  - Requires PhysioNet access.
  - Download MIMIC-IV v3.1 from https://physionet.org/content/mimiciv/3.1/
  - Convert it with `DDR_Bench/scripts/construct_mimic_sqlite.py`.
- **GLOBEM**
  - Requires PhysioNet access.
  - Download from https://physionet.org/content/globem/1.1/
  - Preprocess it with `DDR_Bench/scripts/process_globem.py`.

After downloading and preprocessing the data, create the explore/test split artifacts. For example, for the 10-K scenario:

```bash
python -m DDR_Bench.scripts.prepare_splits 10k \
  --source-db PATH_TO_10K_DB/DDRBench_10K/raw/10k_financial_data.db
```

## Configuration

Configure the benchmark runner in `DDR_Bench/config.yaml`:

- Set `provider.default_provider` and `provider.default_model`.
- Set `evaluation.provider` and `evaluation.model`.
- Set data paths under `scenarios`, such as `scenarios.10k.db_path`, `scenarios.mimic.db_path`, or `scenarios.globem.data_path`.

If you want to generate a DataCOPE skill, also edit `DataCOPE_DDR/config.yaml`:

- Set `scenario`.
- Set `provider.default_provider` and `provider.default_model`.
- Set `evaluation.provider` and `evaluation.model`.
- Set `paths.ddr_bench_dir` and `paths.ddr_data_dir` if the defaults do not match your layout.

Set the required API keys for your selected providers:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
export MINIMAX_API_KEY="your-minimax-api-key"
```

## Generate a DataCOPE Skill

Run the DataCOPE pipeline:

```bash
python -m DataCOPE_DDR.pipeline.main
```

Generated records are saved under:

```text
DataCOPE_DDR/records/<scenario>-<model>/
```

The final skill is copied to:

```text
DataCOPE_DDR/records/<scenario>-<model>/analysis-final/
```

## Run the Agent

Run the agent on a prepared split:

```bash
python -m DDR_Bench.run_agent \
  --scenario 10k \
  --split test \
  --log-dir ./DDR_Bench/logs/10k_<model>_test
```

To run the agent with a generated DataCOPE skill, pass the skill directory with `--skill-dir`:

```bash
python -m DDR_Bench.run_agent \
  --scenario 10k \
  --split test \
  --skill-dir ./DataCOPE_DDR/records/10k-<model>/analysis-final \
  --log-dir ./DDR_Bench/logs/10k_<model>_test
```

We provide in the `skill` directory the skills generated during our iteration process, as well as the skills used in the final testing. You can use them to reproduce the results in the paper.

## Evaluate

Evaluate the generated logs with the LLM-as-a-checker:

```bash
python -m DDR_Bench.run_evaluation \
  --scenario 10k \
  --split test \
  --logs ./DDR_Bench/logs/10k_<model>_test \
  --output ./DDR_Bench/evaluate/results/10k_<model>_test_evaluation_result.json
```

The evaluation script prints the main score summary and writes the full result JSON to the specified output path.

## Acknowledgements

We thank the authors of [DDR_Bench](https://github.com/thinkwee/DDR_Bench) for releasing the benchmark resources and task setting that this code builds on.
