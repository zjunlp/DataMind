"""
Unit test generator for creating per-query executable tests via dataset exploration.

For each query in a dataset, an Agent explores the underlying data (without seeing
the ground-truth answer) and produces a standalone Python unit-test function.
That function later accepts a candidate answer and returns True/False, enabling
automatic verification of any agent's predictions without a hard-coded ground truth.

Typical workflow
----------------
1. Load a dataset (e.g. DSBio).
2. For every sample, send the Agent a prompt that:
   a. Describes the question and data paths.
   b. Asks it to explore the data and derive a *verification logic* rather than an answer.
   c. Returns a self-contained ``def test_answer(prediction) -> bool`` Python function.
3. Save the generated tests alongside the dataset samples.
4. At evaluation time, run each generated test against the agent's prediction.
"""

import ast
import json
import os
import re
import time
import copy
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import concurrent.futures

from tqdm import tqdm

from ...datasets import DatasetRegistry
from ...agents import ReActDSAgent
from ...eval.utils import EvaluationResult

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

UNIT_TEST_SYSTEM_PROMPT = '''You are an expert data scientist, software engineer, and strict Code Verifier specializing in bioinformatics and scientific data analysis.

Your sole task is to explore a dataset and write a Python unit-test function that can automatically verify whether a candidate answer to a given question is mathematically or logically valid. 

CRITICAL ROLE CONSTRAINT: 
You are the VERIFIER. Your goal is to design STRICT, SUFFICIENT conditions that only the true correct answer can pass. 

Do not just check superficial schemas. You must perform TARGETED SEMANTIC VERIFICATION. Extract the specific logical/mathematical constraints from the user's question and explicitly write logic to check if the given `prediction` satisfies ALL these constraints in the real dataset. 
You are allowed to compute necessary metrics to verify the prediction, but avoid blindly solving the whole problem from scratch if verifying the prediction's properties is more efficient.

OUTPUT FORMAT:
You MUST use the following format for your response. Each step must follow this exact structure:

<reasoning>
1. Identify constraints: What specific mathematical, logical, or biological conditions MUST the correct answer satisfy according to the question?
2. Data exploration plan: What data schemas do you need to check to write the verification logic?
3. Adversarial Check: "How could a WRONG prediction maliciously pass this test?" If a wrong answer can easily pass, you MUST tighten the verification logic before writing the code.
</reasoning>
<python>
Write executable Python code to EXPLORE the dataset (e.g., checking column names, data types, unique constraints). 
DO NOT write code here to solve the core problem. Include all necessary imports.
</python>
<information>
The output/results from your Python code will appear here.
This section is read-only - you cannot write here.
</information>

After exploring the data and identifying the necessary invariants, output your final test function inside a <unit_test> XML block:

<reasoning>
1. Constraint Mapping: Explain how your test function strictly enforces each condition identified earlier.
2. Adversarial Check: "How could a WRONG prediction maliciously pass this test?" If a wrong answer can easily pass, you MUST tighten the verification logic before writing the code.
</reasoning>
<unit_test>
def test_answer(prediction) -> bool:
    """
    Verifies whether `prediction` is a valid answer by checking data invariants.

    Args:
        prediction: The candidate answer (string, list, number, etc.)

    Returns:
        True if the prediction satisfies all required data properties/constraints, False otherwise.
    """
    # Your property-based verification logic here
    ...
    return True  # or False
</unit_test>

RULES for the test function:
1. The function signature must be exactly: def test_answer(prediction) -> bool
2. It must be completely self-contained (no reliance on outer variables).
3. NEVER write logic that simply solves the problem to compare `prediction == my_calculated_answer`.
4. Strictly review any restrictions and boundary conditions that the answer must satisfy.
5. Handle common prediction formats (string vs list, case insensitivity, whitespace, etc.).
6. If verification is impossible (e.g., data file missing), return False rather than raising an error.
7. Keep it reasonably concise (< 100 lines of logic).

NOTE:
1. HANDLING RANKING & EXTREMUM (Top-K) QUESTIONS: If the question asks for a global property (e.g., "highest", "lowest", "most common"), you CANNOT verify it by only looking at the `prediction`. You MUST write efficient, vectorized code to compute the required metric for the ENTIRE relevant dataset. Your verification logic must then partition the data and assert the boundary condition:  `min(metric_of_predicted_items) >= max(metric_of_all_other_items)`. Do not sort the whole dataset to check if `prediction == sorted_data[:2]`. Instead, extract the scores of the `prediction` items, extract the scores of the rest, and compare the mathematical bounds.
'''


def _make_unit_test_prompt(
    question: str,
    context: str,
    answer_guideline: str,
    data_paths: List[str],
    domain: str,
) -> str:
    """Build the user prompt that asks the Agent to generate a unit test."""
    paths_str = "\n".join(data_paths) if data_paths else "No data files provided."
    return f"""QUESTION: {question}

DOMAIN: {domain}

CONTEXT: {context}

ANSWER GUIDELINE: {answer_guideline}

DATASET LOCATIONS (use full paths):
{paths_str}

TASK:
1. Explore the data files above using Python.
2. Strictly review any restrictions and boundary conditions that the answer must satisfy for the given question.
3. Write a Python function `test_answer(prediction) -> bool` that:
   - Accepts a candidate answer as input.
   - Returns True if `prediction` satisfies all required data properties/constraints, False otherwise.
4. Output the function inside <unit_test> ... </unit_test> tags.

Start by loading and exploring the data, then write your test function.
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnitTestResult:
    """Result of unit-test generation for a single sample."""
    sample_id: str
    dataset_name: str
    question: str
    answer_guideline: str
    unit_test_code: str          # The generated test function source
    success: bool                # Whether generation succeeded
    error_info: Optional[Dict[str, Any]] = None
    exploration_trajectory: Optional[List[Dict]] = None
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "question": self.question,
            "answer_guideline": self.answer_guideline,
            "unit_test_code": self.unit_test_code,
            "success": self.success,
            "error_info": self.error_info,
            "generation_time": self.generation_time,
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class UnitTestGeneratorConfig:
    """Configuration for unit-test generation."""
    model: str
    backend: str = "litellm"
    temperature: float = 0.2
    max_tokens: int = 16384
    max_workers: int = 16
    max_turns: int = 20
    max_tokens: int = 4096
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dataset_name: str = "dsbio"
    output_dir: str = "./unit_tests"
    run_name: Optional[str] = None
    start_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class UnitTestGenerator:
    """
    Generates per-query Python unit-test functions by having an Agent explore
    the underlying dataset without access to the ground-truth answer.

    Usage::

        from dsgym.synth.generators.unit_test_generator import (
            UnitTestGenerator, UnitTestGeneratorConfig
        )

        config = UnitTestGeneratorConfig(
            model="gpt-4o",
            dataset_name="dsbio",
            output_dir="./unit_tests",
        )
        generator = UnitTestGenerator(config)
        results = generator.generate(limit=5)
    """

    def __init__(self, config: UnitTestGeneratorConfig):
        self.config = config

        # Single-instance backends don't support multi-worker
        if self.config.backend in ("vllm", "sglang") and self.config.max_workers > 1:
            print(
                f"⚠️  Setting max_workers=1 for {self.config.backend} backend "
                f"(was {self.config.max_workers})"
            )
            self.config.max_workers = 1

        self.dataset = None
        self.agent = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_components(self) -> None:
        """Load dataset and initialise agent."""
        self.dataset = DatasetRegistry.load(
            self.config.dataset_name,
        )

        agent_config: Dict[str, Any] = {
            "manager_url": self.config.manager_url,
            "max_turns": self.config.max_turns,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "output_dir": self.config.output_dir,
        }
        if self.config.backend in ("litellm", "openai"):
            if self.config.api_key:
                agent_config["api_key"] = self.config.api_key
            if self.config.base_url:
                agent_config["base_url"] = self.config.base_url

        self.agent = ReActDSAgent(
            backend=self.config.backend,
            model=self.config.model,
            **agent_config,
        )

    # ------------------------------------------------------------------
    # Core: generate test for one sample
    # ------------------------------------------------------------------

    def generate_unit_test_for_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
    ) -> UnitTestResult:
        """
        Ask the Agent to explore the data and produce a unit-test function for
        one dataset sample.

        Args:
            sample: A standardised dataset sample (as returned by dataset.load()).
            sample_idx: Index in the dataset (used for naming artefacts).

        Returns:
            UnitTestResult with the generated test code (or error info).
        """
        start_time = time.time()
        extra_info = sample.get("extra_info", {})

        question = extra_info.get("question", "")
        context = extra_info.get("context", "")
        answer_guideline = extra_info.get("metadata", {}).get("guidelines", "")
        if not answer_guideline:
            answer_guideline = extra_info.get("answer_guideline", "")
        domain = extra_info.get("metadata", {}).get("domain", "data science")
        data_files = extra_info.get("data_files", {})
        virtual_paths: List[str] = data_files.get("virtual", []) if isinstance(data_files, dict) else []
        dataset_name = extra_info.get("source", self.config.dataset_name)
        sample_id = extra_info.get("id", str(sample_idx))

        # Build the prompt for the agent
        user_content = _make_unit_test_prompt(
            question=question,
            context=context,
            answer_guideline=answer_guideline,
            data_paths=virtual_paths,
            domain=domain,
        )

        generation_sample = {
            "prompt": [
                {"role": "system", "content": UNIT_TEST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "ground_truth": "",  # deliberately empty — agent must not see it
            "extra_info": {
                "question": f"Generate unit test for sample {sample_idx}",
                "source": f"{self.config.dataset_name}_unit_test_generation",
                "index": sample_idx,
                "id": str(sample_idx),
            },
        }

        try:
            from ...eval import Evaluator

            temp_evaluator = Evaluator(metrics=[])
            eval_result: EvaluationResult = temp_evaluator._evaluate_single_sample(
                self.agent, generation_sample
            )

            raw_prediction: str = eval_result.prediction or ""
            unit_test_code = self._extract_unit_test_code(raw_prediction)

            success = bool(unit_test_code) and self._validate_unit_test_syntax(unit_test_code)

            return UnitTestResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                question=question,
                answer_guideline=answer_guideline,
                unit_test_code=unit_test_code,
                success=success,
                error_info=None if success else {
                    "error_type": "ValidationError",
                    "error_message": "Generated code did not pass syntax validation or was empty.",
                },
                exploration_trajectory=getattr(eval_result, "conversation", []),
                generation_time=time.time() - start_time,
            )

        except Exception as exc:
            return UnitTestResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                question=question,
                answer_guideline=answer_guideline,
                unit_test_code="",
                success=False,
                error_info={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                generation_time=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Parsing & validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_unit_test_code(raw_text: str) -> str:
        """
        Extract the Python function from within <unit_test>...</unit_test> tags.

        Falls back to searching for ``def test_answer`` directly if tags are absent.
        """
        # Primary: look for XML-style tags
        tag_pattern = re.compile(
            r"<unit_test>(.*?)</unit_test>", re.DOTALL | re.IGNORECASE
        )
        match = tag_pattern.search(raw_text)
        if match:
            return textwrap.dedent(match.group(1)).strip()

        # Fallback: extract starting from "def test_answer"
        func_pattern = re.compile(
            r"(def test_answer\s*\(.*?\)\s*(?:->.*?)?:\s*\n.*?)(?=\n(?:def |\Z))",
            re.DOTALL,
        )
        func_match = func_pattern.search(raw_text)
        if func_match:
            return textwrap.dedent(func_match.group(1)).strip()

        # Last resort: return everything after the last ```python fence
        fence_pattern = re.compile(r"```python\s*(.*?)```", re.DOTALL)
        fences = fence_pattern.findall(raw_text)
        for block in reversed(fences):
            if "def test_answer" in block:
                return textwrap.dedent(block).strip()

        return ""

    @staticmethod
    def _validate_unit_test_syntax(code: str) -> bool:
        """Return True if the code is syntactically valid Python and contains
        ``def test_answer``."""
        if "def test_answer" not in code:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def _wrapper(self, args):
        """Thread-pool wrapper."""
        sample_idx, sample = args
        return sample_idx, self.generate_unit_test_for_sample(sample, sample_idx)

    def generate(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate unit tests for all (or a subset of) samples in the dataset.

        Args:
            samples: Pre-loaded samples.  If None, loads from the configured dataset.
            limit:   Cap on the number of samples to process.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            Dictionary with results, summary statistics, and output file paths.
        """
        start_time = time.time()
        print("🔧 Initializing components...")
        self._initialize_components()

        # Load samples if needed
        if samples is None:
            load_cfg = {"limit": limit} if limit else {}
            samples = self.dataset.load(**load_cfg)

        start_idx = self.config.start_index
        if start_idx > 0:
            samples = samples[start_idx:]
            print(f"📍 Starting from sample index {start_idx}")
        if limit:
            samples = samples[:limit]

        print(f"📊 Generating unit tests for {len(samples)} samples...")
        print(f"🌡️  Temperature: {self.config.temperature}")
        print(f"🤖 Model: {self.config.model}")
        print(f"⚙️  Backend: {self.config.backend}")
        print(f"👷 Max workers: {self.config.max_workers}")

        all_results: List[UnitTestResult] = []

        if self.config.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                futures = [
                    executor.submit(self._wrapper, (start_idx + idx, sample))
                    for idx, sample in enumerate(samples)
                ]
                it = (
                    tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Generating unit tests (parallel)",
                    )
                    if show_progress
                    else concurrent.futures.as_completed(futures)
                )
                results_dict: Dict[int, UnitTestResult] = {}
                for future in it:
                    s_idx, result = future.result()
                    results_dict[s_idx] = result
                for idx in range(len(samples)):
                    all_results.append(results_dict[start_idx + idx])
        else:
            it2 = (
                tqdm(enumerate(samples), total=len(samples), desc="Generating unit tests")
                if show_progress
                else enumerate(samples)
            )
            for idx, sample in it2:
                result = self.generate_unit_test_for_sample(sample, start_idx + idx)
                all_results.append(result)

        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.success)

        summary = {
            "total_samples": len(samples),
            "successful_generations": successful,
            "failed_generations": len(samples) - successful,
            "success_rate": successful / len(samples) if samples else 0.0,
            "total_time": total_time,
            "config": self.config.to_dict(),
        }

        output_paths = self._save_results(all_results, summary)

        print(
            f"✅ Generated {successful}/{len(samples)} unit tests "
            f"in {total_time:.2f}s  "
            f"(success rate: {summary['success_rate']:.1%})"
        )

        return {
            "results": all_results,
            "summary": summary,
            "output_paths": output_paths,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: List[UnitTestResult],
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Persist generated unit tests and summary statistics."""
        run_name = (
            self.config.run_name
            or f"unit_test_{self.config.dataset_name}_{self.config.backend}_"
            f"{self.config.model.replace('/', '_')}"
        )
        run_name = run_name.replace(":", "_")

        out_dir = Path(self.config.output_dir)
        tests_dir = out_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # 1. One JSON file with all results (without trajectory to keep it compact)
        all_tests_file = out_dir / f"{run_name}_unit_tests.json"
        serialisable = [r.to_dict() for r in results]
        with open(all_tests_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)

        # 2. Individual .py files for easy import / inspection
        for result in results:
            if result.success and result.unit_test_code:
                py_file = tests_dir / f"test_sample_{result.sample_id}.py"
                header = (
                    f'"""\nAuto-generated unit test for sample {result.sample_id}\n'
                    f"Dataset : {result.dataset_name}\n"
                    f"Question: {result.question[:120]}{'...' if len(result.question) > 120 else ''}\n"
                    f'"""\n\n'
                )
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(header + result.unit_test_code + "\n")

        # 3. Summary statistics
        summary_file = out_dir / f"{run_name}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {**summary, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                f,
                indent=2,
                ensure_ascii=False,
            )

        paths = {
            "all_tests": str(all_tests_file),
            "individual_tests_dir": str(tests_dir),
            "summary": str(summary_file),
        }

        print(f"📁 Results saved:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

        return paths


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_unit_test_generator(
    model: str,
    dataset_name: str = "dsbio",
    backend: str = "litellm",
    temperature: float = 0.2,
    max_workers: int = 16,
    output_dir: str = "./unit_tests",
    start_index: int = 0,
    **kwargs: Any,
) -> UnitTestGenerator:
    """
    Convenience factory for UnitTestGenerator.

    Args:
        model:        LLM model name (e.g. ``"gpt-4o"``).
        dataset_name: Registered dataset name (e.g. ``"dsbio"``).
        backend:      Inference backend (``"litellm"``, ``"vllm"``, ``"sglang"``).
        temperature:  Sampling temperature.
        max_workers:  Parallel workers.
        output_dir:   Directory for output files.
        start_index:  First sample index to process.
        **kwargs:     Additional UnitTestGeneratorConfig fields.

    Returns:
        Configured :class:`UnitTestGenerator`.
    """
    config = UnitTestGeneratorConfig(
        model=model,
        dataset_name=dataset_name,
        backend=backend,
        temperature=temperature,
        max_workers=max_workers,
        output_dir=output_dir,
        start_index=start_index,
        **kwargs,
    )
    return UnitTestGenerator(config)
