"""
Checklist generator for creating per-query fine-grained evaluation checklists.

For each query in a dataset, an Agent explores the underlying data (without seeing
the ground-truth answer) and produces a structured JSON checklist of verifiable
criteria. Each criterion is a specific, atomic condition that the correct answer
must satisfy, enabling fine-grained evaluation of any agent's predictions.

Unlike unit tests (which produce executable Python), checklists produce human-
readable / LLM-evaluable criteria that can be used by a judge model to score
partial credit, identify failure modes, and provide structured feedback.

Typical workflow
----------------
1. Load a dataset (e.g. DSBio).
2. For every sample, send the Agent a prompt that:
   a. Describes the question and data paths.
   b. Asks it to explore the data and derive atomic verification criteria.
   c. Returns a structured JSON checklist.
3. Save the generated checklists alongside the dataset samples.
4. At evaluation time, use a judge LLM to score each criterion against the
   agent's prediction, producing a fine-grained rubric score.
"""

import json
import os
import re
import time
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

CHECKLIST_SYSTEM_PROMPT = '''You are an expert data analyst and evaluation specialist with broad knowledge spanning data science, statistics, and various applied domains (bioinformatics, finance, etc.). Your task is to explore a dataset and produce a fine-grained checklist of verifiable criteria that a correct answer to a given question MUST satisfy.

CRITICAL ROLE CONSTRAINT:
You are the EVALUATOR. Your goal is to decompose the correctness requirements of a question into atomic, independently verifiable criteria. Each criterion should be:
- **Specific**: Tied to concrete values, ranges, or structures derivable from the data or domain knowledge.
- **Atomic**: Tests exactly one property of the answer.
- **Unambiguous**: A judge can determine pass/fail without subjective interpretation.
- **Necessary**: Failing this criterion means the answer is wrong or incomplete.

TWO MANDATORY DIMENSIONS you must always cover:

[DIMENSION 1 — ANSWER FORMAT & TYPE]
Every checklist MUST include explicit criteria that pin down the exact form of the answer:
- Data type: Is the answer a single string, a list of strings, an integer, a float, a boolean, a dict?
- If numeric: Is it expected to be positive, negative, or can it be either? Is it an integer or a float? What is a plausible order-of-magnitude or range based on the data?
- If string: What naming or representation convention must be followed?
  Examples by domain: gene symbols (uppercase "BRCA1"), column names (exact case from the dataset),
  etc.
- If list/set: Is order significant? Are duplicates allowed? What is the expected length or cardinality?
- If a ratio or percentage: Should it be expressed as a fraction (0-1) or percentage (0-100)?

[DIMENSION 2 — KNOWLEDGE CONSISTENCY & HALLUCINATION GUARD]
Every checklist MUST include criteria that detect answers which, while superficially plausible, conflict with established domain knowledge, common sense, or the factual content of the dataset. Think broadly across these sub-types:

a) **Category / membership conflict**: The question asks for entities of a specific category (e.g., "up-regulated genes"). The answer must only contain entities that truly belong to that category according to the dataset or well-known facts. Any entity outside the category is a hallucination.

b) **Ontology / nomenclature conflict**: The answer uses names, IDs, or labels that do not exist in the dataset or in the relevant reference ontology (e.g., a gene symbol that is not present in the data, a category label that was never defined).

c) **Quantitative plausibility conflict**: The answer contains a numeric value that contradicts established domain knowledge or observable data properties (e.g., a negative count, a p-value = 0 exactly, a revenue that is orders of magnitude outside the dataset range).

d) **Causal / logical conflict**: The answer contradicts a well-known logical or causal relationship implied by the question (e.g., a treatment that is stated to be ineffective is listed as the most effective).

OUTPUT FORMAT:
You MUST use the following format. Each step must follow this exact structure:

<reasoning>
1. Answer decomposition: What are the distinct components of a correct answer? (e.g., correct entity, correct value, correct format, correct ordering)
2. Format & type analysis: What is the exact expected data type, sign, range, and naming convention for this answer?
3. Knowledge consistency check: What domain knowledge, common-sense constraints, or dataset-specific facts must the answer respect? Enumerate which sub-types of hallucination (a-d above) are most likely for this question, and what a plausible-but-wrong answer would look like.
4. Data exploration plan: What columns, value distributions, entity sets, or reference vocabularies do you need to inspect to concretely specify the criteria above?
</reasoning>
<python>
Write executable Python code to EXPLORE the dataset (e.g., check column names, value distributions, boundary values, unique entity names, valid category labels).
DO NOT solve the core problem here. Include all necessary imports.
</python>
<information>
The output/results from your Python code will appear here.
This section is read-only - you cannot write here.
</information>

After exploring the data, output your checklist inside a <checklist> XML block as a JSON array:

<reasoning>
1. Criterion mapping: Explain how each criterion maps to a specific requirement of the question.
2. Format coverage: Confirm you have at least one criterion for data type, sign/range, and string convention (where applicable).
3. Hallucination coverage: For each relevant hallucination sub-type (a-d), confirm you have a criterion that would catch it, or explain why it does not apply to this question.
4. Completeness & independence: Together do these criteria fully characterise a correct answer? Is each criterion independently testable?
</reasoning>
<checklist>
[
  {
    "id": "1",
    "description": "Short human-readable name of this criterion",
    "criterion": "Precise, verifiable statement of what the answer must satisfy",
    "category": "one of: type | format | value | ordering | completeness | correctness | hallucination",
    "hint": "Concrete expected value, range, pattern, valid set, or knowledge fact the judge should check"
  },
  ...
]
</checklist>

RULES for the checklist:
1. Each criterion must be independently verifiable by a judge LLM given only the question, the answer guideline, and the prediction.
2. MANDATORY — include at least one `type` criterion specifying the exact answer type (e.g., "The answer must be a non-empty list of strings", "The answer must be a positive float").
3. MANDATORY — if the answer is numeric, the `type` criterion must explicitly state the expected sign (positive / negative / either) and plausible range derived from the data or domain knowledge.
4. MANDATORY — include at least one `hallucination` criterion that checks whether the answer conflicts with domain knowledge, common sense, or dataset facts. Apply whichever sub-type(s) (a-d) are relevant: category membership, valid entity names, quantitative plausibility, causal logic.
5. Include criteria for the CORE CORRECTNESS of each distinct claim in the answer.
6. Do NOT include criteria that require re-solving the entire problem from scratch.
7. Keep descriptions concise (< 20 words). Keep criterion statements precise but readable.
8. Aim for 5-12 criteria per question; avoid both under-specification and redundancy.
9. If data exploration is impossible (e.g., file missing), still produce criteria based on the question text and domain knowledge alone.

CATEGORY GUIDE:
- type        : Exact data type, sign, range, or structural shape of the answer (ALWAYS include)
- format      : String format, naming convention, or representation style (e.g., uppercase, ID prefix, date format)
- value       : Specific numeric or string values the answer must contain or approximate
- ordering    : Relative or absolute ordering of elements
- completeness: Required number of items, no missing elements
- correctness : Logical or mathematical truth conditions derivable from the data
- hallucination: Answer does not conflict with domain knowledge, common sense, or dataset facts — covers category membership, valid entity names, quantitative plausibility, causal logic
'''


def _make_checklist_prompt(
    question: str,
    context: str,
    answer_guideline: str,
    data_paths: List[str],
    domain: str,
) -> str:
    """Build the user prompt that asks the Agent to generate a checklist."""
    paths_str = "\n".join(data_paths) if data_paths else "No data files provided."
    return f"""QUESTION: {question}

DOMAIN: {domain}

CONTEXT: {context}

ANSWER GUIDELINE: {answer_guideline}

DATASET LOCATIONS (use full paths):
{paths_str}

TASK:
1. Explore the data files above using Python to understand the schema and relevant statistics.
2. Identify ALL distinct properties that a correct answer to the question must have.
3. Decompose these properties into atomic, independently verifiable criteria.
4. Output a JSON checklist inside <checklist> ... </checklist> tags.

Each criterion will later be evaluated by a judge LLM that receives:
- The original question
- The answer guideline
- A candidate prediction
- Your criterion statement

Start by loading and exploring the data, then write your checklist.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChecklistItem:
    """A single verifiable criterion in a checklist."""
    id: str
    description: str
    criterion: str
    category: str
    weight: float
    hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "criterion": self.criterion,
            "category": self.category,
            "weight": self.weight,
            "hint": self.hint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChecklistItem":
        return cls(
            id=str(d.get("id", "")),
            description=str(d.get("description", "")),
            criterion=str(d.get("criterion", "")),
            category=str(d.get("category", "correctness")),
            weight=float(d.get("weight", 1)),
            hint=str(d.get("hint", "")),
        )


@dataclass
class ChecklistResult:
    """Result of checklist generation for a single sample."""
    sample_id: str
    dataset_name: str
    question: str
    answer_guideline: str
    checklist: List[ChecklistItem]       # Parsed checklist items
    checklist_json: str                  # Raw JSON string as generated
    success: bool
    error_info: Optional[Dict[str, Any]] = None
    exploration_trajectory: Optional[List[Dict]] = None
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "question": self.question,
            "answer_guideline": self.answer_guideline,
            "checklist": [item.to_dict() for item in self.checklist],
            "checklist_json": self.checklist_json,
            "success": self.success,
            "error_info": self.error_info,
            "generation_time": self.generation_time,
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ChecklistGeneratorConfig:
    """Configuration for checklist generation."""
    model: str
    backend: str = "litellm"
    temperature: float = 0.2
    max_tokens: int = 8192
    max_workers: int = 16
    max_turns: int = 20
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dataset_name: str = "dsbio"
    output_dir: str = "./checklists"
    run_name: Optional[str] = None
    start_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class ChecklistGenerator:
    """
    Generates per-query fine-grained evaluation checklists by having an Agent
    explore the underlying dataset without access to the ground-truth answer.

    Each checklist is a JSON array of atomic criteria that a judge LLM can
    evaluate against a candidate prediction to produce a structured rubric score.

    Usage::

        from dsgym.synth.generators.checklist_generator import (
            ChecklistGenerator, ChecklistGeneratorConfig
        )

        config = ChecklistGeneratorConfig(
            model="gpt-4o",
            dataset_name="dsbio",
            output_dir="./checklists",
        )
        generator = ChecklistGenerator(config)
        results = generator.generate(limit=5)
    """

    def __init__(self, config: ChecklistGeneratorConfig):
        self.config = config

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
        self.dataset = DatasetRegistry.load(self.config.dataset_name)

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
    # Core: generate checklist for one sample
    # ------------------------------------------------------------------

    def generate_checklist_for_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
    ) -> ChecklistResult:
        """
        Ask the Agent to explore the data and produce an evaluation checklist
        for one dataset sample.

        Args:
            sample:     A standardised dataset sample (as returned by dataset.load()).
            sample_idx: Index in the dataset (used for naming artefacts).

        Returns:
            ChecklistResult with the generated checklist (or error info).
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
        virtual_paths: List[str] = (
            data_files.get("virtual", []) if isinstance(data_files, dict) else []
        )
        dataset_name = extra_info.get("source", self.config.dataset_name)
        sample_id = extra_info.get("id", str(sample_idx))

        user_content = _make_checklist_prompt(
            question=question,
            context=context,
            answer_guideline=answer_guideline,
            data_paths=virtual_paths,
            domain=domain,
        )

        generation_sample = {
            "prompt": [
                {"role": "system", "content": CHECKLIST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "ground_truth": "",  # deliberately empty — agent must not see it
            "extra_info": {
                "question": f"Generate checklist for sample {sample_idx}",
                "source": f"{self.config.dataset_name}_checklist_generation",
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
            checklist_json, checklist_items = self._extract_checklist(raw_prediction)

            success = bool(checklist_items) and self._validate_checklist(checklist_items)

            return ChecklistResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                question=question,
                answer_guideline=answer_guideline,
                checklist=checklist_items,
                checklist_json=checklist_json,
                success=success,
                error_info=None if success else {
                    "error_type": "ValidationError",
                    "error_message": "Generated checklist did not pass validation or was empty.",
                },
                exploration_trajectory=getattr(eval_result, "conversation", []),
                generation_time=time.time() - start_time,
            )

        except Exception as exc:
            return ChecklistResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                question=question,
                answer_guideline=answer_guideline,
                checklist=[],
                checklist_json="",
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
    def _extract_checklist(raw_text: str):
        """
        Extract and parse the JSON checklist from within <checklist>...</checklist> tags.

        Falls back to searching for a bare JSON array if tags are absent.

        Returns:
            (checklist_json: str, checklist_items: List[ChecklistItem])
        """
        # Primary: XML-style tags
        tag_pattern = re.compile(
            r"<checklist>(.*?)</checklist>", re.DOTALL | re.IGNORECASE
        )
        match = tag_pattern.search(raw_text)
        if match:
            raw_json = textwrap.dedent(match.group(1)).strip()
        else:
            # Fallback: look for a JSON array inside a ```json ... ``` fence
            fence_pattern = re.compile(r"```json\s*(\[.*?])\s*```", re.DOTALL)
            fence_match = fence_pattern.search(raw_text)
            if fence_match:
                raw_json = fence_match.group(1).strip()
            else:
                # Last resort: find the first top-level JSON array
                array_pattern = re.compile(r"(\[\s*\{.*?\}\s*\])", re.DOTALL)
                array_match = array_pattern.search(raw_text)
                raw_json = array_match.group(1).strip() if array_match else ""

        if not raw_json:
            return "", []

        try:
            data = json.loads(raw_json)
            if not isinstance(data, list):
                return raw_json, []
            items = [ChecklistItem.from_dict(d) for d in data if isinstance(d, dict)]
            return raw_json, items
        except json.JSONDecodeError:
            return raw_json, []

    @staticmethod
    def _validate_checklist(items: List[ChecklistItem]) -> bool:
        """
        Return True if the checklist is non-empty and every item has the
        required non-empty fields (id, criterion, category).
        """
        if not items:
            return False
        valid_categories = {"type", "format", "value", "ordering", "completeness", "correctness", "domain", "hallucination"}
        for item in items:
            if not item.id or not item.criterion or not item.category:
                return False
            if item.category not in valid_categories:
                # Accept unknown categories rather than hard-failing
                pass
        return True

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def _wrapper(self, args):
        """Thread-pool wrapper."""
        sample_idx, sample = args
        return sample_idx, self.generate_checklist_for_sample(sample, sample_idx)

    def generate(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate evaluation checklists for all (or a subset of) samples.

        Args:
            samples:       Pre-loaded samples. If None, loads from the configured dataset.
            limit:         Cap on the number of samples to process.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            Dictionary with results, summary statistics, and output file paths.
        """
        start_time = time.time()
        print("🔧 Initializing components...")
        self._initialize_components()

        if samples is None:
            load_cfg = {"limit": limit} if limit else {}
            samples = self.dataset.load(**load_cfg)

        start_idx = self.config.start_index
        if start_idx > 0:
            samples = samples[start_idx:]
            print(f"📍 Starting from sample index {start_idx}")
        if limit:
            samples = samples[:limit]

        print(f"📊 Generating checklists for {len(samples)} samples...")
        print(f"🌡️  Temperature: {self.config.temperature}")
        print(f"🤖 Model: {self.config.model}")
        print(f"⚙️  Backend: {self.config.backend}")
        print(f"👷 Max workers: {self.config.max_workers}")

        all_results: List[ChecklistResult] = []

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
                        desc="Generating checklists (parallel)",
                    )
                    if show_progress
                    else concurrent.futures.as_completed(futures)
                )
                results_dict: Dict[int, ChecklistResult] = {}
                for future in it:
                    s_idx, result = future.result()
                    results_dict[s_idx] = result
                for idx in range(len(samples)):
                    all_results.append(results_dict[start_idx + idx])
        else:
            it2 = (
                tqdm(enumerate(samples), total=len(samples), desc="Generating checklists")
                if show_progress
                else enumerate(samples)
            )
            for idx, sample in it2:
                result = self.generate_checklist_for_sample(sample, start_idx + idx)
                all_results.append(result)

        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.success)
        total_criteria = sum(len(r.checklist) for r in all_results if r.success)
        avg_criteria = total_criteria / successful if successful else 0.0

        summary = {
            "total_samples": len(samples),
            "successful_generations": successful,
            "failed_generations": len(samples) - successful,
            "success_rate": successful / len(samples) if samples else 0.0,
            "total_criteria_generated": total_criteria,
            "avg_criteria_per_sample": avg_criteria,
            "total_time": total_time,
            "config": self.config.to_dict(),
        }

        output_paths = self._save_results(all_results, summary)

        print(
            f"✅ Generated checklists for {successful}/{len(samples)} samples "
            f"in {total_time:.2f}s  "
            f"(success rate: {summary['success_rate']:.1%}, "
            f"avg {avg_criteria:.1f} criteria/sample)"
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
        results: List[ChecklistResult],
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Persist generated checklists and summary statistics."""
        run_name = (
            self.config.run_name
            or f"checklist_{self.config.dataset_name}_{self.config.backend}_"
            f"{self.config.model.replace('/', '_')}"
        )
        run_name = run_name.replace(":", "_")

        out_dir = Path(self.config.output_dir)
        items_dir = out_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)

        # 1. One JSON file with all results
        all_checklists_file = out_dir / f"{run_name}_checklists.json"
        with open(all_checklists_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

        # 2. Individual JSON files per sample for easy lookup at eval time
        for result in results:
            if result.success and result.checklist:
                item_file = items_dir / f"checklist_{result.sample_id}.json"
                payload = {
                    "sample_id": result.sample_id,
                    "question": result.question,
                    "answer_guideline": result.answer_guideline,
                    "checklist": [item.to_dict() for item in result.checklist],
                }
                with open(item_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

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
            "all_checklists": str(all_checklists_file),
            "individual_items_dir": str(items_dir),
            "summary": str(summary_file),
        }

        print(f"📁 Results saved:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

        return paths


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_checklist_generator(
    model: str,
    dataset_name: str = "dsbio",
    backend: str = "litellm",
    temperature: float = 0.2,
    max_workers: int = 16,
    output_dir: str = "./checklists",
    start_index: int = 0,
    **kwargs: Any,
) -> ChecklistGenerator:
    """
    Convenience factory for ChecklistGenerator.

    Args:
        model:        LLM model name (e.g. ``"gpt-4o"``).
        dataset_name: Registered dataset name (e.g. ``"dsbio"``).
        backend:      Inference backend (``"litellm"``, ``"vllm"``, ``"sglang"``).
        temperature:  Sampling temperature.
        max_workers:  Parallel workers.
        output_dir:   Directory for output files.
        start_index:  First sample index to process.
        **kwargs:     Additional ChecklistGeneratorConfig fields.

    Returns:
        Configured :class:`ChecklistGenerator`.
    """
    config = ChecklistGeneratorConfig(
        model=model,
        dataset_name=dataset_name,
        backend=backend,
        temperature=temperature,
        max_workers=max_workers,
        output_dir=output_dir,
        start_index=start_index,
        **kwargs,
    )
    return ChecklistGenerator(config)
