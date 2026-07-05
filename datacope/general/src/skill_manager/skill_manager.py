import argparse
import json
import os
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import src.runtimes.agents  # triggers @register_agent decorators
from src.runtimes.registry import AgentRegistry
from src.core.schema import AgentResult

from src.skill_manager.prompt import (
    SYSTEM_PROMPT,
    CATEGORY_SKILL_GENERATE_PROMPT,
    CATEGORY_SKILL_MODIFY_PROMPT,
    NO_CATEGORY_SKILL_CREATE_PROMPT,
    NO_CATEGORY_SKILL_MODIFY_PROMPT,
)
from src.skill_manager.base import BaseSkillManager
from src.skill_manager.utils import SkillModelOutput


output_schema = SkillModelOutput.model_json_schema()


class SkillManager(BaseSkillManager):
    """Manages skill creation and modification using Claude Code as the agent backend."""

    def __init__(
        self,
        model: str,
        agent_type: str,
        task: str,
        data_dir: str,
        current_skill_dir: str,
        category_list: list = None,
        parallel_workers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            agent_type,
            task,
            data_dir,
            current_skill_dir,
            category_list,
            **kwargs,
        )
        self.parallel_workers = parallel_workers
        self.agent = AgentRegistry.load(agent_type, model=model, output_schema=output_schema, **kwargs)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def create_skill(
        self,
        category: str,
        traj_dir: str,
        verifier_prompt: str = "",
    ) -> AgentResult:
        skill_dir = os.path.join(self.current_skill_dir, category)
        Path(skill_dir).mkdir(parents=True, exist_ok=True)

        prompt = CATEGORY_SKILL_GENERATE_PROMPT.format(
            category=category,
            task=self.task,
            traj_dir=traj_dir,
            skill_dir=skill_dir,
            data_dir=self.data_dir,
            verifier_prompt=verifier_prompt,
        )

        return self.agent.solve_task(prompt, system=SYSTEM_PROMPT)

    def modify_skill(
        self,
        category: str,
        traj_dir: str,
        verifier_prompt: str = "",
    ) -> AgentResult:
        skill_dir = os.path.join(self.current_skill_dir, category)
        Path(skill_dir).mkdir(parents=True, exist_ok=True)

        prompt = CATEGORY_SKILL_MODIFY_PROMPT.format(
            category=category,
            task=self.task,
            traj_dir=traj_dir,
            skill_dir=skill_dir,
            data_dir=self.data_dir,
            verifier_prompt=verifier_prompt,
        )

        return self.agent.solve_task(prompt, system=SYSTEM_PROMPT)

    def create_skill_without_category(
        self,
        traj_dir: str,
        verifier_prompt: str = "",
    ) -> AgentResult:
        skill_dir = self.current_skill_dir
        Path(skill_dir).mkdir(parents=True, exist_ok=True)

        prompt = NO_CATEGORY_SKILL_CREATE_PROMPT.format(
            task=self.task,
            traj_dir=traj_dir,
            skill_dir=skill_dir,
            data_dir=self.data_dir,
            verifier_prompt=verifier_prompt,
        )

        return self.agent.solve_task(prompt, system=SYSTEM_PROMPT)

    def modify_skill_without_category(
        self,
        traj_dir: str,
        verifier_prompt: str = "",
    ) -> AgentResult:
        skill_dir = self.current_skill_dir
        Path(skill_dir).mkdir(parents=True, exist_ok=True)

        prompt = NO_CATEGORY_SKILL_MODIFY_PROMPT.format(
            task=self.task,
            traj_dir=traj_dir,
            skill_dir=skill_dir,
            data_dir=self.data_dir,
            verifier_prompt=verifier_prompt,
        )

        return self.agent.solve_task(prompt, system=SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_single_category(
        self,
        category: str,
        traj_dir: str,
        verifier_prompt: str,
        mode: str,
    ) -> tuple[str, AgentResult]:
        if mode == "create":
            result = self.create_skill(category, traj_dir, verifier_prompt)
        else:
            result = self.modify_skill(category, traj_dir, verifier_prompt)
        return category, result

    def _process_without_category(
        self,
        traj_dir: str,
        verifier_prompt: str,
        mode: str,
    ) -> AgentResult:
        if mode == "create":
            return self.create_skill_without_category(traj_dir, verifier_prompt)
        return self.modify_skill_without_category(traj_dir, verifier_prompt)

    def _run_sequential(
        self,
        categories: List[str],
        traj_base_dir: str,
        verifier_prompt: str,
        mode: str,
        show_progress: bool,
    ) -> Dict[str, AgentResult]:
        results: Dict[str, AgentResult] = {}
        iterator = tqdm(categories, desc=f"Skill {mode}") if show_progress else categories

        for category in iterator:
            try:
                traj_dir = os.path.join(traj_base_dir, category)
                cat, result = self._process_single_category(
                    category, traj_dir, verifier_prompt, mode,
                )
                results[cat] = result
            except Exception as e:
                print(f"  [skill manager category {category}] Error: {e}")
                results[category] = AgentResult(error=str(e))

        return results

    def _run_parallel(
        self,
        categories: List[str],
        traj_base_dir: str,
        verifier_prompt: str,
        mode: str,
        show_progress: bool,
    ) -> Dict[str, AgentResult]:
        results: Dict[str, AgentResult] = {}

        def process_one(index_category):
            index, category = index_category
            try:
                traj_dir = os.path.join(traj_base_dir, category)
                return index, *self._process_single_category(
                    category, traj_dir, verifier_prompt, mode,
                )
            except Exception as e:
                print(f"  [skill manager category {category}] Error: {e}")
                return index, category, AgentResult(error=str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = [
                executor.submit(process_one, (i, cat))
                for i, cat in enumerate(categories)
            ]

            completed = concurrent.futures.as_completed(futures)
            if show_progress:
                completed = tqdm(completed, total=len(futures), desc=f"Skill {mode} (parallel)")

            for future in completed:
                _, cat, result = future.result()
                results[cat] = result

        return results

    def run_batch(
        self,
        traj_base_dir: str,
        verifier_prompt: str = "",
        mode: str = "create",
        target_categories: Optional[List[str]] = None,
        show_progress: bool = True,
        use_category: bool = True,
    ) -> Dict[str, AgentResult]:
        if not target_categories:
            use_category = False
        else:
            if target_categories and use_category == False:
                raise ValueError("Cannot specify target_categories when use_category is False.")

        if not use_category:
            start_time = time.time()
            try:
                result = self._process_without_category(
                    traj_base_dir, verifier_prompt, mode,
                )
            except Exception as e:
                print(f"  [skill manager] Error: {e}")
                result = AgentResult(error=str(e))

            prediction = ""
            conversation = result.conversation if result.conversation else []
            turns = result.turns if result.turns else 0
            token_usage = result.metadata.get("token_usage", {}) if result.metadata else {}
            import random
            index = random.randint(0, 1000000)  # Generate a random index for the filename
            self._save_prediction(prediction, {"index": index}, turns, conversation=conversation, token_usage=token_usage)
                         
            total_time = time.time() - start_time
            status = "error" if result.error else "ok"
            preview = (result.raw_response or "")[:120].replace("\n", " ")
            return {"all": result}

        start_time = time.time()

        if self.parallel_workers and self.parallel_workers > 1:
            all_results = self._run_parallel(
                target_categories, traj_base_dir, verifier_prompt, mode, show_progress,
            )
        else:
            all_results = self._run_sequential(
                target_categories, traj_base_dir, verifier_prompt, mode, show_progress,
            )

        total_time = time.time() - start_time

        for cat, res in sorted(all_results.items()):
            prediction = ""
            conversation = res.conversation if res.conversation else []
            turns = res.turns if res.turns else 0
            token_usage = res.metadata.get("token_usage", {}) if res.metadata else {}
            import random
            index = random.randint(0, 1000000)  # Generate a random index
            self._save_prediction(prediction, {"index": index, "category": cat}, turns, filename_prefix=f"prediction_{cat}", conversation=conversation, token_usage=token_usage)

        return all_results
    
    def _save_prediction(self, prediction: str, extra_info: dict, turns: int, filename_prefix: str = "prediction", conversation: Optional[List] = None, token_usage: Optional[Dict] = None):
        """Save prediction to output directory."""
        index = extra_info.get("index", 0)
        category = extra_info.get("category", None)
        if category is not None:
            # Include category in filename: prediction_i_category.json
            filename = f"{filename_prefix}_{index}_category_{category}.json"
        else:
            # Original format: prediction_i.json
            filename = f"{filename_prefix}_{index}.json"

        prediction_data = {
            "prediction": prediction,
            "ground_truth": extra_info.get("ground_truth", ""),
            "query": extra_info.get("query", ""),
            "turns": turns,
            "conversation": conversation,
            "extra_info": extra_info,
            "token_usage": token_usage,
        }

        filepath = os.path.join(self.current_skill_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2, ensure_ascii=False)