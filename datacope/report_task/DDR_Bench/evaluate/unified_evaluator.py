#!/usr/bin/env python3
"""
Unified evaluator for all DDR_Bench scenarios.

Combines BaseEvaluator with scenario-specific prompts.
"""

import json
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

from .base_evaluator import BaseEvaluator
from .prompts import get_prompts, get_entity_prefix

logger = logging.getLogger(__name__)


class UnifiedEvaluator(BaseEvaluator):
    """Unified evaluator supporting all scenarios (MIMIC, 10-K, GLOBEM)."""
    
    def __init__(self, scenario: str = "mimic", **kwargs):
        entity_prefix = get_entity_prefix(scenario)
        super().__init__(scenario=scenario, entity_prefix=entity_prefix, **kwargs)
        
        self.prompts = get_prompts(scenario)
        self.use_qa_accuracy = self.prompts.get("use_qa_accuracy", False)
        self._worker_init_kwargs = dict(kwargs)
    
    def evaluate_qa_with_message_list(self, qa_pair: Dict, insights_list: List[str], filename: str = "unknown") -> Dict[str, Any]:
        """Evaluate QA with individual insights (message-wise context)."""
        question = qa_pair.get("question", "")
        ground_truth = qa_pair.get("answer", "")
        source_text = qa_pair.get("source_text", "")
        
        if not question or not ground_truth:
            return {"error": "Invalid QA pair"}
        
        if not insights_list:
            if self.use_qa_accuracy:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "model_answer": "NO IDEA",
                    "answer_type": "NO_IDEA",
                    "context_type": "message_wise",
                    "source_text": source_text,
                    "supporting_message_indices": [],
                    "contradicting_message_indices": []
                }
            else:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "context_quality": "INSUFFICIENT_INFO",
                    "evidence_text": "",
                    "reasoning": "No insights available",
                    "context_type": "message_wise",
                    "source_text": source_text,
                    "supporting_message_indices": [],
                    "contradicting_message_indices": []
                }
        
        # Build numbered context
        numbered_context = ""
        for idx, insight in enumerate(insights_list):
            numbered_context += f"[Message {idx}]: {insight}\n\n"
        
        # Create prompt
        messages = [
            {"role": "system", "content": self.prompts["message_system"]},
            {"role": "user", "content": self.prompts["message_user"].format(
                numbered_context=numbered_context,
                question=question,
                ground_truth=ground_truth
            )}
        ]
        
        # Call LLM
        result = self.call_llm_api(messages, max_tokens=2048)
        
        if not result:
            logger.warning(f"Failed to get evaluation for question: {question[:100]}...")
            if self.use_qa_accuracy:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "model_answer": "",
                    "answer_type": "API_FAILED",
                    "is_correct": False,
                    "context_type": "message_wise",
                    "source_text": source_text,
                    "supporting_message_indices": [],
                    "contradicting_message_indices": [],
                    "error": "Failed to get evaluation"
                }
            else:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "context_quality": "API_FAILED",
                    "evidence_text": "",
                    "reasoning": "Failed to get evaluation from API",
                    "context_type": "message_wise",
                    "source_text": source_text,
                    "supporting_message_indices": [],
                    "contradicting_message_indices": [],
                    "error": "Failed to get evaluation"
                }
        
        # Parse result based on evaluation type
        if self.use_qa_accuracy:
            return self._parse_qa_accuracy_result(result, question, ground_truth, source_text, "message_wise")
        else:
            return self._parse_context_quality_result(result, question, ground_truth, source_text, "message_wise")
    
    def evaluate_qa_with_context(self, qa_pair: Dict, context: str, context_type: str, filename: str = "unknown") -> Dict[str, Any]:
        """Evaluate QA with full context (chat-wise)."""
        question = qa_pair.get("question", "")
        ground_truth = qa_pair.get("answer", "")
        source_text = qa_pair.get("source_text", "")
        
        if not question or not ground_truth:
            return {"error": "Invalid QA pair"}
        
        # Create prompt
        messages = [
            {"role": "system", "content": self.prompts["chat_system"]},
            {"role": "user", "content": self.prompts["chat_user"].format(
                context=context,
                question=question,
                ground_truth=ground_truth
            )}
        ]
        
        # Call LLM
        result = self.call_llm_api(messages, max_tokens=2048)
        
        if not result:
            logger.warning(f"Failed to get evaluation for question: {question[:100]}...")
            if self.use_qa_accuracy:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "model_answer": "",
                    "answer_type": "API_FAILED",
                    "is_correct": False,
                    "context_type": context_type,
                    "source_text": source_text,
                    "error": "Failed to get evaluation"
                }
            else:
                return {
                    "original_question": question,
                    "ground_truth": ground_truth,
                    "context_quality": "API_FAILED",
                    "evidence_text": "",
                    "reasoning": "Failed to get evaluation from API",
                    "context_type": context_type,
                    "source_text": source_text,
                    "error": "Failed to get evaluation"
                }
        
        # Parse result
        if self.use_qa_accuracy:
            return self._parse_qa_accuracy_result(result, question, ground_truth, source_text, context_type)
        else:
            return self._parse_context_quality_result(result, question, ground_truth, source_text, context_type)
    
    def _parse_context_quality_result(self, result: str, question: str, ground_truth: str, source_text: str, context_type: str) -> Dict[str, Any]:
        """Parse context quality evaluation result (MIMIC/10-K)."""
        result = result.replace("```json", "").replace("```", "")
        
        try:
            data = json.loads(result)
            return {
                "original_question": question,
                "ground_truth": ground_truth,
                "context_quality": data.get("context_quality", "INSUFFICIENT_INFO"),
                "supporting_message_indices": data.get("supporting_message_indices", []),
                "contradicting_message_indices": data.get("contradicting_message_indices", []),
                "evidence_text": data.get("evidence_text", ""),
                "reasoning": data.get("reasoning", ""),
                "context_type": context_type,
                "source_text": source_text
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse result: {e}")
            # Fallback
            upper = result.upper()
            if "CORRECT_INFO" in upper:
                quality = "CORRECT_INFO"
            elif "INCORRECT_INFO" in upper:
                quality = "INCORRECT_INFO"
            else:
                quality = "INSUFFICIENT_INFO"
            
            return {
                "original_question": question,
                "ground_truth": ground_truth,
                "context_quality": quality,
                "supporting_message_indices": [],
                "contradicting_message_indices": [],
                "evidence_text": "",
                "reasoning": result,
                "context_type": context_type,
                "source_text": source_text
            }
    
    def _parse_qa_accuracy_result(self, result: str, question: str, ground_truth: str, source_text: str, context_type: str) -> Dict[str, Any]:
        """Parse QA accuracy evaluation result (GLOBEM)."""
        model_answer = result
        
        # Extract supporting indices
        supporting_indices = []
        indices_match = re.search(r'[Uu]sed messages?:?\s*([\d,\s]+)', result)
        if indices_match:
            indices_str = indices_match.group(1)
            supporting_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
        else:
            message_refs = re.findall(r'\[Message (\d+)\]', result)
            if message_refs:
                supporting_indices = [int(x) for x in message_refs]
        
        # Check for NO IDEA
        if "NO IDEA" in result.upper():
            return {
                "original_question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "answer_type": "NO_IDEA",
                "context_type": context_type,
                "source_text": source_text,
                "supporting_message_indices": supporting_indices,
                "contradicting_message_indices": []
            }
        
        # Check accuracy
        accuracy_messages = [
            {"role": "system", "content": self.prompts["accuracy_system"]},
            {"role": "user", "content": self.prompts["accuracy_user"].format(
                ground_truth=ground_truth,
                model_answer=model_answer
            )}
        ]
        
        accuracy_result = self.call_llm_api(accuracy_messages)
        is_correct = accuracy_result.strip().upper() == "YES" if accuracy_result else False
        
        return {
            "original_question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "answer_type": "CORRECT" if is_correct else "INCORRECT",
            "is_correct": is_correct,
            "context_type": context_type,
            "source_text": source_text,
            "supporting_message_indices": supporting_indices if is_correct else [],
            "contradicting_message_indices": supporting_indices if not is_correct else []
        }
    
    def evaluate_entity(self, entity_id: str, qa_pairs: List[Dict], message_wise_context: str, chat_wise_context: str, message_wise_insights_list: List[str], filename: str = "unknown") -> Dict[str, Any]:
        """Evaluate all QA pairs for an entity."""
        logger.info(f"Evaluating {self.entity_prefix} {entity_id} with {len(qa_pairs)} QA pairs")
        
        results = {
            "entity_id": entity_id,
            "total_qa_pairs": len(qa_pairs),
            "message_wise_context_results": [],
            "chat_wise_context_results": [],
            "summary": {}
        }
        
        # Message-wise evaluation
        if message_wise_insights_list:
            logger.info(f"Evaluating with message-wise context ({len(message_wise_insights_list)} messages)")
            for qa_pair in qa_pairs:
                result = self.evaluate_qa_with_message_list(qa_pair, message_wise_insights_list, filename)
                results["message_wise_context_results"].append(result)
                time.sleep(0.5)
        
        # Chat-wise evaluation
        if chat_wise_context:
            logger.info(f"Evaluating with chat-wise context")
            for qa_pair in qa_pairs:
                result = self.evaluate_qa_with_context(qa_pair, chat_wise_context, "chat_wise", filename)
                results["chat_wise_context_results"].append(result)
                time.sleep(0.5)
        
        results["summary"] = self.calculate_summary_stats(results, self.use_qa_accuracy)
        return results

    def _create_worker_evaluator(self) -> "UnifiedEvaluator":
        """Create an isolated evaluator instance for parallel execution."""
        return UnifiedEvaluator(scenario=self.scenario, **self._worker_init_kwargs)

    def _evaluate_entity_job(
        self,
        entity_id: str,
        qa_pairs: List[Dict],
        message_wise_context: str,
        chat_wise_context: str,
        message_wise_insights_list: List[str],
        filename: str
    ) -> Dict[str, Any]:
        """Evaluate a single entity in a worker thread."""
        worker = self._create_worker_evaluator()
        return worker.evaluate_entity(
            entity_id,
            qa_pairs,
            message_wise_context,
            chat_wise_context,
            message_wise_insights_list,
            filename
        )

    def _log_entity_completion(self, entity_id: str, entity_results: Dict[str, Any]) -> None:
        """Log a concise entity evaluation summary."""
        summary = entity_results.get("summary", {})
        msg_stats = summary.get("message_wise_context", {})
        chat_stats = summary.get("chat_wise_context", {})

        msg_summary = f"Message-wise: {msg_stats.get('correct_percentage', 0):.1f}% correct" if msg_stats else "Message-wise: N/A"
        chat_summary = f"Chat-wise: {chat_stats.get('correct_percentage', 0):.1f}% correct" if chat_stats else "Chat-wise: N/A"
        logger.info(f"Completed {entity_id} - {msg_summary} | {chat_summary}")
    
    def run_evaluation(
        self,
        qa_file: str,
        logs_dir: str,
        output_file: str,
        test_mode: bool = False,
        max_workers: int = 1
    ):
        """Run the complete evaluation."""
        logger.info(f"Starting {self.scenario.upper()} evaluation")
        if test_mode:
            logger.info("TEST MODE: Only evaluating first entity")
        logger.info(f"MAX WORKERS: {max_workers}")
        
        # Load data
        entity_qa_map = self.load_qa_data(qa_file)
        entity_logs_map = self.load_logs_data(logs_dir)
        
        if not entity_qa_map:
            logger.error("No QA data loaded")
            return
        
        if not entity_logs_map:
            logger.error("No logs data loaded")
            return
        
        # Find common entities
        common_entities = sorted(set(entity_qa_map.keys()) & set(entity_logs_map.keys()))
        logger.info(f"Found {len(common_entities)} common entities")

        if not common_entities:
            logger.error("No common entities available for evaluation")
            return
        
        # Prepare evaluation jobs
        total = 1 if test_mode else len(common_entities)
        entity_jobs = []

        for i, entity_id in enumerate(common_entities, 1):
            if test_mode and i > 1:
                break
            
            entity_logs_data = entity_logs_map[entity_id]
            qa_pairs = entity_qa_map[entity_id]
            
            insights_file = entity_logs_data.get("insights_file", "unknown")
            session_stats_file = entity_logs_data.get("session_stats_file", "unknown")
            filename_info = f"insights:{insights_file}, session_stats:{session_stats_file}"
            
            message_wise_context, chat_wise_context, message_wise_insights_list = self.extract_contexts(entity_logs_data)
            
            if not message_wise_context and not chat_wise_context:
                logger.warning(f"Entity {entity_id} has no valid context")
                continue
            
            entity_jobs.append((
                i,
                entity_id,
                qa_pairs,
                message_wise_context,
                chat_wise_context,
                message_wise_insights_list,
                filename_info
            ))

        # Run evaluation
        all_results = []
        normalized_max_workers = max(1, int(max_workers or 1))

        if normalized_max_workers == 1 or len(entity_jobs) <= 1:
            for i, entity_id, qa_pairs, message_wise_context, chat_wise_context, message_wise_insights_list, filename_info in entity_jobs:
                logger.info(f"Processing {entity_id} ({i}/{total})")

                entity_results = self.evaluate_entity(
                    entity_id, qa_pairs,
                    message_wise_context, chat_wise_context,
                    message_wise_insights_list, filename_info
                )
                all_results.append(entity_results)
                self._log_entity_completion(entity_id, entity_results)
        else:
            worker_count = min(normalized_max_workers, len(entity_jobs))
            ordered_entity_ids = [job[1] for job in entity_jobs]
            results_by_entity = {}

            logger.info(f"Running evaluation with up to {worker_count} parallel workers")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_entity = {}
                for i, entity_id, qa_pairs, message_wise_context, chat_wise_context, message_wise_insights_list, filename_info in entity_jobs:
                    logger.info(f"Queueing {entity_id} ({i}/{total})")
                    future = executor.submit(
                        self._evaluate_entity_job,
                        entity_id,
                        qa_pairs,
                        message_wise_context,
                        chat_wise_context,
                        message_wise_insights_list,
                        filename_info
                    )
                    future_to_entity[future] = entity_id

                for future in as_completed(future_to_entity):
                    entity_id = future_to_entity[future]
                    try:
                        entity_results = future.result()
                    except Exception as e:
                        logger.error(f"Evaluation failed for {entity_id}: {e}")
                        continue

                    results_by_entity[entity_id] = entity_results
                    self._log_entity_completion(entity_id, entity_results)

            all_results = [results_by_entity[entity_id] for entity_id in ordered_entity_ids if entity_id in results_by_entity]
        
        # Calculate statistics
        overall_stats = self.calculate_overall_stats(all_results, self.use_qa_accuracy)
        cumulative_stats = self.calculate_cumulative_message_stats(all_results, self.use_qa_accuracy)
        
        # Save results
        output_data = {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": self.scenario,
                "qa_file": qa_file,
                "logs_dir": logs_dir,
                "total_entities_evaluated": len(all_results),
                "test_mode": test_mode,
                "provider": self.provider,
                "max_workers": normalized_max_workers,
            },
            "overall_statistics": overall_stats,
            "cumulative_message_statistics": cumulative_stats,
            "entity_results": all_results
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved evaluation to {output_path}")
        self.print_summary(overall_stats, cumulative_stats)
