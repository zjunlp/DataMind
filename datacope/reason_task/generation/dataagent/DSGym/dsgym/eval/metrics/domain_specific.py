"""
Domain-specific evaluation metrics for different datasets.
"""

import json
import re
import ast
from typing import Optional, Dict, Any, List, Tuple
from .base import BaseMetric, MetricResult

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


# DiscoveryBenchMetric removed - replaced by LLMScoreMetric and HMSScoreMetric


class DABStepMetric(BaseMetric):
    """
    Domain-specific metric for DABStep evaluation.
    """
    
    @property
    def name(self) -> str:
        return "dabstep"
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate DABStep prediction.
        Currently uses exact match, can be enhanced with domain logic.
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        # Simple exact match for now
        # TODO: Implement DABStep-specific evaluation logic
        normalized_pred = prediction.strip().lower()
        normalized_truth = ground_truth.strip().lower()
        
        score = 1.0 if normalized_pred == normalized_truth else 0.0
        
        details = {
            "prediction_normalized": normalized_pred,
            "ground_truth_normalized": normalized_truth,
            "exact_match": score == 1.0,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )


# QRDataMetric removed - QRData uses exact_match metric




class LLMScoreMetric(BaseMetric):
    """
    LLM-based scoring metric for DiscoveryBench evaluation.
    Based on discovery_llm_judge.py implementation.
    """
    
    def __init__(self, model: str = "gpt-4o", **kwargs):
        """
        Initialize LLM score metric.
        
        Args:
            model: LLM model to use for evaluation
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.model = model
        
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LLMScoreMetric. Install with: pip install litellm")
    
    @property
    def name(self) -> str:
        return "llm_score"
    
    def _extract_answer(self, solution_str: str) -> float:
        """Extract answer from LLM response."""
        if "## The final answer is:" not in solution_str:
            return 0.0
            
        answer_str = solution_str.split("## The final answer is:")[1].strip()
        if answer_str == "True":
            return 1.0
        elif answer_str == "False":
            return 0.0
        else:
            return 0.0
    
    def _compute_llm_score_discrete(self, solution_str: str, ground_truth: str, query: str) -> float:
        """Compute LLM-based score for a prediction."""
        if solution_str is None:
            return 0.0
            
        solution_str = str(solution_str)
        ground_truth = str(ground_truth)

        prompt = f"""Please judge whether the generated answer is right or wrong.
Query: {query}

Predicted answer: {solution_str}

True answer: {ground_truth}

Rules for judgment::
- If the answer is numerical, treat it as correct if the relative error < 1% compared with the ground-truth value.
- Otherwise, judge correctness against the provided ground-truth answer. 
- The answer should be clear and complete.
-  Calculation process alone is not considered correct.


Please reply in this format:

Thoughts: <your thoughts here>

## The final answer is: <Output only True or False>
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            score_str = self._extract_answer(response.choices[0].message.content)
            return max(0.0, min(1.0, score_str))
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return 0.0
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate prediction using LLM-based scoring.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query
            **kwargs: Additional context
            
        Returns:
            MetricResult with LLM score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        if query is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No query available for LLM evaluation"}
            )
        
        # Compute LLM score
        score = self._compute_llm_score_discrete(prediction, ground_truth, query)
        
        details = {
            "model_used": self.model,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "query": query,
            "llm_score": score,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )


class HMSScoreMetric(BaseMetric):
    """
    HMS (Hypothesis Matching Score) metric for DiscoveryBench evaluation.
    Based on discovery_hms_score.py implementation.
    """
    
    def __init__(self, model: str = "gpt-4o", **kwargs):
        """
        Initialize HMS score metric.
        
        Args:
            model: LLM model to use for evaluation
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.model = model
        
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for HMSScoreMetric. Install with: pip install litellm")
    
    @property
    def name(self) -> str:
        return "hms_score"
    
    def _get_completion_kwargs(self, json_response: bool = False) -> dict:
        """Get appropriate completion kwargs based on the LLM provider."""
        kwargs = {}
        
        if json_response:
            # For OpenAI models (including via together_ai with OpenAI-compatible format)
            if any(prefix in self.model.lower() for prefix in ['gpt-', 'openai/', 'together_ai/']) or 'gpt' in self.model.lower():
                kwargs['response_format'] = {"type": "json_object"}
            else:
                # For other providers that might still use the old parameter
                kwargs['response_format'] = {"type": "json_object"}
        
        return kwargs
    
    def _prepare_dataset_metadata_json(self, dataset_meta):
        """Prepare dataset metadata in proper JSON format."""
        if dataset_meta == None:
            return [{
                "dataset_description":"",
                "columns": [],
            }]
        
        # If dataset_meta is a string, try to parse it as JSON
        if isinstance(dataset_meta, str):
            try:
                dataset_meta = json.loads(dataset_meta)
            except:
                return [{
                    "dataset_description": dataset_meta,
                    "columns": [],
                }]
        
        # If it's still not a dict after parsing, return default
        if not isinstance(dataset_meta, dict):
            return [{
                "dataset_description":"",
                "columns": [],
            }]
        
        datasets_json = []
        datasets_json.append(
            {
                "dataset_description": dataset_meta.get("dataset_descriptions", ""),
                "columns": dataset_meta.get("columns_info", []),
            }
        )
        return datasets_json
    
    def _get_sub_hypotheses(self, query: str, hypo: str, workflow: str, dataset_meta):
        """Extract sub-hypotheses from main hypothesis."""
        extraction_prompt = f"""\
Given a set of dataset columns, a ground-truth hypothesis, and the analysis workflow used, your task is to extract three dimensions that define the hypothesis: Context, Variables, and Relations. \
Here are the definitions for these dimensions:
- Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., "for men over \
the age of 30", "in Asia and Europe". If the context applies to the full dataset, then extract the context from the dataset_descrption.
- Variables: Known concepts that interact in a meaningful way under a given context to \
produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
- Relations: Interactions between a given set of variables under a given context to produce \
the hypothesis. E.g., "quadratic relationship", "inversely proportional", piecewise conditionals, \
or "None" if there is no interacting relationship.
Make sure to only use the information present in the hypothesis and the workflow. Do not add any new information. \
For each dimension, be specific, and do not omit any important details.

Here is the metadata for the task:
```json
{{
"datasets": %s,
"hypothesis": "%s",
"workflow": "%s"
}}
```

Return your answer as a JSON object in the following format:
```json
{{
"sub_hypo": [
    {{
        "text": the hypothesis in natural language,
        "context": a short text description of the context of the hypothesis,
        "variables": a list of columns involved in the hypothesis,
        "relations": a short text description of the relationship between the variables of the hypothesis
    }},
    ...
]
}}```
"""
        datasets_json = self._prepare_dataset_metadata_json(dataset_meta)
        _prompt = extraction_prompt % (datasets_json, hypo, workflow)
        completion_kwargs = self._get_completion_kwargs(json_response=True)
        
        try:
            response = completion(
                messages=[{"role": "user", "content": _prompt}], 
                model=self.model, 
                temperature=0, 
                **completion_kwargs
            )

            if response != None:
                content = response.choices[0].message.content
                content = content.strip() if content is not None else ""
                
                # Extract JSON from markdown code block if present
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, content, re.DOTALL)
                
                if json_match:
                    content = json_match.group(1).strip()
                elif content.startswith('```json'):
                    # Fallback: find the closing ```
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        content = content[start_idx:end_idx+1]
                elif content.startswith('```'):
                    content = content[3:-3].strip()
                
                try:
                    sub_hypo_json = json.loads(content)
                except Exception as e:
                    print(f"Error parsing JSON: {e}, content: {content}")
                    # Try to find JSON object within the content
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        try:
                            json_content = content[start_idx:end_idx+1]
                            sub_hypo_json = json.loads(json_content)
                        except:
                            sub_hypo_json = {"sub_hypo": []}
                    else:
                        sub_hypo_json = {"sub_hypo": []}
            else:
                sub_hypo_json = {"sub_hypo": []}
        except Exception as e:
            print(f"HMS evaluation error: {e}")
            sub_hypo_json = {"sub_hypo": []}

        sub_hypo_json['full_hypo'] = hypo
        return sub_hypo_json
    
    def _get_score_from_answer(self, answer_type: str, answer: str):
        """Parse answer based on HMS format (from original implementation)."""
        if answer_type == "context":
            answer = answer.replace("Answer:", "").strip()
            if answer.startswith("A)"):
                return 1.0
            elif answer.startswith("B)"):
                return 0.0
            return -1.0

        elif answer_type == "var":
            try:
                # Extract JSON from markdown code block if present
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, answer, re.DOTALL)
                
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    # Try to find JSON object within the content
                    start_idx = answer.find('{')
                    end_idx = answer.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = answer[start_idx:end_idx+1]
                    else:
                        json_content = answer
                
                var_json = json.loads(json_content)
                p = 0.0
                r = 0.0
                f1 = 0.0
                if var_json.get('sizeB'):
                    p = var_json['intersection']/var_json['sizeB']
                if var_json.get('sizeA'):
                    r = var_json['intersection']/var_json['sizeA']
                if p > 0.0 and r > 0.0:
                    f1 = (2 * p * r)/(p + r)
                else:
                    f1 = 0.0
                
                return {
                    "p": p,
                    "r": r,
                    "f1": f1,
                    "sizeA": var_json.get('sizeA', 0),
                    "sizeB": var_json.get('sizeB', 0),
                    "intersection": var_json.get('intersection', 0),
                    "explanation": var_json.get('explanation', ''),
                }
            except Exception as e:
                print(f"Error parsing var JSON: {e}, content: {answer}")
                return {
                    "p": -1.0,
                    "r": -1.0,
                    "f1": -1.0
                }
        elif answer_type == "rel":
            try:
                # Extract JSON from markdown code block if present
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, answer, re.DOTALL)
                
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    # Try to find JSON object within the content
                    start_idx = answer.find('{')
                    end_idx = answer.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = answer[start_idx:end_idx+1]
                    else:
                        json_content = answer
                
                rel_json = json.loads(json_content)
                answer_str = rel_json["answer"].strip() if rel_json.get("answer") is not None else ""
                if answer_str.startswith("A") or "very similar" in answer_str:
                    return 1.0
                elif answer_str.startswith("B") or "similar but general than HypoA" in answer_str:
                    return 0.5
                elif answer_str.startswith("C") or "different" in answer_str:
                    return 0.0
                return -1.0
            except Exception as e:
                print(f"Error parsing rel JSON: {e}, content: {answer}")
                return -1.0
        return -1.0

    def _ask_dimension_question(self, query, gold_hypo, gold_workflow, gen_hypo, gen_workflow, dataset_meta, dimension):
        """Ask dimension-specific questions for HMS evaluation."""
        dimension_question = ""
        answer = ""
        score = 0.0
        if dimension == "var":
            score = {
                "p": -1.0,
                "r": -1.0,
                "f1": -1.0
            }
        num_tokens = 256
        num_retries = 1
        json_response = False

        messages = [
            {"role": "system",
             "content": "You are an AI assistant that helps evaluate a data-driven hypothesis. You are a helpful assistant who is not talkative. You only respond with the exact answer to a query without additional conversation."
             },
        ]
        
        if dimension == "context":
            dimension_question = """\
Question: Is HypoB defined in the same context as HypoA?
(Context refers to assumptions/stratification under which the hypotheses are defined.)
Options: A) same   B) different
What is your answer?"""
        elif dimension == "var":
            dimension_question = """\
Question: For both HypoA and HypoB, what are the different variables found in the hypotheses? \
Return your answer as a JSON object in the following format:
```json
{{
"sizeA": num of variables used in HypoA
"sizeB": num of variables used in HypoB
"intersection": num of variables common in HypoA and HypoB. Use *fuzzy matching* to determine intersection, accounting for paraphrases or slightly different surface forms
"explanation": a short text explanation about the variables
}}```
Answer:"""
            num_tokens = 512
            num_retries = 1
            json_response = True
        elif dimension == "rel":
            dimension_question = """\
Question: Does HypoB exhibit the same relation as HypoA?
Compare using following example hierarchy of relationships (based on specificity): \
"there exists a relationship" > "positive relationship" > "positive AND (linear OR quadratic)" > "positive AND linear".
Options: A) very similar B) similar but general than HypoA C) different
Return your answer as a JSON object in the following format:
```json
{{
"answer": one of the options from A) very similar B) similar but general than HypoA C) different
"explanation": a short text explanation about the relationship comparison
}}```
Answer:"""
            num_tokens = 512
            num_retries = 1
            json_response = True

        datasets_json = self._prepare_dataset_metadata_json(dataset_meta)

        dimension_question_str = f"""\
You are going to compare two natural-language hypotheses HypoA and HypoB accompanied with optional workflows: WorkflowA for HypoA and WorkflowB for HypoB. \
Both the hypotheses answer the natural language query "QUERY" over the dataset(s) described by dataset description(s) and column description(s) below. \
Compare HypoA and HypoB in terms of three aspects: Contexts, Variables, and Relations. \
E.g., for the hypothesis "From 1995 to 2009, the number of sandhill cranes around the tundra (Indigilka River) surged by an astounding ~10X":
* Contexts refer to stratification of the data under which the given hypothesis is True. E.g., "For all women", "From 1995 to 2009".
* Variables refer to the set of variables (either dependent or independent) that are mentioned in the hypothesis. E.g., number of sandhill cranes, location.
* Relations refer to the form of relation between the variables. E.g., "surged by ~10x".

Answer following questions for a given pair of hypotheses, HypoA and HypoB, along with an explanation grounded on the QUERY and the DATASET(S).

Here is the metadata for the task:
```json
{{
"datasets": {datasets_json},
"query": {query},
"HypoA": {gold_hypo},
"WorkflowA": {gold_workflow},
"HypoB": {gen_hypo},
"WorkflowB": {gen_workflow}
}}
```

{dimension_question}"""

        messages.append(
            {"role": "user",
             "content": dimension_question_str
             }
        )
        
        for retry in range(num_retries):
            completion_kwargs = self._get_completion_kwargs(json_response)
            try:
                response = completion(
                        messages=messages,
                        model=self.model,
                        max_tokens=num_tokens,
                        temperature=0,  # 0 for greedy best decoding
                        **completion_kwargs
                )
                if response != None:
                    break
            except Exception as e:
                print(f"Error in dimension question: {e}")
                response = None

        if response != None:
            content = response.choices[0].message.content
            answer = content.strip() if content is not None else ""
            score = self._get_score_from_answer(answer_type=dimension, answer=answer)

        return dimension_question, answer, score

    def _match_context_with_gpt(self, gold_hyp, gold_context, pred_hyp, pred_context):
        """Match context using GPT."""
        prompt = f"""\
Given a gold hypothesis, a gold context, a predicted hypothesis, and a predicted context, your task is \
to determine if the predicted context semantically matches the ground-truth context. \
Here is the definition for Context: Boundary conditions that limit the scope of a sub-hypothesis. E.g., "for men over the age of 30", "in Asia and Europe". If the context applies to the full dataset, then the context is derived from the dataset_descrption. \
If the predicted context matches the gold context, return true, otherwise return false.
If both gold and predicted hypotheses are defined over the context of the full dataset, then also return true.

Here is the metadata for the task:
```json
{{
    "gold_hypothesis": "{gold_hyp}",
    "gold_context": "{gold_context}",
    "predicted_hypothesis": "{pred_hyp}",
    "predicted_context": "{pred_context}"
}}
```

Return your answer as a JSON object in the following format:
```json
{{
    "match": true or false
}}
```"""

        completion_kwargs = self._get_completion_kwargs(json_response=True)
        try:
            output = completion(messages=[{"role": "user", "content": prompt}], model=self.model, temperature=0, **completion_kwargs)
            
            content = output.choices[0].message.content
            content = content.strip() if content is not None else ""
            
            # Extract JSON from markdown code block if present
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                content = json_match.group(1).strip()
            elif content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            # Try to find JSON object within the content if parsing fails
            try:
                result = json.loads(content)
            except:
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_content = content[start_idx:end_idx+1]
                    result = json.loads(json_content)
                else:
                    return False
                    
            return result.get("match", False)
        except Exception as e:
            print(f"Error parsing match context JSON: {e}")
            return False

    def _is_matching_context(self, gold_hyp, gold_context, pred_hyp, pred_context):
        """Check if contexts match."""
        if gold_context == pred_context:
            return True
        if "None" in [gold_context, pred_context]:
            return False
        return self._match_context_with_gpt(gold_hyp, gold_context, pred_hyp, pred_context)

    def _run_eval_gold_vs_gen_NL_subhypo(self, query, gold_hypo, gold_workflow, gen_hypo, gen_workflow, dataset_meta, context_score):
        """Run evaluation for sub-hypotheses."""
        eval_rec = {
            "query": query,
            "HypoA": gold_hypo,
            "WorkflowA": gold_workflow,
            "HypoB": gen_hypo,
            "WorkflowB": gen_workflow,
        }

        for dimension in ['var', 'rel']:
            question, answer, score = self._ask_dimension_question(query, gold_hypo, gold_workflow,
                                   gen_hypo, gen_workflow, dataset_meta, dimension)

            eval_rec[dimension] = {
                "question": question,
                "answer": answer,
                "score": score
            }

        eval_rec['context'] = context_score
        eval_rec['accuracy_score'] = 1.0 * eval_rec['context']['score'] * eval_rec['var']['score']['f1'] * eval_rec['rel']['score']

        return eval_rec

    def _compute_hms_score(self, solution_str: str, ground_truth: str, query: str, metadata):
        """Compute HMS score using the complete original algorithm."""
        try:
            # Input: Dataset Metadata, Query, Gold {Hg, Wg}, Predicted {Hp, Wp}
            # Output: eval_rec json includes final_score
            eval_rec = {
                "query": query,
                "HypoA": ground_truth,
                "WorkflowA": "",
                "HypoB": solution_str,
                "WorkflowB": "",
            }

            gold_sub_hypo_json = self._get_sub_hypotheses(query=query,
                                           hypo=ground_truth, workflow="",
                                           dataset_meta=metadata)
            if len(gold_sub_hypo_json['sub_hypo']) == 0:
                gold_sub_hypo_json['sub_hypo'] = [{"text": ground_truth, "context": "None", "variables": [], "relations": "", "explanation": "unable to segment"}]

            gen_sub_hypo_json = self._get_sub_hypotheses(query=query,
                                           hypo=solution_str, workflow="",
                                           dataset_meta=metadata)
            if len(gen_sub_hypo_json['sub_hypo']) == 0:
                gen_sub_hypo_json['sub_hypo'] = [{"text": solution_str, "context": "None", "variables": [], "relations": "", "explanation": "unable to segment"}]

            eval_rec['gold_sub_hypo'] = gold_sub_hypo_json
            eval_rec['gen_sub_hypo'] = gen_sub_hypo_json

            gold_subh_covered = []
            gen_subh_to_gold_subh = dict()
            gen_gold_subh_to_context = dict()

            for p_id, gen_subh in enumerate(gen_sub_hypo_json['sub_hypo']):
                gen_subh_to_gold_subh[p_id] = -1

                for g_id, gold_subh in enumerate(gold_sub_hypo_json['sub_hypo']):
                    if g_id in gold_subh_covered:
                        continue

                    # match context
                    context_bool = self._is_matching_context(gold_subh["text"], gold_subh.get("context", ""), gen_subh["text"], gen_subh.get("context", ""))
                    if context_bool:
                        context_score = 1.0
                    else:
                        context_score = 0.0

                    if context_score == 1.0: # match only when context_score = 1.0
                        gen_subh_to_gold_subh[p_id] = g_id
                        gold_subh_covered.append(g_id)
                        gen_gold_subh_to_context[f"P{p_id}||G{g_id}"] = {
                            "question": f"""Comparing: GoldH: {gold_subh["text"]}, GoldC: {gold_subh['context']}\nGenH: {gen_subh['text']}, GenC: {gen_subh['context']}""",
                            "answer": context_bool,
                            "score": context_score
                        }
                        break

            eval_rec['gen_subh_to_gold_subh'] = gen_subh_to_gold_subh
            eval_rec['gold_subh_covered'] = gold_subh_covered
            matched_gold_gen_subh_evals = dict()
            sum_accuracy_score = 0.0
            
            for p_id, g_id in gen_subh_to_gold_subh.items():
                if g_id >= 0:
                    key = f"P{p_id}||G{g_id}"
                    context_score = gen_gold_subh_to_context[key]
                    subh_eval_rec = self._run_eval_gold_vs_gen_NL_subhypo(query, ground_truth, "", solution_str, "", metadata, context_score)
                    sum_accuracy_score += subh_eval_rec['accuracy_score']
                    matched_gold_gen_subh_evals[key] = subh_eval_rec

            eval_rec['matched_gold_gen_subh_evals'] = matched_gold_gen_subh_evals
            eval_rec['recall_context'] = len(gold_subh_covered)/len(gold_sub_hypo_json['sub_hypo']) if len(gold_sub_hypo_json['sub_hypo']) else 0.0
            mean_accuracy_score = sum_accuracy_score/len(gen_subh_to_gold_subh) if len(gen_subh_to_gold_subh) else 0.0
            eval_rec['mean_accuracy_score'] = mean_accuracy_score
            final_score = eval_rec['recall_context'] * mean_accuracy_score
            eval_rec['final_score'] = final_score

            return final_score
            
        except Exception as e:
            print(f"HMS score computation error: {e}")
            return 0.0
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate prediction using HMS scoring.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query
            **kwargs: Additional context (should include 'metadata')
            
        Returns:
            MetricResult with HMS score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        if query is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No query available for HMS evaluation"}
            )
        
        # Get metadata from kwargs
        metadata = kwargs.get('metadata', {})
        
        # Compute HMS score
        score = self._compute_hms_score(prediction, ground_truth, query, metadata)
        
        details = {
            "model_used": self.model,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "query": query,
            "metadata": metadata,
            "hms_score": score,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )