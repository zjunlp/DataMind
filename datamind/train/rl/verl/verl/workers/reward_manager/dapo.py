# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
from verl.utils.reward_score.reward_score_helper import process_single_item


@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        max_resp_len=None,
        config=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.config = config

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        pred_csv_result_dir_parent = self.config.get("pred_csv_result_dir_parent", None)
        gold_csv_results_dir = self.config.get("gold_csv_results_dir", None)

        sequences_str_lst = []
        ground_truth_lst = []
        data_source_lst = []
        extra_info_lst = []
        valid_response_len_lst = []
        pred_csv_result_dir_parent_lst = []
        gold_csv_results_dir_lst = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_len_lst.append(valid_response_length)

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequences_str_lst.append(sequences_str)

            ground_truth_lst.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
            data_source_lst.append(data_item.non_tensor_batch["data_source"])
            extra_info_lst.append(data_item.non_tensor_batch.get("extra_info", None))
            pred_csv_result_dir_parent_lst.append(pred_csv_result_dir_parent)
            gold_csv_results_dir_lst.append(gold_csv_results_dir)

        results = [None] * len(sequences_str_lst)
        with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            future_to_idx = {}
            for idx, (sequences, ground_truth, extra_info, data_source, valid_res_len) in enumerate(
                zip(sequences_str_lst, ground_truth_lst, extra_info_lst, data_source_lst, valid_response_len_lst)
            ):
                future = executor.submit(
                    process_single_item, sequences, ground_truth, extra_info, data_source, valid_res_len
                )
                future_to_idx[future] = idx
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=60)
                    results[idx] = result
                except Exception as e:
                    result = {
                        "total_score": -0.1,
                        "answer_score": -1.0,
                        "template_score": -1.0,
                        "accuracy": 0.0,
                        "valid_response_length": valid_response_len_lst[idx],
                        "tool_score": 0.0,
                        "prm_score": 0.0,
                        "reason": f"Worker {idx} failed: {e}"
                    }
                    results[idx] = result
                    print(f"[ERROR] Worker {idx} failed: {e}")

        for i, result in enumerate(results):
            if result is None:
                raise ValueError(f"Result for item {i} is None, indicating a processing error.")

            valid_response_length  = result["valid_response_length"]
            score = result["total_score"]
            # Store the information including original reward
            for key, value in result.items():
                reward_extra_info[key].append(value)

            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor