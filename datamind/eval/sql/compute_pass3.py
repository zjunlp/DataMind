import json

def compute_pass3(eval_results_list):
    assert all(len(r) == len(eval_results_list[0]) for r in eval_results_list), "All results must have the same length."
    
    id_sets = [set(item["id"] for item in res) for res in eval_results_list]
    assert all(s == id_sets[0] for s in id_sets), "All results must have the same set of IDs."

    correct_id_set = set()
    for eval_results in eval_results_list:
        for item in eval_results:
            if item["answer_score"] == 1.0:
                correct_id_set.add(item["id"])
    
    return len(correct_id_set) / len(eval_results_list[0])

def compute_avg(eval_results_list):
    assert all(len(r) == len(eval_results_list[0]) for r in eval_results_list), "All results must have the same length."
    
    id_sets = [set(item["id"] for item in res) for res in eval_results_list]
    assert all(s == id_sets[0] for s in id_sets), "All results must have the same set of IDs."

    correct_id_list = []
    for eval_results in eval_results_list:
        for item in eval_results:
            if item["answer_score"] == 1.0:
                correct_id_list.append(item["id"])
    
    return len(correct_id_list) / (len(eval_results_list) * len(eval_results_list[0]))

if __name__ == "__main__":
    eval_bench = "bird"
    base_filename = "sql_datamind_traj_t0.7_topp0.95_bs5_bird_test"
    file_paths = [
        f"eval_result/{eval_bench}/{base_filename}_0.json",
        f"eval_result/{eval_bench}/{base_filename}_1.json",
        f"eval_result/{eval_bench}/{base_filename}_2.json"
    ]

    eval_results_list = []
    for path in file_paths:
        data = json.load(open(path, "r"))
        eval_results_list.append(data["result"])
    
    pass3 = compute_pass3(eval_results_list)
    avg = compute_avg(eval_results_list)
    print(f"pass@3: {pass3 * 100:.2f}%")
    print(f"avg: {avg * 100:.2f}%")