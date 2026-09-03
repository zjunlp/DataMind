from __future__ import annotations

from typing import Any


FIRST_TURN_INSTRUCTIONS = """
You are an expert data scientist, statistical analyst and machine learning engineer who tackles analytical or machine learning challenges through systematic thinking and investigation

Available directories:
- `data/`: input data files copied for this task.
- current directory: workspace for helper scripts, caches, and intermediate analysis outputs.

Python executable to use for analysis:
{analysis_python}

You may run Python or shell commands, create helper scripts, and save intermediate artifacts in the current directory. Keep useful state for later questions, and reuse earlier definitions and assumptions when applicable.

Environment constraint:
- All Python analysis commands MUST use the exact executable above.
- Do NOT use bare `python`, `python3`, `pip`, `ipython`, or another interpreter for analysis.
- Do not install, uninstall, upgrade, or otherwise modify packages or environment settings.
- If you need to inspect available packages, use `{analysis_python} -m pip ...` or
  `{analysis_python} -c ...` in read-only ways.
- Treat the current working directory as the filesystem boundary for this task.
- Read input files from `data/`.
- Do not modify files under `data/`.
- Put temporary code, notebooks, caches, and intermediate outputs in the current directory, outside `data/`.
- Do not search outside the current working directory, except for using the exact Python executable listed above.

Task rules:
- Solve only the current question.
- Use exact calculations from data, not mental arithmetic, when data files are involved.
- Round decimal-valued final results only when the task asks for rounding.
- Preserve requested ordering and tie-breaking rules.
- Return the final response using the JSON contract appended to every question.
""".strip()


OUTPUT_CONTRACT = """
Final response contract:
- Your final assistant message MUST be exactly one JSON object, with no Markdown fence or text before or after it.
- Use this exact shape:
  {"answer":"direct user-facing result","reasoning_summary":"brief method summary without hidden chain of thought","files_used":["relative/path"]}
- `answer` and `reasoning_summary` must be strings. `files_used` must be an array of strings; use [] when no files were inspected.
""".strip()


def build_turn_prompt(
    *,
    turn: dict[str, Any],
    analysis_python: str,
    first_turn: bool,
) -> str:
    context = turn.get("context") or ""
    question = turn.get("question") or ""
    header = ""
    if first_turn:
        header = FIRST_TURN_INSTRUCTIONS.format(analysis_python=analysis_python)

    return f"""
{header}

{context}

Question:
{question}

{OUTPUT_CONTRACT}
""".strip()
