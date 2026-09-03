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
""".strip()


OUTPUT_CONTRACT = """
Final response format:
- Your final message MUST be a single JSON object and nothing else: no prose before or after, no Markdown code fence.
- Required keys:
{fields}
- Do not add any other key.
""".strip()


SCHEMA_TYPE_NAMES = {
    "string": "string",
    "array": "array",
    "object": "object",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
}


def describe_schema_field(name: str, spec: dict[str, Any]) -> str:
    field_type = SCHEMA_TYPE_NAMES.get(str(spec.get("type")), str(spec.get("type")))
    if field_type == "array":
        item_type = SCHEMA_TYPE_NAMES.get(str((spec.get("items") or {}).get("type")), "")
        if item_type:
            field_type = f"array of {item_type}s"
    description = str(spec.get("description") or "").strip()
    return f'  - "{name}" ({field_type}): {description}'.rstrip(": ")


def build_output_contract(schema: dict[str, Any]) -> str:
    properties = schema.get("properties") or {}
    required = schema.get("required") or list(properties)
    fields = "\n".join(
        describe_schema_field(name, properties.get(name) or {}) for name in required
    )
    return OUTPUT_CONTRACT.format(fields=fields)


def build_turn_prompt(
    *,
    turn: dict[str, Any],
    analysis_python: str,
    first_turn: bool,
    output_contract: str,
) -> str:
    """Build one LongDS turn prompt.

    Qoder CLI has no server-side output schema flag, so the JSON output contract is
    repeated on every turn instead of being enforced by the CLI.
    """
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

{output_contract}
""".strip()
