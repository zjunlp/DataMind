"""
Shared utilities for skill loading and query-category filtering across datasets.
"""

import os
import re
from typing import Any, Dict, List, Optional, Set


def load_skill_content(skills_base_dir: str) -> str:
    skill_file = os.path.join(skills_base_dir, "SKILL.md")
    if not os.path.exists(skill_file):
        raise FileNotFoundError(f"Skill file not found: {skill_file}")
    with open(skill_file) as f:
        content = f.read()
    # Remove leading YAML frontmatter block (--- ... ---)
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    return content.strip()


def build_skill_content(
    skills_base_dir: str,
) -> str:
    """
    Return combined skill text for *skill_categories*.

    Falls back to *default_skill* when *skill_categories* is None or empty.
    Multiple categories are concatenated with a blank line between them.
    """

    skill_content = load_skill_content(skills_base_dir)
    if not skill_content:
        raise ValueError(f"No skill content found in {skills_base_dir}")
    return skill_content.strip()


def filter_by_query_categories(
    tasks: List[Dict[str, Any]],
    query_category_file: str,
    categories: List[str],
    question_key: str = "question",
    category_question_key: str = "question",
) -> List[Dict[str, Any]]:
    """
    Keep only tasks whose *question_key* value appears in the given *categories*
    within *query_category_file*.

    Args:
        tasks: Raw task list to filter.
        query_category_file: Path to the JSON category index file.
        categories: Category names to include.
        question_key: Key in each task dict that holds the question text.
        category_question_key: Key inside each category entry that holds the question text.
    """
    import json

    with open(query_category_file) as f:
        category_data: Dict[str, List[Dict[str, Any]]] = json.load(f)

    allowed: Set[str] = set()
    for cat in categories:
        for entry in category_data.get(cat, []):
            allowed.add(entry[category_question_key])

    return [t for t in tasks if t.get(question_key, "") in allowed]
