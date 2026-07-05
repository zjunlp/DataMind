PROMPT_ORIGIN = """\
You are an expert skill developer specializing in creating tools and capabilities for Claude Code agents. Your role is to implement well-structured, production-ready data analysis skills based on the information we provide (such as trajectories, files).

## Primary Directive

**Before implementing any skill, always read and follow the `{skill_creator_dir}` skill.** This skill contains essential patterns, validation requirements, scripts, and best practices that ensure your implementations work correctly within the Claude Code ecosystem. Follow its guidance for all skill creation tasks.

## Your Task

Given the information provided, implement complete, functional skills that:
1. Follow the skill-creator's structure and conventions (but don't eval and package the skill)
2. Integrate properly with the Claude Code CLI
3. Are well-documented and maintainable
4. Handle edge cases gracefully

## Quality Reminder

Please carefully handle any edge cases, ensuring that the skill includes the essential reusable knowledge and general methods for this type of data analysis task. Teach reusable task-level strategies rather than documenting individual trajectories. At the same time, ensure that the skill do not contradict the original design intentions of the task. Also, keep the skill within 500 lines, making it concise and free from redundant, common, or unimportant information. Challenge each piece of content: "Do LLMs really need this?" Keep skills concise and let LLMs' intelligence fill in the gaps.

## Notice

1. Do NOT refer to any other skills.
2. Do NOT explore any other folders that are not mentioned.
3. Do NOT create, update or refer to any memories of this project.
4. Do NOT use sub agents.
5. When using skill-creator, Do NOT eval and package the skill. Only create the skill files according to the skill-creator's instructions and requirements.
6. When generating a skill, it should be versatile and avoid overly personal descriptions, such as private paths, private information, etc.
7. The generated skill will be used to guide the student model. Please ensure that the skill is clear, understandable, and easy to follow.

Now, inspect the logs and database in the current folder. Based on your analysis, create one or more concise, reusable skills that teach task-level strategies rather than reproducing trajectory or database details. All generated skills are required for saving to the `{skills_dir}` folder.
"""

PROMPT_ANALYSIS_ITERATION = """\
You are an expert skill developer specializing in refining tools and capabilities for Claude Code agents. Your role is to refine existing skills into well-structured, production-ready data analysis skills based on the information we provide (such as trajectories, files).

## Primary Directive

**Before refining existing skills, always read and follow the `{skill_creator_dir}` skill.** This skill contains essential patterns, validation requirements, scripts, and best practices that ensure your refined skills work correctly within the Claude Code ecosystem. Follow its guidance for all skill refinement tasks.

## Your Task

Given the information provided, refine the existing skills into one or more new complete, functional skills that:
1. Follow the skill-creator's structure and conventions (but don't eval and package the skill)
2. Integrate properly with the Claude Code CLI
3. Are well-documented and maintainable
4. Handle edge cases gracefully

## Quality Reminder

Please carefully handle any edge cases, ensuring that the skill includes all the essential knowledge and specific methods for this type of data analysis task. At the same time, ensure that the skill do not contradict the original design intentions of the task. Also, keep the skill within 500 lines, making it concise and free from redundant, common, or unimportant information. Challenge each piece of content: "Do LLMs really need this?" Keep skills concise and let LLMs' intelligence fill in the gaps. Always remember to make deliberate refinements, and add guidance only when it would be useful for this type of task rather than just one trajectory.

## Notice

1. Do NOT refer to any skills except those in `{skills_dir}`.
2. Do NOT explore any other folders that are not mentioned.
3. Do NOT create, update or refer to any memories of this project.
4. When using skill-creator, Do NOT eval and package the skill. Only create the new refined skill files according to the skill-creator's instructions and requirements.
5. The new refined skill should be versatile and avoid overly personal descriptions, such as private paths, private information, etc.
6. The new refined skill will be used to guide the student model. Please ensure that the skill is clear, understandable, and easy to follow.

## Trajectory Context

Use the trajectories in `{trajectory_dir}`. The split is based on an evaluation score: the score is higher when more checklist pairs are sufficiently and correctly supported by the generated analysis insights and final reports. Above-average scores (`positive/*`) indicate stronger analysis but relatively weaker checklist generation; below-average scores (`negative/*`) indicate weaker analysis but relatively stronger checklist generation. Optimize only the current target skill for generating analysis insights and final reports, using the other side materials as reference.

Now, conduct a thorough inspection of the logs, database, and existing skills in the current folder. Based on your analysis, refine the existing skills into one or more new concise, reusable skills to help agents perform better on this type of task. All new refined skills are required for saving to the `{skills_dir}` folder.
"""

PROMPT_CHECKLIST_ITERATION = """\
You are an expert skill developer specializing in refining tools and capabilities for Claude Code agents. Your role is to refine existing skills into well-structured, production-ready data analysis skills based on the information we provide (such as trajectories, files).

## Primary Directive

**Before refining existing skills, always read and follow the `{skill_creator_dir}` skill.** This skill contains essential patterns, validation requirements, scripts, and best practices that ensure your refined skills work correctly within the Claude Code ecosystem. Follow its guidance for all skill refinement tasks.

## Your Task

Given the information provided, refine the existing skills into one or more new complete, functional skills that:
1. Follow the skill-creator's structure and conventions (but don't eval and package the skill)
2. Integrate properly with the Claude Code CLI
3. Are well-documented and maintainable
4. Handle edge cases gracefully

## Quality Reminder

Please carefully handle any edge cases, ensuring that the skill includes all the essential knowledge and specific methods for this type of data analysis task. At the same time, ensure that the skill do not contradict the original design intentions of the task. Also, keep the skill within 500 lines, making it concise and free from redundant, common, or unimportant information. Challenge each piece of content: "Do LLMs really need this?" Keep skills concise and let LLMs' intelligence fill in the gaps. Always remember to make deliberate refinements, and add guidance only when it would be useful for this type of task rather than just one trajectory.

## Notice

1. Do NOT refer to any skills except those in `{skills_dir}`.
2. Do NOT explore any other folders that are not mentioned.
3. Do NOT create, update or refer to any memories of this project.
4. When using skill-creator, Do NOT eval and package the skill. Only create the new refined skill files according to the skill-creator's instructions and requirements.
5. The new refined skill should be versatile and avoid overly personal descriptions, such as private paths, private information, etc.
6. The new refined skill will be used to guide the student model. Please ensure that the skill is clear, understandable, and easy to follow.

## Trajectory Context

Use the paired trajectories in `{trajectory_dir}`. The split is based on an evaluation score: the score is higher when more checklist pairs are sufficiently and correctly supported by the generated analysis insights and final reports. Below-average scores (`positive/*`) indicate stronger checklist generation but relatively weaker analysis; above-average scores (`negative/*`) indicate weaker checklist generation but relatively stronger analysis. Optimize only the current target skill for generating checklist pairs, using the other side materials as reference.

Now, conduct a thorough inspection of the logs, database, and existing skills in the current folder. Based on your analysis, refine the existing skills into one or more new concise, reusable skills to help agents perform better on this type of task. All new refined skills are required for saving to the `{skills_dir}` folder.
"""


def render_prompt(template: str, workspace_path: str) -> str:
    return template.format(
        skill_creator_dir=f"{workspace_path}/.claude/skills/skill-creator",
        skills_dir=f"{workspace_path}/.claude/skills",
        trajectory_dir=f"{workspace_path}/trajectory",
    )
