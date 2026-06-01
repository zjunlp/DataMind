# Using Project-Level Skills in Codex and Claude Code

This document explains how to configure prepared skill files as project-level skills in Codex and Claude Code, and how to use them through automatic triggering in real tasks.

The installation and usage workflow is general and works for any set of task-specific skill files. The examples at the end are data-analysis tasks that use skills, showing the full process from prompting, to triggering a skill, to following the skill workflow, to checking the answer.

## When to Use This

Use this guide when:

```text
You already have a set of skill files.
You want to reuse those skills in Codex or Claude Code.
You want the tool to choose the right skill automatically.
You do not want to paste or upload the skill content every time.
You want to confirm whether a skill was actually loaded and used.
```

## Recommended Layout

Keep the source skills in a normal project directory first:

```text
skills/
  skill_a/
    SKILL.md
  skill_b/
    SKILL.md
  skill_c/
    SKILL.md
```

If the skills need local data, keep the data in a stable project directory:

```text
data/
  dataset.csv
  rules.json
  manual.md
```

Then copy the skills you want to enable into the project-level skill directory for Codex or Claude Code.

## Skill File Requirements

Each skill should be a standalone folder containing `SKILL.md`:

```text
skill_name/
  SKILL.md
```

The top of `SKILL.md` should include metadata:

```yaml
---
name: skill_name
description: "Describe what task this skill should be used for"
---
```

The fields mean:

```text
name: skill name.
description: trigger description used by the tool to decide when to load the skill.
```

Recommendations:

```text
Use one folder per skill.
Keep the file name exactly SKILL.md.
Make the description specific to the task type.
Quote the description if it contains colons or YAML-sensitive punctuation.
Put long references, scripts, or examples in subfolders when needed.
```

## Install in Codex

Create this directory at the project root:

```text
.codex/skills/
```

Copy the skill folders into it:

```text
.codex/skills/
  skill_a/
    SKILL.md
  skill_b/
    SKILL.md
```

Then:

```text
1. Open a new Codex session.
2. Make sure the current working directory is the project root.
3. Check that the project-level skills appear in the skill list.
```

## Install in Claude Code

Create this directory at the project root:

```text
.claude/skills/
```

Copy the skill folders into it:

```text
.claude/skills/
  skill_a/
    SKILL.md
  skill_b/
    SKILL.md
```

Then:

```text
1. Reopen Claude Code if the directory was newly created.
2. Open Claude Code from the project root or a child directory.
3. Confirm that project-level skills can be discovered.
```

## Confirm the Skills Are Configured

After installation, first make sure you are in the correct project location. Project-level skills only apply to the corresponding project. Start Codex / Claude Code from the project root that contains `.codex/skills/` or `.claude/skills/`, or run the check commands from that project root. If you start the tool from another directory, the configured skills may not appear.

The installed project structure can be checked against this layout:

```text
project-root/
  .codex/
    skills/
      skill_a/
        SKILL.md
      skill_b/
        SKILL.md
  .claude/
    skills/
      skill_a/
        SKILL.md
      skill_b/
        SKILL.md
  data/
    dataset.csv
    rules.json
```

If you only use Codex, you only need `.codex/skills/`. If you only use Claude Code, you only need `.claude/skills/`.

There are three ways to confirm the setup.

Claude Code:

```text
1. File check:
   Run from the project root:
   find .claude/skills -maxdepth 2 -name SKILL.md -print

2. In-tool check:
   Start Claude Code from the project root or a child directory, then enter /skills and confirm that the target skill appears in the list.

3. Invocation check:
   Enter /<skill-name>, or explicitly write "Use <skill-name>" in a task.
```

Codex:

```text
1. File check:
   Run from the project root:
   find .codex/skills -maxdepth 2 -name SKILL.md -print

2. In-tool check:
   Start Codex from the project root or a child directory, then check the skill list, or enter /skills to confirm that the target skill appears.

3. Invocation check:
   Explicitly write "Use <skill-name>" in a task, or mention the skill name to trigger the target skill.
```

If your Codex environment uses `.agents/skills/` instead of `.codex/skills/`, replace the directory in the structure diagram and commands with the directory used by your environment. The important point is that the tool launch location, the command execution location, and the skill installation location all belong to the same project.

## How to Use

After installation, do not paste the `SKILL.md` content into the conversation. Ask the task normally and provide the required data path when needed.

Recommended prompt:

```text
Use the data in <data_path> to answer:

<task_question>

Return only the final answer.
```

If the task does not need data files, ask the question directly:

```text
<task_question>
```

You can also explicitly call a specific skill:

```text
Use <skill_name> to answer this question.
```

## Confirm Whether a Skill Was Used

After the task is complete, ask a short follow-up:

```text
Please confirm whether you loaded and used a project-level skill for the previous task. Answer only these fields:

loaded_skill:
skill_name:
followed_skill_instructions:
used_data_files:
evidence:
```

## Example 1: Querying Applicable Rule IDs

The following is an example of using a skill. It shows how a normal user prompt can automatically trigger the corresponding project-level skill and complete the task by following that skill's workflow.

Task:

```text
For the 200th of the year 2023, what are the Fee IDs applicable to Golfclub_Baron_Friso?
```

Prompt:

```text
Use the data in <project-root>/skill/dabstep_data/context to answer:

For the 200th of the year 2023, what are the Fee IDs applicable to Golfclub_Baron_Friso?

Return only the final answer.
```

Expected answer:

```text
64, 123, 304, 381, 384, 454, 473, 572, 595, 678, 813
```

Project-level skill:

```text
Applicable_Fee_IDs
```

This task naturally triggers `Applicable_Fee_IDs` because it asks which Fee IDs apply to a merchant on a specific date.

The skill-guided workflow is:

```text
1. Identify the task type: Merchant + Day Query.
2. Extract the merchant: Golfclub_Baron_Friso.
3. Extract the date: day 200 of 2023.
4. Map the day to its month to compute monthly volume and fraud brackets.
5. Read payments.csv to find the transaction combinations that occurred for this merchant on that day.
6. Read merchant_data.json to get account type, merchant category code, capture delay, and related merchant attributes.
7. Read fees.json and apply the fee matching rules defined by the skill.
8. Collect matched Fee IDs and output them in ascending order.
```

Observed result in Codex and Claude Code:

```text
loaded_skill: yes
skill_name: Applicable_Fee_IDs
used_data_files: yes
model_output: 64, 123, 304, 381, 384, 454, 473, 572, 595, 678, 813
answer_match: yes
```

This example shows that you only need to provide a normal task and a data directory. The tool loads the matching project-level skill, and the skill determines which files to read and which rules to use for the answer.

## Example 2: Computing an Average Fee

Task:

```text
For credit transactions, what would be the average fee that the card scheme NexPay would charge for a transaction value of 10 EUR?
```

Prompt:

```text
Use the data in <project-root>/skill/dabstep_data/context to answer:

For credit transactions, what would be the average fee that the card scheme NexPay would charge for a transaction value of 10 EUR?

Return only the final answer.
```

Expected answer:

```text
0.126459
```

Project-level skill:

```text
Average_Fee_Estimation
```

This task naturally triggers `Average_Fee_Estimation` because it asks for an average processing fee under a specified transaction value, card scheme, and credit-card condition.

The skill-guided workflow is:

```text
1. Identify the task type: Filtered Average.
2. Extract the transaction value: 10 EUR.
3. Extract the card scheme: NexPay.
4. Extract the transaction type: credit transactions.
5. Read fees.json.
6. Filter fee rules where card_scheme is NexPay.
7. Apply the credit-card filtering rule defined by the skill.
8. For each applicable rule, compute fixed_amount + rate * transaction_value / 10000.
9. Average the computed fees across applicable rules.
10. Output the final numeric value in the required format.
```

Observed result in Codex and Claude Code:

```text
loaded_skill: yes
skill_name: Average_Fee_Estimation
used_data_files: yes
model_output: 0.126459
answer_match: yes
```

This example shows that the skill does not only provide background knowledge, but also defines the filtering conditions, calculation formula, and final output format.

## Example 3: General CSV Sales Analysis

The following is a more general data-analysis example. It does not depend on a domain-specific dataset. Instead, it uses a regular `sales.csv` file to show how a general CSV analysis skill can perform grouped aggregation.

This example uses the general-purpose `csv-pipeline` skill:

```text
https://clawhub.ai/skills/csv-pipeline
https://github.com/openclaw/skills/blob/main/skills/gitgoodordietrying/csv-pipeline/SKILL.md
```

Data file:

```text
<project-root>/data/sales.csv
```

Example data structure:

```text
date,region,product,units,unit_price,revenue
2026-01-03,East,Laptop,3,1200,3600
2026-01-04,West,Phone,10,600,6000
...
```

Task:

```text
Use sales.csv to summarize total_revenue by product and region, and identify the product and region with the highest total_revenue.
```

Prompt:

```text
Use the data in <project-root>/data and sales.csv to complete the following analysis:

1. Summarize total_revenue by product, and list every product.
2. Summarize total_revenue by region, and list every region.
3. Identify the product and region with the highest total_revenue.
4. Output the complete summary results as Markdown tables and end with one sentence of conclusion.
```

Expected output example:

```text
Verified: the product summary table has 3 rows, matching the 3 unique product values in sales.csv; the region summary table has 4 rows, matching the 4 unique region values.

| product | total_revenue |
|---|---:|
| Laptop | 9600.00 |
| Phone | 15000.00 |
| Tablet | 11200.00 |

| region | total_revenue |
|---|---:|
| East | 8200.00 |
| North | 6400.00 |
| South | 9200.00 |
| West | 12000.00 |

Conclusion: the product with the highest total_revenue is Phone (15000.00), and the region with the highest total_revenue is West (12000.00).
```

Project-level skill:

```text
csv-pipeline
```

This task naturally triggers `csv-pipeline` because it asks to read a CSV file, group by fields, aggregate a numeric column, and generate Markdown summary tables.

The skill-guided workflow is:

```text
1. Locate the user-specified data file sales.csv.
2. Read the header and sample rows, confirming that the file contains product, region, and revenue.
3. Group by product and sum revenue to compute total_revenue for each product.
4. Group by region and sum revenue to compute total_revenue for each region.
5. Verify that the output table row counts match the number of unique group keys in the source data.
6. Identify the product and region with the highest total_revenue.
7. Output two complete Markdown tables and one sentence of conclusion.
```

Observed result in Codex:

```text
loaded_skill: yes
skill_name: csv-pipeline
used_data_files: yes
model_output: complete product and region summary tables, correctly identifying Phone and West as the highest-revenue entries
answer_match: yes
```

This example shows that project-level skills are not limited to domain-specific business rules. They can also support general CSV data-analysis tasks.

## Troubleshooting

### Skill Does Not Appear

Check:

```text
The skill is under .codex/skills/ or .claude/skills/.
Each skill has its own folder.
The file is named exactly SKILL.md.
The metadata starts and ends with ---.
The metadata contains name and description.
The description is valid YAML.
The tool was reopened after the skill directory was created.
```
