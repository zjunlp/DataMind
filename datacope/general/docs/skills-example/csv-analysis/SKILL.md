---
name: csv-analysis
description: Use this skill for CSV data analysis tasks that require reading a local CSV file, checking row counts and columns, grouping records, computing rates or aggregates, creating a chart, and writing a short Markdown report.
---

# CSV Analysis Skill

Use this workflow when the user asks for analysis over a local CSV file.

## Required Workflow

1. Locate the CSV file from the user prompt.
2. Read the full CSV file, not only the first few rows.
3. Inspect the header, row count, missing values, and key categorical columns.
4. If the task asks for failure rate, use:

```text
failure_rate = number of rows where results == "Fail" / number of rows in the group
```

5. When grouping by `facility_type`, include only groups that meet the user's minimum inspection-count threshold.
6. Sort failure-rate results by:

```text
failure_rate descending, inspection_count descending, facility_type ascending
```

7. Generate the requested chart and save it to the exact output path.
8. Write the requested Markdown report to the exact output path.
9. Verify the result by checking:

```text
source file path
total rows read
number of facility_type groups before and after filtering
chart file exists
report file exists
```

## Output Expectations

For the demo task, create:

```text
outputs/summary.md
outputs/failure_rate_by_facility.png
```

The report should include:

- source file and row count
- metric definition
- top 10 facility types by failure rate
- a short conclusion
- a verification note
