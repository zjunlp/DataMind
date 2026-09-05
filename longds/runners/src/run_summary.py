"""Experiment-level summaries for a single runner invocation (no API calls)."""

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .longds_dataset import dataset_metadata, load_turns


def task_key(task):
    return '/'.join(task[k] for k in ('task_domain', 'dataset_name', 'task_id'))


class RunSummary:
    def __init__(self, args, runner, model, run_name, selected, run_dir):
        self.args = args
        self.runner = runner
        self.model = model
        self.run_name = run_name
        self.selected = selected
        self.run_dir = Path(run_dir)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.outcomes = {}
        self.judge_failures = set()

    def record(self, task, status):
        self.outcomes[task if isinstance(task, str) else task_key(task)] = status

    def judge_failed(self, task):
        self.judge_failures.add(task_key(task))

    def task_summary(self, task):
        key = task_key(task)
        leaf = self.run_dir / key
        row = dict(task=key, status=self.outcomes.get(key, 'not_run'),
                   expected_turns=None, judged_turns=0, judge_errors=0,
                   task_avg_score=None, evaluation_status='not_judged')
        try:
            turns = task.get('tasks')
            if turns is None:
                turns = load_turns(Path(self.args.task_root) / key / 'task.json', self.args.turn_limit)
            ids = [str(t.get('turn_id', i)) for i, t in enumerate(turns, 1)]
            row['expected_turns'] = len(ids)
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate expected turn IDs')
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            row.update(evaluation_status='invalid', evaluation_error=str(exc))
            return row
        if row['status'] == 'dry_run':
            row['evaluation_status'] = 'dry_run'
            return row
        eval_path = leaf / 'results_eval.json'
        if key in self.judge_failures:
            row['evaluation_status'] = 'failed'
            return row
        if not eval_path.exists():
            return row
        try:
            payload = json.loads(eval_path.read_text(encoding='utf-8'))
            if isinstance(payload, dict):
                payload = payload.get('results')
            if not isinstance(payload, list):
                raise ValueError('Expected an evaluation array')
            scores = {}
            seen = set()
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError('Invalid evaluation row')
                if 'summary' in item:
                    continue
                turn_id = str(item.get('turn_id'))
                if turn_id in seen or turn_id not in ids:
                    raise ValueError('Duplicate or unexpected evaluated turn ID')
                seen.add(turn_id)
                judge = item.get('judge') or {}
                score = judge.get('score')
                if judge.get('error'):
                    row['judge_errors'] += 1
                    continue
                if score is None:
                    continue
                if isinstance(score, bool):
                    raise ValueError('Invalid boolean score')
                score = float(score)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Score must be finite and between 0 and 1')
                scores[turn_id] = score
            row['judged_turns'] = len(scores)
            if ids and set(scores) == set(ids):
                row.update(evaluation_status='complete', task_avg_score=mean(scores.values()))
            else:
                row['evaluation_status'] = 'incomplete'
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            row.update(evaluation_status='invalid', evaluation_error=str(exc))
        return row

    def save(self):
        rows = [self.task_summary(task) for task in self.selected]
        counts = Counter(row['status'] for row in rows)
        averages = [row['task_avg_score'] for row in rows if row['task_avg_score'] is not None]
        payload = {
            'schema_version': 1,
            'scope': 'latest_invocation_selected_tasks',
            'runner': self.runner, 'model': self.model, 'run_name': self.run_name,
            **dataset_metadata(self.args),
            'started_at': self.started_at,
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'start_index': self.args.start_index, 'task_limit': self.args.task_limit,
            'turn_limit': self.args.turn_limit,
            'selected_tasks': len(rows),
            'completed_tasks': counts['completed'], 'failed_tasks': counts['failed'],
            'skipped_tasks': counts['skipped'], 'dry_run_tasks': counts['dry_run'],
            'not_run_tasks': counts['not_run'],
            'judged_tasks': len(averages),
            'judge_problem_tasks': sum(r['evaluation_status'] in ('invalid', 'incomplete', 'failed') for r in rows),
            'task_avg_score': mean(averages) if averages else None,
            'score_aggregation': 'equal mean of fully judged task means; unjudged/invalid tasks excluded',
            'tasks': rows,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / 'summary.json'
        # Atomic replacement prevents a half-written overview on interruption.
        fd, temporary = tempfile.mkstemp(prefix='.summary-', suffix='.tmp', dir=self.run_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f'Run summary saved: {path}', flush=True)
        return payload
