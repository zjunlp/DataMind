"""Shared task selection for LongDS runners (standard library only)."""

import json
import re
from pathlib import Path


def add_dataset_arguments(parser, *, task_root_alias=None):
    parser.add_argument('--longds_version', default='v1.1', help='Task version, default: v1.1.')
    parser.add_argument('--split', choices=('full', 'lite'), default='lite')
    flags = ['--task-root'] + ([task_root_alias] if task_root_alias else [])
    parser.add_argument(*flags, type=Path, help='Override the version task directory.')
    parser.add_argument('--task-list-name', help='Override the split task list filename.')
    parser.add_argument('--data-root', type=Path, help='Override shared data/longds directory.')


def _component(value):
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError(f'Expected a single path component, got {value!r}')
    return value


def resolve_dataset(args, dataset_root=None):
    root = Path(dataset_root) if dataset_root else Path(__file__).resolve().parents[2] / 'dataset'
    _component(args.longds_version)
    if args.task_root is None:
        args.task_root = root / 'task' / f'longds_{args.longds_version}'
    else:
        # Explicit paths determine the effective labels as well as the source.
        name = Path(args.task_root).name
        args.longds_version = name.removeprefix('longds_') if name.startswith('longds_') else 'custom'
    _component(args.longds_version)
    args.task_root = Path(args.task_root).expanduser().resolve()
    if args.task_list_name is None:
        args.task_list_name = f'task_list_{args.split}.json'
    else:
        args.split = {'task_list_full.json': 'full', 'task_list_lite.json': 'lite'}.get(
            args.task_list_name, Path(args.task_list_name).stem
        )
    _component(args.task_list_name)
    _component(args.split)
    args.task_list = args.task_root / args.task_list_name
    if not args.task_list.is_file():
        raise FileNotFoundError(f'Task list does not exist (no fallback): {args.task_list}')
    args.data_root = Path(args.data_root or root / 'data' / 'longds').expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f'Data root does not exist: {args.data_root}')
    return dataset_metadata(args)


def dataset_metadata(args):
    return {key: str(getattr(args, key)) for key in (
        'longds_version', 'split', 'task_root', 'task_list', 'data_root'
    )}


def result_root(args):
    """Resolve the output base and shared dataset version/split group."""
    group = f'longds_{args.longds_version}_{args.split}'
    return Path(args.output_dir).expanduser().resolve() / group


def infer_result_metadata(task_dir):
    """Infer identity from new run-first or legacy task-first result paths."""
    path = Path(task_dir).resolve()
    parts = ('', '', '', '') + path.parts
    if re.fullmatch(r'task\d+', path.name, re.IGNORECASE) and not re.fullmatch(
        r'task\d+', path.parent.name, re.IGNORECASE
    ):
        domain, dataset, task_id = parts[-3:]
        run_name = parts[-4]
    else:
        domain, dataset, task_id, run_name = parts[-4:]
    return dict(task_domain=domain, dataset_name=dataset, task_id=task_id, run_name=run_name)


def load_task_list(args):
    tasks = json.loads(args.task_list.read_text(encoding='utf-8'))
    if not isinstance(tasks, list):
        raise ValueError(f'Task list must be a JSON array: {args.task_list}')
    seen = set()
    for task in tasks:
        identity = tuple(_component(task[key]) for key in ('task_domain', 'dataset_name', 'task_id'))
        if identity in seen:
            raise ValueError(f'Duplicate task: {identity}')
        seen.add(identity)
    return tasks


def load_turns(task_json, turn_limit=None):
    turns = json.loads(Path(task_json).read_text(encoding='utf-8'))
    if not isinstance(turns, list):
        raise ValueError(f'Task must contain a JSON array of turns: {task_json}')
    return turns if turn_limit is None else turns[:turn_limit]
