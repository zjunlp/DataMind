#!/usr/bin/env python3
"""Evaluate custom agents through a persistent per-task JSON Lines process."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

import docker_backend

RUNNERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNERS))
from src.longds_dataset import (add_dataset_arguments, resolve_dataset, result_root,
                               load_task_list, load_turns, dataset_metadata)
from src.run_summary import RunSummary, task_key


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class AgentProcess:
    def __init__(self, command, workspace, log, timeout, stop=None):
        self.timeout = timeout
        self.stop = stop or threading.Event()
        self.responses = queue.Queue()
        env = os.environ.copy()
        env.pop('JUDGE_API_KEY', None)
        self.process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=log, text=True,
                                        encoding='utf-8', start_new_session=True, env=env)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def request(self, payload, expected):
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('Agent process exited; see agent.stderr.log') from exc
        deadline = time.monotonic() + self.timeout
        while True:
            if self.stop.is_set():
                raise RuntimeError('Run interrupted')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Agent {payload["type"]} timed out after {self.timeout}s')
            try:
                line = self.responses.get(timeout=min(remaining, 0.2))
                break
            except queue.Empty:
                continue
        if line is None:
            raise RuntimeError('Agent process exited without a response; see agent.stderr.log')
        try:
            response = json.loads(line)
        except ValueError as exc:
            raise ValueError('Agent stdout must contain only JSON Lines; send logs to stderr') from exc
        if not isinstance(response, dict) or response.get('type') != expected:
            raise ValueError(f'Expected protocol response type {expected!r}')
        return response

    def close(self):
        # Terminate the process group, including tool subprocesses on this Linux host.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()


def run_task(args, task, run_dir):
    if args.stop.is_set():
        return 'not_run', False
    leaf = run_dir / task_key(task)
    leaf.mkdir(parents=True)
    workspace = leaf / 'workspace'
    workspace.mkdir()
    metadata = {**dataset_metadata(args), **task, 'run_name': args.run_name,
                'runner': 'custom', 'model': args.model,
                'execution_mode': 'docker' if args.use_docker else 'local',
                'agent': args.agent or 'command', 'workspace_dir': str(workspace)}
    if args.use_docker:
        metadata.update(docker_image=args.docker_image,
                        docker_image_id=getattr(args, 'docker_image_id', None),
                        container_workspace=docker_backend.WORKSPACE)
    write_json(leaf / 'task_metadata.json', metadata)
    process = None
    container = None
    rows = []
    try:
        turns = load_turns(args.task_root / task_key(task) / 'task.json', args.turn_limit)
        source = args.data_root / task_key(task) / 'data'
        if not source.is_dir():
            raise FileNotFoundError(f'Task data directory does not exist: {source}')
        if args.dry_run:
            write_json(leaf / 'preview.json', [
                {k: t.get(k, '') for k in ('turn_id', 'context', 'question')} for t in turns])
            return 'dry_run', False
        print(f'{task_key(task)}: starting {metadata["execution_mode"]} agent; logs: {leaf / "agent.stderr.log"}',
              flush=True)
        if args.use_docker:
            container = docker_backend.TaskContainer(args, workspace, source.resolve())
            metadata['docker_container_name'] = container.name
            write_json(leaf / 'task_metadata.json', metadata)
            command = container.start_command()
            agent_workspace = docker_backend.WORKSPACE
        else:
            shutil.copytree(source, workspace / 'data')
            command = args.command
            agent_workspace = str(workspace)
        with (leaf / 'agent.stderr.log').open('w', encoding='utf-8') as log:
            process = AgentProcess(command, workspace, log, args.timeout, args.stop)
            process.request({'type': 'start', 'workspace': agent_workspace,
                             'data_dir': agent_workspace + '/data', 'config': args.config}, 'ready')
            for index, turn in enumerate(turns, 1):
                request = {'type': 'turn', 'turn_id': turn.get('turn_id', index),
                           'context': turn.get('context') or '', 'question': turn.get('question') or ''}
                started = time.monotonic()
                detail = leaf / 'detail' / f'turn_{index}'
                detail.mkdir(parents=True)
                write_json(detail / 'input.json', request)
                log_offset = (leaf / 'agent.stderr.log').stat().st_size
                try:
                    response = process.request(request, 'answer')
                finally:
                    # Files are bind-mounted in Docker and survive crashes/timeouts.
                    trace_dir = workspace / '.longds' / f'turn_{index}'
                    for name in ('prompt.md', 'trajectory.jsonl'):
                        source_trace = trace_dir / name
                        if source_trace.is_file():
                            shutil.copyfile(source_trace, detail / name)
                    with (leaf / 'agent.stderr.log').open('rb') as stream:
                        stream.seek(log_offset)
                        (detail / 'agent.stderr.log').write_bytes(stream.read())
                if not isinstance(response.get('answer'), str):
                    raise ValueError('Agent answer must be a string; serialize structured answers as JSON text')
                row = {k: v for k, v in request.items() if k != 'type'}
                row.update(solution=response['answer'], success=True,
                           reasoning_summary=response.get('reasoning_summary', ''),
                           files_used=response.get('files_used', []),
                           elapsed_seconds=time.monotonic() - started)
                rows.append(row)
                write_json(leaf / 'results.json', rows)
                write_json(detail / 'result.json', response)
                print(f'{task_key(task)}: {index}/{len(turns)} turns', flush=True)
            process.request({'type': 'end'}, 'done')
            if container is not None:
                container.close()
            process.close()
            process = None
        # Gold is written only after the agent process has finished.
        write_json(leaf / 'results_with_ground_truth.json', [
            {**row, 'ground_truth': turn.get('answer')} for row, turn in zip(rows, turns)])
        judge_failed = False
        if args.judge:
            with (leaf / 'judge.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, str(RUNNERS / 'src/judge.py'),
                    '--run-dir', str(leaf), '--judge-model', args.judge_model], stdout=log, stderr=log)
            judge_failed = result.returncode != 0
        return 'completed', judge_failed
    except Exception as exc:
        write_json(leaf / 'error.json', {'error': str(exc), 'completed_turns': len(rows)})
        print(f'{task_key(task)} failed: {exc}\nTask output: {leaf}', file=sys.stderr, flush=True)
        return 'failed', False
    finally:
        try:
            if container is not None:
                container.close()
        finally:
            if process is not None:
                process.close()
            # Refresh snapshots after shutdown so failed turns retain the final writes too.
            for trace_dir in (workspace / '.longds').glob('turn_*'):
                detail = leaf / 'detail' / trace_dir.name
                if detail.is_dir():
                    for name in ('prompt.md', 'trajectory.jsonl'):
                        if (trace_dir / name).is_file():
                            shutil.copyfile(trace_dir / name, detail / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    docker_backend.add_arguments(parser)
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument('--agent', help='Python file:Class or module:Class with respond(message) -> str; legacy functions also supported')
    agent.add_argument('--agent-command', help='JSON Lines server command, parsed without a shell')
    parser.add_argument('--agent-python', help='Adapter Python (default: host Python locally, python inside Docker)')
    parser.add_argument('--agent-config', type=Path, help='JSON object passed as adapter constructor kwargs')
    parser.add_argument('--model', default='custom', help='Model label for reporting')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--task-limit', type=int)
    parser.add_argument('--turn-limit', type=int)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--run-parallel', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=3600, help='Seconds per turn/start/end request')
    parser.add_argument('--dry-run', action='store_true', help='Validate tasks and write previews without starting agents')
    parser.add_argument('--judge', action='store_true')
    parser.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL', 'deepseek-v4-pro'))
    args = parser.parse_args()
    if not args.agent and not args.agent_command:
        parser.error('Provide --agent my_agent.py:MyAgent with respond(message) -> str, or --agent-command COMMAND')
    try:
        docker_backend.configure(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.stop = threading.Event()
    for name in ('task_limit', 'turn_limit', 'start_index'):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be non-negative')
    if args.run_parallel < 1 or not 0 < args.timeout < float('inf'):
        parser.error('--run-parallel and --timeout must be positive (timeout must be finite)')
    if args.judge and args.dry_run:
        parser.error('--judge cannot be combined with --dry-run')
    if args.judge and not all(os.environ.get(k) for k in ('JUDGE_API_KEY', 'JUDGE_BASE_URL')):
        parser.error('--judge requires JUDGE_API_KEY and JUDGE_BASE_URL')
    try:
        resolve_dataset(args)
        args.config = json.loads(args.agent_config.expanduser().read_text(encoding='utf-8')) if args.agent_config else {}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not isinstance(args.config, dict):
        parser.error('--agent-config must contain a JSON object')
    if args.agent:
        module, separator, name = args.agent.rpartition(':')
        if not separator:
            module, name = args.agent, None
        elif not module or not name:
            parser.error('--agent must be a Python file or module:Class')
        if args.use_docker:
            try:
                container_spec = docker_backend.adapter_spec(args, module, name)
            except ValueError as exc:
                parser.error(str(exc))
            args.command = [args.agent_python or 'python', docker_backend.WORKER, container_spec]
        elif module.endswith('.py'):
            path = Path(module).expanduser().resolve()
            if not path.is_file():
                parser.error(f'Agent file does not exist: {path}')
            args.agent = f'{path}:{name}' if name else str(path)
        if not args.use_docker:
            executable = shutil.which(args.agent_python or sys.executable)
            if executable is None:
                parser.error('--agent-python executable was not found')
            args.command = [str(Path(executable).absolute()), str(Path(__file__).with_name('worker.py')), args.agent]
    else:
        try:
            args.command = shlex.split(args.agent_command)
        except ValueError as exc:
            parser.error(f'Invalid --agent-command: {exc}')
        if not args.command:
            parser.error('--agent-command must not be empty')
    args.run_name = args.run_name or datetime.now().strftime('custom_%Y%m%d_%H%M%S_%f')
    if args.run_name in ('.', '..') or '/' in args.run_name or '\\' in args.run_name:
        parser.error('--run-name must be a single path component')
    tasks = load_task_list(args)[args.start_index:]
    if args.task_limit is not None:
        tasks = tasks[:args.task_limit]
    run_dir = result_root(args) / args.run_name
    # A fresh run avoids mixing scores and silently resetting a persistent agent session.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        parser.error(f'Run already exists; choose a new --run-name: {run_dir}')
    if args.use_docker and not args.dry_run:
        try:
            docker_backend.preflight(args)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            parser.error(str(exc))
        print(f'Docker image: {args.docker_image}\nAdapter command: {shlex.join(args.command)}', flush=True)
    run_dir.mkdir()
    summary = RunSummary(args, 'custom', args.model, args.run_name, tasks, run_dir)
    failed = False
    try:
        with ThreadPoolExecutor(max_workers=args.run_parallel) as pool:
            futures = {pool.submit(run_task, args, task, run_dir): task for task in tasks}
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        status, judge_failed = future.result()
                    except Exception as exc:
                        status, judge_failed = 'failed', False
                        print(f'{task_key(task)}: {exc}', file=sys.stderr)
                    summary.record(task, status)
                    if judge_failed:
                        summary.judge_failed(task)
                        print(f'{task_key(task)}: judge failed; see judge.log', file=sys.stderr)
                    elif args.judge and status == 'completed':
                        evaluation = summary.task_summary(task)
                        if evaluation['evaluation_status'] != 'complete':
                            judge_failed = True
                            print(f'{task_key(task)}: incomplete evaluation; see results_eval.json and judge.log',
                                  file=sys.stderr)
                    failed |= status == 'failed' or judge_failed
                    summary.save()
            except KeyboardInterrupt:
                args.stop.set()
                for future in futures:
                    future.cancel()
                print('Interrupted; stopping agents and removing task containers...', file=sys.stderr)
                raise
    except KeyboardInterrupt:
        # The pool has finished cleanup. Record tasks that completed during shutdown.
        for future, task in futures.items():
            if future.done() and not future.cancelled():
                try:
                    status, judge_failed = future.result()
                    summary.record(task, status)
                    if judge_failed:
                        summary.judge_failed(task)
                except Exception:
                    summary.record(task, 'failed')
        return 130
    finally:
        summary.save()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
