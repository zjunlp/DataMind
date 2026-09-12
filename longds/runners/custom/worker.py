"""Python adapter host. stdout is reserved for the JSON Lines protocol."""

import contextlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


def load_agent(spec):
    module_name, separator, name = spec.rpartition(':')
    if not separator:
        module_name, name = spec, 'respond'
    if module_name.endswith('.py'):
        path = Path(module_name).resolve()
        sys.path.insert(0, str(path.parent))
        module_spec = importlib.util.spec_from_file_location('longds_user_agent', path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    entry = getattr(module, name, None)
    if not separator and entry is None:
        entry = getattr(module, 'Agent', None)
    if not callable(entry):
        raise TypeError(f'{module_name} must define a callable {name} or an Agent class with respond(message: str) -> str')
    return entry


def turn_message(request, workspace, data_dir, first_turn):
    parts = []
    if first_turn:
        parts.append(
            'You are solving a multi-turn data analysis task. Solve only the current question.\n'
            f'Input data (read-only): {data_dir}\n'
            f'Working directory for tools, scripts, and intermediate files: {workspace}\n'
            'Keep useful analysis state for later questions. Use tools to calculate from the data.\n'
            'Do not access benchmark source tasks, reference answers, other tasks, or future questions.\n'
            'Follow the requested rounding and ordering. Return the final answer to the current question.'
        )
        if workspace == '/workspace':
            parts.append('Use /usr/local/bin/python for data analysis in the LongDS analysis image; '
                         'keep agent-specific dependencies in the agent environment.')
    if request['context']:
        parts.append('Context:\n' + request['context'])
    parts.append('Question:\n' + request['question'])
    return '\n\n'.join(parts)


def main():
    # Reserve a private protocol stream; also redirect native libraries and child
    # tools writing directly to fd 1, not just Python print() calls.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1, encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    agent = None
    is_function = False
    first_turn = True
    turn_index = 0
    for line in sys.stdin:
        request = json.loads(line)
        # SDK logs and user print() calls must not corrupt protocol responses.
        with contextlib.redirect_stdout(sys.stderr):
            if request['type'] == 'start':
                entry = load_agent(sys.argv[1])
                is_function = not inspect.isclass(entry)
                if is_function and request['config']:
                    raise ValueError('--agent-config is for class adapters; configure respond() agents in their module or environment')
                agent = entry if is_function else entry(**request['config'])
                respond = agent if is_function else getattr(agent, 'respond', None)
                workspace, data_dir = request['workspace'], request['data_dir']
                if not is_function and hasattr(agent, 'start_task'):
                    agent.start_task(workspace=request['workspace'], data_dir=request['data_dir'])
                response = {'type': 'ready'}
            elif request['type'] == 'turn':
                turn_index += 1
                trace_dir = Path(workspace) / '.longds' / f'turn_{turn_index}'
                trace_dir.mkdir(parents=True, exist_ok=True)
                os.environ['LONGDS_TRACE_DIR'] = str(trace_dir)
                os.environ['LONGDS_TURN_ID'] = str(request['turn_id'])
                if respond is not None:
                    message = turn_message(request, workspace, data_dir, first_turn)
                    (trace_dir / 'prompt.md').write_text(message, encoding='utf-8')
                    answer = respond(message)
                    if not isinstance(answer, str):
                        raise TypeError('respond(message) must return a final answer string (not a coroutine, stream, or dict)')
                else:
                    answer = agent.run_turn(context=request['context'], question=request['question'])
                first_turn = False
                if isinstance(answer, str):
                    answer = {'answer': answer}
                if not isinstance(answer, dict) or not isinstance(answer.get('answer'), str):
                    raise TypeError('run_turn must return a string or a dict with a string answer')
                response = {**answer, 'type': 'answer'}
            elif request['type'] == 'end':
                if not is_function and hasattr(agent, 'end_task'):
                    agent.end_task()
                response = {'type': 'done'}
            else:
                raise ValueError('Unknown request type')
        print(json.dumps(response, ensure_ascii=False), file=protocol, flush=True)
        if request['type'] == 'end':
            return


if __name__ == '__main__':
    main()
