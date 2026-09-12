"""Optional logging helper available to Python agents run by LongDS."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def save(event: str, data: Any) -> None:
    """Append a JSON-serializable event to the current turn's trajectory.

    The runner sets LONGDS_TRACE_DIR per turn. Outside the runner, events go to
    .longds/trajectory.jsonl in the current directory.
    """
    record = json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                         'event': event, 'data': data}, ensure_ascii=False)
    directory = Path(os.environ.get('LONGDS_TRACE_DIR', '.longds'))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(record + '\n')
