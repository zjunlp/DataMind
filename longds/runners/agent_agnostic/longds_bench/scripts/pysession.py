#!/usr/bin/env python3
"""Persistent Python session for LongDS self-evaluation.

LongDS requires CONTINUOUS code execution: variables/data from earlier steps and
earlier turns of the SAME task stay alive. This wraps a long-lived IPython kernel
so the agent's discrete shell calls share one stateful session per task.

Run one kernel per task. Lifecycle:

  # start (run in background; keep it alive for the whole task):
  python pysession.py start --conn /tmp/longds/<key>.json --pidfile /tmp/longds/<key>.pid --cwd <data_dir>

  # each analysis step (the agent's <python> block):
  echo '<code>' > step.py
  python pysession.py exec --conn /tmp/longds/<key>.json --code-file step.py --timeout 300

  # end of task:
  python pysession.py stop --pidfile /tmp/longds/<key>.pid

`exec` prints the combined stdout / results / errors (what goes into <information>)
and exits non-zero if the code raised. Requires `ipykernel` + `jupyter_client`.
"""
import argparse, os, signal, sys, time
from pathlib import Path


def cmd_start(args) -> int:
    from jupyter_client import KernelManager
    conn = str(Path(args.conn).resolve())
    Path(conn).parent.mkdir(parents=True, exist_ok=True)
    km = KernelManager(connection_file=conn)
    # Launch the kernel with THIS interpreter so it inherits our venv (pandas, etc.)
    # rather than whatever `python3` kernelspec happens to be installed.
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    km.start_kernel(cwd=args.cwd or None)
    if args.pidfile:
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")

    def _shutdown(*_):
        try:
            km.shutdown_kernel(now=True)
        finally:
            for p in (args.pidfile, conn):
                try:
                    if p:
                        os.unlink(p)
                except OSError:
                    pass
            os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    print(f"kernel started; connection={conn} pid={os.getpid()}", flush=True)
    while True:
        if not km.is_alive():
            print("kernel died", file=sys.stderr)
            return 1
        time.sleep(1)


def cmd_exec(args) -> int:
    from jupyter_client import BlockingKernelClient
    code = Path(args.code_file).read_text(encoding="utf-8") if args.code_file else sys.stdin.read()
    kc = BlockingKernelClient(connection_file=str(Path(args.conn).resolve()))
    kc.load_connection_file()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=30)
    except RuntimeError as e:
        print(f"ERROR: kernel not ready: {e}", file=sys.stderr)
        return 2
    msg_id = kc.execute(code)
    had_error = False
    deadline = time.time() + args.timeout
    while True:
        if time.time() > deadline:
            print(f"\n[TIMEOUT after {args.timeout}s]", file=sys.stderr)
            had_error = True
            break
        try:
            msg = kc.get_iopub_msg(timeout=1)
        except Exception:
            continue
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mt, c = msg["msg_type"], msg["content"]
        if mt == "stream":
            sys.stdout.write(c.get("text", ""))
        elif mt in ("execute_result", "display_data"):
            txt = c.get("data", {}).get("text/plain")
            if txt:
                sys.stdout.write(txt + "\n")
        elif mt == "error":
            had_error = True
            sys.stdout.write("\n".join(c.get("traceback", [])) + "\n")
        elif mt == "status" and c.get("execution_state") == "idle":
            break
    sys.stdout.flush()
    kc.stop_channels()
    return 1 if had_error else 0


def cmd_stop(args) -> int:
    pid_path = Path(args.pidfile)
    if not pid_path.is_file():
        print(f"no pidfile {pid_path}", file=sys.stderr)
        return 1
    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to {pid}")
    except ProcessLookupError:
        print(f"process {pid} already gone")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Persistent IPython session for LongDS.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("--conn", required=True)
    s.add_argument("--pidfile", default=None); s.add_argument("--cwd", default=None)
    e = sub.add_parser("exec"); e.add_argument("--conn", required=True)
    e.add_argument("--code-file", default=None); e.add_argument("--timeout", type=int, default=300)
    t = sub.add_parser("stop"); t.add_argument("--pidfile", required=True)
    args = ap.parse_args()
    return {"start": cmd_start, "exec": cmd_exec, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
