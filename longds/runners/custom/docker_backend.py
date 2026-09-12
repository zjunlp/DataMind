"""Docker lifecycle for custom agents; no Docker Python SDK required."""

import csv
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

WORKSPACE = '/workspace'
WORKER = '/longds/worker.py'


def add_arguments(parser):
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--use-docker', action='store_true', help='Run each task in its own container (default)')
    mode.add_argument('--local', action='store_true', help='Run in the current host environment instead of Docker')
    parser.add_argument('--requirements', type=Path, help='Install this pip requirements file in a cached agent image')
    parser.add_argument('--docker-image', help='Existing image; implies --use-docker (default: executor-prebuilt)')
    parser.add_argument('--docker-build', type=Path, help='Build this Dockerfile directory once; implies --use-docker')
    parser.add_argument('--docker-user', help='Container UID:GID; default: current host user')
    parser.add_argument('--agent-dir', type=Path, help='Explicit adapter source directory to mount read-only in Docker')
    parser.add_argument('--env', action='append', default=[], metavar='NAME',
                        help='Forward this host environment variable into Docker (repeatable)')
    parser.add_argument('--env-file', type=Path, action='append', default=[],
                        help='Docker agent environment file: NAME=value, one per line (repeatable)')


def configure(args):
    if getattr(args, 'local', False) and (args.use_docker or args.docker_image or args.docker_build
                                        or getattr(args, 'requirements', None)):
        raise ValueError('--local cannot be combined with Docker options or --requirements; install local dependencies yourself')
    args.use_docker = not getattr(args, 'local', False)
    args.agent_mounts = []
    args.container_env = {}
    if not args.use_docker:
        if args.env or args.env_file or args.agent_dir or args.docker_user:
            raise ValueError('--env, --env-file, --agent-dir and --docker-user require Docker mode')
        return
    if getattr(args, 'requirements', None):
        args.requirements = args.requirements.expanduser().resolve()
        if not args.requirements.is_file():
            raise ValueError(f'Requirements file does not exist: {args.requirements}')
    if args.docker_build:
        args.docker_build = args.docker_build.expanduser().resolve()
        if not (args.docker_build / 'Dockerfile').is_file():
            raise ValueError(f'No Dockerfile in {args.docker_build}')
    if not args.docker_image:
        if args.docker_build:
            suffix = hashlib.sha256(str(args.docker_build).encode()).hexdigest()[:12]
            args.docker_image = f'longds-custom-{suffix}:latest'
        else:
            args.docker_image = 'executor-prebuilt'
    if args.agent_dir:
        args.agent_dir = args.agent_dir.expanduser().resolve()
        if not args.agent_dir.is_dir():
            raise ValueError(f'Agent directory does not exist: {args.agent_dir}')
        args.agent_mounts.append((args.agent_dir, '/agent'))
    for path in args.env_file:
        for number, line in enumerate(path.expanduser().read_text(encoding='utf-8').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if not sep:
                raise ValueError(f'{path}:{number}: expected NAME=value')
            args.container_env[key] = value
    for key in args.env:
        if key not in os.environ:
            raise ValueError(f'--env {key}: variable is not set in the host environment')
        args.container_env[key] = os.environ[key]
    for key in args.container_env:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'Invalid environment variable name: {key!r}')
        if key.startswith('JUDGE_') or key in {'HOME', 'PYTHONPATH', 'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE'}:
            raise ValueError(f'{key} is reserved; provide agent credentials under their own name')
    args.docker_user = args.docker_user or f'{os.getuid()}:{os.getgid()}'


def adapter_spec(args, module, name):
    """Map only explicitly selected source files into the container."""
    suffix = f':{name}' if name else ''
    if not module.endswith('.py'):
        return f'{module}{suffix}'
    path = Path(module).expanduser()
    if path.is_file():
        path = path.resolve()
        if args.agent_dir:
            try:
                relative = path.relative_to(args.agent_dir)
            except ValueError as exc:
                raise ValueError('Local adapter must be inside --agent-dir') from exc
            return f'/agent/{relative.as_posix()}{suffix}'
        args.agent_mounts.append((path, '/longds/adapter.py'))
        return f'/longds/adapter.py{suffix}'
    if path.is_absolute():
        # Absolute paths that do not exist on the host refer to image contents.
        return f'{module}{suffix}'
    raise ValueError(f'Agent file does not exist: {path}; use an absolute path for files inside the image')


def preflight(args):
    if shutil.which('docker') is None:
        raise ValueError('Docker is not installed or not on PATH')
    result = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Cannot reach Docker. Start Docker and check socket permissions.\n' + result.stderr.strip())
    if args.docker_image == 'executor-prebuilt' and not args.docker_build and not image_id('executor-prebuilt'):
        source = Path(__file__).resolve().parents[1] / 'DSGym/executors/container_images/longds_image'
        print('Preparing the LongDS analysis image for the first run (subsequent runs reuse it)...', flush=True)
        subprocess.run(['docker', 'build', '-t', 'executor-prebuilt', str(source)], check=True)
    if args.docker_build:
        print(f'Building agent image {args.docker_image} from {args.docker_build}', flush=True)
        subprocess.run(['docker', 'build', '-t', args.docker_image, str(args.docker_build)], check=True)
    base_id = image_id(args.docker_image)
    if not base_id:
        raise ValueError(f'Docker image {args.docker_image!r} is not available locally. '
                         'Build it, pull it, or pass --docker-build DIR. For the default image:\n'
                         'docker build -t executor-prebuilt runners/DSGym/executors/container_images/longds_image')
    args.docker_image_id = base_id
    if getattr(args, 'requirements', None):
        args.docker_image, args.docker_image_id = requirements_image(args.docker_image, base_id, args.requirements)


def image_id(image):
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                            capture_output=True, text=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else None


DEPENDENCIES_DOCKERFILE = '''ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN /usr/local/bin/python -m venv --system-site-packages /opt/longds-agent
COPY requirements.txt /tmp/longds-requirements.txt
RUN /opt/longds-agent/bin/python -m pip --disable-pip-version-check install --no-cache-dir -r /tmp/longds-requirements.txt
ENV PATH="/opt/longds-agent/bin:${PATH}"
'''


def requirements_image(base_image, base_id, requirements):
    content = requirements.read_bytes()
    digest = hashlib.sha256(base_id.encode() + DEPENDENCIES_DOCKERFILE.encode() + content).hexdigest()[:24]
    tag = f'longds-agent-deps:{digest}'
    cached = image_id(tag)
    if cached:
        print(f'Reusing agent dependency image: {tag}', flush=True)
        return tag, cached
    print(f'Installing agent dependencies from {requirements} (cached for later runs)...', flush=True)
    # Only the requested requirements file is sent as build context, never the repository.
    with tempfile.TemporaryDirectory(prefix='longds-agent-deps-') as tmp:
        root = Path(tmp)
        (root / 'Dockerfile').write_text(DEPENDENCIES_DOCKERFILE)
        (root / 'requirements.txt').write_bytes(content)
        subprocess.run(['docker', 'build', '--build-arg', f'BASE_IMAGE={base_image}',
                        '-t', tag, str(root)], check=True)
    built = image_id(tag)
    if not built:
        raise RuntimeError(f'Dependency image was not found after building: {tag}')
    return tag, built


def mount(source, target, readonly=False):
    # Docker --mount uses CSV, so paths containing commas need field quoting.
    stream = io.StringIO()
    fields = ['type=bind', f'source={source}', f'target={target}']
    if readonly:
        fields.append('readonly')
    csv.writer(stream, lineterminator='').writerow(fields)
    return stream.getvalue()


class TaskContainer:
    def __init__(self, args, workspace, data):
        self.args = args
        self.name = f'longds-custom-{uuid.uuid4().hex}'
        self.workspace = workspace
        self.data = data
        self.created = False

    def create_command(self):
        args = self.args
        command = ['docker', 'create', '--interactive', '--init', '--name', self.name,
                   '--label', 'longds.runner=custom', '--workdir', WORKSPACE,
                   '--user', args.docker_user, '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges',
                   '--mount', mount(self.workspace, WORKSPACE),
                   '--mount', mount(self.data, WORKSPACE + '/data', True),
                   '--env', 'HOME=/workspace/.home', '--env', 'PYTHONUNBUFFERED=1',
                   '--env', 'PYTHONDONTWRITEBYTECODE=1']
        mounts = list(args.agent_mounts)
        if args.agent:
            mounts.append((Path(__file__).with_name('worker.py'), WORKER))
            mounts.append((Path(__file__).with_name('longds.py'), '/longds/longds.py'))
        for source, target in mounts:
            command.extend(['--mount', mount(source, target, True)])
        if args.agent_dir:
            command.extend(['--env', 'PYTHONPATH=/agent'])
        for key in sorted(args.container_env):
            # Pass names only. Values are provided through the Docker client's env.
            command.extend(['--env', key])
        command.extend(['--entrypoint', args.command[0],
                        getattr(args, 'docker_image_id', args.docker_image), *args.command[1:]])
        return command

    def start_command(self):
        (self.workspace / '.home').mkdir(exist_ok=True)
        (self.workspace / 'data').mkdir(exist_ok=True)
        env = {**os.environ, **self.args.container_env}
        # Even a timed-out Docker client may have created the container on the daemon.
        self.created = True
        result = subprocess.run(self.create_command(), capture_output=True, text=True, env=env, timeout=60)
        if result.returncode:
            raise RuntimeError(f'Docker container creation failed: {result.stderr.strip()}')
        return ['docker', 'start', '--attach', '--interactive', self.name]

    def close(self):
        if self.created:
            result = subprocess.run(['docker', 'rm', '--force', self.name],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode and 'No such container' not in result.stderr:
                raise RuntimeError(f'Could not remove task container {self.name}: {result.stderr.strip()}')
            self.created = False
