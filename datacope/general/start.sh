#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m src.core.loop --config config.yaml "$@"
