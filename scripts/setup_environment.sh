#!/usr/bin/env bash
set -euo pipefail

echo "Installing nanoGPT experiments and the current pip-resolved WW-PGD dependency"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --upgrade -e .
