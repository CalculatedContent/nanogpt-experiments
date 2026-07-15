#!/usr/bin/env bash
set -euo pipefail
echo "Creating local editable Python environment if requested by user shell"
python -m pip install -e ".[dev]" || python -m pip install -e .
