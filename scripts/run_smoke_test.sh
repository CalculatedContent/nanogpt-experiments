#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: $0 ROOT"; exit 2; fi
echo "Creating invalid-for-science smoke output under $1"
wwgpt smoke-test "$1"
