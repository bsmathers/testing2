#!/usr/bin/env bash
set -eo pipefail

# Forwarding wrapper to train.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train.sh" "$@"
