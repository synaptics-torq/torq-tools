#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.
#
# Set up the dedicated moonshine_streaming environment.
#
# moonshine_streaming requires transformers 5.x, which conflicts with the
# base environment's optimum-based exporters (they cap transformers below
# 4.58), so it must not be installed alongside the base `pip install .`
# dependencies. This script creates a standalone venv with torq-tools
# (installed without its base dependencies) plus the complete pinned
# requirements (shared runtime stack + moonshine, see requirements.txt).
#
# Usage: bash src/torq/models/moonshine_streaming/install_requirements.sh [venv_dir]
#   venv_dir defaults to <repo_root>/.venv-moonshine-streaming

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
VENV_DIR=${1:-$REPO_ROOT/.venv-moonshine-streaming}

python3 -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
"$PIP" install -q -U pip
"$PIP" install -q -e "$REPO_ROOT" --no-deps
# --extra-index-url selects the +cpu torch/torchcodec wheels for CPU-only hosts.
"$PIP" install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$SCRIPT_DIR/requirements.txt"

echo "moonshine_streaming environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
