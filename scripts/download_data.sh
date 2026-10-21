#!/usr/bin/env bash
set -euo pipefail
# Credentials are supplied through Kaggle's supported external configuration.
if [[ -z "${KAGGLE_CONFIG_DIR:-}" && ! -w "$HOME" && -d /workspace ]]; then
    export KAGGLE_CONFIG_DIR=/workspace/.config/kaggle
fi
if [[ -n "${KAGGLE_CONFIG_DIR:-}" ]]; then
    mkdir -p "$KAGGLE_CONFIG_DIR"
fi
destination="${GMM_IMAGE_DIR:-/workspace/datasets/head-ct-hemorrhage}"
mkdir -p "$destination"
kaggle datasets download -d felipekitamura/head-ct-hemorrhage -p "$destination" --unzip
