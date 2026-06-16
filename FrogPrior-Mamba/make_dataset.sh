#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE=${1:-data/processed}

python data/prepare_frog_json.py \
  --source_base "$SOURCE_BASE" \
  --out_dir data \
  --splits train valid test
