#!/usr/bin/env bash
set -euo pipefail

# First verify real S3 shard recovery under the original extraction contract.
# This also prepares the immutable local model assets without duplicate downloads.
bash /opt/ml/code/scripts/run_competition_features.sh
/opt/ml/feature-env/bin/python -m pip install --disable-pip-version-check \
  --constraint /opt/ml/feature-env/native-constraints.txt peft==0.18.0
/opt/ml/feature-env/bin/python -m pip check
export PYTHONPATH=/opt/ml/code/src:/opt/ml/code
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /opt/ml/code
exec /opt/ml/feature-env/bin/python -m scripts.adaptation_worker \
  --plan /opt/ml/code/adaptation_plan.json \
  --config /opt/ml/code/configs/support_adaptation.json \
  --frozen /opt/ml/feature-work/fe1da83fc62341bf608d/representations.npz \
  --model /opt/ml/feature-work/model \
  --output /opt/ml/adaptation-work \
  --bucket "$JIGSAW_BUCKET" --prefix "$JIGSAW_ADAPTATION_PREFIX"
