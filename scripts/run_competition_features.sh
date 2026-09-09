#!/usr/bin/env bash
set -euo pipefail

# The launch receipt pins the CUDA base image by digest. Install the remaining
# dependencies into an isolated environment without replacing its CUDA runtime.
python /opt/ml/code/scripts/bootstrap_gpu.py /opt/ml/feature-env
/opt/ml/feature-env/bin/python -m pip install --disable-pip-version-check \
  --constraint /opt/ml/feature-env/native-constraints.txt \
  transformers==4.57.6 huggingface-hub==0.36.2 accelerate==1.12.0 \
  numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 \
  filelock==3.32.5 joblib==1.5.3 tokenizers==0.22.2 regex==2026.9.10 \
  boto3==1.43.46 botocore==1.43.46
/opt/ml/feature-env/bin/python -m pip check
export PYTHONPATH=/opt/ml/code/src:/opt/ml/code
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
cd /opt/ml/code
exec /opt/ml/feature-env/bin/python -m scripts.competition_worker \
  --input /opt/ml/code/input.csv \
  --config /opt/ml/code/configs/competition_features.json \
  --output /opt/ml/feature-work \
  --bucket "$JIGSAW_BUCKET" --prefix "$JIGSAW_PREFIX"
