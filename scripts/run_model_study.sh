#!/usr/bin/env bash
set -euo pipefail

python /opt/ml/code/scripts/bootstrap_gpu.py /opt/ml/feature-env
/opt/ml/feature-env/bin/python -m pip install --disable-pip-version-check \
  --constraint /opt/ml/feature-env/native-constraints.txt \
  transformers==4.57.6 huggingface-hub==0.36.2 accelerate==1.12.0 peft==0.18.0 \
  numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 \
  filelock==3.32.5 joblib==1.5.3 tokenizers==0.22.2 regex==2026.9.10 \
  boto3==1.43.46 botocore==1.43.46
/opt/ml/feature-env/bin/python -m pip check
export PYTHONPATH=/opt/ml/code/src:/opt/ml/code
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
cd /opt/ml/code
exec /opt/ml/feature-env/bin/python -m scripts.model_study \
  --plan /opt/ml/code/adaptation_plan.json --output /opt/ml/model-study-work \
  --config "/opt/ml/code/$JIGSAW_STUDY_CONFIG" --model-config "/opt/ml/code/$JIGSAW_MODEL_CONFIG" \
  --bucket "$JIGSAW_BUCKET" --prefix "$JIGSAW_STUDY_PREFIX"
