#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export HF_HOME=/home/pokazge/hf_cache
cd /home/pokazge/NativeEntity
echo "PYBIN=$(which python)"
python -u ws_emerge.py
