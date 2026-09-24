#!/bin/bash
#SBATCH --job-name=hicot-sw-fix
#SBATCH --partition=defq
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/hicot_sw_fix_slurm_%j.out

set -e
cd /raid/brunoguilherme/DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
git pull origin slurm-gpu-experiments
source /raid/brunoguilherme/venvs/dgtc/bin/activate

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export HF_HOME=/raid/brunoguilherme/.cache/huggingface
export TRANSFORMERS_CACHE=/raid/brunoguilherme/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=/raid/brunoguilherme/.cache/sentence_transformers
export TORCH_HOME=/raid/brunoguilherme/.cache/torch

python -u scripts/run_hicot_classification_stopword_fix_12ds.py
