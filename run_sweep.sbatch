#!/bin/bash
#SBATCH --job-name=sbert11-sweep
#SBATCH --partition=defq
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/sweep_slurm_%j.out

set -e
cd /raid/brunoguilherme/DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
source /raid/brunoguilherme/venvs/dgtc/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=/raid/brunoguilherme/.cache/huggingface
export TRANSFORMERS_CACHE=/raid/brunoguilherme/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=/raid/brunoguilherme/.cache/sentence_transformers
export TORCH_HOME=/raid/brunoguilherme/.cache/torch
export PIP_CACHE_DIR=/raid/brunoguilherme/.cache/pip
mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

echo "SLURM job $SLURM_JOB_ID starting sweep on $(hostname) at $(date)"
python -u scripts/run_sbert11_sweep.py
echo "SLURM job $SLURM_JOB_ID sweep script exited at $(date)"
