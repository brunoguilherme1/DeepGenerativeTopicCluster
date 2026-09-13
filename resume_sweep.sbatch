#!/bin/bash
#SBATCH --job-name=sbert11-resume
#SBATCH --partition=defq
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/resume_slurm_%j.out

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

echo "SLURM job $SLURM_JOB_ID resuming sweep on $(hostname) at $(date)"
python -u scripts/run_sbert11_sweep.py --run-dir results/sbert11_ecrtm_hicot_20260912_181146
echo "SLURM job $SLURM_JOB_ID resume script exited at $(date)"
