#!/bin/bash
#SBATCH --job-name=bertopic-fm
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/bertopic_fm_slurm_%j.out

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

python -u scripts/run_baseline_sweep_12ds.py --model bertopic --experiments cluster --run-dir results/bertopic_cluster_fullmetrics_futurelab_20260921
