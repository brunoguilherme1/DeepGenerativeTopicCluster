#!/bin/bash
#SBATCH --job-name=hicot-fm-c
#SBATCH --partition=defq
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/hicot_fm_c_slurm_%j.out

set -e
cd /raid/brunoguilherme/DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
git pull origin slurm-gpu-experiments
source /raid/brunoguilherme/venvs/dgtc/bin/activate

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export VAEBM_RESULTS_DIR=/raid/brunoguilherme/DeepGenerativeTopicCluster/results/hicot_cluster_fullmetrics_futurelab_20260921
export HF_HOME=/raid/brunoguilherme/.cache/huggingface
export TRANSFORMERS_CACHE=/raid/brunoguilherme/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=/raid/brunoguilherme/.cache/sentence_transformers
export TORCH_HOME=/raid/brunoguilherme/.cache/torch

for ds in biomedical banking77 m10 pascal_flickr; do
  python -u scripts/run_experiment.py --experiment cluster --models hicot --datasets $ds --seed 42
done
