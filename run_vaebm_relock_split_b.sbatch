#!/bin/bash
#SBATCH --job-name=vaebm-rl-b
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/vaebm_rl_b_slurm_%j.out

set -e
cd /raid/brunoguilherme/DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
git pull origin slurm-gpu-experiments
source /raid/brunoguilherme/venvs/dgtc/bin/activate

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export VAEBM_RESULTS_DIR=/raid/brunoguilherme/DeepGenerativeTopicCluster/results/vaebm_classif_relock_logreg_20260921
export HF_HOME=/raid/brunoguilherme/.cache/huggingface
export TRANSFORMERS_CACHE=/raid/brunoguilherme/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=/raid/brunoguilherme/.cache/sentence_transformers
export TORCH_HOME=/raid/brunoguilherme/.cache/torch
export VAEBM_EMBEDDER=thenlper/gte-large
export VAEBM_UNITS=1024
export VAEBM_DIM_EMB=""
export VAEBM_FREEZE_EMB=1
export VAEBM_ALPHA=0.0
export VAEBM_LR=1e-4
export VAEBM_EPOCHS=1

for ds in m10 pascal_flickr; do
  python -u scripts/run_experiment.py --experiment classification --models vaebm --datasets $ds --split random --seeds 1 2 3 4 5
done
