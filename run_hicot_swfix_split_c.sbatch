#!/bin/bash
#SBATCH --job-name=hicot-sw-c
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/brunoguilherme/DeepGenerativeTopicCluster/hicot_sw_c_slurm_%j.out

set -e
cd /raid/brunoguilherme/DeepGenerativeTopicCluster
git checkout slurm-gpu-experiments
git pull origin slurm-gpu-experiments
source /raid/brunoguilherme/venvs/dgtc/bin/activate

export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export VAEBM_RESULTS_DIR=/raid/brunoguilherme/DeepGenerativeTopicCluster/results/hicot_classif_stopword_fix_20260921
export HF_HOME=/raid/brunoguilherme/.cache/huggingface
export TRANSFORMERS_CACHE=/raid/brunoguilherme/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME=/raid/brunoguilherme/.cache/sentence_transformers
export TORCH_HOME=/raid/brunoguilherme/.cache/torch

for ds in bbc_news tweet stack_overflow biomedical banking77 m10 pascal_flickr; do
  python -u scripts/run_experiment.py --experiment classification --models hicot --datasets $ds --split random --seeds 42
done
