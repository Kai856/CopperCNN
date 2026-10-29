#!/bin/bash
#SBATCH --job-name=make_dataset
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/make_dataset_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

python -u preprocess/make_dataset.py \
  --grid-dir /cluster/scratch/$USER/copper_grid \
  --out-dir /cluster/scratch/$USER/copper_dataset
