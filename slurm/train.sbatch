#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

python -u -m train.train \
  --data-dir /cluster/scratch/$USER/copper_dataset \
  --out-root /cluster/scratch/$USER/copper_runs \
  "$@"
