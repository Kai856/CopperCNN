#!/bin/bash
#SBATCH --job-name=build_grid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_grid_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

python preprocess/build_grid.py \
  --data-dir /cluster/scratch/$USER/sarig_data \
  --out-dir /cluster/scratch/$USER/copper_grid \
  --res 100
