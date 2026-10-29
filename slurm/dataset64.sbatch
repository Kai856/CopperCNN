#!/bin/bash
#SBATCH --job-name=dataset64
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dataset64_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

DATA=/cluster/scratch/$USER/copper_dataset64

python -u preprocess/make_dataset.py \
  --grid-dir /cluster/scratch/$USER/copper_grid \
  --sarig-dir /cluster/scratch/$USER/sarig_data \
  --out-dir "$DATA" --tile 64
python -u preprocess/make_folds.py --data-dir "$DATA"
python -u preprocess/normalize.py --data-dir "$DATA" --figure figures/example_tiles64.png
