#!/bin/bash
#SBATCH --job-name=train_all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_all_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

DATA=/cluster/scratch/$USER/copper_dataset64
RUNS=/cluster/scratch/$USER/copper_runs

run() {
  echo "=== $1 ==="
  python -u -m train.train --data-dir "$DATA" --out-root "$RUNS" --run-name "$@"
}

run base32 --crop 32
run wide64
run wide64_rock --rock
run wide64_rock_depth --rock --depth

python -u -m train.summarize_runs --data-dir "$DATA" --runs-dir "$RUNS" \
  base32 wide64 wide64_rock wide64_rock_depth
