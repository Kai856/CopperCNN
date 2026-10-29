#!/bin/bash
#SBATCH --job-name=random_forest
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/random_forest_%j.out

set -euo pipefail
source ~/venvs/copper/bin/activate
cd ~/CopperCNN

DATA=/cluster/scratch/$USER/copper_dataset64
RUNS=/cluster/scratch/$USER/copper_runs

run() {
  echo "=== $1 ==="
  python -u -m train.random_forest --data-dir "$DATA" --out-root "$RUNS" --run-name "$@"
}

run rf_geo
run rf_geo_rock --rock
run rf_geo_rock_depth --rock --depth

python -u -m train.summarize_runs --data-dir "$DATA" --runs-dir "$RUNS" \
  base32 wide64 wide64_rock wide64_rock_depth rf_geo rf_geo_rock rf_geo_rock_depth
