#!/bin/bash -x
#SBATCH --job-name=adapt
#SBATCH --time=00:10:00
#SBATCH --account=hai_1252
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=1
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=1
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/adapt_refine_%j.out
#SBATCH --error=logs/adapt_refine_%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
# For gpus and and booster partition

CONDA_PATH="miniforge3"             # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"

jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster
module purge
module load CUDA-Python/12
source ~/.bashrc
cd "$PROJECT_DIR/toulouse1"

source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"

cd "${REPO_NAME}"
pwd

python flexcraft/pipelines/tcr/tcrdock/cluster_tcr.py --exec --out_dir data/adapt/clustering_mhc2 --structure_table data/adapt/input_data/tcr3d_data/mhc2.csv --mhc_class 1 --iplddt 0.7 --gpu 1