#! /usr/bin/bash
#SBATCH --job-name=adapt_refine
#SBATCH --time=02:00:00
#SBATCH --account=hai_1252
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=4
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=4
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/adapt_%j.out
#SBATCH --error=logs/adapt_%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1
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

OUT_DIR=/p/home/jusers/toulouse1/juwels/project/toulouse1/flexcraft/data/adapt/full_run_2026-06-11_13:39:50
refine_per_task=5
ADAPT_CONFIG=./adapt_config.json
FAMILY_LIMIT=2
FULL_LIMIT=200

python ./flexcraft/pipelines/tcr/adapt/refine.py --designed_dir $OUT_DIR --refine_steps $refine_per_task --cdrs acdr3 bcdr3  --config $ADAPT_CONFIG --family_limit $FAMILY_LIMIT --full_limit $FULL_LIMIT