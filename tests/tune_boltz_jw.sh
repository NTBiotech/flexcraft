#! /usr/bin/bash
CONDA_PATH="miniforge3"             # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"
jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster
CONC_LIMIT=3

root="${PROJECT_DIR}/toulouse1/flexcraft"
echo "root: ${root}"
test_dir="${root}/tests/test_configs"
out_parent="${root}/data/adapt"

run_config="${test_dir}/run_config_jw.sh"
tmp_config="${test_dir}/run_config_jw_temp.sh"
rm $tmp_config
cp $run_config $tmp_config

# prepare constructs using boltz with msa
construct_config="${test_dir}/construct_adapt_config.json"
prepared_dir="${out_parent}/adapt_tuning_prepared"
cat <<EOF >>$tmp_config
PREPARE=True
PREPARED=False
SEED=0
EOF

cat <<EOF
#! /usr/bin/bash
#SBATCH --job-name=prepare_adapt_tuning
#SBATCH --time=02:00:00
#SBATCH --account=${PROJECT_NAME}
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=4
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=4
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/adapt_tuning_prepare_%j.out
#SBATCH --error=logs/adapt_tuning_prepare_%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
# For gpus and and booster partition

jutil env activate -p "$PROJECT_NAME"

module purge
module load CUDA-Python/12
source ~/.bashrc
cd "$PROJECT_DIR/toulouse1"

source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"

cd ${root}
OUT_DIR=${prepared_dir} ./flexcraft/pipelines/tcr/adapt/full_run_jw.sh $construct_config $tmp_config
EOF

# wait till job finished
while (($(squeue|grep toulouse|wc -l) >= 1)) #TODO: FIX to 1!
do
sleep 10
done

# overwrite BINDERS in run_config to prepared dir
cat <<EOF >> $tmp_config
BINDERS=${prepared_dir}
PREPARED=True
PREPARE=False
N_DESIGN=1
EOF



for c in ${test_dir}/adapt_config_*.json; do
while (($(squeue|grep toulouse|wc -l) >= $CONC_LIMIT))
do
sleep 10
done
echo $c
c_name=$(basename $c ".json")
sbatch <<EOF
#! /usr/bin/bash
#SBATCH --job-name=${c_name}_adapt_tuning
#SBATCH --time=02:00:00
#SBATCH --account=${PROJECT_NAME}
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=4
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=4
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/adapt_tuning_${c_name}_%j.out
#SBATCH --error=logs/adapt_tuning_${c_name}_%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
# For gpus and and booster partition

jutil env activate -p "$PROJECT_NAME"

module purge
module load CUDA-Python/12
source ~/.bashrc
cd "$PROJECT_DIR/toulouse1"

source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"

cd ${root}


OUT_DIR="${out_parent}/adapt_tuning_${c_name}" ./flexcraft/pipelines/tcr/adapt/full_run_jw.sh $c $tmp_config
EOF
# wait 10s to overwrite the config
sleep 10
done

#rm $tmp_config