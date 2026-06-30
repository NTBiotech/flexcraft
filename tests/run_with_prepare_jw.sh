#! /usr/bin/bash
CONDA_PATH="miniforge3"             # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"
jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster
# deprecated for prepared, as sampling from same pool, duplicates designs
N_RUNS=1
current_time=$(date +"%Y-%m-%d_%H:%M:%S")

root="${PROJECT_DIR}/toulouse1/flexcraft"
echo "root: ${root}"
config_dir="${root}/tests/run_configs"
out_parent="${root}/data/adapt/full_run_alt"
mkdir "${out_parent}"

adapt_config="${config_dir}/adapt_config_4.json"
run_config="${config_dir}/run_config_jw_alt.sh"
tmp_config="${config_dir}/run_config_jw_alt_temp.sh"
rm $tmp_config
cp $run_config $tmp_config

# prepare constructs using boltz with msa
construct_config="${config_dir}/construct_adapt_full_run.json"
prepared_dir="${out_parent}/${current_time}_adapt_full_run_prepared"

cat <<EOF >>$tmp_config
PREPARE=True
PREPARED=False
OUT_DIR=${prepared_dir}
EOF
sbatch <<EOF
#! /usr/bin/bash
#SBATCH --job-name=prepare_adapt_tuning_${current_time}
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
#SBATCH --output=logs/${current_time}_adapt_tuning_prepare_%j.out
#SBATCH --error=logs/${current_time}_adapt_tuning_prepare_%j.err
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
SEED=9 ./flexcraft/pipelines/tcr/adapt/full_run_jw.sh $construct_config $tmp_config
EOF
# wait till job finished
sleep 10
while (($(squeue|grep toulouse|wc -l) >= 2))
do
sleep 10
done

# overwrite BINDERS in run_config to prepared dir
cat <<EOF >> $tmp_config
BINDERS=${prepared_dir}
PREPARED=True
PREPARE=False
N_DESIGN=1
OUT_DIR="${out_parent}/${current_time}_adapt_full_run_c4"
EOF

for i in $(seq 1 $N_RUNS); do
sbatch <<EOF
#! /usr/bin/bash
#SBATCH --job-name=${current_time}_adapt_run_${i}
#SBATCH --time=10:00:00
#SBATCH --account=${PROJECT_NAME}
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks=4
# can be omitted if --nodes and --ntasks-per-node
# are given
#SBATCH --ntasks-per-node=4
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --output=logs/${current_time}_${i}_adapt_run_%j.out
#SBATCH --error=logs/${current_time}_${i}_adapt_run_%j.err
#SBATCH --partition=booster
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


SEED=$i ./flexcraft/pipelines/tcr/adapt/full_run_jw.sh $adapt_config $tmp_config
EOF
# reduce concurrent reading by 10s offset
sleep 10

done


# rm $tmp_config