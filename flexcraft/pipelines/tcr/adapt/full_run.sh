#! /usr/bin/bash
# Script for executing a full ADAPT experiment.
#
# example RUN_CONFIG:
# N_TASKS=2
# N_REFINEMENT=100
# PEPTIDE="TLMSAMTNL"
# MHC_ALLELE="A*02:01"
# BINDERS="./data/adapt/input_data/binders_tcr.tsv"
# CDR_FILE="./data/adapt/input_data/paired_human_cdr3s.tsv"
# TYPE="ab"
# OUT_DIR="./data/adapt/full_run_${current_time}"
# WD="."
#
# example ADAPT_CONFIG:
# {
# "op_dir":"./data/adapt/",
# "key":42,
# "boltz_config":{
#     "boltz_docking":true,
#     "boltz_redocking":false,
#     "boltz_parameter_path":"./params/boltz",
#     "boltz_model_name":"boltz2_conf",
#     "boltz_num_recycle":0,
#     "boltz_num_samples":1,
#     "boltz_num_sampling_steps":25,
#     "boltz_deterministic":false},
# "af2_config":{
#     "af2_model":null,
#     "af2_params":null,
#     "af2_model_name":"params_model_2_multimer_v3",
#     "af2_parameter_path":"./params/af/params/",
#     "af2_multimer":null,
#     "af2_num_recycle":0},
# "pmpnn_config":{
#     "pmpnn_model":null,
#     "pmpnn_parameter_path":"./params/pmpnn/v_48_020.pkl",
#     "pmpnn_hparams":{}},
# "mhc_chain_index":0,
# "tcr_chain_index":[2,3],
# "name":"testing_design",
# "out_dir":null,
# "trim":true,
# "chain_cache_len":450
# }
current_time=$(date +"%Y-%m-%d %T")

ADAPT_CONFIG="./adapt_config.json"
RUN_CONFIG="./run_config.sh"
source $RUN_CONFIG

cd $WD
mkdir $OUT_DIR

source ~/.bashrc
#conda init
#conda activate flexcraft
#module load devel/cuda/12.9

# launch designers for all binders
# define array b as binder list
IFS=$'\t' read -ra b <  "$BINDERS"
n_binders=${#b[@]}

binders_per_task=$(($n_binders / $N_TASKS))
echo "binders_per_task $binders_per_task"


for slice in $(seq 0 $binders_per_task $(($n_binders-1))); do
    echo "slice ${slice}"
    binders=${b[@]:$slice:$binders_per_task}
    echo "binders ${binders}"
    if [ "$TYPE" = "ab" ]; then
        echo "python ./flexcraft/pipelines/tcr/adapt/design.py --config $ADAPT_CONFIG --peptide $PEPTIDE --mhc_allele $MHC_ALLELE --binder $binders --cdrs $CDR_FILE --ab &"
    fi
    if [ "$TYPE" = "tcr" ]; then
        echo "python ./flexcraft/pipelines/tcr/adapt/design.py --config $ADAPT_CONFIG --peptide $PEPTIDE --mhc_allele $MHC_ALLELE --binder $binders --cdrs $CDR_FILE &"
    fi
done
# wait for designs to finish
wait

refine_per_task=$(($N_REFINEMENT / $N_TASKS))
for n in $(seq $N_TASKS); do
    echo "python ./flexcraft/pipelines/tcr/adapt/refine.py --designed_dir $OUT_DIR --refine_steps $refine_per_task --cdrs acdr3 bcdr3" &
done

