#!/bin/bash -x
# Script for executing a full ADAPT experiment.
#
# example RUN_CONFIG:
# N_TASKS=2
# N_DESIGN=50 -> designs per binder
# N_REFINEMENT=100
# PEPTIDE="TLMSAMTNL"
# MHC_ALLELE="A*02:01"
# BINDERS="./data/adapt/input_data/binders_tcr.tsv"
# CDR_FILE="./data/adapt/input_data/paired_human_cdr3s.tsv"
# TYPE="tcr"
# OUT_DIR="./data/adapt/full_run_${current_time}"
# WD="."
# FAMILY_LIMIT=10
# FULL_LIMIT=100
# N_GPUS=1
#
# with:
# $ cat $BINDERS
# 5BS0	1OGA	3QDG
# and
# $ head $CDR_FILE
# cdr3a   cdr3b
# CAASFGSNYKLTF   CASSAQSTARYEQYF
# CALSVGQGGSEKLVF CASTLAREQFF
# CAASGLGGNEKLTF  CASSPRGRPTDTQYF
# CAASSSLNYGGSQGNLIF      CASSFAVDRGTYGYTF
# CAMSASSSNTGKLIF CASSDPSMGAGGDNEQFF
# CAASSIYSGNQFYF  CSASGVTGTGELFF
# CIVRVGTSYDKVIF  CASSWVTYEQYF
# CALLNTNAGKSTF   CSARVTGTTYNEQFF
# CAVDLSKGAQKLVF  CASRKQEYNEQFF
# 
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


current_time=$(date +"%Y-%m-%d_%H:%M:%S")

ADAPT_CONFIG=$1
RUN_CONFIG=$2
source $RUN_CONFIG
mkdir $OUT_DIR


# launch designers for all binders
# define array b as binder list
IFS=$'\t' read -ra b <  "$BINDERS"
n_binders=${#b[@]}

binders_per_task=$(($n_binders / $N_TASKS))
# if less binders than tasks set one task per binder
if [ $binders_per_task -le 0 ]; then
    binders_per_task=1
fi
echo "binders_per_task $binders_per_task"

pids=()
i=0

for slice in $(seq 0 $binders_per_task $(($n_binders-1))); do
    echo "slice ${slice}"
    binders=${b[@]:$slice:$binders_per_task}
    echo "binders ${binders}"

    gpu=$(( i % N_GPUS ))
    echo "assigning slice ${slice} to GPU ${gpu}"
    
    if [ "$TYPE" = "ab" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python ./flexcraft/pipelines/tcr/adapt/design.py --config $ADAPT_CONFIG --peptide $PEPTIDE --mhc_allele $MHC_ALLELE --binder $binders --cdrs $CDR_FILE --ab --out_dir $OUT_DIR --random_cdr --design_steps $N_DESIGN --prepare_only &
    fi
    if [ "$TYPE" = "tcr" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python ./flexcraft/pipelines/tcr/adapt/design.py --config $ADAPT_CONFIG --peptide $PEPTIDE --mhc_allele $MHC_ALLELE --binder $binders --cdrs $CDR_FILE --out_dir $OUT_DIR --random_cdr --design_steps $N_DESIGN --prepare_only &
    fi
    pids+=("$!")
    i=$(( i + 1 ))
done
# wait for runs to finish
for pid in ${pids[*]}; do
    wait $pid
done
