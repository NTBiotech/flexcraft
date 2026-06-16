#! /usr/bin/bash
# notes on ressource consumption:
# one task seems to use approx one cpu (always add buffer) and 3GB of memory
# 
N_TASKS=4
N_REFINEMENT=100
PEPTIDE="TLMSAMTNL"
TYPE="tcr"
OUT_DIR="./data/adapt/tuning_run_${current_time}"
WD="/p/home/jusers/toulouse1/juwels/project/toulouse1/flexcraft"
CONDA_PATH="miniforge3"
CONDA_ENV="flexcraft"
FAMILY_LIMIT=2
FULL_LIMIT=10
N_GPUS=4
PREPARED=false
N_DESIGN=2
BINDERS="./data/adapt/input_data/binders_tcr.tsv"
MHC_ALLELE="A*02:01"
CDR_FILE="./data/adapt/input_data/paired_human_cdr3s.tsv"




OUT_DIR=/p/project1/hai_1252/toulouse1/flexcraft/data/adapt/adapt_tuning_prepared
PREPARE=True
PREPARED=False
