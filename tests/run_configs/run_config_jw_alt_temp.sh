#! /usr/bin/bash
# notes on ressource consumption:
# one task seems to use approx one cpu (always add buffer) and 3GB of memory
# 
N_TASKS=4
N_REFINEMENT=100000 # just wait until job time over
PEPTIDE="EVDPIGHLY"
TYPE="tcr"
#OUT_DIR="./data/adapt/tuning_run_${current_time}"
WD="/p/home/jusers/toulouse1/juwels/project/toulouse1/flexcraft"
CONDA_PATH="miniforge3"
CONDA_ENV="flexcraft"
FAMILY_LIMIT=20
FULL_LIMIT=100
N_GPUS=4
PREPARED=false
N_DESIGN=50
BINDERS="./data/adapt/input_data/binders_tcr.tsv"
MHC_ALLELE="A*01:01"
CDR_FILE="./data/adapt/input_data/paired_human_cdr3s.tsv"
SEED=42



PREPARE=True
PREPARED=False
OUT_DIR=/p/project1/hai_1252/toulouse1/flexcraft/data/adapt/full_run/2026-06-28_12:18:30_adapt_full_run_prepared
BINDERS=/p/project1/hai_1252/toulouse1/flexcraft/data/adapt/full_run/2026-06-28_12:18:30_adapt_full_run_prepared
PREPARED=True
PREPARE=False
N_DESIGN=1
OUT_DIR="/p/project1/hai_1252/toulouse1/flexcraft/data/adapt/full_run/2026-06-28_12:18:30_adapt_full_run_c5"
