#! /usr/bin/bash
N_TASKS=2
N_REFINEMENT=10
PEPTIDE="TLMSAMTNL"
MHC_ALLELE="A*02:01"
BINDERS="./data/adapt/input_data/binders_tcr.tsv"
CDR_FILE="./data/adapt/input_data/paired_human_cdr3s.tsv"
TYPE="tcr"
OUT_DIR="./data/adapt/full_run_${current_time}"
WD="/home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft"
CONDA_ENV="flexcraft"