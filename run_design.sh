#!/bin/bash
#SBATCH --job-name=adapt_refine
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --output=logs/adapt_refine_%j.out
#SBATCH --error=logs/adapt_refine_%j.err

source ~/.bashrc

OUT_DIR=/home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft/data/adapt/full_run_2026-06-03_17:16:33
refine_per_task=5
ADAPT_CONFIG=./adapt_config.json
FAMILY_LIMIT=2
FULL_LIMIT=200

conda init
conda activate flexcraft
module load devel/cuda/12.9

cd /home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft

python flexcraft/pipelines/tcr/adapt/design.py --config adapt_config.json --binder 5BS0 --cdrs data/adapt/input_data/paired_human_cdr3s.tsv --peptide TLMSAMTNL --mhc_allele A*02:01 --mhc_class 1 --templates /p/home/jusers/toulouse1/juwels/project/toulouse1/flexcraft/data/adapt/clustering_mhc1/representative --template_mhc_class 1