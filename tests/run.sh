#!/bin/bash
#SBATCH --job-name=adapt_full_run_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:full:2
#SBATCH --output=logs/adapt_full_run_test_%j.out
#SBATCH --error=logs/adapt_full_run_test_%j.err

source ~/.bashrc

conda init
conda activate flexcraft
module load devel/cuda/12.9

cd /home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft

./flexcraft/pipelines/tcr/adapt/full_run.sh
