#!/bin/bash
#SBATCH --job-name=adapt_full_run_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20gb
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:full:1
#SBATCH --output=logs/adapt_full_run_test_%j.out
#SBATCH --error=logs/adapt_full_run_test_%j.err

source ~/.bashrc

conda init
conda activate flexcraft
module load devel/cuda/12.9

cd /home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft

./flexcraft/pipelines/tcr/adapt/full_run.sh
