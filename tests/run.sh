#!/bin/bash
#SBATCH --job-name=adapt_tune
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --output=logs/adapt_tune_%j.out
#SBATCH --error=logs/adapt_tune_%j.err

source ~/.bashrc

conda init
conda activate flexcraft
module load devel/cuda/12.9

cd /home/hgf_dkfz/hgf_dsb0249/workspaces/haicwork/hgf_dsb0249-BinderDesign/flexcraft

python tests/adapt_tuning.py
