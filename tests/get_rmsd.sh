#! /usr/bin/bash
#SBATCH --job-name=adapt_rmsd
#SBATCH --time=02:00:00
#SBATCH --account=hai_1252
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --output=logs/adapt_rmsd%j.out
#SBATCH --error=logs/adapt_rmsd%j.err
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
# For gpus and and booster partition

CONDA_PATH="miniforge3"             # path to miniforge or ... relative to project dir
CONDA_ENV="flexcraft"               # e.g. flexcraft
PROJECT_NAME="hai_1252"             # project id on cluster
REPO_NAME="flexcraft"
CONC_LIMIT=2

jutil env activate -p "$PROJECT_NAME"
PROJECT_DIR="$PROJECT"   # absolute path on cluster
module purge
module load CUDA-Python/12
source ~/.bashrc
cd "$PROJECT_DIR/toulouse1"

source "$CONDA_PATH/bin/activate"
conda activate "$CONDA_ENV"

cd "${REPO_NAME}"

directory=/p/project1/hai_1252/toulouse1/flexcraft/data/adapt

pids=()

for d in "${directory}"/adapt_tuning_adapt_config_*; do
if [ -d "$d" ]; then
if (( ${#pids[@]} >= $CONC_LIMIT )); then
for pid in ${pids[*]}; do
    wait $pid
done
pids=()
fi
echo $d
python <<EOF &
from flexcraft.structure.metrics import RMSD
from flexcraft.files.pdb import PDBFile
from pathlib import Path
import pandas as pd
import numpy as np
from flexcraft.pipelines.tcr.utils import pad_design

files = [f for f in Path("$d").glob("*.pdb")]
out_array = pd.DataFrame(columns=files, index=files)
rmsd = RMSD()
def get_rmsd(file1, file2):
    design1 = PDBFile(path=file1).to_data()
    design1, mask1 =pad_design(550, input_design=design1,covariates= [np.ones(len(design1["aa"]))])
    design2 = PDBFile(path=file2).to_data()
    design2, mask2 =pad_design(550, input_design=design2,covariates= [np.ones(len(design2["aa"]))])
    mask = min([mask1,mask2], key=sum)
    return rmsd(design1, design2, mask=mask)

for n1, f1 in enumerate(files):
    for n2, f2 in enumerate(files):
        print([f1,f2], sep="->\n")
        out_array.loc[f1,f2]=get_rmsd(f1,f2)
print("saving to ",Path("$d")/"rmsd_new.csv")
out_array.to_csv(Path("$d")/"rmsd_new.csv")
EOF

pids+=("$!")
fi
done

for pid in ${pids[*]}; do
    wait $pid
done
