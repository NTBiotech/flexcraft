'''
Script for testing adapt pipeline.
Run in project root!
'''

from flexcraft.pipelines.tcr import *
from flexcraft.utils.rng import Keygen

import os
from pathlib import Path

#load_data(out_dir = "./data/adapt/input_data")

adapt = ADAPT(
    op_dir= "./data/adapt/",
    #af2_parameter_path="./params/af/params",
    boltz_docking=False,
    boltz_num_samples=2,
    af2_model_name="model_2_ptm_ft_binder_20230729",
    key=Keygen(42),
    pmpnn_parameter_path="./params/pmpnn/v_48_030.pkl",
    af2_multimer=False,
    af_num_recycle=0,
    pmpnn_hparams={},
    ab=False,
    mhc_chain_index=0,
    tcr_chain_index=(2,3),
    name="test_adapt_design",
    trim=True,
    #out_dir = Path(os.environ["TMP"])/"test_adapt_design",
)

TCR_STRUCTURES = ["4Y1A","8d5q", "2OI9", "5VCJ"]
#PMHC_STRUCTURES = ["8d5q",]

for pdb_id in TCR_STRUCTURES:
    if not "." in pdb_id:
        pdb_path = download_structure(pdb_id, output_dir="./data/adapt/input_data")
        pdb_path = clean_chothia(pdb_path)
    else:
        pdb_path = pdb_id
    
    with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
        ids = rf.readline().strip("\n").split("\t")
        cdrs = {
            i[-1]+i[:-1]:n
            for i,n in zip(ids, rf.readline().strip("\n").split("\t"))
        }
    print(cdrs)
    adapt.design_trial(
        scaffold=pdb_path,
        pMHC=None,
        cdrs=cdrs
    )