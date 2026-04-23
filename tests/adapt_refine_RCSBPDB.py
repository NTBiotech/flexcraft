'''
Script for testing adapt pipeline.
Run in project root!
'''

from flexcraft.pipelines.tcr import *
from flexcraft.utils.rng import Keygen

import os
import requests
from pathlib import Path

#load_data(out_dir = "./data/adapt/input_data")

adapt = ADAPT(
    op_dir= "./data/adapt/",
    #af2_parameter_path="./params/af/params",
    boltz_docking=True,
    af2_model_name="model_2_ptm_ft_binder_20230729",
    key=Keygen(42),
    pmpnn_parameter_path="./params/pmpnn/v_48_030.pkl",
    af2_multimer=False,
    af_num_recycle=0,
    pmpnn_hparams={},
    ab=False,
    mhc_chain_index=0,
    tcr_chain_index=(2,3),
    name="test_adapt_refine",
    trim=True,
    boltz_redocking=False,
    #out_dir = Path(os.environ["TMP"])/"test_adapt_design",
)


def download_structure(pdb_id: str, file_format: str = "pdb", output_dir: str = "."):
    """
    Download a structure file from RCSB PDB.
    
    file_format: 'pdb', 'cif' (mmCIF), or 'bcif' (BinaryCIF)
    """
    base_urls = {
        "pdb": f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        "cif": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
        "bcif": f"https://models.rcsb.org/{pdb_id.lower()}.bcif",
    }
    url = base_urls[file_format]
    response = requests.get(url)
    response.raise_for_status()

    suffix = {"pdb": ".pdb", "cif": ".cif", "bcif": ".bcif"}[file_format]
    out_path = Path(output_dir) / f"{pdb_id.upper()}{suffix}"
    out_path.write_bytes(response.content)
    return out_path

TCR_STRUCTURES = ["4Y1A","8d5q", "2OI9", "5VCJ"]
PMHC_STRUCTURES = ["4Y1A","8d5q", "2OI9", "5VCJ"]
TMP_DIR = Path(os.environ["TMP"])/"test_adapt_design"

for pdb_id, pmhc_id in zip(TCR_STRUCTURES, PMHC_STRUCTURES):
    if not "." in pdb_id:
        pdb_path = download_structure(pdb_id, output_dir="./data/adapt/input_data")
        pdb_path = clean_chothia(pdb_path)
    else:
        pdb_path = pdb_id
    if not "." in pmhc_id:
        pmhc_path = download_structure(pmhc_id, output_dir="./data/adapt/input_data")
        pmhc_path = clean_chothia(pdb_path)
    else:
        pmhc_path = pmhc_id

    adapt.refine_trial(
        scaffold=pdb_path,
    )