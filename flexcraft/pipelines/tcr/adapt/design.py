from flexcraft.pipelines.tcr.adapt import *
from flexcraft.pipelines.tcr.utils import *
import numpy as np
from typing import List, Tuple, Optional, Dict, Iterable, Callable
from datetime import datetime

from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser(
    usage="Script for running design trials using the ADAPT class."
)

parser.add_argument("--peptide", nargs="*", default=[],)
parser.add_argument("--mhc_allele", nargs="*", default=[],)
parser.add_argument("--binder", nargs="*", default=[],)
parser.add_argument("--cdrs", type=str, default=None)
parser.add_argument("--ab", action="store_true")


parser.add_argument("--config", default="./config.json",)

args = parser.parse_args()
            

if __name__ == "__main__":

    config = json.load(open(args.config, "r"))
    mhcs = list(args.mhc_allele)
    peptides = list(args.peptide)
    binders = list(args.binder)
    cdrs_gen = cdr_parser(args.cdrs)
    
    if (len(mhcs)>1) and (len(peptides)>1):
        out_dir = config.get("out_dir", config.get("op_dir", ".")+f"adapt_design_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}/")
    
    for mhc, peptide in zip(mhcs, peptides):
        if (len(mhcs)>1) and (len(peptides)>1):
            config.update({"out_dir":out_dir+f"{mhc}_{peptide}"})

        adapt = ADAPT(
            **config
        )
    
        print(f"---Designing mhc {mhc} with peptide {peptide}---")
        mhc_seq = get_mhc(name=mhc)

        for binder in binders:
            print(f"Using Binder {binder}...")
            if len(binder)==4:
                # assume pdb id
                binder_path = download_structure(
                    binder,
                    file_format="antibody" if args.ab else "biological assembly",
                    out_dir=config["op_dir"]+"input_data"
                )
            # else assume pdb path
            binder_path = clean_chothia(binder_path)
            
            cdrs = cdrs_gen()

            scaffold, scaffold_name = adapt.make_scaffold(
                receptor=binder_path,
                presenter=mhc_seq,
                antigen=peptide,
                cdrs=cdrs,
                replace_antigen=True,
            )

            adapt.design_trial(
                design=scaffold,
                scaffold_name=scaffold_name,
                cdrs=list(cdrs.keys())
            )
    print(f"Finished design run!")
    if (len(mhcs)>1) and (len(peptides)>1):
        print("Collected results at: ",collect_results(Path(out_dir), pattern=f"**/*{adapt.name}*", save=True))
