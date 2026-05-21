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
parser.add_argument("--out_dir", type=Path, default=Path("."))
parser.add_argument("--random_cdr", action="store_true")
parser.add_argument("--design_steps", type=int, default=1,
    help="Number of design attempts for each binder.")
parser.add_argument("--cdr_length", type=int, default=None,
    help="Pin the cdr length to a specific int")
parser.add_argument("--templates", nargs="*", default=None,
    help="Template TCR-MHC-complexes in pdb files. Directory as input supported.\
        Also set --template_mhc_class when using this feature. Overwrites config templates!")
parser.add_argument("--template_mhc_class", nargs="*", default=None,
    help="Add MHC class of templates. If one element, is broadcasted to all templates.")
parser.add_argument("--mhc_class", type=int, default=None)


parser.add_argument("--config", default="./config.json",)


# parse arguments
args = parser.parse_args()
config = json.load(open(args.config, "r"))
mhcs = list(args.mhc_allele)
peptides = list(args.peptide)
binders = list(args.binder)
out_dir = args.out_dir
templates = args.templates
template_mhc_class = args.template_mhc_class

if not args.mhc_class is None:
    config.update(mhc_class = args.mhc_class)

# cdr generator
cdrs_gen = cdr_parser(args.cdrs, random=args.random_cdr, cdr_length=args.cdr_length, patience=100)
# out directory
if not args.out_dir == Path("."):
    config.update(out_dir=args.out_dir)

if (len(mhcs)>1) and (len(peptides)>1):
    out_dir = config.get("out_dir", config.get("op_dir", ".")+f"adapt_design_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}/")
    if not out_dir.exists():
        out_dir.mkdir()

if not templates is None:
    config.update(templates=list(templates))
    config.update(template_mhc_class=list(template_mhc_class))

for mhc, peptide in zip(mhcs, peptides):
    if (len(mhcs)>1) and (len(peptides)>1):
        config.update({"out_dir":Path(out_dir)/f"{mhc}_{peptide}"})  # pyright: ignore[reportOperatorIssue]

    adapt = ADAPT(
        **config
    )

    print(f"---Designing mhc {mhc} with peptide {peptide}---")
    get_structure = False
    # table with the right columns
    if mhc.endswith(".csv"):
        table:pd.DataFrame = pd.read_csv(mhc)
        if "Bound to TCR" in table.columns:
            table = table[~table["Bound to TCR"].astype(bool)]
        if "Species" in table.columns:
            table = table[table["Species"]=="Human"]
        if "Resolution" in table.columns:
            table = table[table["Resolution"].astype(float)<4]
        mhc = table.sample(1)["PDB ID"]
    # HLA id
    if mhc.startswith("HLA"):
        mhc_seq = get_mhc(accession=mhc)
        get_structure=True
    elif mhc[1] == "*":
        mhc_seq = get_mhc(name=mhc)
        get_structure=True
    # PDB ID
    elif len(mhc) == 4:
        mhc_seq = clean_chothia(download_structure(mhc, file_format="biological assembly", out_dir=config["op_dir"]+"/input_data"))
    # pdb file
    elif mhc.endswith(".pdb"):
        mhc_seq = clean_chothia(mhc)
        

    for binder in binders:
        print(f"Using Binder {binder}...")
        if len(binder)==4:
            # assume pdb id
            binder = download_structure(
                binder,
                file_format="antibody" if args.ab else "biological assembly",
                out_dir=config["op_dir"]+"/input_data"
            )
        # else assume pdb path
        binder_path = clean_chothia(binder)
        print("Components: ",binder_path,mhc_seq,peptide,sep="\n---\n")

        for n in range(args.design_steps):
            print(f"\nDesign step {n}")
            cdrs = cdrs_gen()
            scaffold, scaffold_name = adapt.make_scaffold(
                receptor=binder_path,
                presenter=mhc_seq,
                antigen=peptide,
                cdrs=cdrs,
                replace_antigen=True,
                get_structure=get_structure
            )

            adapt.design_trial(
                design=scaffold,
                scaffold_name=scaffold_name,
                cdrs=list(cdrs.keys())
            )
print(f"Finished design run!\n")
if (len(mhcs)>1) and (len(peptides)>1):
    print("Collected results at: ",collect_results(Path(out_dir), pattern=f"**/*{adapt.name}*", save=True))
