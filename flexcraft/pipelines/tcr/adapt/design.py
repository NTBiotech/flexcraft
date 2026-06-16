from flexcraft.pipelines.tcr.adapt import *
from flexcraft.pipelines.tcr.utils import *
import numpy as np
from typing import List, Tuple, Optional, Dict, Iterable, Callable
from datetime import datetime

from pathlib import Path
import json
import argparse

def _bool(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"{value} not a supported bool format!")

def _str_or_None(value):
    if value.lower()=="none":
        return None
    else:
        return value

parser = argparse.ArgumentParser(
    usage="Script for running design trials using the ADAPT class."
)

parser.add_argument("--peptide", nargs="*", default=None,)
parser.add_argument("--mhc_allele", nargs="*", default=None,)
parser.add_argument("--binder", nargs="*", default=None,)
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

parser.add_argument("--prepared", type=_bool, help="Wether input binders are prepared or need to be constructed.")
parser.add_argument("--prepare_only", type=_bool, help="If True, only scaffolds are prepared in out_dir.")


# parse arguments
args = parser.parse_args()
config = json.load(open(args.config, "r"))
out_dir = args.out_dir
templates = args.templates
template_mhc_class = args.template_mhc_class

# make "None" None
mhcs = [_str_or_None(x) for x in list(args.mhc_allele)]
peptides = [_str_or_None(x) for x in list(args.peptide)]
binders = [_str_or_None(x) for x in list(args.binder)]

# unpack binders if dir
_binders = []
for binder in binders:
    if Path(binder).is_dir():
        _binders.extend([p for p in Path(binder).glob("*.pdb")])
    else:
        _binders.append(binder)
binders = _binders

# extend other components to longest member
longest = max([len(x) for x in [mhcs, peptides]])
def _expand(x, l):
    if len(x) == 1:
        return x*l
    return x
mhcs = _expand(mhcs, longest)
peptides = _expand(peptides, longest)


# cdr generator
cdrs_gen = cdr_parser(args.cdrs, random=args.random_cdr, cdr_length=args.cdr_length, patience=100)

# update config 
if not args.mhc_class is None:
    config.update(mhc_class = args.mhc_class)
if not args.out_dir == Path(".") or "out_dir" not in config.keys():
    # set out directory if not default
    config.update(out_dir=args.out_dir)
if not templates is None:
    config.update(templates=list(templates))
    config.update(template_mhc_class=list(template_mhc_class))

if (len(mhcs)>1) and (len(peptides)>1):
    # make parent for each combi
    out_dir = config.get("out_dir", config.get("op_dir", ".")+f"adapt_design_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}/")
    if not out_dir.exists():
        out_dir.mkdir()

for mhc, peptide in zip(mhcs, peptides):
    print(f"---Designing mhc {mhc} with peptide {peptide}---")
    if (len(mhcs)>1) and (len(peptides)>1):
        # make subdir for each combination
        config.update({"out_dir":Path(out_dir)/f"{mhc}_{peptide}"})  # pyright: ignore[reportOperatorIssue]

    adapt = ADAPT(
        **config
    )
    config["boltz_config"].update(predictor=adapt.boltz_predictor)
    config["af2_config"].update(af2_model=adapt.af2_model)
    config["af2_config"].update(af2_params=adapt.af2_params)
    
    get_structure = False
    if args.prepared:
        peptide = None
        mhc_seq = None
        cdrs_gen = lambda: None
    else:
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
        else:
            mhc_seq = mhc

    for binder in binders:
        print(f"Using Binder {binder}...")
        if len(binder)==4:
            # assume pdb id
            binder_path = Path(config["op_dir"]+f"/input_data/{binder}.pdb")
            if not binder_path.exists():
                binder = download_structure(
                    binder,
                    file_format="antibody" if args.ab else "biological assembly",
                    out_dir=config["op_dir"]+"/input_data"
                )
            else:
                # assume path to pdb
                binder = binder_path
        if Path(binder).exists():
            # else assume pdb path
            binder_path = clean_chothia(binder)
        else:
            # assume sequence
            binder_path = binder

        print("Components: ",binder_path,mhc_seq,peptide,sep="\n---\n")

        for n in range(args.design_steps):
            print(f"\nDesign step {n}")
            cdrs = cdrs_gen()
            scaffold, scaffold_name = adapt.make_scaffold(
                receptor=binder_path,
                presenter=mhc_seq,
                antigen=peptide,
                cdrs=cdrs,
                get_structure=get_structure
            )
            if args.prepare_only:
                scaffold.save_pdb((config["out_dir"]/scaffold_name).with_suffix(".pdb"))
            else:
                adapt.design_trial(
                    design=scaffold,
                    scaffold_name=scaffold_name,
                    cdrs=list(cdrs.keys())
                )
print(f"Finished design run!\n")
if (len(mhcs)>1) and (len(peptides)>1):
    print("Collected results at: ",collect_results(Path(out_dir), pattern=f"**/*{adapt.name}*", save=True))
