from flexcraft.pipelines.tcr.adapt import *
from flexcraft.pipelines.tcr.utils import *
import numpy as np
from typing import List, Tuple, Optional, Dict, Iterable, Callable

from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser(
    usage="Script for running design trials using the ADAPT class."
)

parser.add_argument("--designed_dir", type=Path)
parser.add_argument("--refine_steps", type=int, default=100)
parser.add_argument("--cdrs", nargs="*")


parser.add_argument("--config", default="./config.json",)

args = parser.parse_args()


if __name__ == "__main__":

    config = json.load(open(args.config, "r"))
    # check if subdirs exist
    out_dir = args.designed_dir
    if not (out_dir/"scores.csv").exists():
        raise FileNotFoundError(f"No scores.csv in designed_dir {out_dir}!")
    config.update(out_dir=out_dir)
    adapt = ADAPT(
    **config
    )
    for n in range(args.refine_steps):
        design, pdb_path, design_name = adapt.get_design(
            index="random"
        )

        adapt.refine_trial(
            scaffold=design,
            scaffold_name=design_name,
            cdrs=list(args.cdrs)
        )

    print(f"Finished design run!\nCollecting results...")
