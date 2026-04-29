from flexcraft.pipelines.tcr import *
import numpy as np

from pathlib import Path


class DesignQueue:
    def __init(
    input_dir:Path,
    cdr_file:str,
    tcr_dir:str,
    pmhc_dir:str|None = None,
    ):
        self.out = []
        zipper = []
        keys = []
        # check if everything is here
        if not (input_dir/tcr_dir).exists():
            raise FileNotFoundError(f"tcr_dir {tcr_dir} not found in input_dir ({input_dir})!")
        keys.append("scaffold")
        zipper.append((input_dir/tcr_dir).glob(".pdb"))
        if not cdr_file is None:
            if not (input_dir/cdr_file).exists():
                raise FileNotFoundError(f"cdr_file {cdr_file} not found in input_dir ({input_dir})!")
            keys.append("cdr3s")
            zipper.append((
                l
                for l in cdr_file
                ))
        if not pmhc_dir is None:
            if not (input_dir/pmhc_dir).exists():
                raise FileNotFoundError(f"pmhc_dir {pmhc_dir} not found in input_dir ({input_dir})!")
        self.input_dir = input_dir
        self.cdr_file = cdr_file
        self.tcr_dir = tcr_dir
        self.pmhc_dir = pmhc_dir
        self.out = [

            for n in zip(*zipper)
        ]
    def __call__():
        return self.out.pop()

    return 
    pass
