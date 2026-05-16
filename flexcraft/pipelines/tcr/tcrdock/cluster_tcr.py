from flexcraft.pipelines.tcr.utils import download_structure
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def count_chains(pdb_path:Path,)->int:
    n=0
    with open(pdb_path, "r") as rf:
        l=rf.readline()
        while l:
            if l.startswith("TER"):
                n+=1
            l = rf.readline()
    return n

def build_database(df:pd.DataFrame, out_dir:Path, ab:bool=False, chain_number:int|None=None):
    out_dir.mkdir(exist_ok=True)
    drop=[]
    for pdb_id in df["PDB ID"].to_list():
        path = download_structure(pdb_id=pdb_id, file_format="antibody" if ab else "biological assembly", out_dir=out_dir)
        if not path is None:
            if (path.parent/(path.name+".gz")).exists():
                (path.parent/(path.name+".gz")).unlink()
            if not chain_number is None:
                chain_count = count_chains(path)
                if chain_count!=chain_number:
                    path.unlink()
                    print(f"Skipping {pdb_id} with {chain_count}!")
                    drop.append(pdb_id)
        else:
            drop.append(pdb_id)
    df=df[(df["PDB ID"].to_numpy()[:, None]==np.array(drop)[None,:]).any(axis=1)]
    df.to_csv(out_dir/"annotation.csv")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        usage="Cluster TCR-MHC complexes using foldseek."
    )

    parser.add_argument("--structure_table", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--exec", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)



    args = parser.parse_args()
    table_path = (args.structure_table).resolve()
    out_dir = (args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(table_path)
    table = table[table["Bound to TCR"].astype(bool)]
    table = table[table["Species"]=="Human"]
    table = table[table["Resolution"].astype(float)<3]
    table = table.sort_values("Release date", ascending=False)[:400]


    build_database(table, out_dir/"pdb_files", ab=False, chain_number=5)
    cmd = f"foldseek easy-cluster {out_dir/'pdb_files'} {out_dir/'cluster_result'} $TMP -c 0.9 --gpu {args.gpu}"
    if args.exec:
        import os
        os.system(cmd)
    else:
        # execute manually
        print("Execute this command: ", cmd, sep="\n")
        from time import sleep
        while not (out_dir/'cluster_result_cluster.tsv').exists():
            sleep(5)
    clusters = pd.read_csv(out_dir/'cluster_result_cluster.tsv', delimiter="\t", header=None)
    clusters.columns = ["rep", "member"]
    counts = clusters["rep"].value_counts()[:4]
    reps = counts.index.map(lambda x: x.split("_")[0]).to_list()

    rep_dir = out_dir/"representative"
    rep_dir.mkdir()
    for pdb_id in reps:
        Path(out_dir/f"pdb_files/{pdb_id}.pdb").rename(out_dir/f"representative/{pdb_id}.pdb")