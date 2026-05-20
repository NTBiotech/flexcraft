from flexcraft.pipelines.tcr.utils import download_structure, clean_chothia, number_anarci
from flexcraft.files import PDBFile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil


def count_chains(pdb_path:Path,)->int:
    n=0
    with open(pdb_path, "r") as rf:
        l=rf.readline()
        while l:
            if l.startswith("TER"):
                n+=1
            l = rf.readline()
    return n

def build_database(df:pd.DataFrame, out_dir:Path, ab:bool=False, chain_number:int|None=None, mhc_class:int=1):
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
                    print(f"Skipping {pdb_id} with {chain_count}!")
                    path.unlink()
                    drop.append(pdb_id)#
                    continue
        else:
            drop.append(pdb_id)
            continue
        # check if actually tcr and trim to variable chains
        path = clean_chothia(path)
        design = PDBFile(path=path).to_data()
        design, params = number_anarci(design, trim=True, mhc_class=mhc_class)
        print(params)
        if None in params["tcr_chain_index"]:#
            print(f"Not all tcr chains found, dropping {pdb_id}!")
            path.unlink()
            (out_dir/pdb_id).with_suffix(".pdb").unlink()
            drop.append(pdb_id)
            continue
        chains = np.unique(design["chain_index"])
        chains = chains[~(chains[:,None]==np.concatenate([c for c in params.values()])[None,:]).any(axis=1)]

        if len(chains)>1:
            lengths = (design["chain_index"][:,None]==chains[None,:]).sum(axis=0)
            chains = chains[np.argmin(lengths)][None,]
        print(chains)
        print(params["tcr_chain_index"])
        tcr_peptide_contact = check_contact(design, params["tcr_chain_index"], chains, 8, )
        if not tcr_peptide_contact:
            print(f"No TCR Peptide contact, dropping {pdb_id}!")
            path.unlink()
            (out_dir/pdb_id).with_suffix(".pdb").unlink()
            drop.append(pdb_id)
            continue

        path.unlink()
        design.save_pdb((out_dir/pdb_id).with_suffix(".pdb"))

    df=df[(df["PDB ID"].to_numpy()[:, None]==np.array(drop)[None,:]).any(axis=1)]
    df.to_csv(out_dir/"annotation.csv")

def check_contact(input_design, tcr_chains, peptide_chains, distance_threshold=8, residue_threshold=3):
    tcr = input_design[(np.array(input_design["chain_index"])[:,None]==tcr_chains[None,:]).any(axis=1)]
    tcr_atoms = np.where(np.stack([tcr["atom_mask"]]*3, axis=-1), tcr["atom_positions"], np.stack([tcr["atom_positions"][:,1,:]]*14, axis=1))
    tcr_atoms = tcr_atoms[:,4]

    peptide = input_design[(input_design["chain_index"][:,None]==peptide_chains[None,:]).any(axis=1)]
    peptide_atoms = np.where(np.stack([peptide["atom_mask"]]*3, axis=-1), peptide["atom_positions"], np.stack([peptide["atom_positions"][:,1,:]]*14, axis=1))
    peptide_atoms = peptide_atoms[:,4]
    # pairwise distance matrix
    distance = np.sqrt(np.sum(np.square(tcr_atoms[None,:,:] - peptide_atoms[:,None,:]), axis=-1))
    return (distance<distance_threshold).any(axis=0).sum()>residue_threshold

def main():
    import argparse

    parser = argparse.ArgumentParser(
        usage="Cluster TCR-MHC complexes using foldseek."
    )

    parser.add_argument("--structure_table", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--exec", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chain_number", type=int, default=5)
    parser.add_argument("--mhc_class", type=int, default=1)


    args = parser.parse_args()
    if args.out_dir is None:
        out_dir = Path(f"./").resolve()
    else:
        out_dir = (args.out_dir).resolve()
    out_dir = out_dir/f"clustering_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}_{args.mhc_class}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = (args.structure_table).resolve()
    table:pd.DataFrame = pd.read_csv(table_path)
    table = table[table["Bound to TCR"].astype(bool)]
    table = table[table["Species"]=="Human"]
    table = table[table["Resolution"].astype(float)<3]
    table:pd.DataFrame = table.sort_values("Release date", ascending=False)[:400]


    build_database(df=table, out_dir=out_dir/"pdb_files", ab=False, chain_number=args.chain_number, mhc_class=args.mhc_class)
    cmd = f"foldseek easy-cluster {out_dir/'pdb_files'} {out_dir/'cluster_result'} $TMP \
        -e 0.01 -c 0.0 --cov-mode 0 --interface-lddt-threshold 0.9 --alignment-type 0 --cluster-reassign 1 -v 2  \
            --gpu {args.gpu}"
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
    # make subdirectories for the clusters
    cluster_dir = out_dir/"clusters"
    cluster_dir.mkdir()
    for cluster in np.unique(clusters["rep"]):
        (cluster_dir/cluster).mkdir()
        print(clusters.query(f"rep=='{cluster}'")["member"])
        for pdb in clusters.query(f"rep=='{cluster}'")["member"]:
            shutil.copy(out_dir/"pdb_files"/f"{pdb.split('_')[0]}.pdb", cluster_dir/cluster)
    # make subdirectory with representative structures for the 4 most populates clusters 
    counts = clusters["rep"].value_counts()[:4]
    reps = counts.index.map(lambda x: x.split("_")[0]).to_list()
    print(f"Representatives: {counts}")
    rep_dir = out_dir/"representative"
    rep_dir.mkdir()
    for pdb_id in reps:
        shutil.copy(out_dir/f"pdb_files/{pdb_id}.pdb", rep_dir)

    print(f"Saved representative structures in {rep_dir}")
    return rep_dir
if __name__ =="__main__":
    main()