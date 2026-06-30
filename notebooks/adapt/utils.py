import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

def read_json(p):
    with open(p, "r") as rf:
        d = json.load(rf)
    return dict(d)

def apply_to_all(func, inputs, *covariates):
    res = []
    for t in zip(inputs, *covariates):
        try:
            res.append(func(*t))
        except Exception as ex:
            print("Encountered: ", ex)
            res.append(None)
    return res

def process_scores(df, intervall = 100, minutes=True):
    if len(df) == 0:
        print("DataFrame empty")
        return None
    #df.columns = ["design"]+df.columns.to_list()[1:]
    print(f"{df['in_pool'].sum()}/{df.shape[0]} structures in pool!")
    min_time=datetime.strptime(df["time"].iloc[0],"%Y-%d-%b_%H:%M:%S")
    for c in df.columns:
        if "time" in c:
            df[c] = df[c].map(lambda x: (datetime.strptime(x,"%Y-%d-%b_%H:%M:%S")-min_time).total_seconds()/[1,60][minutes] if isinstance(x,str) else x)

    df["cdr_len"] = df[["acdr3", "bcdr3"]].apply(lambda x: sum([len(i) for i in x.to_list()]), axis=1)
    df["gly_ala"] = df[["acdr3", "bcdr3"]].apply(lambda x: sum(["".join(x.to_list()).count(l) for l in ["A", "G"]]), axis=1)
    df["gly_ala"] = df["gly_ala"]/df["cdr_len"]
    df[["receptor", "mhc", "antigen", "cdr3a", "cdr3b"]] = df["scaffold"].str.split("+", expand=True)
    df["cdr3b"] = df["cdr3b"].map(lambda x: x.split("_")[0])
    bins = np.arange(0, ((df["time"].max()//intervall)+2)*intervall, intervall)
    i = np.arange(len(bins))

    #df["binned_time"] = df["time"].map(lambda x: i[bins>x][0]*intervall)
    return df

def compare(scores, family_limit=20, full_limit=100):
    
    for family in np.unique(scores["scaffold"]):
        mask = scores["scaffold"]==family
        if (mask).sum()<=family_limit:
            # if not enough designs for family limit skip
            continue
        else:
            # if over family limit set score threshold
            min_score = scores.loc[mask].sort_values("score", ascending=True).iloc[family_limit]["score"]
            scores.loc[mask, "in_pool"] = scores.loc[mask, "score"].map(lambda x: x<min_score)
    # check total limit
    if len(scores) <= full_limit:
        # if limit not reached dont drop any
        return scores
    else:
        min_score = scores.sort_values("score", ascending=True).iloc[full_limit]["score"]
        scores["in_pool"] = scores["score"].map(lambda x: x<min_score)
    return scores

def in_pool_time(df, max_time=None, bins=1000, func=np.mean):
    df = df.copy()
    time_cols=[c for c in df.columns if "time" in c.lower()]
    if max_time is None:
        max_time = df[time_cols].max(axis=None)
    df[time_cols] = df[time_cols].fillna(max_time)

    t = np.array([(max_time/bins) * n for n in range(bins)])
    mask = (df["time"].to_numpy()[None,:]<t[:,None])#&(df["out_time"].to_numpy()[None,:]>t[:,None])
    return t,[func(compare(df.loc[m]).query("in_pool==True")["score"]) for m in mask]

def get_label(config):
    return f'boltz:{config["boltz_config"]["docking"]}\nmsa:{config["boltz_config"]["msa"]}\ntemplates:{not config["templates"]is None}'

import tempfile
import subprocess

from rapidfuzz.distance import Levenshtein
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


import Bio 
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import logomaker

#Clustering
def malign(seqs, tmp_dir:Path|None=None):
    seqs = [SeqRecord(Seq(s)) for s in seqs]

    if tmp_dir is None:
        tmpdir_ = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmpdir_.name)
    # write to fasta
    (tmp_dir/'input_mafft.fa').unlink(missing_ok=True)
    with open(tmp_dir/'input_mafft.fa', 'a') as handle:
        for i in seqs:
            SeqIO.write(i, handle ,'fasta')
    # align using mafft
    out_path = tmp_dir/"mafft_out.fa"
    cmd = ['mafft', '--auto', '--inputorder', f'{tmp_dir/"input_mafft.fa"}']
    
    try:
        with open(out_path, "w") as out_f:
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE,
                text=True, check=True)

        with open(out_path, "r") as rf:
            align = AlignIO.read(rf, "fasta")
        #tmpdir_.cleanup()
        return align
    except subprocess.CalledProcessError as e:
        print(f"Failed with code {e.returncode}")
        raise e

def get_logo(seqs, **kwargs):
    align = malign(seqs, tmp_dir=Path("."))
    return logomaker.Logo(pd.DataFrame(align.alignment.frequencies), **kwargs)

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Pairwise distance matrix
def distance_matrix(seqs):
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # normalized edit distance -> length-robust
            d = Levenshtein.normalized_distance(seqs[i], seqs[j])
            D[i, j] = D[j, i] = d
    return D

# Cluster 
def cluster_seqs(seqs, threshold=0.3, criterion="distance"):
    D = distance_matrix(seqs)
    Z = linkage(squareform(D, checks=False), method="average")
    dendrogram(Z)
    labels = fcluster(Z, t=threshold, criterion=criterion)
    return labels
