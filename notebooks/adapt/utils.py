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

def process_scores(df, intervall = 100):
    if len(df) == 0:
        print("DataFrame empty")
        return None
    df.columns = ["design"]+df.columns.to_list()[1:]
    print(f"{df['in_pool'].sum()}/{df.shape[0]} structures in pool!")
    min_time=datetime.strptime(df["time"].iloc[0],"%Y-%d-%b_%H:%M:%S")
    for c in df.columns:
        if "time" in c:
            df[c] = df[c].map(lambda x: (datetime.strptime(x,"%Y-%d-%b_%H:%M:%S")-min_time).total_seconds() if isinstance(x,str) else x)

    df["gly_ala"] = df[["acdr3", "bcdr3"]].apply(lambda x:sum(["".join(x.to_list()).count(l) for l in ["A", "G"]]), axis=1)
    
    bins = np.arange(0, ((df["time"].max()//intervall)+2)*intervall, intervall)
    i = np.arange(len(bins))

    df["binned_time"] = df["time"].map(lambda x: i[bins>x][0]*intervall)
    return df

def in_pool_time(df, max_time=None, bins=1000):
    df = df.copy()
    time_cols=[c for c in df.columns if "time" in c.lower()]
    if max_time is None:
        max_time = df[time_cols].max(axis=None)
    df[time_cols] = df[time_cols].fillna(max_time)

    t = np.array([(max_time/bins) * n for n in range(bins)])
    mask = (df["time"].to_numpy()[None,:]<t[:,None])&(df["out_time"].to_numpy()[None,:]>t[:,None])
    return t,[df["score"][m].mean() for m in mask]

def get_label(config):
    return f'boltz:{config["boltz_config"]["docking"]}\nmsa:{config["boltz_config"]["msa"]}\ntemplates:{not config["templates"]is None}'