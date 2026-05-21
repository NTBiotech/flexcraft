from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Callable
import numpy as np

from tree import K


def load_data(out_dir:str|Path=Path("./data/adapt/input_data"),
    url = "https://zenodo.org/records/17488258/files/",
    files = [
        "paired_human_cdr3s.tsv",
        "model_2_ptm_ft_binder_20230729.pkl",
        #"RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt",
        "zenodo_design_models.zip"
    ],
    ):
    '''Load Data used in the original ADAPT paper.'''
    from urllib.request import urlretrieve
    from zipfile import ZipFile
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    existing = {x.name for x in out_dir.iterdir() if x.is_file()}
    for file in files:
        file_url = url + file
        print(file_url)
        if file not in existing:
            urlretrieve(file_url, str(out_dir/file))
        else:
            print(f"{file} exists, skipping download")

        if file.endswith(".zip"):
            zip_dir = out_dir/file.split(".")[0]
            print(f"Extracting {file} to {zip_dir}")
            if zip_dir.exists():
                print(f"{zip_dir} exists, skipping unzipping {file}")
                continue
            with ZipFile(out_dir/file, 'r') as zip_ref:
                zip_ref.extractall(out_dir)


def print_dd(dd, name:str="", keys:list=["aa"]):
    '''Print configurable attributes of the DesignData object dd'''
    try:
        print(f"\n---{name}---",
            *[f"{k}:{dd.to_sequence_string()}\n\t shape: {v.shape}" for k,v in dd.data.items() if k in keys],
            sep="\n"
            )
    except KeyError:
        print("Key not found")


def clean_chothia(file)->Path:
    '''Removes annotations, duplicate chains and HETATMs.'''
    if isinstance(file, str):
        file = Path(file)
    if file.name.endswith("_clean.pdb"):
        print("File already clean")
        return file
    out_path = Path(file.with_suffix("").__str__()+"_clean.pdb")
    with open(out_path, "w") as wf:
        with open(file, "r") as rf:
            l = "init value"
            while l:
                l = rf.readline()
                if l.startswith("ATOM"):
                    wf.write(l[:26]+" "+l[27:])
                elif not l.startswith(("HETATM", "MODEL", "ENDMDL")):
                    wf.write(l)
    return out_path

def download_structure(pdb_id: str, file_format: str = "biological assembly", out_dir: str = ".")->Path|None:
    """
    Download a structure file from RCSB PDB.
    
    file_format: 'pdb', 'cif' (mmCIF), 'bcif' (BinaryCIF), biological assembly (pdb1.gz) or antibody (.pdb from SAbDab).
    """
    from urllib.request import urlretrieve
    from urllib.error import HTTPError
    base_urls = {
        "biological assembly": f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb1.gz",
        "pdb": f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        "cif": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
        "bcif": f"https://models.rcsb.org/{pdb_id.lower()}.bcif",
        "antibody": f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id.lower()}/?scheme=imgt",
    }
    url = base_urls[file_format]
    suffix = {"pdb": ".pdb", "cif": ".cif", "bcif": ".bcif", "biological assembly":".pdb.gz" ,"antibody":".pdb"}.get(file_format, ".pdb")
    out_path = Path(out_dir) / f"{pdb_id.upper()}{suffix}"
    try:
        urlretrieve(url, out_path)
        if out_path.suffix == ".gz":
            # decompress
            import gzip
            import shutil
            with gzip.open(out_path, 'rb') as f_in:
                with open(out_path.with_suffix(''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return out_path.with_suffix("")
    except HTTPError as e:
        print(f"Encountered HTTPError for {pdb_id}: {e}")
        return None
    return out_path

def get_mhc_by_accession(accession, base="https://www.ebi.ac.uk/cgi-bin/ipd/api/allele")->Dict[str,str]|int:
    import requests
    response = requests.get(f"{base}/{accession}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"WARNING: response status code: {response.status_code}!")
        return response.status_code

def query_mhc_by_name(name, base="https://www.ebi.ac.uk/cgi-bin/ipd/api/allele", limit:int=10)->Dict[str,str]|int:
    import requests
    params={"query":f"startsWith(name, '{name}')" ,"limit":limit}
    response = requests.get(f"{base}", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"WARNING: response status code: {response.status_code}!")
        return response.status_code

def get_mhc(accession:str|None=None, name:str|None=None)->str|None:
    '''Query the EBI HLA database for mhc sequence with accession number and/or WHO notation (name).'''
    if accession is None:
        response = query_mhc_by_name(name, limit=1)
        if isinstance(response,dict):
            accession = response["data"][0]["accession"]
            print(f"Found accession {accession} corresponding to name {name}")
    if not accession is None:
        response = get_mhc_by_accession(accession)
        if isinstance(response,dict):
            return response["sequence"]["protein"]
    print(f"No protein found for accession {accession} with name {name}!")
    return None

def collect_results(directory:Path, pattern="**/*", in_file:str="scores.csv", save:bool=True):
    '''Collect result csv files from subdirectories recursively and concatenate to one pandas DataFrame.'''
    import pandas as pd
    scores = {}
    for d in directory.glob(pattern):
        if not d.is_dir():
            continue
        if (d/in_file).exists():
            print(d/in_file)
            scores[d] = pd.read_csv(d/in_file, header=0, index_col=0)
        else:
            sub_dir = collect_results(d, pattern="*", in_file=in_file, save=False)
            if not sub_dir is None:
                scores[d] = sub_dir
    if not scores:
        return None
    df = pd.concat(scores)
    if save:
        df.to_csv((directory/(in_file.split(".")[0]+"collected.csv")))
        return directory/(in_file.split(".")[0]+"collected.csv")
    return df

def cdr_parser(cdrs:str|None, random:bool=False, cdr_length:int|tuple|None=None, patience:int=100)->Callable:
    '''
    Creates a generator for cdr dicts from either an existing path or a json encoded string.
    Always returns None, if creating the generator fails.
    '''
    import json
    if random:
        from numpy import random as nprandom
    if cdrs is None:
        def _inner():
            return None

    elif Path(cdrs).exists():
        cdr_file = open(cdrs, "r")
        keys = cdr_file.readline().strip("\n").split("\t")
        keys = [i[-1]+i[:-1] for i in keys]
        if random:
            file_size = Path(cdrs).stat().st_size
            def _inner():
                # set the pointer at random position (-200 as buffer from file end)
                cdr_file.seek(nprandom.randint(0,file_size-200), 0)
                cdr_file.readline()
                return {
                        k:n
                        for k,n in zip(keys, cdr_file.readline().strip("\n").split("\t"))
                    }
        else:
            def _inner():
                return {
                        i:n
                        for i,n in zip(keys, cdr_file.readline().strip("\n").split("\t"))
                    }
    else:
        try:
            out = json.loads(cdrs)
            if isinstance(out, dict):
                keys = [k for k in out.keys()]
                def _inner():
                    return out
            elif isinstance(out, list):
                global cdr_iter
                cdr_iter=-1
                assert isinstance(out[cdr_iter], dict)
                keys = [k for k in out[1].keys()]
                if random:
                    def _inner():
                        return out[nprandom.randint(0,len(out))]
                else:
                    def _inner():
                        global cdr_iter
                        cdr_iter+=1
                        return out[cdr_iter]
        except json.JSONDecodeError:
            raise ValueError(f"Not able to interpret cdrs {cdrs}!")

    if not cdr_length is None:
        if isinstance(cdr_length, int):
            cdr_length = (cdr_length, cdr_length)
        def _fixed_length_inner():
            cdrs = _inner()
            n=1
            while len(cdrs[keys[0]])!=cdr_length[0] or len(cdrs[keys[1]])!=cdr_length[1]:
                cdrs = _inner()
                n+=1
                if n>patience:
                    raise ValueError(f"No cdr of fixed length {cdr_length} found for {patience} runs")
            return cdrs
        return _fixed_length_inner
    return _inner

def number_anarci(
    input_design,
    mhc_class:int|None=None,
    code:str|None=None,
    scheme:str="imgt",
    trim=False,
    accept_ab=True,
    ):
    '''
    Basic numbering of AB or TCR sequences.
    Returns:
        (DesignData, dict): numbered design and dict containing "mhc_chain_index" and "tcr_chain_index" numpy arrays
    '''
    from flexcraft.sequence.aa_codes import AF2_CODE, decode
    import anarci
    import jax.numpy as jnp
    if code is None:
        code = AF2_CODE
    params = {"tcr_chain_index":np.array([None,None]),
    "mhc_chain_index":np.array([None]),}
    chains = np.unique(input_design["chain_index"])

    chain_types = (["A", "L"],["B", "H"])
    if not accept_ab:
        chain_types = (["A"],["B"])

    for chain in chains:
        chain_mask = np.array(input_design["chain_index"]) == chain
        # Pass only this chain's sequence to anarci
        chain_aa = np.array(input_design["aa"])[chain_mask]
        seq = decode(chain_aa, code=code)
        numbering = anarci.number(sequence=seq, scheme=scheme)
        if numbering[0]:
            chain_type = numbering[-1]
            if chain_type in chain_types[0]:
                print(f"Setting chain {chain} to {chain_type}!")
                params["tcr_chain_index"][0] = chain
            elif chain_type in chain_types[1]:
                print(f"Setting chain {chain} to {chain_type}!")
                params["tcr_chain_index"][1] = chain
            else:
                print(f"Unknown chain type {chain_type} of chain {chain}!")
                continue
            # Build IMGT position strings (e.g. "1", "111", "111A") then convert to int
            numbering = [f"{x[0][0]}{x[0][1].strip()}" for x in numbering[0] if x[1] != "-"]
            numbering = [int(x) if x.isnumeric() else int(x[:-1]) for x in numbering]

            residue_index = np.array(input_design["residue_index"])
            if len(residue_index[chain_mask])>len(numbering):
                # extend variable region by constant region
                if trim:
                    print(f"Trimming chain {chain} to length {len(numbering)}.")
                    index = np.arange(len(residue_index))[chain_mask]
                    start = index[0]
                    stop = index[-1]+1
                    mask = np.ones(len(residue_index), dtype=np.bool_)
                    mask[start+len(numbering):stop] = False
                    input_design = input_design[mask]
                    chain_mask = np.array(input_design["chain_index"]) == chain
                    residue_index = np.array(input_design["residue_index"])
                else:
                    numbering += np.arange(numbering[-1]+1,numbering[-1]+1+chain_mask.sum()-len(numbering)).tolist()
            residue_index[chain_mask] = numbering
            input_design = input_design.update(residue_index=np.array(residue_index))

    # check if chain indices correct
    if params["tcr_chain_index"][0]==params["tcr_chain_index"][1] and not params["tcr_chain_index"][0] is None:
        raise ValueError("TCR chains identical! Currently only 2 chain tcrs supported.")
    if isinstance(mhc_class, int):
        # fix mhc chain index to longest non-tcr chain
        chains = np.unique(input_design["chain_index"])
        # mask out tcr chains
        tcr_mask = ~(chains[:,None]==params["tcr_chain_index"][None,:]).any(axis=1)
        chains = chains[tcr_mask]
        # get chain lengths
        chain_lengths =  (input_design["chain_index"][:,None] == chains[None,:]).sum(axis=0)
        # take the n longest chain indices, where n the number of non-tcr chains -1 (for the peptide chain) 
        # take all non-tcr-chains except for smalles (peptide, hopefully)
        if mhc_class == 1:
            params["mhc_chain_index"] = chains[np.argsort(chain_lengths)[::-1][0]][None,].astype(int)
        elif mhc_class==2:
            params["mhc_chain_index"] = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-len(chains):-1]],dtype=int
            )
        else:
            raise ValueError(f"MHC class {mhc_class} not a real thing! Or is it?")
        assert (np.array([a.ndim for a in params.values()])==1).all(), ValueError(f"Param dimensions should be 1: {np.array([a.ndim for a in params.values()])}")
        assert len(params["tcr_chain_index"])==2, ValueError(f"len(params[tcr_chain_index]): ", len(params["tcr_chain_index"]))
        assert len(params["mhc_chain_index"]) >0, ValueError(f"len(params[mhc_chain_index]): ", len(params["mhc_chain_index"]))
        if trim:
            target_chains=np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[::-1][:2]],dtype=int
            )
            target_length=90
            if mhc_class == 1:
                target_length=180
            input_design = trim_mhc(input_design, target_chains, target_length=target_length, mhc_class=mhc_class)
    print("Classified params: ", params)
    return input_design, params

def trim_mhc(input_design, mhc_chains, target_length=90, mhc_class=1):
    #TODO fix trimming of mhc chains!
    print("trim_design mhc_chains",mhc_chains)
    for n, chain in enumerate(mhc_chains):
        if mhc_class == 1:
            n = 0
        print("trim_design chain",chain)
        chain_mask = input_design["chain_index"]==chain
        trim_mask = np.ones(len(input_design["aa"]), dtype=np.bool_)
        if chain_mask.sum() < 110 or chain_mask.sum() < target_length:
            trim_mask[chain_mask]=False
            print(f"WARNING! Chain {chain} smaller than 110 residues. Removing chain fully!")
        else:
            trim_mask[chain_mask] = np.concatenate((np.ones(target_length), np.zeros(int(chain_mask.sum()-target_length))))[::(1-(0*2))]
            print(f"Trimming chain {chain} to {target_length} AAs.")
        input_design = input_design[trim_mask]
    return input_design
