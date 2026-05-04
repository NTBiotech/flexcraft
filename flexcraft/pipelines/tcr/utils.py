from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Callable

from flexcraft.data.data import DesignData

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


def print_dd(dd:DesignData, name:str="", keys:list=["chain_index"]):
    try:
        print(f"\n---{name}---",
            *[f"{k}:{v}\n\t shape: {v.shape}" for k,v in dd.data.items() if k in keys],
            sep="\n"
            )
    except KeyError:
        print("Key not found")


def clean_chothia(file)->Path:
    '''Removes annotations, duplicate chains and HETATMs.'''
    if isinstance(file, str):
        file = Path(file)
    if file.name.endswith("clean.pdb"):
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

def download_structure(pdb_id: str, file_format: str = "biological assembly", out_dir: str = ".")->Path:
    """
    Download a structure file from RCSB PDB.
    
    file_format: 'pdb', 'cif' (mmCIF), 'bcif' (BinaryCIF), biological assembly (pdb1.gz) or antibody (.pdb from SAbDab).
    """
    from urllib.request import urlretrieve
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
    urlretrieve(url, out_path)
    if out_path.suffix == ".gz":
        # decompress
        import gzip
        import shutil
        with gzip.open(out_path, 'rb') as f_in:
            with open(out_path.with_suffix(''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out_path.with_suffix("")
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

def collect_results(directory:Path, pattern:str, in_file:str="scores.csv", save:bool=True):
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

def cdr_parser(cdrs:str|None)->Callable:
    '''
    Creates a generator for cdr dicts from either an existing path or a json encoded string.
    Always returns None, if creating the generator fails.
    '''
    if cdrs is None:
        def _inner():
            return None

    elif Path(cdrs).exists():
        cdr_file = open(cdrs, "r")
        keys = cdr_file.readline().strip("\n").split("\t")
        def _inner():
            return {
                    i[-1]+i[:-1]:n
                    for i,n in zip(keys, cdr_file.readline().strip("\n").split("\t"))
                }
    else:
        try:
            out = json.loads(cdrs)
            if isinstance(out, dict):
                def _inner():
                    return out
            elif isinstance(out, list):
                global cdr_iter
                cdr_iter=-1
                def _inner():
                    global cdr_iter
                    cdr_iter+=1
                    return out[cdr_iter]
        except json.JSONDecodeError:
            print(f"Not able to interpret cdrs {cdrs}! Returning None!")
            def _inner():
                return None
    
    return _inner