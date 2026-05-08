
from shutil import ExecError
import sys
import os

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from flexcraft.data.data import DesignData
from flexcraft.files import PDBFile
from flexcraft.pipelines.tcr.utils import *
from flexcraft.sequence.aa_codes import AF2_CODE, decode, PMPNN_CODE
import anarci
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import tempfile

from scipy.spatial.transform import Rotation

## 1-indexed
## numbered wrt the sequence in 3pqyA.fasta ie class1_template_seq below
## to get 3pqy PDB numbers just add 1
## 4...10 are upward facing in the N-terminal central strand
## 23...25 are upward facing in the neighboring N-terminal strand
## 94...100 are upward facing in the C-terminal central strand
## 113.115 are upward facing in the neighboring C-terminal
##

# for class 1 blosum align
class1_template_seq = 'PHSMRYFETAVSRPGLEEPRYISVGYVDNKEFVRFDSDAENPRYEPRAPWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGSDWRLLRGYLQFAYEGRDYIALNEDLKTWTAADMAAQITRRKWEQSGAAEHYKAYLEGECVEWLHRYLKNGNATLLRTDSPKAHVTHHPRSKGEVTLRCWALGFYPADITLTWQLNGEELTQDMELVETRPAGDGTFQKWASVVVPLGKEQNYTCRVYHEGLPEPLTLRWEP'

class1_template_core_positions_1indexed = [4, 6, 8, 10, 23, 25, 94, 96, 98, 100, 113, 115]
class1_template_core_positions_0indexed = [x-1 for x in class1_template_core_positions_1indexed]
# for class 2
class2_alfas_positions_0indexed = {
    'A': [2, 4, 7, 9, 18, 20],
    'B': [2, 4, 6, 8, 21, 23], # 8 is disulfide posn
}

class2_db = "/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_B_chains_v2.fasta"



# Und für tcr:
imgt_mapper = {
    "acdr1":(27,38),
    "acdr2":(56, 65),
    "acdr3":(105,117),
    "bcdr1":(27,38),
    "bcdr2":(56,65),
    "bcdr3":(105,117),
    }

core_positions_generic_1x = [
21, 23, 25, ## 23 is C
39, 41, ## 41 is W
53, 54, 55,
78, ## maybe also 80?
89, ## 89 is L
102, 103, 104 ## 104 is C
]
# (1 indexed)

def align_seq(x,y):
    aligner = Align.PairwiseAligner(scoring="blastp")
    alignments = aligner.align(x,y)
    alignment = max(alignments, key=lambda x: x.score)
    if alignment.score<1:
        raise ValueError(f"Alignment score of {alignment.score} is too low!")
    align = {}
    for i,(a,b) in enumerate(zip(*alignment)):
        if a!= '-' and b!='-':
            #assert begin <= i <= end
            pos1 = i-alignment[0][:i].count('-')
            pos2 = i-alignment[1][:i].count('-')
            align[pos1] = pos2
    return align

def get_mhc1_positions(
    design,
    params,
    ):
    chain_mask = design["chain_index"]==params["mhc_chain_index"]
    mhc_seq = decode(design["aa"][chain_mask], AF2_CODE)

    t = align_seq(class1_template_seq, mhc_seq)
    positions = [t[x] for x in class1_template_core_positions_0indexed]
    index = np.arange(len(design["aa"]))[chain_mask]
    return np.array([index[x] for x in positions])

# make db from fasta
def db_from_fasta(fasta:Path, blast_exe:Path, dbtype:str="prot"):
    assert fasta.exists(), f"{fasta} does not exist!"
    print(f"{(blast_exe/'makeblastdb').resolve()} -in {fasta.resolve()} -dbtype {dbtype} -parse_seqids")
    out = os.system(f"{(blast_exe/'makeblastdb').resolve()} -in {fasta.resolve()} -dbtype {dbtype} -parse_seqids")

    if out != 0:
        raise ExecError(f"Could not create database from fasta {fasta}.")
    return fasta.resolve()

def blastp(
    blast_exe:Path,
    query:Path,
    db:Path,
    out_file:Path,
    num_alignments:int=5,
    cols:list=[
        "evalue", "bitscore", "qaccver", "saccver", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "qlen", "qseq", "sstart", "send", "slen", "sseq",
        ]
    ):
    cmd = f"{blast_exe/'blastp'} -query {query.resolve()} -db {db.resolve()} -num_alignments {num_alignments} -outfmt \"10 delim=, {' '.join(cols)}\" >> {out_file.resolve()}"
    with open(out_file, "w") as wf:
        wf.write(",".join(cols)+"\n")
    os.system(cmd)
    return out_file.resolve()

def align_from_blast(hit:pd.Series):
    align = {}
    for ii,(a,b) in enumerate(zip(hit.qseq, hit.sseq)):
        if a!= '-' and b!='-':
            apos = hit.qstart + ii - hit.qseq[:ii].count('-') - 1 #0-idx
            bpos = hit.sstart + ii - hit.sseq[:ii].count('-') - 1 #
            align[int(bpos)] = int(apos)
    return align

def get_mhc2_positions(
    design:DesignData,
    params:dict,
    db_files:Dict[str,Path],
    blast_exe:Path,
    reverse=False
    ):
    try:
        positions = {}
        chain_masks = design["chain_index"][None,:]==params["mhc_chain_index"][:,None]
        
        for c, mask, chain_index in zip(["A", "B"][::-1 if reverse else 1],chain_masks, params["mhc_chain_index"]):
            
            assert db_files[c].suffix==".fasta"
            # check for blast database
            if not db_files[c].with_suffix(".fasta.phr").exists():
                print(f"Created blastdb for {db_from_fasta(db_files[c], blast_exe, dbtype='prot')}")

            seq = SeqRecord(Seq(decode(design["aa"][mask], AF2_CODE)), id=params["pdb_id"], name="mhc 2")
            
            # write seq to fasta
            with tempfile.TemporaryDirectory(prefix="tcrdock_") as tmp_dir:
                tmp_dir = Path(tmp_dir)
                # write query to fasta
                query_path = tmp_dir/"query.fasta"
                with open(query_path, "w") as wf:
                    SeqIO.write(seq, handle=wf, format="fasta")
                out_path=tmp_dir/"blast_out.csv"
                
                blastp(
                    blast_exe=blast_exe,
                    query=query_path,
                    db=db_files[c],
                    out_file=out_path
                )

                hits = pd.read_csv(out_path, header=0)
                if len(hits)<1:
                    raise AttributeError("No hits found!")
                hit = hits.iloc[0]
                if float(hit["evalue"])>0.05:
                    raise AttributeError("E-Value too low!")

            align = align_from_blast(hit)
            _positions = [align[x] for x in class2_alfas_positions_0indexed[c]]
            index = np.arange(len(design["aa"]))[mask]
            positions.update({int(chain_index):np.array([index[x] for x in _positions])})
    except AttributeError as e:
        print(e, "Running in reverse mode.")
        if not reverse:
            return get_mhc2_positions(
                design=design,
                params=params,
                db_files=db_files,
                blast_exe=blast_exe,
                reverse=True
            )
        else:
            raise e
    return positions


#--- Geometry ---

def centroid(points:np.ndarray):
    '''Calculate centroid of points in 3d space.'''
    return points.mean(axis=0)
def proj(a:np.ndarray,b:np.ndarray):
    '''Projection of vector a onto vector b'''
    return (np.dot(a,b)/np.linalg.norm(b))*b
def orthogonalize(a:np.ndarray,b:np.ndarray):
    return np.array(a-proj(a,b))

def get_axes(a,b):
    rotation, rssd = Rotation.align_vectors(a, b[::1])
    axis1=rotation.inv().as_mrp()
    axis2=orthogonalize(centroid(a)-centroid(b), axis1)
    axis3 = np.cross(axis2,axis1)
    return axis1,axis2,axis3

def plot_axes(a:np.ndarray,b:np.ndarray, plot_points:bool=False, normalize:bool|float|int=True, angle:None|float|int=None):
    n=1
    if isinstance(normalize, (float, int)):
        n = normalize
        normalize=False
    fig = plt.figure()
    center = centroid(np.concatenate([a,b], axis=0))
    ax = fig.add_subplot(projection='3d', )
    axis1, axis2, axis3 = get_axes(a,b)
    if plot_points:
        ax.scatter(*a.T,c="blue")
        ax.scatter(*centroid(a), c="darkblue",)
        ax.scatter(*b.T,c="red")
        ax.scatter(*centroid(b), c="darkred",)
    ax.scatter(*center, c="black")
    #ax.quiver(*center, *(centroid(a)-centroid(b)), normalize=True, pivot="middle",)
    #ax.quiver(*center,color="purple", *(rotation.apply(centroid(a)-centroid(b))), normalize=True, pivot="middle")
    ax.quiver(*center, *((axis1/np.linalg.norm(axis1))*n), normalize=normalize, pivot="tail",)
    ax.scatter(*((center+axis1)/np.linalg.norm(center+axis1)), alpha=0)
    ax.quiver(*center, *((axis2/np.linalg.norm(axis2))*n), normalize=normalize, color="purple")
    ax.scatter(*((center+axis2)/np.linalg.norm(center+axis2)), alpha=0)
    ax.quiver(*center, *((axis3/np.linalg.norm(axis3))*n), normalize=normalize, color="yellow")
    ax.scatter(*((center+axis3)/np.linalg.norm(center+axis3)), alpha=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if not plot_points:
        ax.set_xlim(center[0]-1,center[0]+1)
        ax.set_ylim(center[1]-1,center[1]+1)
        ax.set_zlim(center[2]-1,center[2]+1)
    ax.set_title("Orientation Axes")
    if isinstance(angle, (float, int)):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
        else:
            elev = azim = roll = angle_norm

        # Update the axis view and title
        ax.view_init(elev, azim, roll)
    return fig

def get_mhc_coords(design:DesignData, positions:np.ndarray, atoms:list=["CA"]):
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    return design["atom_positions"][(np.arange(len(design["aa"]))[:,None]==positions[None,:]).any(axis=1)][:,atom_index,:].squeeze()

def get_tcr_coords(design:DesignData, tcr_chain_index:np.ndarray, atoms:list=["CA"]):
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    chain_mask = (design["chain_index"][:,None]==tcr_chain_index[None,:]).any(axis=1)
    positions_mask = (design["residue_index"][:,None]==np.array(core_positions_generic_1x)[None,:]).any(axis=1)
    mask = positions_mask&chain_mask
    return design["atom_positions"][mask][:,atom_index,:].squeeze()
