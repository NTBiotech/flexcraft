
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

import jax.numpy as jnp

from scipy.spatial.transform import Rotation



#---Alignments---

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
''' installation of ncbi blast
if sys.platform == 'linux':
    address = ('https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/latest/ncbi-blast-2.17.0+-x64-linux.tar.gz')
elif sys.platform == 'darwin':
    address = ('https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/latest/ncbi-blast-2.17.0+-x64-macosx.tar.gz')
else:
    print('unrecognized platform type:', sys.platform,'expected "linux" or "darwin"')
    exit()'''

def align_seq(x,y):
    aligner = Align.PairwiseAligner(scoring="blastp")
    alignments = aligner.align(x,y)
    alignment = max(alignments, key=lambda x: x.score)
    print(f"Alignment score: {alignment.score}")
    print(alignment)
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
    params=None,
    ):
    if params is None:
        chain_mask = np.ones(len(design["chain_index"]), dtype=np.bool_)
    else:    
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
    db_files:Dict[str,Path],
    blast_exe:Path,
    params:dict|None=None,
    reverse=False
    ):
    try:
        positions = {}
        if params is None:
            chain_masks = design["chain_index"][None,:]==np.unique(design["chain_index"])[:,None]
        else:
            chain_masks = design["chain_index"][None,:]==params["mhc_chain_index"][:,None]
        
        for c, mask, chain_index in zip(["A", "B"][::-1 if reverse else 1],chain_masks, params["mhc_chain_index"]):
            
            assert db_files[c].suffix==".fasta"
            # check for blast database
            if not db_files[c].with_suffix(".fasta.phr").exists():
                print(f"Created blastdb for {db_from_fasta(db_files[c], blast_exe, dbtype='prot')}")

            seq = SeqRecord(Seq(decode(design["aa"][mask], AF2_CODE)), id="mhc2_query", name="mhc 2")
            
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
                print(f"Found hit with evalue {hit['evalue']}")
                if float(hit["evalue"])>0.5 and not reverse:
                    raise AttributeError(f"E-Value too low: {hit['evalue']}")

            align = align_from_blast(hit)
            _positions = [align[x] for x in class2_alfas_positions_0indexed[c]]
            index = np.arange(len(design["aa"]))[mask]
            positions.update({c:np.array([index[x] for x in _positions])})
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

def centroid(points:np.ndarray, where=np._NoValue):
    '''Calculate centroid of points in 3d space.'''
    return np.mean(points,axis=0, where=where)
def proj(a:np.ndarray,b:np.ndarray):
    '''Projection of vector a onto vector b'''
    return (np.dot(a,b)/np.linalg.norm(b))*b
def orthogonalize(a:np.ndarray,b:np.ndarray):
    return np.array(a-proj(a,b))
def gram_schmidt(arr:np.ndarray):
    '''Orthogonalize rows of arr.'''
    orth = []
    for n,v in enumerate(arr):
        u = v.copy()
        for r in range(n):
            u-=proj(arr[r], v)
        orth.append(u)
    return np.stack(orth, axis=0)

def get_axes(a,b)->Tuple[np.ndarray, np.ndarray]:
    '''Calculates normalized axes from two sets of points.'''
    
    center = centroid(np.concat([a,b]))
    rotation, rssd = Rotation.align_vectors(a - center, b - center)
    # TODO: inner product with cdr or peptide
    # if negative invert
    axis1 = rotation.as_mrp()
    axis1 = axis1/np.linalg.norm(axis1)
    axis2 = orthogonalize(centroid(a)-centroid(b), axis1)
    axis2 = axis2/np.linalg.norm(axis2)
    axis3 = np.cross(axis2,axis1)
    axis3 = axis3/np.linalg.norm(axis3)
    return gram_schmidt(np.array((axis3,axis2,axis1))).T, center

def check_direction(axis:np.ndarray, center:np.ndarray, reference:np.ndarray, covariate:np.ndarray|None=None):
    print("Before correction: ",axis, covariate)
    if len(reference.shape)>1:
        reference = np.mean(reference,axis=(0,1))
    reference -= center
    reference /= np.linalg.norm(reference)
    print(f"Correcting to {reference}")
    cosine = np.inner(axis, reference)
    print(f"Angle is {np.degrees(np.arccos(cosine))}")
    if cosine<0:
        axis=-axis
        if not covariate is None:
            covariate = -covariate
    print("After correction: ",axis, covariate)
    if covariate is None:
        return axis
    return axis, covariate


def plot_axes(
    axes:None|np.ndarray=None,
    center:None|np.ndarray=None,
    a:np.ndarray|None=None,
    b:np.ndarray|None=None,
    plot_points:bool=False,
    normalize:bool|float|int=True,
    angle:None|float|int=None,
    ax=None,
    color=False,
    lim:tuple=(-1,1),
    ref:np.ndarray|None=None
    ):
    n=1
    if isinstance(normalize, (float, int)):
        n = normalize
        normalize=False
    if ax is None:
        print("New plot axis")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', )
    if axes is None or center is None:
        print("Calculating Axes")
        axes, center = get_axes(a,b)

    axis1, axis2, axis3 = axes.T
    if not ref is None:
        axis3, axis1 = check_direction(axis3, center,ref, axis1)
    if plot_points and not a is None and not b is None:
        ax.scatter(*a.T,c="blue" if not color else color)
        ax.scatter(*centroid(a), c="darkblue" if not color else color)
        ax.scatter(*b.T,c="red" if not color else color)
        ax.scatter(*centroid(b), c="darkred" if not color else color)

    ax.scatter(*center, c="black")
    #ax.quiver(*center, *(centroid(a)-centroid(b)), normalize=True, pivot="middle",)
    #ax.quiver(*center,color="purple", *(rotation.apply(centroid(a)-centroid(b))), normalize=True, pivot="middle")
    ax.quiver(*center, *((axis1/np.linalg.norm(axis1))*n), normalize=normalize, pivot="tail", color="red" if not color else color, label="axis1")
    ax.scatter(*((center+axis1)), alpha=0)
    ax.quiver(*center, *((axis2/np.linalg.norm(axis2))*n), normalize=normalize, color="purple" if not color else color, label="axis2")
    ax.scatter(*((center+axis2)), alpha=0)
    ax.quiver(*center, *((axis3/np.linalg.norm(axis3))*n), normalize=normalize, color="yellow" if not color else color, label="axis3")
    ax.scatter(*((center+axis3)), alpha=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    if not plot_points:
        ax.set_xlim(center[0]+lim[0],center[0]+lim[1])
        ax.set_ylim(center[1]+lim[0],center[1]+lim[1])
        ax.set_zlim(center[2]+lim[0],center[2]+lim[1])
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
    return ax

def get_mhc_coords(design:DesignData, positions:np.ndarray, atoms:list=["CA"]):
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    return design["atom_positions"][(np.arange(len(design["aa"]))[:,None]==positions[None,:]).any(axis=1)][:,atom_index,:].squeeze()

def get_tcr_coords(design:DesignData, tcr_chain_index:np.ndarray, atoms:list=["CA"]):
    '''Get the TCR coordinates for stub calculation. Expects IMGT numbered residue_index for the tcr chains.'''
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    chain_mask = (design["chain_index"][:,None]==tcr_chain_index[None,:]).any(axis=1)
    positions_mask = (design["residue_index"][:,None]==np.array(core_positions_generic_1x)[None,:]).any(axis=1)
    mask = positions_mask&chain_mask
    return design["atom_positions"][mask][:,atom_index,:].squeeze()

def test_axes_operations(axes1:np.ndarray, center1:np.ndarray, axes2:np.ndarray, center2:np.ndarray, get_ax_op:Callable, apply_ax_op:Callable,):
    op = get_ax_op(axes1, center1, axes2, center2)
    axesop, centerop = apply_ax_op(axes1, center1, op)
    assert np.isclose(axes2,axesop, atol=0.00001).all()
    assert np.isclose(center2,centerop, atol=0.00001).all()
    rev_axes, rev_center = rev_ax_op(axesop, centerop, op)
    assert np.isclose(axes1,rev_axes, atol=0.00001).all()
    assert np.isclose(center1,rev_center, atol=0.00001).all()

def get_rotation(axes1, axes2):
    return axes2@axes1.T

def get_centering(center1, center2):
    return center1-center2

def get_ax_op(axes1, center1, axes2, center2):
    # rotation
    r = get_rotation(axes1, axes2)
    # centering
    d = get_centering(center1, center2)
    return r,d

def apply_ax_op(axes, center, op:tuple):
    return op[0]@axes, center-op[1]

def rev_ax_op(axes, center, op:tuple):
    return op[0].T@axes, center + op[1]

#---wrapper functions---

def number_anarci(
    input_design:DesignData,
    mhc_class:int|None=None,
    code:str=AF2_CODE,
    scheme:str="imgt",
    )->DesignData:
    '''
    Basic numbering of AB or TCR sequences.
    Returns:
        (DesignData, dict): numbered design and dict containing "mhc_chain_index" and "tcr_chain_index" numpy arrays
    '''
    params = {"tcr_chain_index":np.array([0,1]),
    "mhc_chain_index":np.array([2]),}
    chains = np.unique(input_design["chain_index"])

    for chain in chains:
        chain_mask = np.array(input_design["chain_index"]) == chain
        # Pass only this chain's sequence to anarci
        chain_aa = np.array(input_design["aa"])[chain_mask]
        seq = decode(chain_aa, code=code)
        numbering = anarci.number(sequence=seq, scheme=scheme)
        if numbering[0]:
            chain_type = numbering[-1]
            if chain_type in ["A", "L"]:
                print(f"Setting chain {chain} to {chain_type}!")
                params["tcr_chain_index"][0] = chain
            elif chain_type in ["B", "H"]:
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
                numbering += np.arange(numbering[-1]+1,numbering[-1]+1+chain_mask.sum()-len(numbering)).tolist()
            residue_index[chain_mask] = numbering
            input_design = input_design.update(residue_index=np.array(residue_index))
        else:
            print(f"No numbering found for chain {chain}!")
    # check if chain indices correct
    if params["tcr_chain_index"][0]==params["tcr_chain_index"][1]:
        raise ValueError("TCR chains identical! Currently only 2 chain tcrs supported.")
    if not mhc_class is None:
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
            params["mhc_chain_index"] = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-(len(chains)-1):-1]],dtype=int
            )
        elif mhc_class==2:
            params["mhc_chain_index"] = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-len(chains):-1]],dtype=int
            )
        print(f"Classified chains {params['mhc_chain_index']} as MHC/antigen chains")
    return input_design, params

def convert_chains(input_design:DesignData, d:dict|None=None):
    if d is None:
        d = {}
        for x,y in zip(np.sort(np.unique(input_design["chain_index"])), range(len(np.unique(input_design["chain_index"])))):
            d[int(x)]=int(y)
    print(d)
    design = input_design.update(chain_index=np.array([d[int(x)] for x in input_design["chain_index"]]))
    return design, d

def get_cdr_mask(
        input_design:DesignData,
        tcr_chain_index:int,
        cdr_ids:Iterable[str]=[x for x in imgt_mapper.keys() if x.startswith("a")],
        chain_index:int|None=None,
        ):
        '''
        Get a float mask (1.0 for CDR, 0.0 for framework) for cdr_ids on chain chain_index.
        Args:
            input_design:DesignData, must contain TCR chain in "chain_index" key
            chain_index:int, chain_index value for TCR/Ab chain
            cdr_ids:iterable[str], ids for cdrs (all must share the same chain letter prefix)
        '''
        cdr_ids = list(cdr_ids)
        assert (np.array([x[0] for x in cdr_ids])[:,None]==np.array([x[0] for x in cdr_ids])[None,:]).all(), \
            ValueError("All cdrs must be on the same chain!")
        if chain_index is None:
            chain_index = tcr_chain_index[int(cdr_ids[0].startswith("h") or cdr_ids[0].startswith("b"))]

        chain_mask = np.array(input_design["chain_index"]) == chain_index

        positions = [imgt_mapper[k] for k in cdr_ids]

        residue_index = np.array(input_design["residue_index"])[chain_mask]
        all_cdr_positions = np.concatenate([np.arange(s, e) for s, e in positions])
        mask = (residue_index[:, None] == all_cdr_positions[None, :]).any(axis=1)

        chain_mask[chain_mask] = mask

        return chain_mask.astype(float)

def parse_structure(
    design,
    mhc_class,
    ):

    design = design.copy()
    # convert chain indices
    design,_ = convert_chains(design)
    design, params = number_anarci(design, mhc_class=mhc_class)

    # get tcr stub
    tcr_axes, tcr_center = get_axes(
    *[get_tcr_coords(design, np.array([chain])) for chain in params["tcr_chain_index"]]
    )
    ref = get_tcr_ref(design, params)
    axis3, axis1 = check_direction(tcr_axes[:,-1], tcr_center, ref, tcr_axes[:,0])
    tcr_axes[:,-1] = axis3
    tcr_axes[:,0] = axis1
    # get mhc stub
    if mhc_class==1:
        mhc_positions = get_mhc1_positions(
            design=design,
            params=params,
            )
        mhc_coords_0 = get_mhc_coords(
            design=design,
            positions=mhc_positions[:6])
        mhc_coords_1 = get_mhc_coords(
            design=design,
            positions=mhc_positions[6:])
    elif mhc_class==2:
        mhc_positions = get_mhc2_positions(
            design=design,
            params=params,
            db_files={
                "A":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_A_chains_v2.fasta"),
                "B":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_B_chains_v2.fasta")
            },
            blast_exe=Path("../../../ncbi-blast-2.17.0+/bin"),
            )
        mhc_coords_0 = get_mhc_coords(
            design=design,
            positions=mhc_positions["A"])
        mhc_coords_1 = get_mhc_coords(
            design=design,
            positions=mhc_positions["B"])
    else:
        raise ValueError(f"Invalid mhc_class {mhc_class}!")
    print(mhc_positions)
    mhc_axes, mhc_center = get_axes(
            mhc_coords_0,
            mhc_coords_1
        )
    
    ref = get_mhc_ref(design, params)
    axis3, axis1 = check_direction(mhc_axes[:,-1], mhc_center, ref, mhc_axes[:,0])
    mhc_axes[:,-1] = axis3
    mhc_axes[:,0] = axis1
    op = get_ax_op(
        tcr_axes,
        tcr_center,
        mhc_axes,
        mhc_center)

    return op

def apply_atom_op(atom_positions:np.ndarray, op:tuple, atom_mask=np._NoValue):
    atom_positions=atom_positions.squeeze()
    # center before rotation
    a_center = np.mean(atom_positions,axis=0, where=atom_mask)
    atom_positions -= a_center
    # rotate about 000
    atom_positions =  np.einsum("xy,...y", op[0],atom_positions)
    # restore to target center
    atom_positions = atom_positions-op[1] + a_center
    return atom_positions

def rev_atom_op(atom_positions:np.ndarray, op:tuple, atom_mask=np._NoValue):
    atom_positions=atom_positions.squeeze()
    # center before rotation
    a_center = np.mean(atom_positions,axis=0, where=atom_mask)
    atom_positions -= a_center
    # rotate at origin
    atom_positions = np.einsum("...y, yx",atom_positions, op[0])
    return atom_positions + op[1] + a_center


def apply_op_full_scaffold(
    design:DesignData,
    op:np.ndarray,
    chains:np.ndarray|Iterable,
    design_op:None|np.ndarray=None,
    mhc_class:int=1,
    ):
    chains = np.array(chains)
    design = design.copy()
    #scaler = Scaler(design)
    #design = scaler.transform(design)
    chain_mask = (design["chain_index"][:,None]==chains[None,:]).any(axis=1)
    subset_design = design[chain_mask]
    atom_positions = np.array(subset_design["atom_positions"].squeeze())
    atom_mask = np.repeat(subset_design["atom_mask"][...,None],3, axis=-1).astype(bool)
    # get the operation
    if design_op is None:
        design_op = parse_structure(design, mhc_class)
    # center and align on mhc axes
    atom_positions = apply_atom_op(atom_positions, design_op, atom_mask=atom_mask)
    # apply op in reverse to mimic binding position
    atom_positions = rev_atom_op(atom_positions, op, atom_mask=atom_mask)
    # reinsert atom_positions
    subset_design = subset_design.update(atom_positions=jnp.array(atom_positions))
    subset_design.data = {k:jnp.array(v) for k,v in subset_design.data.items()}
    design.data = {k:jnp.array(v) for k,v in design.data.items()}
    design[chain_mask] = subset_design
    #design = scaler.reverse(design)
    return design

def apply_op(
    design:DesignData,
    op:np.ndarray,
    chains:np.ndarray|Iterable|None=None,
    reverse:bool=False
    ):
    design = design.copy()
    if not chains is None:
        chains = np.array(chains)
        chain_mask = (design["chain_index"][:,None]==chains[None,:]).any(axis=1)
        subset_design = design[chain_mask]
    else:
        subset_design = design
    atom_positions = np.array(subset_design["atom_positions"].squeeze())
    atom_mask = np.repeat(subset_design["atom_mask"][...,None],3, axis=-1).astype(bool)
    # get the operation
    # apply op in reverse to mimic binding position
    if reverse:
        atom_positions = rev_atom_op(atom_positions, op, atom_mask=atom_mask)
    else:
        atom_positions = apply_atom_op(atom_positions, op, atom_mask=atom_mask)
    # reinsert atom_positions
    subset_design = subset_design.update(atom_positions=jnp.array(atom_positions))
    subset_design.data = {k:jnp.array(v) for k,v in subset_design.data.items()}
    design.data = {k:jnp.array(v) for k,v in design.data.items()}
    if chains is None:
        return subset_design
    else:
        design[chain_mask] = subset_design
        return design
    
def center_atom_op(atom_positions, axes, center):
    atom_positions -= center
    return np.einsum("ij,...j->...i",axes.T,atom_positions)

def uncenter_atom_op(atom_positions, axes, center):
    atom_positions = np.einsum("ij,...j->...i",axes,atom_positions)
    return atom_positions + center

def _apply_atom_op(atom_positions:np.ndarray, op:tuple, atom_mask=np._NoValue):
    atom_positions=atom_positions.squeeze()
    # center before rotation
    #a_center = np.mean(atom_positions,axis=0, where=atom_mask)
    #atom_positions -= a_center
    # rotate about 000
    atom_positions =  np.einsum("xy,...y", op[0],atom_positions)
    atom_positions = atom_positions+op[1]
    # restore to target center
    return atom_positions

def translate_pose(
    design:DesignData,
    target_pose,
    current_pose=None,
    chains:np.ndarray|Iterable|None=None,
    mhc_class=1,
    ):
    
    design = design.copy()
    if not chains is None:
        chains = np.array(chains)
        chain_mask = (design["chain_index"][:,None]==chains[None,:]).any(axis=1)
        subset_design = design[chain_mask]
    else:
        subset_design = design
    atom_positions = np.array(subset_design["atom_positions"].squeeze())
    atom_mask = np.repeat(subset_design["atom_mask"][...,None],3, axis=-1).astype(bool)  # pyright: ignore
    # get the operation
    if current_pose is None:
        # if no axes, center given, attempt to infer
        current_pose = parse_structure(design, mhc_class=mhc_class)
    # center
    atom_positions = center_atom_op(atom_positions, *current_pose)
    atom_positions = _apply_atom_op(atom_positions, target_pose, atom_mask=atom_mask)
    # reinsert atom_positions
    subset_design = subset_design.update(atom_positions=jnp.array(atom_positions))
    subset_design.data = {k:jnp.array(v) for k,v in subset_design.data.items()}
    design.data = {k:jnp.array(v) for k,v in design.data.items()}
    if chains is None:
        return subset_design
    else:
        design[chain_mask] = subset_design
        return design

def get_mhc_ref(design, params, atoms:list=["CA"]):
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    known_chains = np.concatenate([v for v in params.values()])
    peptide_chain_index = np.array([k for k in np.unique(design["chain_index"]) if k not in known_chains])
    if len(peptide_chain_index>1):
        # if more than on unknown chain take shortest as peptide
        chain_lengths = (design["chain_index"][:,None]==peptide_chain_index[None,:]).sum(axis=0)
        peptide_chain_index = peptide_chain_index[np.argmin(chain_lengths)]
    peptide_mask = design["chain_index"]==peptide_chain_index
    return design["atom_positions"][peptide_mask][:,atom_index,:]

def get_tcr_ref(design, params, atoms:list=["CA"]):
    order = ["N", "CA", "C", "O", "CB"]
    atom_index = np.array([order.index(a) for a in atoms], dtype=np.int32)
    cdr_mask = get_cdr_mask(design, params["tcr_chain_index"], cdr_ids=[x for x in imgt_mapper.keys() if x.startswith("a")])
    cdr_mask = (cdr_mask + get_cdr_mask(design, params["tcr_chain_index"], cdr_ids=[x for x in imgt_mapper.keys() if x.startswith("b")]))>0
    return design["atom_positions"][cdr_mask][:,atom_index,:]

def check_pose_direction(pose, ref):
    axes = pose[0]
    axis3, axis1 = check_direction(axes[:,-1], pose[1], ref, axes[:,0])
    axes[:,-1] = axis3
    axes[:,0] = axis1
    return (axes, pose[1])

def superpose(tcr_pose, mhc_pose, tcr_design, mhc_design, mhc_class=1):
    '''
    Move Align designs on input poses.
    Uses an adaptation of parse_structure to get mhc and tcr separately.
    '''
    # tcr
    tcr_design, params = number_anarci(tcr_design, mhc_class=None)

    tcr_target_pose=get_axes(
        *[get_tcr_coords(
            tcr_design,
            np.array([chain])
        ) for chain in params["tcr_chain_index"]]
    )

    target_tcr_params = dict(tcr_chain_index=np.unique(tcr_design["chain_index"]))
    tcr_target_pose = check_pose_direction(tcr_target_pose, get_tcr_ref(tcr_design, target_tcr_params))
    tcr_op = get_ax_op(
        *tcr_target_pose,
        *tcr_pose,
    )
    tcr_design = translate_pose(tcr_design, target_pose=tcr_pose, current_pose=tcr_target_pose, chains=None)

    
    # mhc
    if mhc_class==1:
        chains = np.unique(mhc_design["chain_index"])
        chain_lengths = (mhc_design["chain_index"][:, None]==chains[None,:]).sum(axis=0)
        target_mhc_params = dict(mhc_chain_index=chains[np.argmax(chain_lengths)][None])
        mhc_positions = get_mhc1_positions(
            design=mhc_design,
            params=None,
            )
        mhc_coords_0 = get_mhc_coords(
            design=mhc_design,
            positions=mhc_positions[:6])
        mhc_coords_1 = get_mhc_coords(
            design=mhc_design,
            positions=mhc_positions[6:])
    elif mhc_class==2:
        chain_lengths = (mhc_design["chain_index"][:, None]==chains[None,:]).sum(axis=0)
        target_mhc_params = dict(mhc_chain_index=chains[np.argsort(chain_lengths)][::-1][:2])
        mhc_positions = get_mhc2_positions(
            design=mhc_design,
            params=target_mhc_params,
            db_files={
                "A":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_A_chains_v2.fasta"),
                "B":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_B_chains_v2.fasta")
            },
            blast_exe=Path("../../../ncbi-blast-2.17.0+/bin"),
            )
        mhc_coords_0 = get_mhc_coords(
            design=mhc_design,
            positions=mhc_positions["A"])
        mhc_coords_1 = get_mhc_coords(
            design=mhc_design,
            positions=mhc_positions["B"])
    else:
        raise ValueError(f"Invalid mhc_class {mhc_class}!")
    mhc_target_pose = get_axes(
            mhc_coords_0,
            mhc_coords_1,
        )
    mhc_target_pose = check_pose_direction(mhc_target_pose, get_mhc_ref(mhc_design, target_mhc_params))
    
    
    mhc_design = translate_pose(mhc_design, target_pose=mhc_pose, current_pose=mhc_target_pose, chains=None)

    return DesignData.concatenate((tcr_design, mhc_design), sep_chains=False, sep_batch=False)

def _parse_structure(
    design,
    mhc_class,
    ):

    design = design.copy()
    # convert chain indices
    design,_ = convert_chains(design)
    design, params = number_anarci(design, mhc_class=mhc_class)

    # get tcr stub
    tcr_pose = get_axes(
    *[get_tcr_coords(design, np.array([chain])) for chain in params["tcr_chain_index"]]
    )
    tcr_ref = get_tcr_ref(design, params)
    tcr_pose = check_pose_direction(tcr_pose, tcr_ref)
    # get mhc stub
    if mhc_class==1:
        mhc_positions = get_mhc1_positions(
            design=design,
            params=params,
            )
        mhc_coords_0 = get_mhc_coords(
            design=design,
            positions=mhc_positions[:6])
        mhc_coords_1 = get_mhc_coords(
            design=design,
            positions=mhc_positions[6:])
    elif mhc_class==2:
        mhc_positions = get_mhc2_positions(
            design=design,
            params=params,
            db_files={
                "A":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_A_chains_v2.fasta"),
                "B":Path("/home/ntbiotech/Documents/Current_projects/BinderDesign/TCRdock/tcrdock/db/both_class_2_B_chains_v2.fasta")
            },
            blast_exe=Path("../../../ncbi-blast-2.17.0+/bin"),
            )
        mhc_coords_0 = get_mhc_coords(
            design=design,
            positions=mhc_positions["A"])
        mhc_coords_1 = get_mhc_coords(
            design=design,
            positions=mhc_positions["B"])
    else:
        raise ValueError(f"Invalid mhc_class {mhc_class}!")
    print(mhc_positions)
    mhc_pose = get_axes(
            mhc_coords_0,
            mhc_coords_1
        )
    mhc_ref = get_mhc_ref(design, params)
    mhc_pose = check_pose_direction(mhc_pose, ref=mhc_ref)

    return design, params, mhc_pose, tcr_pose

def set_tcr_pose(design, target_pose, mhc_class):
    '''Apply a '''

    design, params, mhc_pose, tcr_pose = _parse_structure(design, mhc_class=mhc_class)

    # align tcr to mhc_pose
    design = translate_pose(
        design,
        target_pose=mhc_pose,
        current_pose=tcr_pose,
        chains=params["tcr_chain_index"],
    )
    origin_pose=(
        np.array([[1,0,0], [0,1,0], [0,0,1]]), np.array([0,0,0])
    )
    # align to origin
    design = translate_pose(
        design,
        target_pose=origin_pose,
        current_pose=mhc_pose,
        chains=None
    )

    return translate_pose(
        design,
        target_pose=target_pose,
        current_pose=origin_pose,
        chains=params["tcr_chain_index"]
    )

def get_tcr_pose(design, mhc_class):
    '''Get the tcr pose, with mhc aligned to the origin.'''
    
    design, params, mhc_pose, tcr_pose = _parse_structure(design, mhc_class=mhc_class)

    origin_pose=(
        np.array([[1,0,0], [0,1,0], [0,0,1]]), np.array([0,0,0])
    )

    # get operation to align mhc to origin
    center_mhc = get_ax_op(
        *mhc_pose,
        *origin_pose
    )

    return apply_ax_op(*tcr_pose, center_mhc)