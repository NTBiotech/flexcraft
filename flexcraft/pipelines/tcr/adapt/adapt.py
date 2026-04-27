'''
Codebase for running the ADAPT pipeline for TCR design
Notes:
    To install ANARCI for IMGT numbering, go to your project directory run:
        """
        conda update -n base -c conda-forge conda
        conda install -c bioconda hmmer=3.3.2 -y
        git clone https://github.com/oxpig/ANARCI.git
        cd ANARCI
        python setup.py install
        """
TODO:
- add pmpnn hparams
(- implement current design attribute)
'''
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import *
from flexcraft.structure.metrics import *
from flexcraft.structure.boltz import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.aa_codes import AF2_CODE, decode
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *

from colabdesign.af.alphafold.model import utils as af_utils
from jax._src.pjit import JitWrapped
import anarci

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Callable
from datetime import datetime

import numpy as np
import pandas as pd


class ADAPT:
    '''
    ADAPT (Antigen-receptor Design Against Peptide-MHC Targets)
    adapted from:
    Motmaen, A., Jude, K.M., Wang, N., Minervina, A., Feldman, D., Lichtenstein, M.A.,
    Ebenezer, A., Correnti, C., Thomas, P.G., Garcia, K.C., Baker, D., Bradley, P., 2025.
    Targeting peptide-MHC complexes with designed T cell receptors and antibodies.
    https://doi.org/10.1101/2025.11.19.689381
    '''
    def __init__(
        self,
        op_dir:Path|str,
        key,
        pmpnn_model:Callable|None=None,
        af2_model:None|JitWrapped=None,
        af2_params:None|dict=None,
        boltz_docking:bool=False,
        boltz_redocking:bool=False,
        boltz_parameter_path:Path|str=Path("./params/boltz"),
        boltz_model_name:str="boltz2_conf",
        boltz_num_recycle = 2,
        boltz_num_samples = 1,
        boltz_num_sampling_steps = 25,
        boltz_deterministic = False,
        af2_model_name:str|None=None,
        af2_parameter_path:str|Path|None=None,
        af2_multimer:bool|None=None,
        af_num_recycle:int=0,
        pmpnn_parameter_path:str|None=None,
        pmpnn_hparams:dict={},
        ab:bool = False,
        mhc_chain_index:int|Tuple[int]=0,
        tcr_chain_index:int|Tuple[int]=(2,3),
        name="AdaptTrial",
        out_dir:None|str|Path=None,
        trim:bool=True,
        chain_cache_len:int=650, # how long to pad for af
    ):
        '''
        Initialize ADAPT class for TCR design and refinement.
        Args:
            op_dir:Path|str, directory containing input_data subdir with TCR, pMHC and CDR files
            key, Key object used for AF inference
            pmpnn_model:JitWrapped|None=None, ProteinMPNN model for sequence prediction.
                If None, the model is loaded from params in the input_dir.
            af2_model:None|JitWrapped=None, AF2 Model for structure prediction.
                If None, the model is loaded from params in the input_dir.
            af2_params:None|dict=None, Dictionary of AF2 parameters. 
                If not supplied, parameters are loaded from the input_dir.
            boltz_docking:bool=False, Wether to use Boltz-2 for the initial docking step instead of AF2.
            boltz_redocking:bool=False, Wether to use Boltz-2 for the re-docking step instead of AF2.
            boltz_parameter_path:Path|str=Path("./params/boltz"), path to directory containing Boltz-2 parameters.
            boltz_model_name:str="boltz2_conf", name for Boltz-2 model for loading parameters.
            boltz_num_recycle = 2, number of recycle steps in Boltz-2 inference.
            boltz_num_samples = 1, number of samples generated in Boltz-2 docking.
            boltz_num_sampling_steps = 25, number of Boltz-2 sampling steps.
            boltz_deterministic = False, wether to load Boltz-2 in deterministic mode
            af2_model_name:str|None=None, AF2 model name for loading config (and parameters).
            af2_parameter_path:str|Path|None=None, path to AF2 parameter file (.npy)
            af2_multimer:bool|None=None, wether AF2 model is multimer. If None, is infered from model name
            af_num_recycle:int=0, number of AF2 recycle steps.
            pmpnn_parameter_path:str|None=None, path to pMPNN parameter file (.pkl).
            pmpnn_hparams:dict={}, pMPNN hyperparameters.
            ab:bool = False, wether to work on Antibodies of TCRs
            mhc_chain_index:int|Tuple[int]=0, index of MHC chains in DesignData input.
                Is generally infered automatically
            tcr_chain_index:int|Tuple[int]=(2,3), index
            name="AdaptTrial",
            out_dir:None|str|Path=None,
            trim:bool=False,
            chain_cache_len:int=400, # how long to pad for af
        Attr:

        '''
        # directory organization
        if not isinstance(op_dir, Path):
            op_dir = Path(op_dir)
        self.op_dir = op_dir
        self.in_dir = self.op_dir/"input_data"
        self.name=name
        
        if not self.in_dir.exists():
            raise FileNotFoundError(f"Input dir {self.in_dir} does not exist!")
        
        if out_dir:
            self.out_dir = out_dir
        else:
            self.out_dir = self.op_dir/(datetime.now().strftime("%Y-%d-%b_%H:%M:%S")+f"_{name}_0")
            n=0
            while self.out_dir.exists():
                n+=1
                self.out_dir = self.op_dir/(self.out_dir.name[:-1]+str(n))
        
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        
        if not self.out_dir.exists():
            print(f"Creating Output Directory at {self.out_dir}")
            self.out_dir.mkdir()

        # map cdr id to start, end tuple (end exclusive)
        self.ab = ab
        self.imgt_mapper = {
            "acdr1":(27,38),
            "acdr2":(56, 65),
            "acdr3":(105,117),
            "bcdr1":(27,38),
            "bcdr2":(56,65),
            "bcdr3":(105,117),
            }
        if self.ab:
            self.imgt_mapper = {
                "lcdr1":(27,38),
                "lcdr2":(56, 65),
                "lcdr3":(105,117),
                "hcdr1":(27,38),
                "hcdr2":(56,65),
                "hcdr3":(105,117),}

        self.trim = trim
        self.columns=["score","scaffold","in_pool",
            *[
            k for k in self.imgt_mapper.keys()
            ],
            *[
            f"{k}_coords" for k in self.imgt_mapper.keys()
            ],
            ]
        self.columns_default = {"in_pool":True}
        self.scores = self.out_dir/"scores.csv"
        pd.DataFrame(columns=self.columns).to_csv(self.scores, header=True)
        self.cdr_coords = copy(self.imgt_mapper)

        self.chain_cache_len = chain_cache_len
        # chain indices
        # pack in array for comparative operations
        if isinstance(mhc_chain_index, int):
            mhc_chain_index = (mhc_chain_index,)
        self.mhc_chain_index = np.array(mhc_chain_index)
        if isinstance(tcr_chain_index, int):
            tcr_chain_index = (tcr_chain_index,)
        self.tcr_chain_index = np.array(tcr_chain_index)
        
        
        self.setup_pmpnn(
            pmpnn_model=pmpnn_model,
            pmpnn_parameter_path=pmpnn_parameter_path,
            pmpnn_hparams=pmpnn_hparams,
        )
        
        # AlphaFold
        self.key = key
        if af2_model is None or af2_params is None:
            self.setup_af2(
                af2_model_name=af2_model_name,
                af2_parameter_path=af2_parameter_path,
                af2_multimer=af2_multimer,
            )
        else:
            self.af2_model = af2_model
            self.af2_params = af2_params

        # Boltz2
        self.boltz_docking = boltz_docking
        self.boltz_redocking = boltz_redocking
        if self.boltz_docking or self.boltz_redocking:
            self.boltz_parameter_path = boltz_parameter_path
            self.boltz_model_name = boltz_model_name
            self.boltz_model = Joltz2(model=self.boltz_model_name+".ckpt", cache=self.boltz_parameter_path)
            self.boltz_num_recycle = boltz_num_recycle
            self.boltz_num_samples = boltz_num_samples
            self.boltz_num_sampling_steps = boltz_num_sampling_steps
            self.boltz_deterministic = boltz_deterministic
            self.boltz_predictor = self.boltz_model.predictor_adhoc(
                num_recycle=self.boltz_num_recycle,
                num_samples=self.boltz_num_samples,
                num_sampling_steps=self.boltz_num_sampling_steps,
                deterministic=self.boltz_deterministic
            )
        else:
            self.boltz_parameter_path = None
            self.boltz_model_name = None

        self.rmsd = RMSD()
    
    def get_scores(self):
        return pd.read_csv(self.scores, header=0, index_col=0)
    def write_scores(self, scores:pd.DataFrame):
        return scores[self.columns].to_csv(self.scores, header=True)
    def append_scores(self, row:pd.Series, name:str|None=None):
        for c in self.columns:
            if c not in row.index:
                row[c] = self.columns_default.get(c, None)
        row.name = name
        return row.to_frame().T[self.columns].to_csv(self.scores, header=False, mode="a")

    def trim_design(
        self,
        input_design:DesignData
        ):
        '''Trim the length of the construct to cache length.'''
        design_length = len(input_design["aa"])
        if design_length<=self.chain_cache_len:
            return input_design
        trim = ((design_length-self.chain_cache_len)//len(self.mhc_chain_index))+1
        # trim the longest mhc chain
        chains, counts = np.unique(input_design["chain_index"], return_counts=True)
        mhc_mask = (chains[:, None]==self.mhc_chain_index[None,:]).any(axis=1)
        print("trim_design self.mhc_chain_index",self.mhc_chain_index)
        for chain in self.mhc_chain_index:
            print("trim_design chain",chain)
            chain_mask = input_design["chain_index"]==chain
            if chain_mask.sum()-trim< 200:
                print("WARNING! trimmed mhc chain to less than 200 AAs! Manual trimming may be necessary.")
            trim_mask = np.ones(len(input_design["aa"]), dtype=np.bool_)
            chain_end = np.arange(len(chain_mask))[chain_mask][-1]+1
            # trim from the chain end
            trim_mask[chain_end-trim:chain_end] = False
            input_design = input_design[trim_mask]
            print(f"Trimming chain {chain} by {trim} from {chain_mask.sum()} to a total of {len(input_design['aa'])} residues.")
        return input_design

    def pad_design(
        self,
        input_design:DesignData,
        *covariates,
        ):
        '''
        Pad the design to conserve AF input length in order to avoid recompilation.
        Covariates are padded with same length as input_design with zeros
        '''
        pad_length = self.chain_cache_len-len(input_design["aa"])
        if pad_length==0:
            return input_design, pad_length, *covariates
        if pad_length<0:
            print(f"Design with length {len(input_design['aa'])} exceeds chain_cache_len {self.chain_cache_len}!\nSkipping padding step.")
            return input_design, pad_length, *covariates
        print(f"Padding design by {pad_length}...")
        atom_format = input_design["atom_mask"].shape[1]
        input_design = DesignData.concatenate(
            [input_design, DesignData.from_length(pad_length).update(
                aa=jnp.full((pad_length,), 7, dtype=jnp.int32),
                mask=jnp.zeros((pad_length,), dtype=jnp.bool_),
                atom_positions=jnp.zeros((pad_length, atom_format, 3), dtype=jnp.float32),
                atom_mask=jnp.zeros((pad_length, atom_format), dtype=jnp.bool_),
                )],
            sep_chains=False, sep_batch=False
        )
        if covariates:
            covariates = [np.concatenate((c.copy(), np.zeros((pad_length,), dtype=c.dtype)), axis=0) for c in covariates]
            return input_design, pad_length, *covariates
        return input_design, pad_length
    
    def rm_pad(
        self,
        input_design:DesignData,
        pad_length:int,
        *covariates
        ):
        '''Remove padding from pad_design. Removes pad from all keys with same second dimension too.'''
        print(f"Removing {pad_length} residues long pad!")
        for k,v in input_design.items():
            if len(v.shape)>1:
                if v.shape[0]==v.shape[1]:
                    input_design = input_design.update(**{k:v[:,:-pad_length]})
        if covariates:
            covariates = [c[:-pad_length] for c in covariates]
            return input_design[:-pad_length], *covariates
        return input_design[:-pad_length]


    def setup_pmpnn(
        self,
        pmpnn_model:None|JitWrapped=None,
        pmpnn_parameter_path:str|None=None,
        pmpnn_hparams:dict={},
        ):

        # Protein MPNN TODO: add hparams
        self.pmpnn = jax.jit(make_pmpnn(pmpnn_parameter_path, eps=0.05))
        self.pmpnn_hparams = {
            "temperature":0.1,
            "model_name":'v_48_020',
            "num_seq_per_target":3,
            "n_edges":48,
            "training_noise":0.2,#Å,
            "center_logits":False,
        }
        self.pmpnn_hparams.update(pmpnn_hparams)
        self.pmpnn_transform = lambda center: transform_logits((
            toggle_transform(
                center_logits(center), use=self.pmpnn_hparams["center_logits"]),
            scale_by_temperature(self.pmpnn_hparams["temperature"]),
            forbid("C", aas.PMPNN_CODE),
            norm_logits
        ))


    def setup_af2(
        self,
        af2_model_name:str|None=None,
        af2_parameter_path:str|Path|None=None,
        af2_multimer:bool|None=None,
        af_num_recycle:int=0,
        ):

        self.af2_model_name = af2_model_name
        self.af2_parameter_path = af2_parameter_path
        assert (not af2_model_name is None) or (not af_parameter_path is None), ValueError("Specify either model name or parameter path!")
        if self.af2_parameter_path is None:
            self.af2_parameter_path = (self.in_dir/self.af2_model_name).with_suffix(".pkl")
        if self.af2_model_name is None:
            self.af2_model_name = self.af2_parameter_path.stem
        if isinstance(self.af2_parameter_path, str):
            self.af2_parameter_path = Path(self.af2_parameter_path)
        if self.af2_parameter_path.suffix == ".pkl":
            # if params in pickle we expect fine-tuned parameters from zenodo, see load_data
            import pickle
            with open(self.af2_parameter_path, "rb") as rf:
                params = pickle.load(rf)
            # filer param keys and adjust formatting
            clean_params = {}
            for k,v in params.items():
                if not "/" in k:
                    print(f"Skipping {k}")
                    continue
                if not isinstance(v, dict):
                    print(f"{k} has no dict")
                    continue
                for n,i in v.items():
                    clean_params[f"{k}//{n}"] = i
            self.af2_params = af_utils.flat_params_to_haiku(params=clean_params, fuse=True)
            # adjust model name
            self.af2_model_name = "_".join(self.af2_model_name.split("_")[:3])
        else:
            self.af2_params = get_model_haiku_params(
                    model_name=self.af2_model_name,
                    data_dir=af2_parameter_path.__str__(), fuse=True)


        self.af_num_recycle = af_num_recycle
        if af2_multimer is None:
            self.use_multimer = "multimer" in self.af2_model_name
        else:
            self.use_multimer = af2_multimer
        self.af2_config = model_config(self.af2_model_name)
        self.af2_config.model.global_config.use_dgram = False
        self.af2_model = jax.jit(make_predict(
            make_af2(self.af2_config, use_multimer=self.use_multimer),
            num_recycle=self.af_num_recycle))
        
        return self.af2_model, self.af2_params


    def af_infer(self, af_input:AFInput) -> AFResult:
        '''
        Wrapper for af model, handles non-multimer models.
        '''
        if not self.use_multimer:
            num_chains = len(jnp.unique(af_input.chain_index))
            af_input_masked = af_input.block_diagonal(num_sequences=num_chains)
            af_result: AFResult = self.af2_model(self.af2_params, self.key(), af_input_masked)
        else:
            af_result: AFResult = self.af2_model(self.af2_params, self.key(), af_input)
        return af_result


    def af_docking_step(
        self,
        input_design:DesignData,
        evaluate:bool=False,
        is_target:np.ndarray|None=None,
        templates:List[DesignData]|None=None,
        off_target_template:bool=True,
        save_structure:bool|Path|str=False,
        ) -> DesignData|Tuple[DesignData, float]:
        '''
        Predict structures of CDRs.
        Args:
            input_design: DesignData, input sequence of spliced TCR with pMHC
            evaluate:bool=False, if True calculate scoring for design and output
            is_target:np.ndarray|None, boolean mask for CDR3 positions (required when evaluate=True)
        Returns:
            structure: DesignData, predicted structure and sequence of the input_design
            score: float, output score if evaluate
        '''
        # pad to avoid recompilation
        design, pad_length, is_target = self.pad_design(input_design, is_target)
        design = design.update(residue_index = np.arange(
            len(design["aa"])
        ))
        af_input = AFInput.from_data(design)

        if off_target_template:
            if is_target is None:
                print("No is_target input. Not adding template!")
            else:
                af_input = af_input.add_template(design, where=~is_target)
        
        if not templates is None:
            for t in templates:
                af_input = af_input.add_template(t)

        af_result = self.af_infer(af_input=af_input)
        design, is_target = self.rm_pad(af_result.to_data(), pad_length, is_target)
        if evaluate:
            score = self.evaluate_step(result=design, input_design=input_design, is_target=is_target)
            if save_structure:
                if isinstance(save_structure, bool):
                    save_structure = "evaluated_structure.pdb"
                design.save_pdb(self.out_dir/save_structure)
            return design, score
        if save_structure:
            if isinstance(save_structure, bool):
                save_structure = "docked_structure.pdb"
            design.save_pdb(self.out_dir/save_structure)
        # remove pad
        return design


    def boltz_docking_step(
        self,
        input_design:DesignData,
        save_structure:bool=False,
        evaluate:bool=False,
        is_target:np.ndarray|None=None,
        )->List[DesignData]|Tuple[List[DesignData], float]:
        '''
        Predict protein structure using Boltz-2.
        Returns:
            structure: List[DesignData], predicted structure and sequence of the input_design for each sample in self.boltz_num_samples
            score: float, output score if evaluate
        '''
        input_design, pad_length = self.pad_design(input_design=input_design)
        chain_masks = (input_design["chain_index"][None,:] == np.unique(input_design["chain_index"])[:, None])
        boltz_prediction = self.boltz_predictor(
            self.key(), 
            *[
                {
                    "sequence": decode(input_design["aa"][c], AF2_CODE),
                    "kind":"protein"
                }
                for c in chain_masks
            ]
            )
        if save_structure:
            for n in range(self.boltz_num_samples):
                if isinstance(save_structure, str):
                    save_structure = Path(save_structure)
                if isinstance(save_structure, bool):
                    if evaluate:
                        save_structure = Path("boltz_evaluated_structure.pdb")
                    else:
                        save_structure = Path("boltz_docked_structure.pdb")
                save_structure = f"{save_structure.with_suffix('')}_{n}.pdb"
                boltz_prediction.save_pdb(self.out_dir/save_structure, n)
        
        boltz_result = boltz_prediction.result
        print_dd(boltz_result.to_data(),"boltz_result")
        atom24, mask24 = boltz_result.atom24_samples
        
        if self.boltz_num_samples>1:
            out = [
                self.rm_pad(
                    DesignData(data=dict(
                    atom_positions=atom24[n],
                    atom_mask=mask24,
                    aa=boltz_result.restype,
                    mask=mask24.any(axis=1),
                    residue_index=boltz_result.residue_index,
                    chain_index=boltz_result.chain_index,
                    batch_index=jnp.zeros_like(boltz_result.residue_index),
                    plddt=boltz_result.plddt[n] if len(boltz_result.plddt.shape) == 2 else boltz_result.plddt,)
                    ).untie(),
                    pad_length,)
                for n in range(self.boltz_num_samples)
            ]
        else:
            out = [self.rm_pad(boltz_result.to_data(), pad_length),]
        if evaluate:
            score = self.evaluate_step(result=boltz_result, input_design=input_design, is_target=is_target)
            return out, score
        
        return out


    def design_step(self, input_design:DesignData, target_mask:np.ndarray|None):
        '''
        Design a sequence from a structure using ProteinMPNN.
        Args:
            input_design: DesignData, containing predicted structure
            target_mask: np.ndarray|None, if None, all cdrs are redesigned
        '''
        input_design, pad_length, target_mask = self.pad_design(input_design, target_mask)
        if target_mask is None:
            # mask all cdrs: use canonical CDR names by chain
            if not self.ab:
                alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]
            else:
                alpha_cdrs = [f"lcdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"hcdr{n}" for n in range(1, 4)]
            target_mask = np.zeros(len(input_design["aa"]))
            target_mask += self.cdr_mask(
                input_design=input_design,
                chain_index=self.tcr_chain_index[0],
                cdr_ids=alpha_cdrs,
            )
            target_mask += self.cdr_mask(
                input_design=input_design,
                chain_index=self.tcr_chain_index[1],
                cdr_ids=beta_cdrs,
            )
            target_mask = target_mask > 0

        target_aa = np.array(input_design["aa"])
        target_aa[target_mask] = 20
        input_design = input_design.update(aa=jnp.array(target_aa))
        
        logit_center = self.pmpnn(self.key(), input_design)["logits"].mean(axis=0)
        pmpnn_sampler = sample(self.pmpnn, logit_transform=self.pmpnn_transform(logit_center))

        pmpnn_result, _ = pmpnn_sampler(self.key(), input_design)
        pmpnn_result = input_design.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        return self.rm_pad(pmpnn_result, pad_length)

    def evaluate_step(
        self,
        result:AFResult|JoltzResult|DesignData,
        input_design:DesignData,
        is_target:np.ndarray,
        ) -> float:
        '''
        Calculate score for protein design.
        '''
        cdr3_rmsd = self.cdr3_rmsd(
            result=result, input_design=input_design, is_target=is_target)
        ipae = self.ipae(result=result)
        return 2*ipae+0.5*cdr3_rmsd


    def ipae(
        self,
        result:AFResult|JoltzResult|DesignData,
    )-> float:
        '''
        Compute the mean PAE(predicted aligned error) for all (pMHC, TCR)x(TCR, pMHC) residue pairs.
        result needs to contain "pae" and chain_index keys/attributes.
        '''
        if isinstance(result, DesignData):
            pae_matrix=result["pae"]
        else:
            pae_matrix = result.pae

        mask = (result.chain_index[:,None] == self.tcr_chain_index[None,:]).sum(axis=1)>0
        mask = mask[:,None]!=mask[None,:]
        return (mask*pae_matrix).sum()/np.max([1,mask.sum()])


    def cdr3_rmsd(
        self,
        result:AFResult|JoltzResult,
        input_design: DesignData,
        is_target:np.ndarray,
        ) -> float:
        '''
        Calculate the RMSD for CDR3 chains after alignment of mhc chain.
        Args:
            result: AFResult|JoltzResult, result of af prediction. Has inputs and result attribute.
            is_target: np.ndarray, bool array that is True for all CDR3 positions.
        '''
        # only include MHC in alignment
        # only calculate rmsd for cdr3
        rmsd = self.rmsd(
            x=result,
            y=input_design,
            weight=(result.chain_index[:,None]==self.mhc_chain_index[None,:]).any(axis=1),
            eval_mask=is_target,
            )
        return rmsd


    def design_trial(
        self,
        scaffold:str|Path|DesignData,
        cdrs:Dict[str,str],
        pMHC:str|Path|DesignData|None=None,
        redesign_all_cdrs:bool=False,
    ):
        '''Run a design step for recombination of TCRs with CDR3s'''
        # Save original identifiers for output naming
        scaffold_name = Path(scaffold).stem.split("_")[0] if isinstance(scaffold, (str, Path)) else "scaffold"
        pmhc_name = Path(pMHC).stem if isinstance(pMHC, (str, Path)) else "pmhc"
        scaffold_name += "+"+pmhc_name
        pmhc_is_scaffold = pMHC is None or pMHC == scaffold
        print(
            f"\n!---Design Trial for {scaffold_name}---!\n",
        )
        # process input
        if not isinstance(scaffold, DesignData):
            # load tcr
            if isinstance(scaffold, str):
                scaffold = Path(scaffold)
            if scaffold.parent == self.in_dir:
                scaffold = PDBFile(path=self.in_dir/scaffold.name).to_data()
            else:
                scaffold = PDBFile(path=scaffold).to_data()

        if pmhc_is_scaffold:
            # check if all chains are present in the single structure
            assert len(np.unique(scaffold["chain_index"])) >= (len(self.mhc_chain_index) + len(self.tcr_chain_index) + 1), \
                ValueError(f"No pMHC provided and TCR with incorrect chain number!")
            design = scaffold
        else:
            if not isinstance(pMHC, DesignData):
                # load pmhc
                if isinstance(pMHC, str):
                    pMHC = Path(pMHC)
                if pMHC.parent == self.in_dir:
                    pmhc = PDBFile(path=self.in_dir/pMHC.name).to_data()
                else:
                    pmhc = PDBFile(path=pMHC).to_data()
            else:
                pmhc = pMHC
            if len(np.unique(pmhc["chain_index"]))>3:
                pmhc = self.number_anarci(pmhc, trim=False)
                pmhc = pmhc[(pmhc["chain_index"][:,None] != self.tcr_chain_index[None,:]).any(axis=1)]
            # concatenate with split_chains
            design = DesignData.concatenate([pmhc, scaffold], sep_chains=True)
        print_dd(design, "Input")
        # imgt numbering
        design = self.number_anarci(design, trim=self.trim)
        
        sequences = ({},{})
        # load cdr
        for k,v in cdrs.items():
            if k in self.imgt_mapper:
                sequences[k.startswith("h") or k.startswith("b")].update({k:v})

        print("CDRs: ", sequences)
        print_dd(design, "Numbered")
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[0],
            sequences=sequences[0]
            )
        print_dd(design, "Inserted1")
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[1],
            sequences=sequences[1]
            )
        if self.trim:
            design = self.trim_design(design)
        print_dd(design, "Inserted")

        if redesign_all_cdrs:
            pre = ["b","h"][int(self.ab)]
            alpha_cdrs = [k for k in self.imgt_mapper.keys() if not k.startswith(pre)]
            beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith(pre)]
        else:
            alpha_cdrs = [k for k in sequences[0].keys()]
            beta_cdrs = [k for k in sequences[1].keys()]
        
        target_mask = (
            self.cdr_mask(design, chain_index=self.tcr_chain_index[0], cdr_ids=alpha_cdrs)
            + self.cdr_mask(design, chain_index=self.tcr_chain_index[1], cdr_ids=beta_cdrs)
        ) > 0

        # docking step (structure prediction without evaluation)
        if self.boltz_docking:
            boltz_designs = self.boltz_docking_step(input_design=design)
            design = boltz_designs.pop()
        else:
            design = self.af_docking_step(input_design=design, is_target=target_mask)

        print_dd(design, "Docked")
        # redesign step: redesign the CDR positions
        design = self.design_step(input_design=design, target_mask=target_mask)
        print_dd(design, "Redesigned")
        # save design as unique file name
        file_name = f"{scaffold_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{n}.pdb"
        # redocking + evaluation step
        if self.boltz_redocking:
            boltz_designs, score = self.boltz_docking_step(
                input_design=design,
                evaluate=True,
                is_target=target_mask,
                save_structure=file_name,
            )
        else:
            templates = None
            if self.boltz_docking:
                templates = boltz_designs
            design, score = self.af_docking_step(
                input_design=design,
                evaluate=True,
                is_target=target_mask,
                templates=templates,
                save_structure=file_name,
            )
        print_dd(design, "Redocked")
        print(f"Saving design with score {score} to {file_name}!")
        # Use a named Series so missing CDR1/CDR2 columns get NaN automatically
        row = {"score": score, "scaffold": scaffold_name,
                **{cdr:self.get_cdr_seq(design, cdr) for cdr in self.imgt_mapper.keys()},
                **{f"{k}_coords":v for k, v in self.cdr_coords.items()}}
        print(row)
        self.append_scores(pd.Series(row), file_name)
        return score


    def get_design(
        self,
        index:str,
        family:str|None=None
        ):
        '''
        Query the scores table for a design and adjust the cdr coords.
        '''
        scores = self.get_scores()
        if not family is None:
            scores = scores.loc[scores["scaffold"]==family]
        if index=="random":
            row = scores.sample(1)
        elif index=="max":
            row = scores.sort_values("score", ascending=False).iloc[0]
        elif index=="min":
            row = scores.sort_values("score", ascending=True).iloc[0]
        else:
            row = scores.loc[index]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        to_tuple = lambda s: tuple([int(n) for n in s.strip("()").split(",")])
        self.cdr_coords = {k:to_tuple(row[f"{k}_coords"]) if row[f"{k}_coords"] else self.imgt_mapper[k] for k in self.imgt_mapper.keys()}
        design = PDBFile(path=self.out_dir/row.name).to_data()
        return design, row.name


    def refine_trial(
        self,
        scaffold:DesignData|str|Path,
        scaffold_name:str|None=None,
        cdrs:List[str]|None=None,
        redesign_all_cdrs:bool=False,
        ):
        if cdrs is None:
            cdrs = ["lcdr3", "hcdr3"] if self.ab else ["acdr3", "bcdr3"]

        # filter ids
        for cdr in cdrs:
            if cdr not in self.imgt_mapper.keys():
                raise KeyError(f"Invalid cdr id {cdr}! Choose one of {[f'{k}, ' for k in self.imgt_mapper.keys()]} or change self.ab.")

        # fetch list of candidates
        # process input
        
        scaffold_name = Path(scaffold).stem.split("_")[0] if isinstance(scaffold, (str, Path)) else scaffold_name
        if not isinstance(scaffold, DesignData):
            # load tcr
            if isinstance(scaffold, str):
                scaffold = Path(scaffold)
            if scaffold.parent==self.in_dir:
                scaffold = scaffold.name
            scaffold = PDBFile(path=self.in_dir/scaffold).to_data()
        print(
            f"\n!---Design Trial for {scaffold_name}---!\n",
        )
        print(f"CDRs: {cdrs}")
        # imgt numbering
        print_dd(scaffold, "loaded")
        scaffold = self.number_anarci(scaffold, trim=self.trim)
        if self.trim:
            scaffold = self.trim_design(scaffold)
        print_dd(scaffold, "trimmed")
        # mutate 2 cdr positions
        scaffold, mutated_cdrs = self.mutate_cdrs(
            input_design=scaffold,
            cdrs=cdrs,
            n_mutations=2,
            )

        if redesign_all_cdrs:
            pre = ["b","h"][int(self.ab)]
            alpha_cdrs = [k for k in self.imgt_mapper.keys() if not k.startswith(pre)]
            beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith(pre)]
        else:
            pre = ["b","h"][int(self.ab)]
            alpha_cdrs = [cdr for cdr in cdrs if not cdr.startswith(pre)]
            beta_cdrs = [cdr for cdr in cdrs if cdr.startswith(pre)]
        target_mask = (
            self.cdr_mask(scaffold, chain_index=self.tcr_chain_index[0], cdr_ids=alpha_cdrs)
            + self.cdr_mask(scaffold, chain_index=self.tcr_chain_index[1], cdr_ids=beta_cdrs)
        ) > 0

        # docking step (structure prediction without evaluation)
        if self.boltz_docking:
            boltz_designs = self.boltz_docking_step(input_design=scaffold)
            scaffold = boltz_designs.pop()
        else:
            scaffold = self.af_docking_step(input_design=scaffold, is_target=target_mask)
        print_dd(scaffold, "Docked")

        # redesign step: redesign the CDR positions
        scaffold = self.design_step(input_design=scaffold, target_mask=target_mask)
        print_dd(scaffold, "Redesigned")
        # save design as unique file name
        file_name = f"{scaffold_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{n}.pdb"
        # redocking + evaluation step
        if self.boltz_redocking:
            scaffold, score = self.boltz_docking_step(
                input_design=scaffold,
                evaluate=True,
                is_target=target_mask,
                save_structure=file_name,
            )
        else:
            templates = None
            if self.boltz_docking:
                templates = boltz_designs
            scaffold, score = self.af_docking_step(
                input_design=scaffold,
                evaluate=True,
                is_target=target_mask,
                templates=templates,
                save_structure=file_name,
            )
        print_dd(scaffold, "Redocked")
        print(f"Saving design with score {score} to {file_name}!")
        # Use a named Series so missing CDR1/CDR2 columns get NaN automatically
        row = {"score": score, "scaffold": scaffold_name,
            **{cdr:self.get_cdr_seq(scaffold, cdr) for cdr in self.imgt_mapper.keys()},
            **{f"{k}_coords":v for k, v in self.cdr_coords.items()}}

        # compare to existing
        print("Replacing ",self.compare(file_name,row))


    def compare(
        self,
        file_name:str|Path,
        specs:pd.Series|dict,
        family_limit:int=10,
        full_limit:int=20,
        delete_file=False,
    ):
        '''
        Compare design with previous designs. Adds specs to the self.scores DataFrame, to then remove the worst performing design.
        Args:
            specs:pd.Series|dict, specs to add to self.scores
            family_limit: int, the maximum number of designs from the same pMHC-TCR pair in the pool
        Returns:
            pd.Series: removed specs Series
        '''
        if isinstance(specs, dict):
            specs = pd.Series(specs)
        # add to pool and remove worst performing
        self.append_scores(specs, file_name)
        scores = self.get_scores()
        family:pd.DataFrame = scores.loc[scores["scaffold"]==specs["scaffold"]]
        if len(family.loc[family["in_pool"]]) > family_limit:
            out_name = family.loc[family["in_pool"]].sort_values("score", ascending=True).iloc[0].name
        elif scores["in_pool"].sum() < full_limit:
            return None
        else:
            out_name = scores.loc["in_pool"].sort_values("score", ascending=True).iloc[0].name

        scores.loc[out_name, "in_pool"] = False
        print(f"Removing worst design {out_name} and adding {specs}.")
        if delete_file:
            if not isinstance(file_name, Path):
                out_name = Path(out_name)
            if out_name.parent != self.out_dir:
                out_name = self.out_dir/out_name
            out_name.unlink()
        self.write_scores(scores)
        return out_name


    def mutate_cdrs(
        self,
        input_design:DesignData,
        cdrs:List[str]|str,
        n_mutations:List[int]|int,
        mutate_all=False,
        use_imgt_mapper:bool=False
        ):
        """
        Function to mutate custom positions.
        Args:
            input_design: DesignData, sequence to mutate
            cdrs: List[str], cdr ids for mutating. Alternative modes are "all" and "random", which are equivalent to mutate_all and mutating 1 cdr per chain respectively.
            n_muations: List[int]|int, number of mutations per cdr
            mutate_all: bool, if True, overrides cdrs and applies n_mutations to all cdrs (if n_mutations mismatches, the mean is applied to all cdrs)
        Returns:
            (DesignData, dict): the input design with mutated cdrs and a dict with the mutated cdrs
        """
        out_cdrs = {}
        pre = ["b","h"][int(self.ab)]
        alpha_cdrs = [k for k in self.imgt_mapper.keys() if not k.startswith(pre)]
        beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith(pre)]

        if isinstance(cdrs, str):
            if cdrs == "all":
                mutate_all = True
            elif cdrs == "random":
                alpha_cdrs = np.random.choice(alpha_cdrs, 1, replace=False).tolist()
                beta_cdrs = np.random.choice(beta_cdrs, 1, replace=False).tolist()
                mutate_all = True
            else:
                cdrs = [cdrs,]

        if mutate_all:
            cdrs = alpha_cdrs+beta_cdrs
            if not isinstance(n_mutations, int):
                if len(n_mutations)!=len(cdrs):
                    n_mutations = np.mean(n_mutations)

        if isinstance(n_mutations, int):
            n_mutations = np.full(len(cdrs),n_mutations)

        for cdr, n in zip(cdrs, n_mutations):
            chain_index = self.tcr_chain_index[int(cdr.startswith("h") or cdr.startswith("b"))]
            # create new insert
            o_insert = self.get_cdr_seq(input_design=input_design, cdr=cdr, decode_seq=False)
            positions = np.arange(len(o_insert))
            if n > len(positions):
                print(f"CDR {cdr} has length {len(positions)} and {n} mutations have been requested. Mutating the whole CDR!")
                n = len(positions)
            # select random positions in range
            m_positions = np.random.choice(positions, size=n, replace=False)
            insert_mask = (positions[:,None]==m_positions[None,:]).any(axis=1)
            # select new bases
            i_insert = np.random.randint(0,20, size=len(o_insert))
            i_insert = np.where(insert_mask, i_insert, o_insert)
            input_design,_ = self.insert_cdr(
                input_design=input_design,
                chain_index=chain_index,
                sequences={cdr:decode(i_insert, AF2_CODE)}
            )

            #self.number_anarci(
            #    input_design=input_design,
            #    chains=[chain_index,],
            #    trim=False,
            #)
            out_cdrs.update({cdr:decode(i_insert, AF2_CODE)})
        return input_design, out_cdrs


    def get_cdr_seq(
        self,
        input_design:DesignData,
        cdr:str,
        decode_seq:bool=True,
        use_imgt_mapper:bool=False
        ):
        target_mask = self.cdr_mask(
            input_design=input_design,
            cdr_ids=[cdr],
            use_imgt_mapper=use_imgt_mapper,
        ).astype(bool)
        if decode_seq:
            return decode(input_design["aa"][target_mask], AF2_CODE)
        return input_design["aa"][target_mask]

    def cdr_mask(self,
        input_design:DesignData,
        cdr_ids:Iterable[str],
        chain_index:int|None=None,
        use_imgt_mapper:bool=False
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
            chain_index = self.tcr_chain_index[int(cdr_ids[0].startswith("h") or cdr_ids[0].startswith("b"))]

        chain_mask = np.array(input_design["chain_index"]) == chain_index

        if use_imgt_mapper:
            positions = [self.imgt_mapper[k] for k in cdr_ids]

            residue_index = np.array(input_design["residue_index"])
            all_cdr_positions = np.concatenate([np.arange(s, e) for s, e in positions])
            mask = (residue_index[:, None] == all_cdr_positions[None, :]).any(axis=1)
        else:
            positions = [self.cdr_coords[k] for k in cdr_ids]
            mask = np.zeros(chain_mask.sum(), dtype=np.bool_)
            for s,e in positions:
                mask[s:e] = True
        chain_mask[chain_mask] = mask

        return chain_mask.astype(float)


    def insert_cdr(self,
        input_design:DesignData,
        sequences:Dict[str,str],
        chain_index:int|None=None,
        use_imgt_mapper:bool=False
        ):
        '''
        Insert CDR sequences in TCR chain, replacing existing IMGT-numbered CDR positions.
        Also adjusts cdr_coords.
        Args:
            input_design:DesignData, must contain TCR chain in "chain_index" key
            chain_index:int, chain_index value for TCR/Ab chain
            sequences:Dict[str,str], mapping cdr id (e.g. "acdr3") to amino acid sequence
        Returns:
            out:DesignData, design with CDRs replaced by the given sequences
            target_mask:np.ndarray, float mask: 1.0 for inserted CDR positions, 0.0 for framework
        '''
        if chain_index is None:
            chain_index = self.tcr_chain_index[int(cdr_ids[0].startswith("h") or cdr_ids[0].startswith("b"))]

        inserts = [DesignData.from_sequence(cdr).update(
            chain_index=jnp.full(len(cdr), chain_index)
            ) for cdr in sequences.values()]

        # Sort by IMGT start position
        if use_imgt_mapper:
            positions = [self.imgt_mapper[k] for k in sequences.keys()]
        else:
            positions = [self.cdr_coords[k] for k in sequences.keys()]
        
        sorter = np.argsort([s for s, _ in positions])
        positions = np.array(positions)[sorter]
        inserts = [inserts[p] for p in sorter]

        for i in inserts:
            print_dd(i, "Insert")

        cdr_mask = self.cdr_mask(
            input_design=input_design,
            cdr_ids = [k for k in sequences.keys()],
            chain_index=chain_index,
            use_imgt_mapper=use_imgt_mapper,
            ).astype(np.bool_)
        # Build framework (non-CDR) segment slices — corrected off-by-one
        fw_slices = []
        in_cdr = False
        fw_start = 0
        for i, current in enumerate(cdr_mask):
            if current and not in_cdr:
                fw_slices.append(slice(fw_start, i))
                in_cdr = True
            elif not current and in_cdr:
                fw_start = i
                in_cdr = False
        if not in_cdr:
            fw_slices.append(slice(fw_start, None))
        # Assemble output DesignData: framework[0], CDR1, framework[1], CDR2, ...
        parts = [input_design[fw_slices[0]]]
        for insert, fw_slice in zip(inserts, fw_slices[1:]):
            parts.append(insert)
            parts.append(input_design[fw_slice])
        for n,p in enumerate(parts):
            print_dd(p, n)
        out = DesignData.concatenate(parts, sep_batch=False, sep_chains=False)
        print_dd(out, "out insert_cdr_concat")

        # Attempt to Update residue index with IMGT numbering for the modified chain TODO: still needed?
        if use_imgt_mapper:
            out = self.number_anarci(out, chains=(chain_index,), trim=False)
        else:
            # correct self.cdr_coords
            offsets = {
                k:len(seq)-(self.cdr_coords[k][1]-self.cdr_coords[k][0])
                for k, seq in  sequences.items()}
            total_offset = sum([s for s in offsets.values()])
            # dict tracking dif per cdr
            for k, offset in offsets.items():
                self.cdr_coords[k] = (
                    self.cdr_coords[k][0],
                    self.cdr_coords[k][1]+offset)
                # affected downstream cdrs
                for i_k in self.imgt_mapper.keys():
                    if i_k.startswith(k[0]) and int(i_k[-1])> int(k[-1]):
                        self.cdr_coords[i_k] = (
                            self.cdr_coords[i_k][0]+offset,
                            self.cdr_coords[i_k][1]+offset)
        cdr_mask = self.cdr_mask(
            input_design=out,
            cdr_ids = [k for k in sequences.keys()],
            chain_index=chain_index,
            use_imgt_mapper=use_imgt_mapper,
            ).astype(np.bool_)
        assert len(out["aa"]) == len(cdr_mask), "CDR mask does not have the same length as construct!"
        assert len(input_design["aa"])==(len(out["aa"])-total_offset), f"Expected offset:{total_offset} vs actual offset {len(out['aa'])-len(input_design['aa'])} !"
        return out, cdr_mask


    def number_anarci(
        self,
        input_design:DesignData,
        chains:Tuple[int]|None=None,
        code:str=AF2_CODE,
        scheme:str="imgt",
        trim:bool=False,
        classify_all:bool = False
        )->DesignData:
        '''
        Update the residue index with standardized numbering. Trims recognized chains.
        Args:
            input_design: DesignData
            chains: int, chain_id from chain_index to number
            code: str, code for converting encoded amino-acids
            scheme: str, numbering scheme
            trim: bool, wether to trim non-numbered parts of recognized chains (i.e. constant regions)
        Returns:
            DesignData
        '''
        if chains is None:
            classify_all=True
            chains = np.unique(input_design["chain_index"])
        if isinstance(chains, int):
            chains = tuple(chains,)
        for chain in chains:
            chain_mask = np.array(input_design["chain_index"]) == chain
            # Pass only this chain's sequence to anarci
            chain_aa = np.array(input_design["aa"])[chain_mask]
            seq = decode(chain_aa, code=code)
            numbering = anarci.number(sequence=seq, scheme=scheme)
            if numbering[0]:
                print(f"Classified chain {chain} as {numbering[-1]}.")
                chain_type = numbering[-1]
                if chain_type == "A":
                    print(f"Setting chain {chain} to tcr chain 0!")
                    self.tcr_chain_index[0] = chain
                elif chain_type == "B":
                    print(f"Setting chain {chain} to tcr chain 1!")
                    self.tcr_chain_index[1] = chain
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
                input_design = input_design.update(residue_index=jnp.array(residue_index))

                # update cdr coords dict
                pre = [["a", "b"],
                    ["l", "h"],][self.ab][chain_type=="B"]
                chain_mask = np.array(input_design["chain_index"]) == chain
                for k,v in self.cdr_coords.items():
                    if k.startswith(pre):
                        print("Configuring cdr_coords for ", k)
                        index = np.arange(chain_mask.sum())
                        mask = (input_design["residue_index"][chain_mask][:, None]==np.arange(*self.imgt_mapper[k])[None,:]).any(axis=1)
                        self.cdr_coords[k] = (index[mask][0], index[mask][-1]+1)
                
            else:
                print(f"No numbering found for chain {chain}!")
                if chain in self.tcr_chain_index:
                    print("Sequence: ", seq)
        # check if chain indices correct
        if self.tcr_chain_index[0]==self.tcr_chain_index[1]:
            raise ValueError("TCR chains identical! Currently only 2 chain tcrs supported.")
        if (self.mhc_chain_index[:,None] == self.tcr_chain_index[None,:]).any() or classify_all:
            # fix mhc chain index to longest non-tcr chain
            chains = np.unique(input_design["chain_index"])
            # mask out tcr chains
            tcr_mask = ~(chains[:,None]==self.tcr_chain_index[None,:]).any(axis=1)
            chains = chains[tcr_mask]
            # get chain lengths
            chain_lengths =  (input_design["chain_index"][:,None] == chains[None,:]).sum(axis=0)
            # take the n longest chain indices, where n the number of non-tcr chains -1 (for the peptide chain) 
            self.mhc_chain_index = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-(len(chains)):-1]]
            )
            print(f"Classified chains {self.mhc_chain_index} as MHC chains")
        return input_design


def load_data(out_dir:str|Path=Path("./data/adapt/input_data"),
    url = "https://zenodo.org/records/17488258/files/",
    files = [
        "paired_human_cdr3s.tsv",
        "model_2_ptm_ft_binder_20230729.pkl",
        #"RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt",
        "zenodo_design_models.zip"
    ],
    ):
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


def print_dd(dd:DesignData, name:str="", keys:list=[]):
    try:
        print(f"\n---{name}---",
            *[f"{k}:{v}\n\t shape: {v.shape}" for k,v in dd.data.items() if k in keys],
            sep="\n"
            )
    except KeyError:
        print("Key not found")


def clean_chothia(file):
    '''Removes annotations, duplicate chains and HETATMs.'''
    if isinstance(file, str):
        file = Path(file)
    if file.name.endswith("clean.pdb"):
        print("File already clean")
        return file
    out_path = Path(file.with_suffix("").__str__()+"_clean.pdb")
    doubled_chains = []
    with open(out_path, "w") as wf:
        with open(file, "r") as rf:
            l = "init value"
            while l:
                l = rf.readline()
                if l.startswith("ATOM"):
                    wf.write(l[:26]+" "+l[27:])
                elif not l.startswith("HETATM"):
                    wf.write(l)
    return out_path

def download_structure(pdb_id: str, file_format: str = "pdb", output_dir: str = "."):
    """
    Download a structure file from RCSB PDB.
    
    file_format: 'pdb', 'cif' (mmCIF), or 'bcif' (BinaryCIF)
    """
    import requests
    base_urls = {
        "pdb": f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        "cif": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
        "bcif": f"https://models.rcsb.org/{pdb_id.lower()}.bcif",
    }
    url = base_urls[file_format]
    response = requests.get(url)
    response.raise_for_status()

    suffix = {"pdb": ".pdb", "cif": ".cif", "bcif": ".bcif"}[file_format]
    out_path = Path(output_dir) / f"{pdb_id.upper()}{suffix}"
    out_path.write_bytes(response.content)
    return out_path