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
- profile to see pmpnn vs. af
- boltz msa or template .pdb
(- implement current design attribute)
- design cdrs with salad/boltz
- add structure templates from canonical tcrs
    : get mhc 1 and mhc 2 +tcr structures from pdb
    : align on mhc
    : cluster by tcr rmsd 
    : sample from each cluster for representative binding mode set

'''
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import AFInput, AFResult, make_predict, make_af2
from flexcraft.structure.metrics import RMSD
from flexcraft.structure.boltz import Joltz2, JoltzResult, JoltzPrediction
from flexcraft.utils import Keygen, parse_options, data_from_protein
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.aa_codes import AF2_CODE, decode
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *

from flexcraft.pipelines.tcr.utils import print_dd
from filelock import FileLock

from colabdesign.af.alphafold.model import utils as af_utils
from colabdesign.af.alphafold.model.data import get_model_haiku_params
from colabdesign.af.alphafold.model.config import model_config

from jax._src.typing import Array
import anarci

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Callable
from datetime import datetime
import tempfile



from jax import jit
import jax.numpy as jnp
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
        key:Array|int,
        boltz_config:dict={},
        af2_config:dict={},
        pmpnn_config:dict={},
        mhc_chain_index:int|tuple[int]=0,
        tcr_chain_index:int|tuple[int]=(2,3),
        name="AdaptTrial",
        out_dir:None|str|Path=None,
        trim:bool=True,
        chain_cache_len:int=650, # how long to pad for af
        redesign_all_cdrs=False,
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
            af2_num_recycle:int=0, number of AF2 recycle steps.
            pmpnn_parameter_path:str|None=None, path to pMPNN parameter file (.pkl).
            pmpnn_hparams:dict={}, pMPNN hyperparameters.
            ab:bool = False, wether to work on Antibodies of TCRs
            mhc_chain_index:int|Tuple[int]=0, index of MHC chains in DesignData input. For antibodies, this denotes the antigen chain index.
                Is generally infered automatically
            tcr_chain_index:int|Tuple[int]=(2,3), index
            name="AdaptTrial",
            out_dir:None|str|Path=None,
            trim:bool=False,
            chain_cache_len:int=400, # how long to pad for af

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
        self.redesign_all_cdrs = redesign_all_cdrs
        self.imgt_mapper = {
            "acdr1":(27,38),
            "acdr2":(56, 65),
            "acdr3":(105,117),
            "bcdr1":(27,38),
            "bcdr2":(56,65),
            "bcdr3":(105,117),
            }
        self.trim = trim
        self.columns=["score","scaffold",
            "time",
            "in_pool",
            "tcr_chain_index",
            "mhc_chain_index",
            *[
            k for k in self.imgt_mapper.keys()
            ],
            *[
            f"{k}_coords" for k in self.imgt_mapper.keys()
            ],
            ]
        self.columns_default = {"in_pool":True}
        self.scores = self.out_dir/"scores.csv"
        self.lock = FileLock(self.scores.with_suffix(".lock"))
        if not (self.scores).exists():
            lock.aquire()
            pd.DataFrame(columns=self.columns).to_csv(self.scores, header=True)
            lock.release()
        else:
            self.columns = list(pd.read_csv(self.scores, header=0, index_col=0).columns)
        self.cdr_coords = self.imgt_mapper.copy()

        self.chain_cache_len = chain_cache_len
        # chain indices
        # pack in array for comparative operations
        if isinstance(mhc_chain_index, int):
            mhc_chain_index = (mhc_chain_index,)
        self.mhc_chain_index = np.array(mhc_chain_index)
        if isinstance(tcr_chain_index, int):
            tcr_chain_index = (tcr_chain_index,)
        self.tcr_chain_index = np.array(tcr_chain_index)
        
        self.key = key
        if isinstance(self.key, int):
            self.key = Keygen(self.key)
        
        self.setup_pmpnn(
            **pmpnn_config
        )
        
        # AlphaFold
        self.setup_af2(
            **af2_config
        )

        # Boltz2
        self.setup_boltz(
            **boltz_config
        )

        self.rmsd = RMSD()
    
    def setup_boltz(self,
        **boltz_config
        ):

        config = {
            "docking":False,
            "redocking":False,
            "parameter_path":Path("./params/boltz"),
            "model_name":"boltz2_conf",
            "num_recycle":2,
            "num_samples":1,
            "num_sampling_steps":25,
            "deterministic":False,
            "predictor":None}
        config.update(boltz_config)

        self.boltz_docking = config["docking"]
        self.boltz_redocking = config["redocking"]
        if self.boltz_docking or self.boltz_redocking:
            self.boltz_parameter_path = config["parameter_path"]
            self.boltz_model_name = config["model_name"]
            self.boltz_num_recycle = config["num_recycle"]
            self.boltz_num_samples = config["num_samples"]
            self.boltz_num_sampling_steps = config["num_sampling_steps"]
            self.boltz_deterministic = config["deterministic"]
            if config["predictor"] is None:
                self.boltz_model = Joltz2(model=self.boltz_model_name+".ckpt", cache=self.boltz_parameter_path)
                self.boltz_predictor = self.boltz_model.predictor_adhoc(
                    num_recycle=self.boltz_num_recycle,
                    num_samples=self.boltz_num_samples,
                    num_sampling_steps=self.boltz_num_sampling_steps,
                    deterministic=self.boltz_deterministic
                )
            else:
                self.boltz_predictor = config["predictor"]
        else:
            self.boltz_parameter_path = None
            self.boltz_model_name = None

    def get_scores(self):
        with self.lock:
            out = pd.read_csv(self.scores, header=0, index_col=0)
        return out
    def write_scores(self, scores:pd.DataFrame):
        with self.lock:
            scores[self.columns].to_csv(self.scores, header=True)
    def append_scores(self, row:pd.Series, name:str|None=None):
        with self.lock:
            for c in self.columns:
                if c not in row.index:
                    row[c] = self.columns_default.get(c, None)
            row.name = name
            row.to_frame().T[self.columns].to_csv(self.scores, header=False, mode="a")

    def trim_design(
        self,
        input_design:DesignData
        ):
        '''Trim the length of the mhc chains to fit the construct to cache length.'''
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
            if (chain_mask.sum()-trim) < 50:
                print("WARNING! trimmed mhc chain to less than 50 AAs! Manual trimming may be necessary.")
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
        **pmpnn_config
        ):
        
        config = {
        "pmpnn_model":None,
        "pmpnn_parameter_path":None,
        "pmpnn_hparams":{},
        "pmpnn_n_per_target":3}
        config.update(pmpnn_config)

        self.pmpnn_hparams = {
            "temperature":0.1,
            "model_name":'v_48_020',
            "n_edges":48,
            "training_noise":0.2,#Å,
            "center_logits":False,
        }
        self.pmpnn = config["pmpnn_model"]
        self.pmpnn_parameter_path = config["pmpnn_parameter_path"]
        self.pmpnn_hparams.update(config["pmpnn_hparams"])
        self.pmpnn_n_per_target = config["pmpnn_n_per_target"]

        if self.pmpnn is None:
            self.pmpnn = jit(make_pmpnn(self.pmpnn_parameter_path, eps=0.05))

        self.pmpnn_transform = lambda center: transform_logits((
            toggle_transform(
                center_logits(center), use=self.pmpnn_hparams["center_logits"]),
            scale_by_temperature(self.pmpnn_hparams["temperature"]),
            forbid("C", aas.PMPNN_CODE),
            norm_logits
        ))


    def setup_af2(
        self,
        **af2_config
        ):
        
        
        config = {
        "af2_model":None,
        "af2_params":None,
        "af2_model_name":None,
        "af2_parameter_path":None,
        "af2_multimer":True,
        "af2_num_recycle":0,
        }

        config.update(af2_config)
        self.af2_model = config["af2_model"]
        self.af2_params = config["af2_params"]
        self.af2_model_name = config["af2_model_name"]
        self.af2_parameter_path = config["af2_parameter_path"]
        self.af2_multimer = config["af2_multimer"]
        self.af2_num_recycle = config["af2_num_recycle"]

        if self.af2_model is None or self.af2_params is None:
            assert (not self.af2_model_name is None) or (not self.af2_parameter_path is None), ValueError("Specify either model name or parameter path!")
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
                        data_dir=self.af2_parameter_path.__str__(), fuse=True)


            if self.af2_multimer is None:
                self.af2_multimer = "multimer" in self.af2_model_name
            self.af2_config = model_config(self.af2_model_name)
            self.af2_config.model.global_config.use_dgram = False
            self.af2_model = jit(make_predict(
                make_af2(self.af2_config, use_multimer=self.af2_multimer),
                num_recycle=self.af2_num_recycle))
        
        return self.af2_model, self.af2_params


    def af_infer(self, af_input:AFInput) -> AFResult:
        '''
        Wrapper for af model, handles non-multimer models.
        '''
        if not self.af2_multimer:
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
        template:DesignData|None=None,
        )->List[DesignData]|Tuple[List[DesignData], float]:
        '''
        Predict protein structure using Boltz-2.
        Returns:
            structure: List[DesignData], predicted structure and sequence of the input_design for each sample in self.boltz_num_samples
            score: float, output score if evaluate
            template: DesignData, template with same corresponding chain_index
        '''
        chain_index = input_design["chain_index"]
        input_design, pad_length = self.pad_design(input_design=input_design)
        chain_masks = (input_design["chain_index"][None,:] == np.unique(input_design["chain_index"])[:, None])
        if not template is None:
            template_dir = tempfile.gettempdir()
            template_files = []
            for chain in np.unique(input_design["chain_index"]):

                template[template["chain_index"]==chain].to_pdb(f"{template_dir}/template_{chain}.pdb")
                template_files.append(f"{template_dir}/template_{chain}.pdb")
        else:
            template_files=np.full(len(chain_masks), None)

        boltz_prediction = self.boltz_predictor(
            self.key(), 
            *[
                {
                    "sequence": decode(input_design["aa"][c], AF2_CODE),
                    "kind":"protein",
                    "use_msa": template is None,
                    "template_file":template_path
                }
                for c, template_path in zip(chain_masks, template_files)
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
                    pad_length,).update(chain_index=chain_index)
                for n in range(self.boltz_num_samples)
            ]
        else:
            out = [self.rm_pad(boltz_result.to_data(), pad_length).update(chain_index=chain_index),]
        if evaluate:
            if len(out)>1:
                print("WARNING: evaluate_step currently only accepts one sample!")
            score = self.evaluate_step(result=out[0], input_design=input_design, is_target=is_target)
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
            alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
            beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]

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

        # predict on all aas to calculate center
        logit_center = self.pmpnn(self.key(), input_design)["logits"].mean(axis=0)
        pmpnn_sampler = sample(self.pmpnn, logit_transform=self.pmpnn_transform(logit_center))
        results = []
        for n in range(self.pmpnn_n_per_target):

            pmpnn_result, _ = pmpnn_sampler(self.key(), input_design)
            results.append(self.rm_pad(input_design.update(
                aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE)), pad_length))
        return results

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
        Calculate the RMSD for CDR3 chains after alignment of MHC chain. For antibodies, the antigen is used for alignment.
        Args:
            result: AFResult|JoltzResult, result of af prediction. Has inputs and result attribute.
            input_design: DesignData, original structure before prediction
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
        design:DesignData,
        scaffold_name:str,
        cdrs:List[str]=["acdr3", "bcdr3"],
        ):
        '''Run a design step for recombination of TCRs with CDR3s'''
        print(
            f"\n!---Design Trial for {scaffold_name}---!\n",
        )


        # filter ids
        for cdr in cdrs:
            if cdr not in self.imgt_mapper.keys():
                raise KeyError(f"Invalid cdr id {cdr}! Choose one of {[f'{k}, ' for k in self.imgt_mapper.keys()]}.")

        # process input
        print_dd(design, "Input")

        # mask cdrs and 
        pre = "b"
        if self.redesign_all_cdrs:
            alpha_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("a")]
            beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("b")]
        else:
            alpha_cdrs = [k for k in cdrs if k.startswith("a")]
            beta_cdrs = [k for k in cdrs if k.startswith("b")]

        target_mask = (
            self.cdr_mask(design, chain_index=self.tcr_chain_index[0], cdr_ids=alpha_cdrs)
            + self.cdr_mask(design, chain_index=self.tcr_chain_index[1], cdr_ids=beta_cdrs)
        ) > 0

        # docking step (structure prediction without evaluation)
        if self.boltz_docking:
            boltz_designs = self.boltz_docking_step(input_design=design, template_path=None)
            design = boltz_designs.pop()
        else:
            design = self.af_docking_step(input_design=design, is_target=target_mask)

        print_dd(design, "Docked")
        # redesign step: redesign the CDR positions
        
        designs = self.design_step(input_design=design, target_mask=target_mask)
        structures = []
        scores = []
        print_dd(designs[0], "Redesigned")
        for n,design in enumerate(designs):

            # redocking + evaluation step
            if self.boltz_redocking:
                design, score = self.boltz_docking_step(
                    input_design=design,
                    evaluate=True,
                    is_target=target_mask,
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
                )
            scores.append(score)
            structures.append(design)
        design = structures[np.argmax(scores)]
        score = max(scores)
        print_dd(design, "Redocked")
        # save design as unique file name
        file_name = f"{scaffold_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{n}.pdb"
        design.save_pdb(self.out_dir/file_name)
        print(f"Saving design with score {score} to {file_name}!")
        # Use a named Series so missing CDR1/CDR2 columns get NaN automatically
        row = {"score": score, "scaffold": scaffold_name, "time":datetime.now().strftime("%Y-%d-%b_%H:%M:%S"),
            "tcr_chain_index":(*[int(i) for i in self.tcr_chain_index],),"mhc_chain_index":(*[int(i) for i in self.mhc_chain_index],),
            **{cdr:self.get_cdr_seq(design, cdr) for cdr in self.imgt_mapper.keys()},
            **{f"{k}_coords":v for k, v in self.cdr_coords.items()},}
        print(row)
        self.append_scores(pd.Series(row), file_name)
        return score


    def get_design(
        self,
        index:str,
        family:str|None=None
        )->Tuple[DesignData, str, str]:
        '''
        Query the scores table for a design and adjust the cdr coords.
        Returns:
            (DesignData, str, str); design, path to the pdb, family
        '''
        with self.lock:
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
        # load structure descriptors
        to_tuple = lambda s: tuple([int(n) for n in s.strip("()").split(",") if n])
        self.cdr_coords = {k:to_tuple(row[f"{k}_coords"]) if row[f"{k}_coords"] else self.imgt_mapper[k] for k in self.imgt_mapper.keys()}
        self.tcr_chain_index = np.array(to_tuple(row["tcr_chain_index"]))
        self.mhc_chain_index = np.array(to_tuple(row["mhc_chain_index"]))
        # loading the file collapses the chain indices
        design = PDBFile(path=self.out_dir/row.name).to_data()
        design = self.convert_chains(design)
        print(f"Loaded Design {row.name} with cdr_coords {self.cdr_coords}!")
        print_dd(design, "Loaded Design!")
        return design, row.name, row["scaffold"]

    def convert_chains(self, input_design:DesignData):
        d = {}
        for x,y in zip(np.sort(np.unique(input_design["chain_index"])), range(len(np.unique(input_design["chain_index"])))):
            d[int(x)]=int(y)
        print(d)
        design = input_design.update(chain_index=jnp.array([d[int(x)] for x in input_design["chain_index"]]))
        self.tcr_chain_index = np.array([d[int(x)] for x in self.tcr_chain_index])
        self.mhc_chain_index = np.array([d[int(x)] for x in self.mhc_chain_index])
        return design

    def refine_trial(
        self,
        scaffold:DesignData,
        scaffold_name:str,
        cdrs:List[str]=["acdr3", "bcdr3"],
        ):
        
        # filter ids
        for cdr in cdrs:
            if cdr not in self.imgt_mapper.keys():
                raise KeyError(f"Invalid cdr id {cdr}! Choose one of {[f'{k}, ' for k in self.imgt_mapper.keys()]}.")

        # fetch list of candidates
        # process input

        print(
            f"\n!---Refine Trial for {scaffold_name}---!\n",
        )
        print(f"CDRs: {cdrs}")
        print_dd(scaffold, "Input")
        # mutate 2 cdr positions
        scaffold, mutated_cdrs = self.mutate_cdrs(
            input_design=scaffold,
            cdrs=cdrs,
            n_mutations=2,
            )
        print_dd(scaffold, "Mutated")

        if self.redesign_all_cdrs:
            alpha_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("a")]
            beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("b")]        
        else:
            alpha_cdrs = [cdr for cdr in cdrs if cdr.startswith("a")]
            beta_cdrs = [cdr for cdr in cdrs if cdr.startswith("b")]
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

        designs = self.design_step(input_design=scaffold, target_mask=target_mask)
        print_dd(designs[0], "Redesigned")
        structures = []
        scores = []
        print_dd(designs[0], "Redesigned")
        for n,design in enumerate(designs):

            # redocking + evaluation step
            if self.boltz_redocking:
                design, score = self.boltz_docking_step(
                    input_design=design,
                    evaluate=True,
                    is_target=target_mask,
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
                )
            scores.append(score)
            structures.append(design)
        design = structures[np.argmax(scores)]
        score = max(scores)
        print_dd(design, "Redocked")
        # save design as unique file name
        file_name = f"{scaffold_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{n}.pdb"
        design.save_pdb(self.out_dir/file_name)
        print(f"Saving design with score {score} to {file_name}!")
        # Use a named Series so missing CDR1/CDR2 columns get NaN automatically
        row = {"score": score, "scaffold": scaffold_name, "time":datetime.now().strftime("%Y-%d-%b_%H:%M:%S"),
            "tcr_chain_index":(*[int(i) for i in self.tcr_chain_index],),"mhc_chain_index":(*[int(i) for i in self.mhc_chain_index],),
            **{cdr:self.get_cdr_seq(scaffold, cdr) for cdr in self.imgt_mapper.keys()},
            **{f"{k}_coords":v for k, v in self.cdr_coords.items()},}

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
        with self.lock:
            self.append_scores(specs, file_name)
            scores = self.get_scores()
            family:pd.DataFrame = scores.loc[scores["scaffold"]==specs["scaffold"]]
            if len(family.loc[family["in_pool"]]) > family_limit:
                out_name = family.loc[family["in_pool"]].sort_values("score", ascending=False).iloc[0].name
            elif scores["in_pool"].sum() < full_limit:
                return None
            else:
                out_name = scores.loc["in_pool"].sort_values("score", ascending=False).iloc[0].name

            print(scores)
            scores.loc[out_name, "in_pool"] = False
            print(scores)
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
        alpha_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("a")]
        beta_cdrs = [k for k in self.imgt_mapper.keys() if k.startswith("b")]

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
        out = DesignData.concatenate(parts, sep_batch=False, sep_chains=False)

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
                    int(self.cdr_coords[k][0]),
                    int(self.cdr_coords[k][1]+offset))
                # affected downstream cdrs
                for i_k in self.imgt_mapper.keys():
                    if i_k.startswith(k[0]) and int(i_k[-1])> int(k[-1]):
                        self.cdr_coords[i_k] = (
                            int(self.cdr_coords[i_k][0]+offset),
                            int(self.cdr_coords[i_k][1]+offset))
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
                if chain_type in ["A", "L"]:
                    print(f"Setting chain {chain} to {chain_type}!")
                    self.tcr_chain_index[0] = chain
                elif chain_type in ["B", "H"]:
                    print(f"Setting chain {chain} to {chain_type}!")
                    self.tcr_chain_index[1] = chain
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
                input_design = input_design.update(residue_index=jnp.array(residue_index))

                # update cdr coords dict
                pre = ["a","b",][chain_type=="B"or chain_type=="H"]
                chain_mask = np.array(input_design["chain_index"]) == chain
                for k in self.imgt_mapper.keys():
                    if k.startswith(pre):
                        print("Configuring cdr_coords for ", k)
                        index = np.arange(chain_mask.sum())
                        mask = (input_design["residue_index"][chain_mask][:, None]==np.arange(*self.imgt_mapper[k])[None,:]).any(axis=1)
                        self.cdr_coords[k] = (int(index[mask][0]), int(index[mask][-1]+1))
                
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
            # take all non-tcr-chains except for smalles (peptide, hopefully)
            self.mhc_chain_index = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-(len(chains)):-1]]
            )
            print(f"Classified chains {self.mhc_chain_index} as MHC/antigen chains")
        return input_design

    def _convert_input_peptide(self, peptide:DesignData|Path|str|None)->DesignData|None:
        if isinstance(peptide, (DesignData|None)):
            return peptide

        elif isinstance(peptide, Path):
            if not peptide.suffix in [".pdb", ".cif"]:
                peptide = peptide.with_suffix(".pdb")
            if peptide.parent != self.in_dir:
                peptide = self.in_dir/peptide
            return PDBFile(path=peptide).to_data()

        elif isinstance(peptide, str):
            return DesignData.from_sequence(peptide)

        else:
            print(f"Unknown format of peptide {peptide}!")
            return None

    def make_scaffold(
        self,
        receptor:DesignData|Path|str,
        antigen:DesignData|Path|str,
        presenter:DesignData|Path|str|None=None,
        cdrs:Dict[str,str]|Path|None=None,
        replace_antigen:bool=False,
        ):

        receptor_name = Path(receptor).stem.split("_")[0][:10] if isinstance(receptor, (str, Path)) else ""
        pmhc_name = Path(presenter).stem[:10] if isinstance(presenter, (str, Path)) else ""
        antigen_name = Path(antigen).stem[:10] if isinstance(antigen, (str, Path)) else ""
        scaffold_name = "+".join([receptor_name, pmhc_name, antigen_name])
        
        # load all constructs
        receptor = self._convert_input_peptide(receptor)

        # remove non- (mhc or tcr/ab) chains
        receptor = self.number_anarci(receptor, trim=False)
        receptor = receptor[(
            receptor["chain_index"][:, None]==self.tcr_chain_index[None,:]).any(axis=1)]
        print_dd(receptor,"Receptor(peptide removed)")

        antigen = self._convert_input_peptide(antigen)
        presenter = self._convert_input_peptide(presenter)
        if not presenter is None:
            if len(np.unique(presenter["chain_index"]))>2:
                presenter = self.number_anarci(presenter, trim=False)
                presenter = presenter[(
                    presenter["chain_index"][:, None]==self.mhc_chain_index[None,:]
                ).any(axis=1)]

        scaffold = DesignData.concatenate(
            [d for d in [receptor, antigen, presenter] if not d is None],
            sep_chains=True, sep_batch=False
        )
        # index with imgt numbering
        scaffold = self.number_anarci(input_design=scaffold, trim=self.trim)
        scaffold = self.convert_chains(scaffold)

        if not cdrs is None:
            # insert the cdrs
            sequences = ({},{})
            # load cdr
            # sort by tcr/ab chain
            for k,v in cdrs.items():
                if k in self.imgt_mapper:
                    sequences[k.startswith("h") or k.startswith("b")].update({k:v})

            print("CDRs: ", sequences)

            scaffold, _ = self.insert_cdr(
                input_design=scaffold,
                chain_index=self.tcr_chain_index[0],
                sequences=sequences[0]
                )
            print_dd(scaffold, "Inserted1")
            scaffold, _ = self.insert_cdr(
                input_design=scaffold,
                chain_index=self.tcr_chain_index[1],
                sequences=sequences[1]
                )
        
        if self.trim:
            scaffold =  self.trim_design(scaffold)
        return scaffold, scaffold_name
