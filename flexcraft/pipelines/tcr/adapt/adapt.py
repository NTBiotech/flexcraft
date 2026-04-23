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
- implement length conservation with masking
- implement model/params as input to avoid recompilation
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
import anarci

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable
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
        af2_model_name:str,
        key,
        pmpnn_parameter_path:str,
        boltz_docking:bool=False,
        boltz_redocking:bool=False,
        boltz_parameter_path:Path|str=Path("./params/boltz"),
        boltz_model_name:str="boltz2_conf",
        boltz_num_recycle = 2,
        boltz_num_samples = 1,
        boltz_num_sampling_steps = 25,
        boltz_deterministic = False,
        af2_parameter_path:str|Path|None=None,
        af2_multimer:bool|None=None,
        af_num_recycle:int=0,
        pmpnn_hparams:dict={},
        ab:bool = False,
        mhc_chain_index:int|Tuple[int]=0,
        tcr_chain_index:int|Tuple[int]=(2,3),
        name="AdaptTrial",
        out_dir:None|str|Path=None,
        trim:bool=False,
    ):
        # directory organization
        if not isinstance(op_dir, Path):
            op_dir = Path(op_dir)
        self.op_dir = op_dir
        self.in_dir = self.op_dir/"input_data"
        
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
        columns=["score","scaffold",
            *[
            k for k in self.imgt_mapper.keys()
            ]]
        self.scores = pd.DataFrame(columns=columns)

        # chain indices
        # pack in array for comparative operations
        if isinstance(mhc_chain_index, int):
            mhc_chain_index = (mhc_chain_index,)
        self.mhc_chain_index = np.array(mhc_chain_index)
        if isinstance(tcr_chain_index, int):
            tcr_chain_index = (tcr_chain_index,)
        self.tcr_chain_index = np.array(tcr_chain_index)
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
        self.center_logits = self.pmpnn_hparams["center_logits"]
        self.pmpnn_transform = lambda center, do_center, T: transform_logits((
            toggle_transform(
                center_logits(center), use=do_center),
            scale_by_temperature(self.pmpnn_hparams["temperature"]),
            forbid("C", aas.PMPNN_CODE),
            norm_logits
        ))

        self.pmpnn_sampler = sample(self.pmpnn, logit_transform=center_logits())
        
        # AlphaFold
        self.af2_model_name = af2_model_name
        self.key = key
        if af2_parameter_path is None:
            af2_parameter_path = (self.in_dir/self.af2_model_name).with_suffix(".pkl")
        self.af2_parameter_path = af2_parameter_path
        if isinstance(self.af2_parameter_path, str):
            self.af2_parameter_path = Path(self.af2_parameter_path)
        if self.af2_parameter_path.is_file():
            # if params in pickle
            if self.af2_parameter_path.suffix != ".pkl":
                raise DeprecationWarning(f"AF_parameter filetype of {self.af2_parameter_path} not supported!")
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
            self.use_multimer = "multimer" in af2_model_name
        else:
            self.use_multimer = af2_multimer
        self.af2_config = model_config(self.af2_model_name)
        self.af2_config.model.global_config.use_dgram = False
        self.af2_model = jax.jit(make_predict(
            make_af2(self.af2_config, use_multimer=self.use_multimer),
            num_recycle=self.af_num_recycle))

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
        save_structure:bool|Path|str=True,
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

        input_design = input_design.update(residue_index = np.arange(
            len(input_design["aa"])
        ))
        af_input = AFInput.from_data(input_design)

        if off_target_template:
            if is_target is None:
                print("No is_target input. Not adding template!")
            else:
                af_input = af_input.add_template(input_design, where=~is_target)
        
        if not templates is None:
            for t in templates:
                af_input = af_input.add_template(t)

        af_result = self.af_infer(af_input=af_input)

        if evaluate:
            score = self.evaluate_step(result=af_result, input_design=input_design, is_target=is_target)
            if save_structure:
                if isinstance(save_structure, bool):
                    save_structure = "evaluated_structure.pdb"
                af_result.to_data().save_pdb(self.out_dir/save_structure)
            return af_result.to_data(), score
        if save_structure:
            if isinstance(save_structure, bool):
                save_structure = "docked_structure.pdb"
            af_result.to_data().save_pdb(self.out_dir/save_structure)
        return af_result.to_data()

    def boltz_docking_step(
        self,
        input_design:DesignData,
        save_structure:bool=True,
        evaluate:bool=False,
        is_target:np.ndarray|None=None,
        )->List[DesignData]|Tuple[List[DesignData], float]:
        '''
        Predict protein structure using Boltz-2.
        Returns:
            structure: List[DesignData], predicted structure and sequence of the input_design for each sample in self.boltz_num_samples
            score: float, output score if evaluate
        '''
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
                DesignData(data=dict(
                atom_positions=atom24[n],
                atom_mask=mask24[n],
                aa=boltz_result.restype,
                mask=mask24[n].any(axis=1),
                residue_index=boltz_result.residue_index,
                chain_index=boltz_result.chain_index,
                batch_index=jnp.zeros_like(boltz_result.residue_index),
                plddt=boltz_result.plddt[n] if len(boltz_result.plddt.shape) == 2 else boltz_result.plddt,)
                ).untie()
                for n in range(self.boltz_num_samples)
            ]
        else:
            out = [boltz_result.to_data(),]
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

        if self.center_logits:
            # calculate logit center over non-masked positions
            logit_center = self.pmpnn(self.key(), input_design)["logits"].mean(axis=0)

        pmpnn_result, _ = self.pmpnn_sampler(self.key(), input_design)
        pmpnn_result = input_design.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        return pmpnn_result

    def evaluate_step(
        self,
        result:AFResult|JoltzResult,
        input_design:DesignData,
        is_target:np.ndarray,
        ) -> float:
        '''Calculate score for protein design.'''
        cdr3_rmsd = self.cdr3_rmsd(
            result=result, input_design=input_design, is_target=is_target)
        ipae = self.ipae(result=result)
        return 2*ipae+0.5*cdr3_rmsd
    
    def ipae(
        self,
        result:AFResult|JoltzResult,
    )-> float:
        '''
        Compute the mean PAE(predicted aligned error) for all (pMHC, TCR)x(TCR, pMHC) residue pairs.
        '''
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

    def _design_trial(
        self,
        scaffold:str|Path|DesignData,
        cdr3s:str|Path|Tuple[str],
        pMHC:str|Path|DesignData|None=None,
        redesign_all_cdrs:bool=False,
    ):
        '''Run a design step for recombination of TCRs with CDR3s'''
        # Save original identifiers for output naming
        scaffold_name = Path(scaffold).stem if isinstance(scaffold, (str, Path)) else "scaffold"
        pmhc_name = Path(pMHC).stem if isinstance(pMHC, (str, Path)) else "pmhc"
        pmhc_is_scaffold = pMHC is None or pMHC == scaffold

        # process input
        if not isinstance(scaffold, DesignData):
            # load tcr
            if isinstance(scaffold, str):
                scaffold = Path(scaffold)
            if scaffold.parent == self.in_dir:
                scaffold = PDBFile(path=self.in_dir/scaffold.name).to_data()
            else:
                scaffold = PDBFile(path=scaffold).to_data()
        # imgt numbering
        scaffold = self.number_anarci(scaffold, trim=self.trim)

        if pmhc_is_scaffold:
            # check if all chains are present in the single structure
            assert len(np.unique(scaffold["chain_index"])) == (len(self.mhc_chain_index) + len(self.tcr_chain_index) + 1), \
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
            # concatenate with split_chains
            design = DesignData.concatenate([pmhc, scaffold], sep_chains=True)

        # load cdr3s
        if not isinstance(cdr3s, tuple):
            if isinstance(cdr3s, str):
                cdr3s = Path(cdr3s)
            if cdr3s.parent == self.in_dir:
                cdr3s = cdr3s.name
            sep = "\t"
            if "csv" in cdr3s.suffix:
                sep = ","
            df = pd.read_csv(self.in_dir/cdr3s, delimiter=sep)
            cdr3a, cdr3b = df.sample(n=1).iloc[0].values
        else:
            cdr3a, cdr3b = cdr3s
        # insert cdr3s
        sequences = ({"acdr3": cdr3a}, {"bcdr3": cdr3b})
        print("CDR3s: ", sequences)
        print_dd(design, "Loaded")
        if self.ab:
            sequences = ({"lcdr3": cdr3a}, {"hcdr3": cdr3b})
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[0],
            sequences=sequences[0]
            )
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[1],
            sequences=sequences[1]
            )
        print_dd(design, "Inserted")
        # Build target mask from CDR IMGT positions on the final (renumbered) design.
        # The two insertions may change the sequence length by different amounts, so
        # masks from individual insert_cdr calls cannot be summed directly.
        if redesign_all_cdrs:
            if not self.ab:
                alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]
            else:
                alpha_cdrs = [f"lcdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"hcdr{n}" for n in range(1, 4)]
        else:
            if not self.ab:
                alpha_cdrs = [f"acdr3",]
                beta_cdrs = [f"bcdr3",]
            else:
                alpha_cdrs = [f"lcdr3",]
                beta_cdrs = [f"hcdr3",]
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
        file_name = f"{scaffold_name}_{pmhc_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{pmhc_name}_{n}.pdb"
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
        if not self.ab:
            row = {"out_file": file_name, "score": score, "scaffold": scaffold_name, "pMHC": pmhc_name,
                   "acdr3": cdr3a, "bcdr3": cdr3b}
        else:
            row = {"out_file": file_name, "score": score, "scaffold": scaffold_name, "pMHC": pmhc_name,
                   "lcdr3": cdr3a, "hcdr3": cdr3b}
        self.scores.loc[len(self.scores)] = pd.Series(row)
        return score


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
            assert len(np.unique(scaffold["chain_index"])) == (len(self.mhc_chain_index) + len(self.tcr_chain_index) + 1), \
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
        
        # imgt numbering
        design = self.number_anarci(design, trim=self.trim)

        sequences = ({},{})
        # load cdr
        for k,v in cdrs.items():
            if not k in self.imgt_mapper:
                out_key = cdrs.pop(k)
                print(f"{out_key} not a known cdr. Proceeding without!")
            sequences[k.startswith("h") or k.startswith(b)].update({k:v})
            
        print("CDRs: ", sequences)
        print_dd(design, "Loaded")
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[0],
            sequences=sequences[0]
            )
        design, _ = self.insert_cdr(
            input_design=design,
            chain_index=self.tcr_chain_index[1],
            sequences=sequences[1]
            )
        print_dd(design, "Inserted")

        if redesign_all_cdrs:
            if not self.ab:
                alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]
            else:
                alpha_cdrs = [f"lcdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"hcdr{n}" for n in range(1, 4)]
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
        if not self.ab:
            row = {"score": score, "scaffold": scaffold_name,
                   **{cdr:self.get_cdr_seq(design, cdr) for cdr in self.imgt.keys()}}
        else:
            row = {"score": score, "scaffold": scaffold_name,
                   **{cdr:self.get_cdr_seq(design, cdr) for cdr in self.imgt.keys()}}
        self.scores.loc[file_name] = pd.Series(row)
        return score

    def refine_trial(
        self,
        scaffold:DesignData|str|Path,
        redesign_all_cdrs:bool=False,
        ):
        # fetch list of candidates
        # process input
        scaffold_name = Path(scaffold).stem.split("_")[0] if isinstance(scaffold, (str, Path)) else "scaffold"
        if not isinstance(scaffold, DesignData):
            # load tcr
            if isinstance(scaffold, str):
                scaffold = Path(scaffold)
            if scaffold.parent==self.in_dir:
                scaffold = scaffold.name
            scaffold = PDBFile(path=self.in_dir/scaffold).to_data()
        # imgt numbering
        scaffold = self.number_anarci(scaffold)
        # mutate 2 cdr positions
        scaffold, mutated_cdrs = self.mutate_cdrs(
            input_design=scaffold,
            cdrs="random",
            n_mutations=2,
            )

        if redesign_all_cdrs:
            if not self.ab:
                alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]
            else:
                alpha_cdrs = [f"lcdr{n}" for n in range(1, 4)]
                beta_cdrs = [f"hcdr{n}" for n in range(1, 4)]
        else:
            if not self.ab:
                alpha_cdrs = [f"acdr3",]
                beta_cdrs = [f"bcdr3",]
            else:
                alpha_cdrs = [f"lcdr3",]
                beta_cdrs = [f"hcdr3",]
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
        if not self.ab:
            row = {"score": score, "scaffold": scaffold_name,
                   **{cdr:self.get_cdr_seq(scaffold, cdr) for cdr in self.imgt_mapper.keys()}}
        else:
            row = {"score": score, "scaffold": scaffold_name,
                   **{cdr:self.get_cdr_seq(scaffold, cdr) for cdr in self.imgt_mapper.keys()}}

        # compare to existing
        print("Replacing ",self.compare(file_name,row))

    def compare(
        self,
        file_name:str|Path,
        specs:pd.Series|dict,
        family_limit:int=10,
        delete_file=True,
    ):
        '''
        Compare design with previous designs. Adds specs to the self.scores DataFrame, to then remove the worst performing design.
        Args:
            specs:pd.Series|dict, specs to add to self.scores
            family_limit: int, the maximum number of designs from the same pMHC-TCR pair in the pool
        Returns:
            pd.Series: removed specs Series
        '''
        # add to pool and remove worst performing
        self.scores.loc[file_name] = specs
        family:pd.DataFrame = self.scores.loc[self.scores["scaffold"]==specs["scaffold"]]
        if len(family)>=family_limit:
            out_name = family.sort_values("score", ascending=True).iloc[0].name
        else:
            out_name = self.scores.sort_values("score", ascending=True).iloc[0].name

        self.scores = self.scores.drop(index=out_name)
        print(f"Removing worst design {out_name} and adding {specs}.")
        if delete_file:
            if not isinstance(file_name, Path):
                out_name = Path(out_name)
            if out_name.parent != self.out_dir:
                out_name = self.out_dir/out_name
            out_name.unlink()
        return out_name

    def mutate_cdrs(
        self,
        input_design:DesignData,
        cdrs:List[str]|str,
        n_mutations:List[int]|int,
        mutate_all=False
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
        if not self.ab:
            alpha_cdrs = [f"acdr{n}" for n in range(1, 4)]
            beta_cdrs = [f"bcdr{n}" for n in range(1, 4)]
        else:
            alpha_cdrs = [f"lcdr{n}" for n in range(1, 4)]
            beta_cdrs = [f"hcdr{n}" for n in range(1, 4)]

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
            target_mask = self.cdr_mask(
                input_design=input_design,
                cdr_ids=[cdr],
                chain_index=chain_index
            )
            positions = np.arange(target_mask.sum())
            if n > len(positions):
                print(f"CDR has length {len(positions)} and {n} mutations have been requested. Mutating the whole CDR!")
                n = len(positions)
            # select random positions in range
            m_positions = np.random.choice(positions, size=n, replace=False)
            insert_mask = (positions[:,None]==m_positions[None,:]).any(axis=1)
            o_insert = input_design["aa"][target_mask.astype(bool)]
            # select new bases
            i_insert = np.random.randint(0,20, size=len(o_insert))
            print(
                f"self.imgt_mapper[cdr]: {self.imgt_mapper[cdr]}",
                f"cdr: {cdr}",
                f"chain_index: {chain_index}",
                f"insert_mask: {insert_mask}",
                f"i_insert: {i_insert}",
                f"o_insert: {o_insert}",
                f"target_mask.sum(): {target_mask.sum()}",
                sep="\n",
                )

            i_insert = np.where(insert_mask, i_insert, o_insert)
            input_design,_ = self.insert_cdr(
                input_design=input_design,
                chain_index=chain_index,
                sequences={cdr:decode(i_insert, AF2_CODE)}
            )
            self.number_anarci(
                input_design=input_design,
                chains=[chain_index,],
                trim=False,
            )
            out_cdrs.update({cdr:decode(i_insert, AF2_CODE)})
        return input_design, out_cdrs

    def get_cdr_seq(
        self,
        input_design:DesignData,
        cdr:str,
        decode:bool=True,
        ):
        target_mask = self.cdr_mask(
            input_design=input_design,
            cdr_ids=[cdr],
        )
        if decode:
            return decode(input_design["aa"][target_mask], AF2_CODE)
        return input_design["aa"][target_mask]

    def cdr_mask(self,
        input_design:DesignData,
        cdr_ids:Iterable[str],
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
            chain_index = self.tcr_chain_index[int(cdr_ids.startswith("h") or cdr_ids.startswith("b"))]
        positions = [self.imgt_mapper[k] for k in cdr_ids]

        chain_mask = np.array(input_design["chain_index"]) == chain_index
        residue_index = np.array(input_design["residue_index"])
        all_cdr_positions = np.concatenate([np.arange(s, e) for s, e in positions])
        mask = (residue_index[:, None] == all_cdr_positions[None, :]).any(axis=1)
        mask = mask & chain_mask
        return mask.astype(float)

    def insert_cdr(self,
        input_design:DesignData,
        chain_index:int,
        sequences:Dict[str,str],
        ):
        '''
        Insert CDR sequences in TCR chain, replacing existing IMGT-numbered CDR positions.
        Args:
            input_design:DesignData, must contain TCR chain in "chain_index" key
            chain_index:int, chain_index value for TCR/Ab chain
            sequences:Dict[str,str], mapping cdr id (e.g. "acdr3") to amino acid sequence
        Returns:
            out:DesignData, design with CDRs replaced by the given sequences
            target_mask:np.ndarray, float mask: 1.0 for inserted CDR positions, 0.0 for framework
        '''
        positions = [self.imgt_mapper[k] for k in sequences.keys()]
        inserts = [DesignData.from_sequence(cdr).update(
            chain_index=jnp.full(len(cdr), chain_index)
            ) for cdr in sequences.values()]
        # Sort by IMGT start position
        sorter = np.argsort([s for s, _ in positions])
        positions = np.array(positions)[sorter]
        inserts = [inserts[p] for p in sorter]

        for i in inserts:
            print_dd(i, "Insert")
        chain_mask = np.array(input_design["chain_index"]) == chain_index
        residue_index = np.array(input_design["residue_index"])
        all_cdr_positions = np.concatenate([np.arange(s, e) for s, e in positions])
        mask = (residue_index[:, None] == all_cdr_positions[None, :]).any(axis=1)
        mask = mask & chain_mask
        # Build framework (non-CDR) segment slices — corrected off-by-one
        fw_slices = []
        in_cdr = False
        fw_start = 0
        for i, current in enumerate(mask):
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

        # Build target mask: 1.0 for inserted CDR, 0.0 for retained framework
        mask_parts = [np.zeros(len(input_design[fw_slices[0]]["aa"]))]
        for insert, fw_slice in zip(inserts, fw_slices[1:]):
            mask_parts.append(np.ones(len(insert["aa"])))
            mask_parts.append(np.zeros(len(input_design[fw_slice]["aa"])))
        target_mask = np.concatenate(mask_parts)

        print_dd(out, "out")
        # Update residue index with IMGT numbering for the modified chain
        out = self.number_anarci(out, chains=(chain_index,))

        assert len(out["aa"]) == len(target_mask), "CDR mask does not have the same length as construct!"
        return out, target_mask

    def number_anarci(
        self,
        input_design:DesignData,
        chains:Tuple[int]|None=None,
        code:str=AF2_CODE,
        scheme:str="imgt",
        trim:bool=False,
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
                if chain_type == "A" and chain != self.tcr_chain_index[0]:
                    self.tcr_chain_index[0] = chain
                elif chain_type == "B" and chain != self.tcr_chain_index[1]:
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
            else:
                print(f"No numbering found for chain {chain}!")
                if chain in self.tcr_chain_index:
                    print("Sequence: ", seq)
        # check if chain indices correct
        if self.tcr_chain_index[0]==self.tcr_chain_index[1]:
            raise ValueError("TCR chains identical! Currently only 2 chain tcrs supported.")
        if self.mhc_chain_index in self.tcr_chain_index:
            # fix mhc chain index to longest non-tcr chain
            chains = np.unique(input_design["chain_index"])
            # mask out tcr chains
            tcr_mask = ~(chains[:,None]==self.tcr_chain_index[None,:]).any(axis=1)
            chains = chains[tcr_mask]
            # get chain lengths
            chain_lengths =  (input_design["chain_index"][:,None] == chains[None,:]).sum(axis=0)
            # take the n longest chain indices, where n the number of mhc chains
            self.mhc_chain_index = np.array(
                [chains[r]
                for r in np.argsort(chain_lengths)[:-(len(self.mhc_chain_index)+1):-1]]
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
    print(f"---{name}---",
        *[f"{k}: {v}\n\t shape: {v.shape}" for k,v in dd.data.items() if k in keys],
        sep="\n"
        )

def clean_chothia(file):
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
                elif not l.startswith("HETATM"):
                    wf.write(l)
    return out_path
