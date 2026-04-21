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
'''
from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import *
from flexcraft.structure.metrics import *
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
    Targeting peptide–MHC complexes with designed T cell receptors and antibodies.
    https://doi.org/10.1101/2025.11.19.689381
    '''
    def __init__(
        self,
        op_dir:Path|str,
        af2_model_name:str,
        key,
        pmpnn_parameter_path:str,
        af2_parameter_path:str|Path|None=None,
        af2_multimer:bool|None=None,
        num_recycle:int=0,
        pmpnn_hparams:dict={},
        ab:bool = False,
        mhc_chain_index:int|Tuple[int]=0,
        tcr_chain_index:int|Tuple[int]=(2,3),
        name="AdaptTrial",
        out_dir:None|str|Path=None,
    ):
        print("initializing")
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

        self.ab = ab
        columns=["out_file","score","TCR","pMHC",]
        if self.ab:
            for n in range(1,4):
                columns.append(f"lcdr{n}")
                columns.append(f"hcdr{n}")
        else:
            for n in range(1,4):
                columns.append(f"acdr{n}")
                columns.append(f"bcdr{n}")
        self.scores = pd.DataFrame(columns=columns)


        # map cdr id to start, end tuple (end exclusive)
        self.imgt_mapper = {
            "lcdr1":(27,38),
            "lcdr2":(56, 65),
            "lcdr3":(105,117),
            "hcdr1":(27,38),
            "hcdr2":(56,65),
            "hcdr3":(105,117),
            "acdr1":(27,38),
            "acdr2":(56, 65),
            "acdr3":(105,117),
            "bcdr1":(27,38),
            "bcdr2":(56,65),
            "bcdr3":(105,117),
            }
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
            self.af2_params = af_utils.flat_params_to_haiku(params=clean_params)
            # adjust model name
            self.af2_model_name = "_".join(self.af2_model_name.split("_")[:3])
        else:
            self.af2_params = get_model_haiku_params(
                    model_name=self.af2_model_name,
                    data_dir=af2_parameter_path.__str__(), fuse=True)

        self.num_recycle = num_recycle
        if af2_multimer is None:
            self.use_multimer = "multimer" in af2_model_name
        else:
            self.use_multimer = af2_multimer
        self.af2_config = model_config(self.af2_model_name)
        self.af2_config.model.global_config.use_dgram = False
        self.af2_model = jax.jit(make_predict(
            make_af2(self.af2_config, use_multimer=self.use_multimer),
            num_recycle=self.num_recycle))

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

    def docking_step(
        self,
        input_design:DesignData,
        evaluate:bool=False,
        is_target:np.ndarray|None=None,
        off_target_template:bool=True,
        save_structue:bool=True,
        ) -> DesignData|Tuple[DesignData, float]:
        '''
        Predict structures of CDRs.
        Args:
            input_design: DesignData, input sequence of spliced TCR with pMHC
            evaluate:bool=False, if True calculate scoring for design and output
            is_target:np.ndarray|None, boolean mask for CDR3 positions (required when evaluate=True)
        Returns:
            structure: DesignData, predicted structure and sequence of the input_design
            Score: float, output score if evaluate
        '''

        input_design = input_design.update(residue_index = np.arange(
            len(input_design["aa"])
        ))
        af_input = AFInput.from_data(input_design)

        if off_target_template:
            if is_target is None:
                print("No is_target input. Not adding template!")
            else:
                print("template mask: ", ~is_target)
                af_input = af_input.add_template(input_design, where=~is_target)

        af_result = self.af_infer(af_input=af_input)

        if evaluate:
            score = self.evaluate_step(af_result=af_result, input_design=input_design, is_target=is_target)
            if save_structue:
                af_result.to_data().save_pdb(self.out_dir/"evaluated_structure.pdb")
            return af_result.to_data(), score
        if save_structue:
            af_result.to_data().save_pdb(self.out_dir/"docked_structure.pdb")
        return af_result.to_data()

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
        af_result:AFResult,
        input_design: DesignData,
        is_target:np.ndarray,
        ) -> float:
        '''Calculate score for protein design.'''
        cdr3_rmsd = self.cdr3_rmsd(
            af_result=af_result, input_design=input_design, is_target=is_target)
        af_pae = self.af_pae(af_result=af_result)
        return 2*af_pae+0.5*cdr3_rmsd
    
    def af_pae(
        self,
        af_result:AFResult,
    )-> float:
        '''
        Compute the mean PAE(predicted aligned error) for all (pMHC, TCR)x(TCR, pMHC) residue pairs.
        '''
        pae_matrix = af_result.result["predicted_aligned_error"]["logits"]
        #mask = (af_result.result["chain_index"][:,None] == np.concat([self.mhc_chain_index,self.tcr_chain_index])[None,:]).sum(axis=1)>0
        #mask = mask[:,None]*mask[None,:]
        #return np.where(mask, pae_matrix, 0).sum()/mask.sum()
        return pae_matrix.mean()

    def cdr3_rmsd(
        self,
        af_result:AFResult,
        input_design: DesignData,
        is_target:np.ndarray,
        ) -> float:
        '''
        Calculate the RMSD for CDR3 chains after alignment of mhc chain.
        Args:
            af_result:AFResult, result of af prediction. Has inputs and result attribute.
            is_target:np.ndarray, bool array that is True for all CDR3 positions.
        '''
        # only include MHC in alignment
        # only calculate rmsd for cdr3
        rmsd = self.rmsd(
            x=af_result,
            y=input_design,
            weight=(af_result.chain_index[:,None]==self.mhc_chain_index[None,:]).any(axis=1),
            eval_mask=is_target,
            )
        return rmsd

    def design_trial(
        self,
        scaffold:str|Path|DesignData,
        pMHC:str|Path|DesignData|None,
        cdr3s:str|Path|Tuple[str],
        redesign_all_cdrs:bool=True,
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
        scaffold = self.number_anarci(scaffold)

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
        print("target mask: ",target_mask)
        # docking step (structure prediction without evaluation)
        design = self.docking_step(input_design=design, is_target=target_mask)
        print_dd(design, "Docked")
        # redesign step: redesign the CDR positions
        design = self.design_step(input_design=design, target_mask=target_mask)
        print_dd(design, "Redesigned")
        # redocking + evaluation step
        design, score = self.docking_step(input_design=design, evaluate=True, is_target=target_mask)
        print_dd(design, "Redocked")
        # save design as unique file name
        file_name = f"{scaffold_name}_{pmhc_name}_0.pdb"
        n = 0
        while (self.out_dir/file_name).exists():
            n += 1
            file_name = f"{scaffold_name}_{pmhc_name}_{n}.pdb"
        print(f"Saving design with score {score} to file {file_name}")
        # Use a named Series so missing CDR1/CDR2 columns get NaN automatically
        if not self.ab:
            row = {"out_file": file_name, "score": score, "TCR": scaffold_name, "pMHC": pmhc_name,
                   "acdr3": cdr3a, "bcdr3": cdr3b}
        else:
            row = {"out_file": file_name, "score": score, "TCR": scaffold_name, "pMHC": pmhc_name,
                   "lcdr3": cdr3a, "hcdr3": cdr3b}
        self.scores.loc[len(self.scores)] = pd.Series(row)
        design.save_pdb(path=self.out_dir/file_name)
        return score

    def refine_trial(
        self,
        scaffold,
        ):
        # fetch list of candidates
        # process input
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

        # perform design step

        # compare to existing

        # add to pool and remove worst performing
    
        pass

    def mutate_cdrs(
        self,
        input_design:DesignData,
        cdrs:List[str],
        n_mutations:List[int]|int,
        mutate_all=False
        ):
        """Function to mutate custom positions."""
        if isinstance(n_mutations, int):
            n_mutations = np.full(len(cdrs),n_mutations)
        for cdr, n in zip(cdrs, n_mutations):
            chain = self.tcr_chain_index[int(cdr.startswith("h") or cdr.startswith("b"))]
            positions = np.arange(*self.imgt_mapper[cdr])
            # select random positions in range
            m_positions = np.sort(np.random.choice(positions, size=n, replace=False))
            insert_mask = (positions[:,None]==m_positions[None,:]).any(axis=1)
            # create new insert
            target_mask = self.cdr_mask(
                input_design=input_design,
                cdr_ids=[cdr]
            )
            o_insert = input_design["aa"][target_mask]
            # select new bases
            i_insert = np.random.randint(0,20, size=len(o_insert))
            i_insert = np.where(insert_mask, i_insert, o_insert)
                            



    def cdr_mask(self,
        input_design:DesignData,
        chain_index:int,
        cdr_ids:Iterable[str],
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
        print(f"mask: {mask}")
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
        print("fw_slices: ",fw_slices)
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
        )->DesignData:
        '''
        Update the residue index with standardized numbering.
        '''
        if chains is None:
            chains = np.unique(input_design["chain_index"])
        if isinstance(chains, int):
            chains = tuple(chains,)
        for chain in chains:
            mask = np.array(input_design["chain_index"]) == chain
            # Pass only this chain's sequence to anarci
            chain_aa = np.array(input_design["aa"])[mask]
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
                if len(residue_index)>len(numbering):
                    # extend variable region by constant region
                    numbering += np.arange(numbering[-1]+1,numbering[-1]+1+mask.sum()-len(numbering)).tolist()
                residue_index[mask] = numbering
                input_design = input_design.update(residue_index=jnp.array(residue_index))
            else:
                if chain in self.tcr_chain_index:
                    print(seq)
                print(f"No numbering found for chain {chain}!")
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
