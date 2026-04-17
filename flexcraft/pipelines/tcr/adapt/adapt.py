from flexcraft.data.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import *
from flexcraft.structure.metrics import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *


from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable
from datetime.datetime import now

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
        op_dir:Path|str,
        af2_parameter_path:str,
        af2_model_name:str,
        key,
        pmpnn_parameter_path:str,
        num_recycle:int=0,
        pmpnn_hparams:dict={},
        ab:bool = False,
        mhc_chain_index:int=0,
        tcr_chain_index:Tuple[int]|List[int]=(2,3),
        name="AdaptTrial"
    ):
        # directory organization
        if not isinstance(op_dir, Path):
            op_dir = Path(op_dir)
        self.op_dir = op_dir
        self.in_dir = self.op_dir/"input_data"
        if not self.in_dir.exists():
            raise FileNotFoundError(f"Input dir {self.in_dir} does not exist!")
        self.out_dir = self.in_dir/(now().__str__()+"AdaptTrial_0")
        n=0
        while self.out_dir.exists():
            n+=1
            self.out_dir = self.in_dir/(self.out_dir.name[:-1]+str(n))
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
        self.mhc_chain_index = mhc_chain_index
        self.tcr_chain_index = tcr_chain_index
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
        self.af2_parameter_path = af2_parameter_path
        self.num_recycle = num_recycle
        self.use_multimer = "multimer" in af2_model_name
        self.af2_params = get_model_haiku_params(
                model_name=model,
                data_dir=af2_parameter_path, fuse=True)
        self.af2_config = model_config(self.model[0])
        self.af2_config.model.global_config.use_dgram = False
        self.af2_model = jax.jit(make_predict(
            make_af2(self.af2_config, use_multimer=self.use_multimer),
            num_recycle=self.num_recycle))

        self.rmsd = RMSD()
        

    def af_infer(self, af_input:AFInput) -> AFResult:
        '''Wrapper for af model, handles non-multimer models.'''
        if not self.use_multimer:
            num_chains = len(jnp.unique(af_input.data["chain_index"]))
            af_input_masked = af_input.block_diagonal(num_sequences=num_chains)
            af_result: AFResult = self.af2_model(params, self.key(), af_input_masked)
        else:
            af_result: AFResult = self.af2_model(params, self.key(), af_input)
        return af_result

    def docking_step(
        self,
        input_design:DesignData,
        evaluate:bool=False,
        ) -> DesignData|Tuple[DesignData, float]:
        '''
        Predict structures of CDRs.
        Args:
            input_design: DesignData, input sequence of spliced TCR with pMHC
            evaluate:bool=False, if True calculate scoring for design and output
        Returns:
            structure: DesignData, predicted structure and sequence of the input_design
            Score: float, output score if evaluate
        '''

        af_input = AFInput.from_data(input_design)

        af_result = self.af_infer(af_input=af_input)

        if evaluate:
            score = self.evaluate_step(af_result=af_result)
            return af_result.to_data(), score

        return af_result.to_data()

    def design_step(self, input_design:DesignData, target_mask:np.ndarray|None):
        '''
        Design a sequence from a structure using ProteinMPNN.
        Args:
            input_design: DesignData, containing predicted structure
            target_mask: np.ndarray|None, if None, all cdrs are redesigned
        '''
        if not target_mask:
            # mask all cdrs
            target_mask = np.zeros(len(input_design["aa"]))
            target_mask += self.cdr_mask(
                input_design=DesignData,
                cdr_ids=self.scores.columns[3:6]
            )
            target_mask += self.cdr_mask(
                input_design=DesignData,
                cdr_ids=self.scores.columns[6:]
            )
            target_mask = target_mask>0

        target_aa = design["aa"]
        target_aa[target_mask] = 20
        input_design = input_design.update(aa=target_aa)
        
        if self.center_logits:
            # calculate logit center 
            logit_center = self.pmpnn(self.key(), input_design)["logits"][target_size:].mean(axis=0)

        pmpnn_result, _ = self.pmpnn_sampler(key(), input_design)
        pmpnn_result = input_design.update(
            aa=aas.translate(pmpnn_result["aa"], aas.PMPNN_CODE, aas.AF2_CODE))
        return pmpnn_result

    def evaluate_step(
        self,
        af_result:AFResult,
        is_target:np.ndarray,
        ) -> float:
        '''Calculate score for protein design.'''
        cdr3_rmsd = self.cdr3_rmsd(af_result=af_result, is_target=is_target)
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
        mask = (af_result.result["chain_index"][:,None] == np.array([self.mhc_chain_index, *self.tcr_chain_index])[None,:]).sum(axis=1)>0
        mask = mask[:,None]*mask[None,:]
        return np.where(mask, pae_matrix, 0).sum()/mask.sum()

    def cdr3_rmsd(
        self,
        af_result:AFResult,
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
            x=af_result.inputs,
            y=af_result.result,
            weight=af_result["chain_index"]==self.mhc_chain_index,
            eval_mask=is_target,
            )
        return rmsd

    def design_trial(
        scaffold:str|Path,
        pMHC:str|Path,
        cdr3s:str|Path
    ):
        '''Run a design step for recombination of TCRs with CDR3s'''
        if isinstance(scaffold, str):
            scaffold = Path(scaffold)
        if scaffold.parent==self.in_dir:
            scaffold = scaffold.name
        if isinstance(pMHC, str):
            pMHC = Path(pMHC)
        if pMHC.parent==self.in_dir:
            pMHC = pMHC.name
        if isinstance(cdr3s, str):
            cdr3s = Path(cdr3s)
        if cdr3s.parent==self.in_dir:
            cdr3s = cdr3s.name
        # process input
        # load tcr
        scaffold = PDBFile(path=self.in_dir/scaffold)
        # load pmhc
        pmhc = PDBFile(path=self.in_dir/scaffold)
        # concatenate with split_chains
        design = DesignData.concat([pmhc, scaffold])
        # load cdr3s
        sep="\t"
        if cdr3s.endswith("csv"):
            sep = ","
        df = pd.read_csv(self.in_dir/cdr3s, sep=sep)
        cdr3a, cdr3b = df.sample(n=1).iloc[0].values
        # insert cdr3s
        design, target_mask_a = self.insert_cdr(
            input_design=design,
            chain_index=2,
            sequences={"lcdr3":cdr3a,}
            )
        design, target_mask_b = self.insert_cdr(
            input_design=design,
            chain_index=2,
            sequences={"hcdr3":cdr3b})
        target_mask = (target_mask_a+target_mask_b)>0
        # docking step
        design = self.docking_step(input_design=design)
        # redesign step
        design = self.design_step(input_design=design)
        # redocking step
        # ranking step
        design, score = self.docking_step(input_design=design, evaluate=True)
        # save design as unique file name
        file_name = f"{scaffold.split('.')[0]}_{pMHC.split('.')[0]}_0.pdb"
        n=0
        while (self.out_dir/file_name).exists():
            n+=1
            file_name = file_name.split(".")[0][:-1]+str(n)+".pdb"
        print(f"Saving design with score {score} to file {file_name}")
        self.scores.loc[len(self.scores)] = [file_name, score, scaffold, pMHC, cdr3a, cdr3b]
        design.save_pdb(path=self.out_dir/file_name)
        return ranking

    def refine_trial(
        self,
        design,
        ):

        # fetch list of candidates
        
        # randomly select a candidate
        
        # remove candidate from available list
        
        # mutate 2 cdr positions

        # perform design step

        # compare to existing

        # add to pool and remove worst performing
    
        pass

    def cdr_mask(self,
        input_design:DesignData,
        chain_index:int,
        cdr_ids:Iterable[str],
        ):
        '''
        Get a mask for cdr_ids on chain chain_index.
        Args:
            input_design:DesignData, must contain TCR chain in "chain_index" key
            chain_index:int, chain_index value for TCR/Ab chain
            cdr_ids:iterable[str], ids for cdrs
        '''
        assert (np.array([x[0] for x in cdr_ids])[:,None]==np.array([x[0] for x in cdr_ids])[None,:]).all(), ValueError("All cdrs must be on the same chain!")

        positions = [self.imgt_mapper[k] for k in cdr_ids]

        # sort by positions
        sorter = np.argsort([s for s,_ in positions])
        positions = np.array(positions)[sorter]
        cdr_ids = np.array(cdr_ids)[sorter]

        chain_mask = input_design["chain_index"]==chain_index

        index = np.arange(len(input_design["aa"]))
        mask = (input_design["residue_index"][:,None]==np.concat(
            [np.arange(s,e) for s,e in positions])
            ).any(axis=1)
        mask *= chain_mask
        l = []
        start = 0
        last = start
        for i, current in enumerate(mask):
            if last != current:
                if current:
                    l.append(slice(start, i))
                else:
                    start = i
            last = current
        l.append(slice(start,-1))

        ones = np.ones(len(input_design["aa"]))
        target_mask = np.concatenate(
            [ones[l[0]],
            *[
                np.concatenate([np.zeros(len(insert)), ones[next_slice]])
            for insert, next_slice in zip(cdrs, l[1:])
            ]],
        )
        return target_mask

    def insert_cdr(self,
        input_design:DesignData,
        chain_index:int,
        sequences:Dict[str,str|Iterable[str|int]],
        ):
        '''
        Insert CDR sequences in TCR chain
        Args:
            input_design:DesignData, must contain TCR chain in "chain_index" key
            chain_index:int, chain_index value for TCR/Ab chain
            sequences:Dict[str:str|list], mapping cdr number to sequence
        '''
        
        positions = [self.imgt_mapper[k] for k in sequences.keys()]

        # sort by positions
        sorter = np.argsort([s for s,_ in positions])
        cdrs = np.array(cdrs)[sorter] # type: ignore
        positions = np.array(positions)[sorter]
        
        inserts = [DesignData.from_sequence(cdr).update(
            chain_index=jnp.full(len(cdr), chain_index)
            ) for cdr in sequences.values()]
        
        chain_mask = input_design["chain_index"]==chain_index

        index = np.arange(len(input_design["aa"]))
        mask = (input_design["residue_index"][:,None]==np.concat(
            [np.arange(s,e) for s,e in positions])
            ).any(axis=1)
        mask *= chain_mask
        l = []
        start = 0
        last = start
        for i, current in enumerate(mask):
            if last != current:
                if current:
                    l.append(slice(start, i))
                else:
                    start = i
            last = current
        l.append(slice(start,-1))

        out = DesignData.concatenate(
            [input_design[l[0]],
            *[
                DesignData.concatenate([insert, input_design[next_slice]], sep_batch=False, sep_chains=False)
            for insert, next_slice in zip(inserts, l[1:])
            ]],
            sep_batch=False,
            sep_chains=False,
        )
        ones = np.ones(len(input_design["aa"]))
        target_mask = np.concatenate(
            [ones[l[0]],
            *[
                np.concatenate([np.zeros(len(insert)), ones[next_slice]])
            for insert, next_slice in zip(cdrs, l[1:])
            ]],
        )

        # update residue index
        out = out.update(residue_index = jnp.arange(len(out["aa"])))

        assert len(out["aa"]) == len(target_mask), "CDR mask does not have the same length as construct!"
        return out, target_mask

def load_data(out_dir=Path("./data/adapt/input_data"),
    url = "https://zenodo.org/records/17488258/files/",
    files = [
        "paired_human_cdr3s.tsv",
        "model_2_ptm_ft_binder_20230729.pkl",
        "RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt",
        "zenodo_design_models.zip"
    ],
    ):
    from urllib.request import urlretrieve
    from zipfile import ZipFile
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
