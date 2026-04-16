from flexcraft.data import DesignData
from flexcraft.files.pdb import PDBFile
from flexcraft.structure.af import *
from flexcraft.utils import Keygen, parse_options, data_from_protein
import flexcraft.sequence.aa_codes as aas
from flexcraft.sequence.mpnn import make_pmpnn
from flexcraft.sequence.sample import *


from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

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
        dir:Path|str,
        af2_parameter_path:str,
        af2_model_name:str,
        key,
        pmpnn_parameter_path:str,
        num_recycle:int=0,
        pmpnn_hparams:dict={},
        ab:bool = False,
    ):
        if not isinstance(dir, Path):
            dir = Path(dir)
        self.dir = dir
        self.ab = ab
        # map cdr id to start, end tuple (end exclusive)
        self.imgt_mapper = {
            "lcdr1":(27,38),
            "lcdr2":(56, 65),
            "lcdr3":(105,117),
            "hcdr1":(27,38),
            "hcdr2":(56,65),
            "hcdr3":(105,117),
            }
        
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
        ) -> DesignData:
        '''
        Predict structures of CDRs.
        Args:
            input_design: DesignData, input sequence of spliced TCR with pMHC
        Returns:
            DesignData: predicted structure and sequence of the input_design
        '''

        af_input = AFInput.from_data(input_design)

        af_result = self.af_infer(af_input=af_input)

        return af_result.to_data()

    def design_step(self, input_design:DesignData, target_mask:np.ndarray):
        '''
        Design a sequence from a structure using ProteinMPNN.
        Args:
            input_design: DesignData, containing predicted structure
        '''
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

    def evaluate_step(self, input_design:DesignData):
        pass

    def design_trial(
        scaffold:str,
        pMHC:str,
        cdr3s:str
    ):
        '''Run a design step for recombination of TCRs with CDR3s'''
        # process input
        in_dir = self.dir/"input_data"
        # load tcr
        scaffold = PDBFile(path=in_dir/scaffold)
        # load pmhc
        pmhc = PDBFile(path=in_dir/scaffold)
        # concatenate with split_chains
        design = DesignData.concat([pmhc, scaffold])
        # load cdr3s
        sep="\t"
        if cdr3s.endswith("csv"):
            sep = ","
        df = pd.read_csv(in_dir/cdr3s, sep=sep)
        cdr3a, cdr3b = df.sample(n=1).iloc[0].values
        # insert cdr3a
        design, target_mask = self.insert_cdr(
            input_design=design,
            chain_index=2,
            sequences={"lcdr3":cdr3a}
        )
        # insert cdr3b
        design, target_mask_2 = self.insert_cdr(
            input_design=design,
            chain_index=3,
            sequences={"hcdr3":cdr3b}
        )

        target_mask = (target_mask_2 + target_mask)>0
        # docking step
        design = self.docking_step(input_design=design)
        # redesign step
        design = self.design_step(input_design=design, target_mask=target_mask)
        # redocking step
        design = self.docking_step(input_design=design)
        # ranking step
        # TODO write evaluation_step
        return ranking

    def refine_trial(
        n_instances:int=5,
        n_steps:int=10,
        ):
        # load candidate list
        # start parrallel simulations
        instances = {str(i):asyncio.create_task(n_steps) for i in n_instances}
        
        asyncio.gather(instances)



    async def refiner(n_steps):
        for n in n_steps:
            pass
            # fetch list of candidates
            
            # randomly select a candidate
            
            # remove candidate from available list
            
            # mutate 2 cdr positions

            # perform design step

            # compare to existing

            # add to pool and remove worst performing
        
        pass

    def insert_cdr(self,
        input_design:DesignData,
        chain_index:int,
        sequences:Dict[str:str|Iterable[str|int]],
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
        # add 0 to start the sequence for out
        positions = np.array(positions)[sorter]

        inserts = [DesignData.from_sequence(cdr).update(
            chain_index=jnp.full(len(cdr), chain_index)
            ) for cdr in sequences.values()]

        chain_mask = input_design["chain_index"]==chain_index
        print(np.concat([np.arange(s,e) for s,e in positions]))
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
        print(l)
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
