from flexcraft.pipelines.tcr.adapt import *
from flexcraft.pipelines.tcr.utils import *
from flexcraft.utils.rng import Keygen

from datetime import datetime
from pathlib import Path

#--- test the design pipeline ---
out_dir = Path(f"./data/adapt/tuning_trial_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}")
out_dir.mkdir()
config = dict(
    op_dir="./data/adapt/",
    key = Keygen(42),
    pmpnn_model=None,
    af2_model=None,
    af2_params=None,
    boltz_docking=True,
    boltz_redocking=False,
    boltz_parameter_path=Path("./params/boltz"),
    boltz_model_name="boltz2_conf",
    boltz_num_recycle = 0,
    boltz_num_samples = 1,
    boltz_num_sampling_steps = 25,
    boltz_deterministic = False,
    af2_model_name=None,
    af2_parameter_path=None,
    af2_multimer=None,
    af2_num_recycle=0,
    pmpnn_parameter_path="./params/pmpnn/v_48_030.pkl",
    pmpnn_hparams={},
    ab = False,
    mhc_chain_index=0,
    tcr_chain_index=(2,3),
    name="tuning",
    out_dir=None,
    trim=True,
    chain_cache_len=450, # how long to pad for af
)
unique_targets = [
    ("A*02:01", "TLMSAMTNL"),   # PAP
    #("A*01:01", "EVDPIGHLY"),   # MAGE-A3
    #("A*02:01", "ALYDKTKRI"),   # TdT
    #("A*02:01", "GLMWLSYFV"),   # SARS
    #("A*02:01", "HMTEVVRHC"),   # p53
    #("A*02:01", "LLWNGPIAV"),   # YFV
    #("A*03:01", "ALHGGWTTK"),   # PIK3CA
    #("A*11:01", "VVVGADGVGK"),  # KRAS
    #("B*44:02", "SEITKQEKDF"),  # PIK3CA
]
#TCR_STRUCTURES = ["4Y1A", "5BS0","8d5q", "2OI9", "5VCJ"]
TCR_STRUCTURES = ["5BS0",]# "1OGA", "3QDG", "7OW6", "3GSN", "7RRG", "7N2R", "5EU6"]
AB_STRUCTURES = ["7YV1", ]#"6YIO", "7LSG", "7KQL", "7SSC"]
# test af_parameters
params_path = Path("./params/af/params")
af_parameter_paths=[Path("data/adapt/input_data/model_2_ptm_ft_binder_20230729.pkl"),
    *[p for p in params_path.glob("*.npz")]]

n_refine_steps=50
n_design_steps=50

scores = []

adapt = ADAPT(
    **_config
)
for (mhc_allele, antigen) in unique_targets:
    with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
        ids = rf.readline().strip("\n").split("\t")
        for pdb_id, is_ab in zip(
            TCR_STRUCTURES+AB_STRUCTURES,
            np.concatenate([np.zeros(len(TCR_STRUCTURES), dtype=np.bool_),
                    np.ones(len(AB_STRUCTURES), dtype=np.bool_)])):
            
            cdrs = {
                i[-1]+i[:-1]:n
                for i,n in zip(ids, rf.readline().strip("\n").split("\t"))
            }
            scaffold = clean_chothia(
                    download_structure(
                        pdb_id=pdb_id, out_dir="./data/adapt/input_data",
                        file_format="antibody" if is_ab else "biological assembly"))
            scaffold, scaffold_name = adapt.make_scaffold(
                receptor=scaffold,
                antigen=antigen,
                presenter=None if is_ab else scaffold,
                cdrs=cdrs,
                is_ab=is_ab,
                trim=True,
                replace_antigen=True,
            )

            adapt.design_trial(
                design=scaffold,
                scaffold_name=scaffold_name
            )

    for n in range(n_refine_steps):
        print(f"refinement step {n}")

        scaffold, pdb_path, scaffold_name = adapt.get_design("random")

        adapt.refine_trial(
            scaffold,
            scaffold_name=scaffold_name
        )
    scores.append(adapt.get_scores()["score"].min())
    _config.update(
    af2_params = adapt.af2_params,
    af2_model = adapt.af2_model,
    )
    config.update(
    pmpnn_model=adapt.pmpnn,
    )

