from flexcraft.pipelines.tcr import *
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
    boltz_config = dict(
        boltz_docking=True,
        boltz_redocking=False,
        boltz_parameter_path=Path("./params/boltz"),
        boltz_model_name="boltz2_conf",
        boltz_num_recycle = 0,
        boltz_num_samples = 1,
        boltz_num_sampling_steps = 25,
        boltz_deterministic = False,),
    af2_config = dict(
        af2_model=None,
        af2_params=None,
        af2_model_name=None,
        af2_parameter_path=None,
        af2_multimer=None,
        af2_num_recycle=0,),
    pmpnn_config = dict(
        pmpnn_model=None,
        pmpnn_parameter_path="./params/pmpnn/v_48_020.pkl",
        pmpnn_hparams={},),
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

n_refine_steps=10
boltz_sample_range = [1,3,7,10]

scores = []
for n, af_parameter_path in enumerate(af_parameter_paths):
    _config = config.copy()
    _config["af2_config"].update(af2_parameter_path=af_parameter_path)
    _config.update(out_dir=out_dir/f"af_params_{af_parameter_path.stem}")
    print(_config)
    for (mhc_allele,antigen) in unique_targets:
        adapt = ADAPT(
            **_config
        )
        mhc_seq = get_mhc(name=mhc_allele)
        with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
            ids = rf.readline().strip("\n").split("\t")
            for pdb_id, is_ab in zip(TCR_STRUCTURES+AB_STRUCTURES,
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
                    presenter=mhc_seq,
                    cdrs=cdrs,
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
        _config["af2_config"].update(
        af2_params = adapt.af2_params,
        af2_model = adapt.af2_model,
        )
        config["pmpnn_config"].update(
        pmpnn_model=adapt.pmpnn,
        )

best_af = af_parameter_paths[np.argmin(scores)]
best_af_score = min(scores)
config["af2_config"].update(af2_parameter_path=best_af)

# test boltz vs af in docking
config["boltz_config"].update(boltz_docking=True)
boltz_scores = []
af2_params = None
af2_model = None
for n, num_samples in enumerate(boltz_sample_range):
    _config = config.copy()
    _config["boltz_config"].update(boltz_num_samples=num_samples)
    _config.update(out_dir=out_dir/f"boltz_numsamples_{num_samples}")
    print(_config)
    
    adapt = ADAPT(
        **_config
        )
    for (mhc_allele,antigen) in unique_targets:
        adapt = ADAPT(
            **_config
        )
        mhc_seq = get_mhc(name=mhc_allele)
        with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
            ids = rf.readline().strip("\n").split("\t")
            for pdb_id, is_ab in zip(TCR_STRUCTURES+AB_STRUCTURES,
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
                    presenter=mhc_seq,
                    cdrs=cdrs,
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
        boltz_scores.append(adapt.get_scores()["score"].min())

    _config["af2_config"].update(
        af2_params = adapt.af2_params,
        af2_model = adapt.af2_model,
        )

best_boltz = boltz_sample_range[np.argmin(boltz_scores)]
best_boltz_score = min(boltz_scores)

print(
    "---AF Scores---",
    *[f'{n}:{s}'for n, s in zip(af_parameter_paths, scores)],
    f"Best Model: {best_af} with score {best_af_score}",
    "---Boltz Scores---",
    *[f'{n}:{s}'for n, s in zip(boltz_sample_range, boltz_scores)],
    f"Best samples: {best_boltz} with score {best_boltz_score}",
)

print(collect_results(out_dir, pattern="*", save=True))