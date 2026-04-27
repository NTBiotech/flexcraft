from flexcraft.pipelines.tcr import *
from flexcraft.utils.rng import Keygen

from datetime import datetime
from pathlib import Path

#--- test the design pipeline ---
out_dir = Path(f"./data/adapt/tuning_trial_{datetime.now().strftime('%Y-%d-%b_%H:%M:%S')}")
out_dir.mkdir()
config = dict(
    op_dir="./data/adapt/",
    key = Keygen(42),
    pmpnn_sampler=None,
    af2_model=None,
    af2_params=None,
    boltz_docking=False,
    boltz_redocking=False,
    boltz_parameter_path=Path("./params/boltz"),
    boltz_model_name="boltz2_conf",
    boltz_num_recycle = 0,
    boltz_num_samples = 1,
    boltz_num_sampling_steps = 25,
    boltz_deterministic = False,
    af2_model_name="model_2_ptm_ft_binder_20230729",
    af2_parameter_path=None,
    af2_multimer=None,
    af_num_recycle=0,
    pmpnn_parameter_path="./params/pmpnn/v_48_030.pkl",
    pmpnn_hparams={},
    ab = False,
    mhc_chain_index=0,
    tcr_chain_index=(2,3),
    name="tuning",
    out_dir=None,
    trim=True,
    chain_cache_len=650, # how long to pad for af
)

TCR_STRUCTURES = ["4Y1A","8d5q", "2OI9", "5VCJ"]

# test af_parameters
params_path = Path("./params/af/params")
af_parameter_paths=[Path("data/adapt/input_data/model_2_ptm_ft_binder_20230729.pkl"),
    *[p for p in params_path.glob("*.npz")]]

n_refine_steps=5
boltz_sample_range = np.linspace(1,10,4, dtype=int)

scores = []
for af_parameter_path in af_parameter_paths:
    _config = copy(config)
    _config.update(af2_parameter_path=af_parameter_path)
    _config.update(out_dir=out_dir/f"af_params_{af_parameter_path.stem}")
    print(_config)
    adapt = ADAPT(
        **_config
    )
    with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
        ids = rf.readline().strip("\n").split("\t")
        for pdb_id in TCR_STRUCTURES:
            if not "." in pdb_id:
                pdb_path = download_structure(pdb_id, output_dir="./data/adapt/input_data")
                pdb_path = clean_chothia(pdb_path)
            else:
                pdb_path = pdb_id
            cdrs = {
                i[-1]+i[:-1]:n
                for i,n in zip(ids, rf.readline().strip("\n").split("\t"))
            }
            print(cdrs)
            adapt.design_trial(
                pdb_path,
                cdrs=cdrs
            )
        for n in range(n_refine_steps):
            adapt.refine_trial(
                adapt.get_design("random")
            )
        scores.append(adapt.get_scores()["score"].max())

best_af = af_parameter_paths[np.argmax(scores)]
best_af_score = max(scores)
config.update(af_parameter_path=best_af)

# test boltz vs af in docking
config.update(boltz_docking=True)
boltz_scores = []
for num_samples in boltz_sample_range:
    _config = copy(config)
    _config.update(boltz_num_samples=num_samples)
    _config.update(out_dir=out_dir/f"boltz_numsamples_{num_samples}")
    print(_config)
    adapt = ADAPT(
        **_config
        )
    with open("./data/adapt/input_data/paired_human_cdr3s.tsv", "r") as rf:
        ids = rf.readline().strip("\n").split("\t")
        for pdb_id in TCR_STRUCTURES:
            if not "." in pdb_id:
                pdb_path = download_structure(pdb_id, output_dir="./data/adapt/input_data")
                pdb_path = clean_chothia(pdb_path)
            else:
                pdb_path = pdb_id
            cdrs = {
                i[-1]+i[:-1]:n
                for i,n in zip(ids, rf.readline().strip("\n").split("\t"))
            }
            print(cdrs)
            adapt.design_trial(
                pdb_path,
                cdrs=cdrs
            )
        for n in range(n_refine_steps):
            adapt.refine_trial(
                adapt.get_design("random")
            )
        boltz_scores.append(adapt.get_scores()["score"].max())
    

best_boltz = boltz_sample_range[np.argmax(boltz_scores)]
best_boltz_score = max(boltz_scores)




print(
    "---AF Scores---",
    *[f'{n}:{s}'for n, s in zip(af_parameter_paths, scores)],
    f"Best Model: {best_af} with score {best_af_score}",
    "---Boltz Scores---",
    *[f'{n}:{s}'for n, s in zip(boltz_sample_range, boltz_scores)],
    f"Best samples: {best_boltz} with score {best_boltz_score}",
)
