import os, yaml, sys
import numpy as np
from scipy.spatial.distance import squareform

sys.path.append("..")
from general_utils.utils import print_wise, TimeSeries, dRSA, load_img_natraster, check_attributes, subsample_RDM



"""
init_whole_neural_RDM
Initializes and computes the full neural RDM time series for a given brain area.

1) Creates a dRSA object using the RDM metric specified in the configuration
2) Computes the signal RDM at each time point of the input TimeSeries
3) Converts each vectorized RDM into its square (matrix) form

INPUT:
- area_rasters: TimeSeries -> neural activity restricted to a single brain area
- cfg: Cfg -> configuration object with required attributes:
    * RDM_metric: str

OUTPUT:
- drsa_obj: dRSA -> dRSA object containing the signal RDM time series
- whole_RDM_signal: list[np.ndarray] -> list of square-form RDMs (one per time point)
"""
def init_whole_neural_RDM(area_rasters, cfg):
    check_attributes(cfg, "RDM_metric")
    drsa_obj = dRSA(cfg.RDM_metric)
    drsa_obj.compute_RDM_timeseries(area_rasters, "signal")
    whole_RDM_signal = [squareform(RDM_t) for RDM_t in drsa_obj.get_RDM_timeseries("signal")]
    return drsa_obj, whole_RDM_signal
# EOF

"""
similarity_subsamples_loop
Computes static dRSA time courses for multiple random subsamples of trials.

For each requested subsample size, the function:
1) Randomly samples trials without replacement
2) Subsamples neural and model RDMs accordingly
3) Computes the static dRSA time series
4) Repeats the procedure for a fixed number of iterations

INPUT:
- n_samples: np.ndarray[int] -> array of subsample sizes
- iter_dict: dict[int, np.ndarray] -> dictionary mapping sample size to results array
- whole_RDM_signal: list[np.ndarray] -> list of square-form neural RDMs (one per time point)
- whole_RDM_model: np.ndarray -> square-form model RDM
- cfg: Cfg -> configuration object with required attributes:
    * n_iter: int
    * RDM_metric: str
    * new_fs: float

OUTPUT:
- iter_dict: dict[int, np.ndarray] -> updated dictionary where each key k maps to
  an array of shape (n_iter, n_timepoints) containing static dRSA values
"""
def similarity_subsamples_loop(n_samples: np.ndarray[int], iter_dict: dict[int, np.ndarray], whole_RDM_signal: list[np.ndarray], whole_RDM_model: np.ndarray, cfg):
    check_attributes(cfg, "n_iter", "RDM_metric", "new_fs")
    drsa_obj = dRSA(cfg.RDM_metric)
    n_trials = whole_RDM_signal[0].shape[0]
    n_tpts = len(whole_RDM_signal)
    for k in n_samples: 
        curr_ns = np.empty((cfg.n_iter, n_tpts))
        for i_iter in range(cfg.n_iter):
            curr_idx = np.random.choice(n_trials, size=k, replace=False)
            assert len(np.unique(curr_idx)) == k # sanity check that we have unique elements
            curr_neu = [squareform(subsample_RDM(RDM_t, curr_idx)) for RDM_t in whole_RDM_signal]
            curr_neu = TimeSeries(curr_neu, cfg.new_fs)
            drsa_obj.set_RDM_timeseries(curr_neu, "signal")
            curr_mod = squareform(subsample_RDM(whole_RDM_model, curr_idx))
            drsa_obj.set_RDM(curr_mod, "model")
            curr_ns[i_iter, :] = drsa_obj.compute_static_dRSA().array
        # end for i_iter in range(n_iter):
        iter_dict[k] = curr_ns
    # end for k in n_samples: 
    return iter_dict
# EOF


"""
similarity_subsamples_par
Parallel wrapper for subsampled static dRSA computation for a single model layer.

1) Loads model features and computes the full model RDM
2) Computes the static dRSA using all trials (reference condition)
3) Runs subsampled dRSA computations across multiple sample sizes
4) Saves the results to disk in compressed NPZ format

INPUT:
- paths
- rank: int -> process rank (used for controlled logging in parallel execution)
- layer_name: str -> name of the model layer being analyzed
- drsa_obj: dRSA -> initialized dRSA object
- whole_RDM_signal: list[np.ndarray] -> list of square-form neural RDMs (one per time point)
- n_samples: np.ndarray[int] -> array of subsample sizes
- cfg: Cfg -> configuration object with required attributes:
    * new_fs: float
    * monkey_name: str
    * model_name: str
    * img_size: int
    * n_iter: int
    * brain_area: str

OUTPUT:
- None
Side effects:
- Saves a compressed .npz file containing the subsampling results
"""
def similarity_subsamples_par(paths: dict[str: str], rank: int, layer_name: str, drsa_obj: dRSA, whole_RDM_signal: list[np.ndarray], n_samples: np.ndarray[int], cfg):
    check_attributes(cfg, "new_fs", "monkey_name", "model_name", "img_size", "n_iter", "brain_area")
    dict_savename = f"{paths['livingstone_lab']}/tiziano/results/subsampling_{cfg.new_fs}Hz_{n_samples[0]}-{n_samples[-1]}_{cfg.n_iter}iter_{cfg.monkey_name}_{cfg.date}_{cfg.brain_area}_{cfg.model_name}_{cfg.img_size}_{layer_name}.npz"
    if os.path.exists(dict_savename):
        print_wise(f"model already exists at {dict_savename}", rank=rank)
    else:
        feats_filename = f"{paths['livingstone_lab']}/tiziano/models/{cfg.monkey_name}_{cfg.date}_{cfg.model_name}_{cfg.img_size}_{layer_name}_features_{cfg.pooling}pool.npz"
        feats = np.load(feats_filename)["arr_0"]
        drsa_obj.compute_RDM(feats, "model")
        whole_RDM_model = squareform(drsa_obj.get_RDM("model"))
        iter_dict = {ns: np.empty((0,)) for ns in n_samples}
        n_trials = feats.shape[1]
        iter_dict[n_trials] = drsa_obj.compute_static_dRSA().array
        print_wise(f"computed the whole static dRSA for layer {layer_name}", rank=rank)
        del drsa_obj
        iter_dict = similarity_subsamples_loop(n_samples, iter_dict, whole_RDM_signal, whole_RDM_model, cfg)
        np.savez_compressed(dict_savename, **{str(k): v for k, v in iter_dict.items()})
        print_wise(f"computed all iterations for layer {layer_name}, \nsaved at {dict_savename}", rank=rank)
    # end if os.path.exists(dict_savename):
# EOF
