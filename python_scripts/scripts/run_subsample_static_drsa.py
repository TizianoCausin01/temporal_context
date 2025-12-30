import os, yaml, sys
import argparse

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_relevant_output_layers, compute_samples_sizes, load_img_natraster
from general_utils.static_dRSA import init_whole_neural_RDM, similarity_subsamples_par
from parallel.parallel_funcs import master_workers_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--brain_area", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--max_size", type=int) # the maximal number of samples we'll take
    parser.add_argument("--step_samples", type=int) # of how many samples we jump every time
    parser.add_argument("--n_iter", type=int) # how many time we repeat the sampling from each size
    parser.add_argument("--RDM_metric", type=str)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--new_fs", type=int) 
    parser.add_argument("--pkg", type=str)

    cfg = parser.parse_args()

    task_list = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    area_rasters = load_img_natraster(paths, cfg)
    n_trials = area_rasters.array.shape[2]
    n_samples = compute_samples_sizes(cfg)
    drsa_obj, whole_RDM_signal = init_whole_neural_RDM(area_rasters, cfg)
    master_workers_queue(task_list, paths, similarity_subsamples_par, *(drsa_obj, whole_RDM_signal, n_samples, cfg)) 
