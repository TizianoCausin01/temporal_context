import os, yaml, sys
import argparse

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_relevant_output_layers, compute_samples_sizes, load_img_natraster
from general_utils.utils import BrainAreas, get_triu_perms, load_img_natraster
from general_utils.dRSA import across_areas_dRSA
from parallel.parallel_funcs import master_workers_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--RDM_metric", type=str)
    parser.add_argument("--RSA_metric", type=str, default="correlation")
    parser.add_argument("--new_fs", type=int) 

    cfg = parser.parse_args()

    area_rasters = load_img_natraster(paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs)
    ba_obj = BrainAreas(cfg.monkey_name)
    brain_areas = ba_obj.get_brain_areas()
    task_list = get_triu_perms(brain_areas)
    raster = load_img_natraster(paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs)
    master_workers_queue(task_list, paths, across_areas_dRSA, *(raster, ba_obj, cfg)) 
