import os, yaml, sys
import argparse
import numpy as np
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise, get_relevant_output_layers



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--pooling", type=str)
    
    cfg = parser.parse_args()
    layers = get_relevant_output_layers(cfg.model_name, pkg=cfg.pkg)
    for l in layers:
        feats_save_name = f"{paths['livingstone_lab']}/tiziano/models/{cfg.monkey_name}_{cfg.date}_{cfg.model_name}_{cfg.img_size}_{l}_features_{cfg.pooling}pool.npz"
        arr = np.load(feats_save_name)["arr_0"]
        if arr.shape[0] == 4377: # i.e. if samples are in the rows
            np.savez_compressed(feats_save_name, arr.T)
            print_wise(f"transposed layer {l}")
