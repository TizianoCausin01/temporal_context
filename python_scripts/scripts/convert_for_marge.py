import os, yaml, sys
import numpy as np
from scipy.io import savemat
import argparse
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])

parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
parser.add_argument("--layer_name", type=str)
cfg = parser.parse_args()

save_dir = f"{paths['livingstone_lab']}/tiziano/for_marge"
save_path = f"{save_dir}/manyOO_vit_l_16_384_{cfg.layer_name}_features_meanpool.mat"
p = f"/n/data2/hms/neurobio/livingstone/tiziano/models/manyOO_vit_l_16_384_{cfg.layer_name}_features_meanpool.npz"
f = np.load(p)["arr_0"]
savemat(save_path, {'features': f})
