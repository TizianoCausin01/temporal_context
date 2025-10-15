import os, yaml, sys
import numpy as np
from scipy.io import loadmat
import h5py
import copy
import pickle
from scipy.stats import zscore

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from data_preprocessing.preprocessing import wrapper_load_and_save 
from general_utils.utils import print_wise, get_experiment_parameters, update_experiments_log

parms = get_experiment_parameters()

resolution_Hz = parms['resolution_Hz']
imec = 0
for name in parms['monkey_names']:
    for day in parms['dates']:
        experiment_name = f"{name}_25{day}"
        if name == "paul": # he also has a plexon array
            wrapper_load_and_save(paths, f"{name}_2025{day}", imec, resolution_Hz, npx=False)
        # end if name == paul:
        wrapper_load_and_save(paths, experiment_name, imec, resolution_Hz, npx=True)
        if name == "paul" or name == "venus": # they also have another array
            wrapper_load_and_save(paths, experiment_name, 1, resolution_Hz, npx=True)
        # end if name == "paul" or name == "venus":
