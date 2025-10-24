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
update_experiments_log(parms['experiment_name'])
resolution_Hz = parms['resolution_Hz']
imec = 0
foreperiod_len = 300 # in ms
for name in parms['monkey_names']:
    for day in parms['dates']:
        experiment_name = f"{name}_25{day}"
        if name == "paul" or name == "red": # he also has a plexon array
            wrapper_load_and_save(paths, f"{name}_25{day}", imec, resolution_Hz, foreperiod_len, npx=False)
        # end if name == paul:
        if name != "red":
            wrapper_load_and_save(paths, experiment_name, imec, resolution_Hz, foreperiod_len, npx=True)
        if name == "paul" or name == "venus": # they also have another array
            wrapper_load_and_save(paths, experiment_name, 1, resolution_Hz, foreperiod_len, npx=True)
        # end if name == "paul" or name == "venus":
