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
from data_preprocessing.preprocessing import format_in_trials
from general_utils.utils import print_wise, get_experiment_parameters, update_experiments_log

parms = get_experiment_parameters()

data_path = f"{paths['data_formatted']}/{parms['experiment_name']}_experiment.mat"
d = loadmat(data_path)
trials = d["Trials"]
stimuli = d["Stimuli"]
print_wise("Start loading rasters...")
if parms['npx'] == False:
    rasters_path = f"{paths['data_formatted']}/{parms['experiment_name']}-rasters.h5"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["rasters"][:]
elif parms['npx'] == True:
    rasters_path = f"{paths['data_neuropixels']}/{parms['experiment_name']}/catgt_{parms['experiment_name']}_g0/{parms['experiment_name']}_g0_imec{parms['imec']}/{parms['experiment_name']}-imec{parms['imec']}-mua_cont.h5"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["mua_cont"][:]
# end if npx == False:
print_wise("Finished loading rasters")

s = np.concatenate(stimuli["filename"])
file_list = [str(x[0]) for x in s]
len_window_firing_rate = 1000/parms['resolution_Hz']
neural, gaze = format_in_trials(file_list, len_window_firing_rate, rasters, trials, stimuli)

with open(f"{paths['livingstone_lab']}/tiziano/data/neural_{parms['experiment_name']}_{parms['resolution_Hz']}Hz.pkl", "wb") as f:
    pickle.dump(neural, f)
with open(f"{paths['livingstone_lab']}/tiziano/data/gaze_{parms['experiment_name']}_{parms['resolution_Hz']}Hz.pkl", "wb") as f:
    pickle.dump(gaze, f)
