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
from general_utils.utils import print_wise


data_path = f"{paths['data_formatted']}/{exp_name}_experiment.mat"
d = loadmat(data_path)
trials = d["Trials"]
stimuli = d["Stimuli"]
print_wise("Start loading rasters...")
if npx == False:
    rasters_path = f"{paths['data_formatted']}/{exp_name}-rasters.h5"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["rasters"][:]
elif npx == True:
    rasters_path = f"{paths['data_neuropixels']}/{exp_name}/catgt_{exp_name}_g0/{exp_name}_g0_imec{imec_n}/{exp_name}-imec{imec_n}-mua_cont.h5"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["mua_cont"][:]
# end if npx == False:
print_wise("Finished loading rasters")

s = np.concatenate(stimuli["filename"])
file_list = [str(x[0]) for x in s]
len_window_firing_rate = 1000/res_Hz
neural, gaze = format_in_trials(file_list, len_window_firing_rate, rasters, trials, stimuli)

with open(f"{paths['livingstone_lab']}/tiziano/data/neural_{exp_name}_{res_Hz}Hz.pkl", "wb") as f:
    pickle.dump(neural, f)
with open(f"{paths['livingstone_lab']}/tiziano/data/gaze_{exp_name}_{res_Hz}Hz.pkl", "wb") as f:
    pickle.dump(gaze, f)
