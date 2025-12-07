import sys, os, yaml
import numpy as np
from scipy.io import loadmat
import pickle
import cv2
sys.path.append("..")
from general_utils.utils import print_wise, get_upsampling_indices, delete_empty_keys
from image_processing.utils import get_video_dimensions, load_stimuli_models
from data_preprocessing.preprocessing import min_max_normalization

def extract_fixations_onset(fixations, foreperiod_len_timepts=30):
    differences = np.diff(fixations[foreperiod_len_timepts:], prepend=0)
    fixations_onsets = np.where(differences == 1)[0]
    return fixations_onsets


def load_monkey_data(paths, monkey_name, day, month, resolution_Hz, npx=True, imec_n=0):
    data_path = f"{paths['livingstone_lab']}/tiziano/data"
    if month == "aug":
        month = "08"
    elif month == "sep":
        month = "09"
    if npx:
        neural_path = f"{data_path}/neural_{monkey_name}_25{month}{day}_imec{imec_n}_{resolution_Hz}Hz.pkl"
        gaze_path = f"{data_path}/gaze_{monkey_name}_25{month}{day}_imec{imec_n}_{resolution_Hz}Hz.pkl"
    else:
        neural_path = f"{data_path}/neural_{monkey_name}_25{month}{day}_plx_{resolution_Hz}Hz.pkl"
        gaze_path = f"{data_path}/gaze_{monkey_name}_25{month}{day}_plx_{resolution_Hz}Hz.pkl"
    # end if npx:
    
    # loading the data
    with open(neural_path, "rb") as f:
        neural_data = pickle.load(f)
    with open(gaze_path, "rb") as f:
        gaze_data = pickle.load(f)

    return neural_data, gaze_data


def get_start_end_chunk(center, foreperiod_len_timepts, n_timepts_bef, n_timepts_aft):
    start = (center + foreperiod_len_timepts) - n_timepts_bef
    end = (center + foreperiod_len_timepts) + n_timepts_aft
    return start, end

def extract_face_fixation_responses(n_norm, gaze_data, all_models, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=30):
    day_non_face_fix = []
    day_face_fix = []
    day_occluded_face_fix = []
    day_rep_face_fix = []
    for fn in n_norm.keys(): # loops through all the stimuli
        for i in range(n_norm[fn].shape[2]): # loops thourgh all the repetitions
            current_neural = n_norm[fn][:,:,i]
            current_gaze = gaze_data[fn][:,:,i]
            current_model = all_models[fn]
            fixations_vector = current_gaze[2,:]
            fixation_onsets = extract_fixations_onset(fixations_vector, foreperiod_len_timepts=foreperiod_len_timepts)
            on_face = 0
            for onset in fixation_onsets:
                start, end = get_start_end_chunk(onset, foreperiod_len_timepts, n_timepts_bef, n_timepts_aft)
                if not (start < 0 or end > current_neural.shape[1]): # if it's not too close to the beginning or end
                    fix_resp = current_neural[:,start:end] # indexing period around fixation
                    x_gaze, y_gaze = current_gaze[:2, onset+foreperiod_len_timepts] 
                    mod_position = current_model[:, onset]
                    x1, y1, x2, y2 = mod_position[2:]
                    if mod_position[0] == 0: # check if there is no face in the frame
                        on_face = 0
                        day_non_face_fix.append(fix_resp)
                    elif mod_position[0] ==1: # if there is a face in the frame
                        if (x_gaze >= x1) and (x_gaze <= x2) and (y_gaze >= y1) and (y_gaze <= y2):
                            if on_face == 0:
                                on_face = 1
                                day_face_fix.append(fix_resp)
                            elif on_face == 1:
                                day_rep_face_fix.append(fix_resp)
                            # if on_face == 0:
                        else:
                            on_face = 0
                            day_non_face_fix.append(fix_resp)
                    elif mod_position[0] ==2: # if there is a face in the frame
                        if (x_gaze >= x1) and (x_gaze <= x2) and (y_gaze >= y1) and (y_gaze <= y2):
                            if on_face == 0:
                                on_face = 1
                                day_occluded_face_fix.append(fix_resp)
                            elif on_face == 1:
                                day_rep_face_fix.append(fix_resp)
                            # if on_face == 0:
                        else:
                            on_face = 0
                            day_non_face_fix.append(fix_resp)
                        # if (x_gaze >= x1) and (x_gaze <= x2) and (y_gaze >= y1) and (y_gaze <= y2):
                    # if mod_position[0] == 0: # check if there is no face in the frame
                # if not (start < 0 or end > current_neural.shape[1]): # if it's not too close to the beginning or end
            # for onset in fixation_onsets:
        # end for i in range(n_norm[fn].shape[2]): # loops thourgh all the repetitions
    # end for fn in n_norm.keys(): # loops through all the stimuli
    face_fix = np.stack(day_face_fix, axis=-1)
    occluded_face_fix = np.stack(day_occluded_face_fix, axis=-1)
    non_face_fix = np.stack(day_non_face_fix, axis=-1)
    rep_face_fix = np.stack(day_rep_face_fix, axis=-1)
    return face_fix, occluded_face_fix, non_face_fix, rep_face_fix


def face_fixations(paths, monkey_name, days, month, npx, imec_n, resolution_Hz, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts, model_name, normalization):
    tot_face_fixation = []
    tot_occluded_face_fixation = []
    tot_rep_face_fixation = []
    tot_non_face_fixation = []
    for day in days: # loops through the days of recording # to be changed 
        day = str(day)
        neural_data, gaze_data = load_monkey_data(paths, monkey_name, day, month, resolution_Hz, npx=npx, imec_n=imec_n)
        # normalizes them 
        if normalization == "min_max":
            n_norm = min_max_normalization(neural_data)
        # ADD zscore? robust?
        elif normalization == None:
            n_norm = delete_empty_keys(neural_data)
        
        # loads and upsamples the model 
        all_models = load_stimuli_models(paths, model_name, n_norm.keys(), resolution_Hz)
        day_face_fix, day_occluded_face_fix, day_non_face_fix, day_rep_face_fix = extract_face_fixation_responses(n_norm, gaze_data, all_models, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=foreperiod_len_timepts)
#        day_non_face_fix = day_non_face_fix[:,:,:day_face_fix.shape[2]] # to make their size even I'd have to do a random choice actually
        tot_face_fixation.append(day_face_fix)
        tot_occluded_face_fixation.append(day_occluded_face_fix)
        tot_non_face_fixation.append(day_non_face_fix)
        tot_rep_face_fixation.append(day_rep_face_fix)

        print_wise(f"computed day {day} of monkey {monkey_name}")
    tot_face_fixation = np.concatenate(tot_face_fixation, axis=2)
    tot_occluded_face_fixation = np.concatenate(tot_occluded_face_fixation, axis=2)
    tot_non_face_fixation = np.concatenate(tot_non_face_fixation, axis=2)    
    tot_rep_face_fixation = np.concatenate(tot_rep_face_fixation, axis=2)    
    return tot_face_fixation, tot_occluded_face_fixation, tot_non_face_fixation, tot_rep_face_fixation


"""
random_trials_per_rep
Samples randomly each repetition of each stimulus recorded in n_norm. It is used for random subsampling of the pseudo-trials.
INPUT:
    - n_norm: Dict{str, np.array} -> it's the normalized neural dictionary. Keys: filenames, values: trials associated to them (channels, timepoints, repetitions).
    - gaze_data: Dict{str, np.array} -> it's the gaze/fixations dictionary. Keys: filenames, values: trials associated to them (x-y-fixations, timepoints, repetitions).
    - all_models: Dict{str, np.array} -> it's the models dictionary. Keys: filenames, values: visual information associated to them (features, timepoints, repetitions). 
    - trial_len: int -> the len in timepoints of the pseudotrials
    - trials_per_rep: int -> how many pseudo-trials we sample per repetition
    - foreperiod_len_timepts: int -> how many timepoints we added before the beginning of the repetition in the neural and gaze data.

OUTPUT:
    - neural_trials, gaze_trials, model_trials: np.ndarray -> (features, timepoints, trials) the sampled data to run dRSA
"""
def random_trials_per_rep(n_norm, gaze_data, all_models, trial_len, trials_per_rep=1, foreperiod_len_timepts=30):
    neural_trials = []
    gaze_trials = []
    model_trials = []
    for fn in list(n_norm.keys()): # loops through all the stimuli
        for i in range(n_norm[fn].shape[2]): # loops thourgh all the repetitions
            current_neural = n_norm[fn][:,:,i]
            current_gaze = gaze_data[fn][:2,:,i]
            current_model = all_models[fn]
            indices = np.random.choice(np.arange(foreperiod_len_timepts, current_neural.shape[1]-trial_len), size=trials_per_rep, replace=False)
            for start_idx in indices:
                current_neural_trial = current_neural[:, start_idx:start_idx+trial_len]
                current_gaze_trial = current_gaze[:, start_idx:start_idx+trial_len]
                current_model_trial = current_model[:, start_idx-foreperiod_len_timepts:start_idx+trial_len-foreperiod_len_timepts] # - foreperiod_len_timepts because the movie doesn't have any foreperiod
                neural_trials.append(current_neural_trial)
                gaze_trials.append(current_gaze_trial)
                model_trials.append(current_model_trial)
    neural_trials = np.stack(neural_trials, axis=-1)
    gaze_trials = np.stack(gaze_trials, axis=-1)
    model_trials = np.stack(model_trials, axis=-1)    
    return neural_trials, gaze_trials, model_trials



"""
extract_fixation_responses_sing_trial
Extracts all the fixations in a single trial, it's the building block for the 2 functions below. To run the time-locked dRSA
INPUT:
    - n_norm: Dict{str, np.array} -> it's the normalized neural dictionary. Keys: filenames, values: trials associated to them (channels, timepoints, repetitions).
    - gaze_data: Dict{str, np.array} -> it's the gaze/fixations dictionary. Keys: filenames, values: trials associated to them (x-y-fixations, timepoints, repetitions).
    - all_models: Dict{str, np.array} -> it's the models dictionary. Keys: filenames, values: visual information associated to them (features, timepoints, repetitions). 
    - fn: str -> the filename to be analyzed
    - i_rep: int -> which repetition of the stimulus we have to extract
    - n_timepts_bef: int -> how many timepoints to preserve before the fixation
    - n_timepts_aft: int -> how many timepoints to preserve after the fixation
    - foreperiod_len_timepts: int -> how many timepoints we added before the beginning of the repetition in the neural and gaze data.
    - stack: int -> either 0 or 1, if stack==0, it leaves the fixations as a list of np.ndarray (features, timepts), otherwise it stacks the list into a 3d array (see output) 
OUTPUT:
    - neural_trials, gaze_trials, model_trials: np.ndarray -> (features, timepoints, fixations) the sampled data to run dRSA
    or 
    - neural_trials, gaze_trials, model_trials: list of np.ndarray-> (features, timepoints) fixations) the sampled data to run dRSA
"""
def extract_fixation_responses_sing_trial(n_norm, gaze_data, all_models, fn, i_rep, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=30, stack=0):
    all_neural_fix = []
    all_gaze_fix = []
    all_model_fix = []
    current_neural = n_norm[fn][:,:,i_rep]
    current_gaze = gaze_data[fn][:,:,i_rep]
    current_model = all_models[fn]
    fixations_vector = current_gaze[2,:] 
    fixation_onsets = extract_fixations_onset(fixations_vector, foreperiod_len_timepts=foreperiod_len_timepts)
    for onset in fixation_onsets:
        start, end = get_start_end_chunk(onset, foreperiod_len_timepts, n_timepts_bef, n_timepts_aft)
        if not (start < 0 or end > current_neural.shape[1]): # if it's not too close to the beginning or end
            if not (onset - n_timepts_bef < 0): # checks that the onset doesn't start before the stimulus onset (because we don't have the model for that)
                neural_fix = current_neural[:,start:end] # indexing period around fixation
                gaze_fix = current_gaze[:2, start:end] 
                model_fix = current_model[:, onset-n_timepts_bef:onset+n_timepts_aft]
                all_neural_fix.append(neural_fix)
                all_gaze_fix.append(gaze_fix)
                all_model_fix.append(model_fix)
    if stack == 1:
        all_neural_fix = np.stack(all_neural_fix, axis=-1)
        all_gaze_fix = np.stack(all_gaze_fix, axis=-1)
        all_model_fix = np.stack(all_model_fix, axis=-1)  
    # end if stack == 1:
    return all_neural_fix, all_gaze_fix, all_model_fix


"""
extract_fixation_responses_sing_stim
Extracts all the fixations in a single stimulus, also across repetitions. To run the time-locked dRSA
INPUT:
    - n_norm: Dict{str, np.array} -> it's the normalized neural dictionary. Keys: filenames, values: trials associated to them (channels, timepoints, repetitions).
    - gaze_data: Dict{str, np.array} -> it's the gaze/fixations dictionary. Keys: filenames, values: trials associated to them (x-y-fixations, timepoints, repetitions).
    - all_models: Dict{str, np.array} -> it's the models dictionary. Keys: filenames, values: visual information associated to them (features, timepoints, repetitions). 
    - fn: str -> the filename to be analyzed
    - n_timepts_bef: int -> how many timepoints to preserve before the fixation
    - n_timepts_aft: int -> how many timepoints to preserve after the fixation
    - foreperiod_len_timepts: int -> how many timepoints we added before the beginning of the repetition in the neural and gaze data.
    - stack: int -> either 0 or 1, if stack==0, it leaves the fixations as a list of np.ndarray (features, timepts), otherwise it stacks the list into a 3d array (see output) 
OUTPUT:
    - neural_trials, gaze_trials, model_trials: np.ndarray -> (features, timepoints, fixations) the sampled data to run dRSA
    or 
    - neural_trials, gaze_trials, model_trials: list of np.ndarray-> (features, timepoints) fixations) the sampled data to run dRSA
"""
def extract_fixation_responses_sing_stim(n_norm, gaze_data, all_models, fn, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=30, stack=0):
    all_neural_fix = []
    all_gaze_fix = []
    all_model_fix = []
    for i_rep in range(n_norm[fn].shape[2]): # loops thourgh all the repetitions
        neural_fix, gaze_fix, model_fix = extract_fixation_responses_sing_trial(n_norm, gaze_data, all_models, fn, i_rep, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=foreperiod_len_timepts, stack=0)
        all_neural_fix.extend(neural_fix)
        all_gaze_fix.extend(gaze_fix)
        all_model_fix.extend(model_fix)
    # for i_rep in range(n_norm[fn].shape[2]): # loops thourgh all the repetitions
    if stack == 1:
        all_neural_fix = np.stack(all_neural_fix, axis=-1)
        all_gaze_fix = np.stack(all_gaze_fix, axis=-1)
        all_model_fix = np.stack(all_model_fix, axis=-1)  
    # end if stack == 1:  
    return all_neural_fix, all_gaze_fix, all_model_fix



"""
extract_all_fixation_responses
Extracts all the fixations in a single stimulus, also across repetitions. To run the time-locked dRSA
INPUT:
    - n_norm: Dict{str, np.array} -> it's the normalized neural dictionary. Keys: filenames, values: trials associated to them (channels, timepoints, repetitions).
    - gaze_data: Dict{str, np.array} -> it's the gaze/fixations dictionary. Keys: filenames, values: trials associated to them (x-y-fixations, timepoints, repetitions).
    - all_models: Dict{str, np.array} -> it's the models dictionary. Keys: filenames, values: visual information associated to them (features, timepoints, repetitions). 
    - n_timepts_bef: int -> how many timepoints to preserve before the fixation
    - n_timepts_aft: int -> how many timepoints to preserve after the fixation
    - foreperiod_len_timepts: int -> how many timepoints we added before the beginning of the repetition in the neural and gaze data.
    - stack: int -> either 0 or 1, if stack==0, it leaves the fixations as a list of np.ndarray (features, timepts), otherwise it stacks the list into a 3d array (see output) 
OUTPUT:
    - neural_trials, gaze_trials, model_trials: np.ndarray -> (features, timepoints, fixations) the sampled data to run dRSA
    or 
    - neural_trials, gaze_trials, model_trials: list of np.ndarray-> (features, timepoints) fixations) the sampled data to run dRSA
"""
def extract_all_fixation_responses(n_norm, gaze_data, all_models, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=30, stack=0):
    all_neural_fix = []
    all_gaze_fix = []
    all_model_fix = []
    for fn in n_norm.keys(): # loops through all the stimuli
        neural_fix, gaze_fix, model_fix = extract_fixation_responses_sing_stim(n_norm, gaze_data, all_models, fn, n_timepts_bef, n_timepts_aft, foreperiod_len_timepts=foreperiod_len_timepts, stack=0)
        all_neural_fix.extend(neural_fix)
        all_gaze_fix.extend(gaze_fix)
        all_model_fix.extend(model_fix)
    # end for fn in n_norm.keys()[:2]: # loops through all the stimuli
    if stack == 1:
        all_neural_fix = np.stack(all_neural_fix, axis=-1)
        all_gaze_fix = np.stack(all_gaze_fix, axis=-1)
        all_model_fix = np.stack(all_model_fix, axis=-1)      
    # end if stack == 1:
    return all_neural_fix, all_gaze_fix, all_model_fix
