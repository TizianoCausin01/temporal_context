import sys, os
import numpy as np
import copy
import h5py
import pickle
import cv2
from scipy.io import loadmat
sys.path.append("..")
from general_utils.utils import print_wise

"""
format_in_trials
From the raster, stimuli and trials objects, it gives two dicts (neural and gaze dicts) with the formatted trials.
INPUT: 
    - paths: dict{str, str} -> the dict specified in the config.yaml file
    - file_list: list{str} -> a list with all the stimulus names
    - len_avg_window: np.float -> how much we want to average out our 1000Hz neural and eye-tracking signals
    - rasters: np.ndarray (timepts x channels) -> the whole experiment neural signal 
    - trials, stimuli -> the objects loaded from the experiment.mat file

OUTPUT:
    - final_res_neural, final_res_gaze: dict{str, np.ndarray{3D}} -> dictionaries with the stimuli names as keys and 3D np.ndarrays (channels x timepts x trials)  
"""
def format_in_trials(paths, file_list, len_avg_window, rasters, trials, stimuli):
    unique_stimuli_names = set(file_list)
    final_res_neural = {name : [] for name in unique_stimuli_names}
    final_res_gaze = copy.deepcopy(final_res_neural)

    # correctly estimates trials durations
    for idx, fn in enumerate(file_list):  # range(len(stimuli)): 
        trial_number = (int(stimuli[idx]["trial_number"][0].item()) - 1)  # extracts the trial number to which the stimulus corresponds (-1 because of python indexing)
        
        if trials[trial_number]["success"] == 1 and stimuli[idx]["filename"] == fn:
            trial_start = stimuli[idx]["start_time"][0].item()
            trial_end = stimuli[idx]["stop_time"][0].item()
            trial_duration = trial_end - trial_start
            stim_onset_delay = trial_start - trials[trial_number]["start_time"][0].item()
            stim_onset_delay = round(stim_onset_delay) - 1  # -1 for python indexing
            gaze_signal = trials[trial_number]["eye_data"][0]
            end_gaze = min(stim_onset_delay + round(trial_duration), len(gaze_signal))
            gaze_signal = gaze_signal[stim_onset_delay:end_gaze, :].T # extracts gaze from the stimulus onset till the end of the trial
            trial_start_int = round(trial_start)
            trial_end_int = round(trial_end)
            bins = create_bins(trial_duration, len_avg_window)
            neural_signal = rasters[trial_start_int:trial_end_int, :].T  # slices the trial from raster
            trial_firing_rate = get_firing_rate(bins, neural_signal)
            trial_gaze = get_firing_rate(bins, gaze_signal)
            trial_gaze = convert_gaze_coordinates(trial_gaze)
            trial_gaze = append_fixations(trial_gaze, trial_number, trials, len_avg_window, stim_onset_delay)
            final_res_neural[fn].append(trial_firing_rate)
            final_res_gaze[fn].append(trial_gaze)
        # if trials[trial_number]["success"] == 1 and stimuli[idx]["filename"] == fn:
    # end for i in range(len(stimuli)):
    final_res_neural = cut_excess_timepoints(final_res_neural)
    final_res_gaze = cut_excess_timepoints(final_res_gaze)
    final_res_neural, final_res_gaze = cut_short_movies(paths, final_res_neural, final_res_gaze, len_avg_window)
    return final_res_neural, final_res_gaze

"""
create_bins
Creates the bins for computing the firing rate by defining a range with a certain step and converting to int (because we assume the resolution of the signal is at 1000Hz.
INPUT:
    - trial_duration: int -> the duration of the trial
    - len_avg_window: np.float -> how much we want to average out our 1000Hz neural and eye-tracking signals
OUTPUT:
    - bins: np.ndarray -> the onsets of each bin to average in get_firing_rate
"""
def create_bins(trial_duration, len_avg_window):
    bins = np.round(np.arange(0, trial_duration, len_avg_window)).astype(int)  # bins the target trial with the required resolution, convert to int for later indexing
    if bins[-1] != trial_duration:
        bins = np.append(bins, int(trial_duration))  # adds the last time bin
    return bins
# EOF


"""
get_firing_rate
Smooths and downsamples the neural or gaze signal to the desired resolution, identified by bins.
INPUT:
    - bins: np.ndarray -> defined in create bins, they are the onset indeces 
    - neural_signal: np.ndarray -> the neural signal of a trial
OUTPUT: 
    - trial_firing_rate: np.ndarray -> the firing rate of a trial
"""
def get_firing_rate(bins, neural_signal):
    trial_firing_rate = []
    
    for idx_bin, bin_start in enumerate(bins[:-1]): # the last el in bin is just the end of the trial, that's why the [:-1] indexing
        bin_end = bins[idx_bin + 1]
        curr_chunk = neural_signal[:,bin_start:bin_end]  # slices the current chunk
        curr_firing_rate = np.mean(curr_chunk, axis=1)  # computes the mean firing rate over the chunk
        trial_firing_rate.append(curr_firing_rate)    
    # end for idx, bin_start in enumerate(bins[:-1]):
    trial_firing_rate = np.stack(trial_firing_rate, axis=1) # stacks time in the columns
    return trial_firing_rate
# EOF


"""
cut_excess_timepoints
Cuts the excess datapoints in the data dict lists before stacking them together. It's needed due to duration timing inconsistencies between different repetitions of the same trial.
INPUT:
    - data_dict: dict{str, list} -> the data dictionary still with lists as values (each element of the list is a separate trial of the same stimulus, represented as features x timepoints)

OUTPUT:
    - data_dict: dict{str, np.ndarray} -> the data dictionary with 3D arrays as values (features x timepoints x trials). We trim the last datapoints (usually 0 or 1) to make trials of the same length
"""

def cut_excess_timepoints(data_dict):
    dict_keys = list(data_dict.keys())
    if type(data_dict[dict_keys[0]]) != list: # checks if the arg is of the right type
        print_wise(f"Warning! The values of the data dictionary are not lists, cut_excess_timepoints might have been already applied. Returning the same dict.")
        return data_dict
    # end if type(data_dict[dict_keys[0]]) != list:
    for key in dict_keys:
        len_timepts_list = [data_dict[key][i].shape[1] for i in range(len(data_dict[key]))]
        if len(len_timepts_list) > 0:
            min_time_pts = min(len_timepts_list)
            if len(set(len_timepts_list)) != 1:   
                print_wise(f"Warning! {key} has different time-points across trials {len_timepts_list}")
            # end if len(set(len_timepts_list)) != 1:   
            for i_rep in range(len(data_dict[key])): # loops_through the repetitions of the video within the day
                data_dict[key][i_rep] = data_dict[key][i_rep][:, :min_time_pts]
            # end for i in range(len(data_dict[key])):
            data_dict[key] = np.stack(data_dict[key], axis=2)
        else:
            print_wise(f"Warning, {key} doesn't have any successful trial")
            data_dict[key] = np.array([])
        # end if len(len_timepts_list) > 0:
    # end for key in data_dict.keys():
    return data_dict
# EOF


"""
convert_gaze_coordinates
Converts the eye-tracking signal into pixels coordinates (1080 x 1920 resolution).
Since the initial reference frame has the origin at the center of the screen we have to add width/2 on the x and subtract height/2 on the y. 
The *32 is a multiplicative factor to convert the original unit of measure for gaze into pixels of that size.
INPUT:
    - gaze: np.ndarray (2, timepoints)
OUTPUT:
    - gaze: np.ndarray (2, timepoints)
"""
def convert_gaze_coordinates(gaze):
    gaze[0, :] = 960 + gaze[0, :]*32
    gaze[1, :] = 540 - gaze[1, :]*32
    return gaze 
# EOF


"""
movie_paths
Gets the right folder of the movies of that experiment
INPUT: 
    - paths: dict{str, str} -> the dict specified in the config.yaml file
    - stimuli_names: list -> the list with the stimuli that were presented
OUTPUT:
    - movies_folder: str -> the folder were the movies are
"""
def movie_paths(paths, stimuli_names):
    stimuli_folder = f"{paths['livingstone_lab']}/Stimuli/movies"
    if "IMG_4692.mp4" in stimuli_names:
        return f"{stimuli_folder}/peoplePPE"
    elif "IMG_4655.mp4" in stimuli_names:
        return f"{stimuli_folder}/cagemonkeys"
    elif "YDXJ0085.MP4" in stimuli_names:
        return f"{stimuli_folder}/occluded_videos"
    elif "anna1_10s.mp4" in stimuli_names:
        return f"{stimuli_folder}/faceswap_4"
    elif "steve1.mp4" in stimuli_names:
        return f"{stimuli_folder}/faceswap_5"
    else:
        print_wise(f"I couldn't find the right path for these stimuli {stimuli_names}")


"""
get_video_duration_fps
Extracts metadata from the movie (in order to account for movies that were shorter than the trial duration)
INPUT:
    - video_path: str -> the path to the video
OUTPUT:
    - fps: np.float -> the framerate of the video
    - duration: np.float -> the duration of the video in ms
"""
def get_video_duration_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))        
    duration = frame_count *1000 / fps # in ms, that's why we multiply by 1000
    cap.release()
    return np.round(fps, 2), duration


"""
cut_short_movies 
Cuts the excess datapoints in a dict, it's there to account for the fact that some movies were much less than 10s.
INPUT:
    - paths: dict{str, str} -> the dict specified in the config.yaml file
    - neural_dict: dict{str, np.ndarray} -> the neural dict with the trials names and data
    - gaze_dict: dict{str, np.ndarray} -> the gaze dict with the trials names and data
    - len_avg_window: int -> 1000/resolution_Hz, i.e. the length in ms of the window within which we average
OUTPUT:
    - data_dict: dict{str, np.ndarray} -> the dict with the trials names and data, can be both gaze and neural with the datapoints cut now
"""
def cut_short_movies(paths, neural_dict, gaze_dict, len_avg_window):
    stimuli_names = list(neural_dict.keys())
    movies_folder = movie_paths(paths, stimuli_names)
    for fn in stimuli_names:
        if neural_dict[fn].size: 
            fps, vid_duration = get_video_duration_fps(f"{movies_folder}/{fn}")
            vid_duration = round(vid_duration / len_avg_window)
            if vid_duration < neural_dict[fn].shape[1]:
                neural_dict[fn] = neural_dict[fn][:,:vid_duration, :]
            if vid_duration < gaze_dict[fn].shape[1]:
                gaze_dict[fn] = gaze_dict[fn][:,:vid_duration, :]
    return neural_dict, gaze_dict


"""
append_fixations
Appends the aligned fixation times as the third (and binary) row to trial_gaze
INPUT:
    - trial_gaze: np.ndarray -> 2 x trial_duration, gaze data for the current trial 
    - trial_number: int -> the number of the trial
    - trials -> the trials file
    - len_avg_window: np.float -> how much we want to average out our 1000Hz neural and eye-tracking signals
    - stim_onset_delay: int -> the delay between the start of the trial and the start of the stimulus presentation
OUTPUT:
    - trial_gaze_fixations: np.ndarray -> 3 x trial_duration, gaze data for the current trial with the fixations in the third row
"""
def append_fixations(trial_gaze, trial_number, trials, len_avg_window, stim_onset_delay):
    gaze_len = trial_gaze.shape[1]
    fixation_times = trials[trial_number]["fixation_times"]
    fixation_times_delay = fixation_times[0].astype(int) - stim_onset_delay - 1 # -1 for python indexing, the other for the fact that we are subtracting 1 to stimulus_duration
    downsampled_fixation_times = np.round(fixation_times_delay/len_avg_window).astype(int)
    fixation_mask = (downsampled_fixation_times >= 0) & (downsampled_fixation_times <= gaze_len)
    fixation_masked = downsampled_fixation_times[np.any(fixation_mask, axis=1)]
    fixation_masked = fixation_masked 
    if fixation_masked[0,0] < 0:
        fixation_masked[0,0] = 0
    if fixation_masked[-1, 1] >= gaze_len:
        fixation_masked[-1, 1] = gaze_len -1
    fixation_binary = np.zeros(gaze_len)
    for onset, offset in fixation_masked:
        fixation_binary[onset:offset] = 1  
    trial_gaze_fixations = np.concatenate((trial_gaze, fixation_binary[np.newaxis,:]), axis=0)      
    return trial_gaze_fixations

def wrapper_load_and_save(paths, experiment_name, imec, resolution_Hz, npx=True):
    if npx == True:
        neural_out_fn=f"{paths['livingstone_lab']}/tiziano/data/neural_{experiment_name}_imec{imec}_{resolution_Hz}Hz.pkl"
        gaze_out_fn=f"{paths['livingstone_lab']}/tiziano/data/gaze_{experiment_name}_imec{imec}_{resolution_Hz}Hz.pkl"
    else:
        neural_out_fn=f"{paths['livingstone_lab']}/tiziano/data/neural_{experiment_name}_plx_{resolution_Hz}Hz.pkl"
        gaze_out_fn=f"{paths['livingstone_lab']}/tiziano/data/gaze_{experiment_name}_plx_{resolution_Hz}Hz.pkl"

    # end if npx == True:
    if os.path.exists(neural_out_fn) & os.path.exists(gaze_out_fn):
        print_wise(f"paths {neural_out_fn} already exist")
        return 
    # end if os.path.exists(neural_out_fn) & os.path.exists(gaze_out_fn):

    data_path = f"{paths['data_formatted']}/{experiment_name}_experiment.mat"
    if "red" in experiment_name: # red is saved differently also in the experiment.mat file
        exp_name_plx = experiment_name[:4] + "20" + experiment_name[4:]
        data_path = f"{paths['data_formatted']}/{exp_name_plx}_experiment.mat"
    d = loadmat(data_path)
    trials = d["Trials"]
    stimuli = d["Stimuli"]
    print_wise(f"Start loading rasters of {experiment_name}...")
    if npx == False:
        if "paul" in experiment_name:
            exp_name_plx = experiment_name[:5] + "20" + experiment_name[5:] # because plx saves files with 2025 instead of 25
        elif "red" in experiment_name:
            pass # we have already defined exp_name_plx
        rasters_path = f"{paths['data_formatted']}/{exp_name_plx}-rasters.h5"
        with h5py.File(rasters_path, "r") as f:
            rasters = f["rasters"][:]
    elif npx == True:
        rasters_path = f"{paths['data_neuropixels']}/{experiment_name}/catgt_{experiment_name}_g0/{experiment_name}_g0_imec{imec}/{experiment_name}-imec{imec}-mua_cont.h5"
        with h5py.File(rasters_path, "r") as f:
            rasters = f["mua_cont"][:]
    # end if npx == False:
    print_wise("Finished loading rasters")

    s = np.concatenate(stimuli["filename"])
    file_list = [str(x[0]) for x in s]
    len_window_firing_rate = 1000/resolution_Hz
    neural, gaze = format_in_trials(paths, file_list, len_window_firing_rate, rasters, trials, stimuli)

    with open(neural_out_fn, "wb") as f:
        pickle.dump(neural, f)
        print_wise(f"file saved at {neural_out_fn}")
    with open(gaze_out_fn, "wb") as f:
        pickle.dump(gaze, f)
# EOF
