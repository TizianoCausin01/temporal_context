import sys, os
import numpy as np
import copy
sys.path.append("..")
from general_utils.utils import print_wise


def format_in_trials(file_list, len_avg_window, rasters, trials, stimuli):
    unique_stimuli_names = set(file_list)
    final_res_neural = {name : [] for name in unique_stimuli_names}
    final_res_gaze = copy.deepcopy(final_res_neural)

    # correctly estimates trials durations
    for idx, fn in enumerate(file_list):  # range(len(stimuli)): 
        # print(stimuli[idx]["trial_number"],stimuli[idx]["trial_number"].shape)
        trial_number = (int(stimuli[idx]["trial_number"][0].item()) - 1)  # extracts the trial number to which the stimulus corresponds (-1 because of python indexing)
        
        if trials[trial_number]["success"] == 1 and stimuli[idx]["filename"] == fn:
            trial_start = stimuli[idx]["start_time"][0].item()
            trial_end = stimuli[idx]["stop_time"][0].item()
            trial_duration = trial_end - trial_start
            #print(fn, trial_duration)
            stim_onset_delay = trial_start - trials[trial_number]["start_time"][0].item()
            stim_onset_delay = int(stim_onset_delay) - 1  # -1 for python indexing
            gaze_signal = trials[trial_number]["eye_data"][0]
            end_gaze = min(stim_onset_delay + int(trial_duration), len(gaze_signal))
            gaze_signal = gaze_signal[stim_onset_delay:end_gaze, :].T # extracts gaze from the stimulus onset till the end of the trial
            trial_start_int = int(trial_start)
            trial_end_int = int(trial_end)
            bins = create_bins(trial_duration, len_avg_window)
            neural_signal = rasters[trial_start_int:trial_end_int, :].T  # slices the trial from raster
            trial_firing_rate = get_firing_rate(bins, neural_signal)
            trial_gaze = get_firing_rate(bins, gaze_signal)
            final_res_neural[fn].append(trial_firing_rate)
            final_res_gaze[fn].append(trial_gaze)
        # if trials[trial_number]["success"] == 1 and stimuli[idx]["filename"] == fn:
    # end for i in range(len(stimuli)):
    final_res_neural = cut_excess_timepoints(final_res_neural)
    final_res_gaze = cut_excess_timepoints(final_res_gaze)
    return final_res_neural, final_res_gaze

"""
Creates the bins for computing the firing rate
"""
def create_bins(trial_duration, len_avg_window):
    bins = np.round(np.arange(0, trial_duration, len_avg_window)).astype(int)  # bins the target trial with the required resolution, convert to int for later indexing
    bins = np.append(bins, int(trial_duration))  # adds the last time bin
    return bins
# EOF


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


def cut_excess_timepoints(data_dict):
    dict_keys = list(data_dict.keys())
    if type(data_dict[dict_keys[0]]) != list: # checks if the arg is of the right type
        print_wise(f"Warning! The values of the data dictionary are not lists, cut_excess_timepoints might have been already applied. Returning the same dict.")
        return data_dict
    # end if type(data_dict[dict_keys[0]]) != list:
    for key in dict_keys:
        len_timepts_list = [data_dict[key][i].shape[1] for i in range(len(data_dict[key]))]
        min_time_pts = min(len_timepts_list)
        if len(set(len_timepts_list)) != 1:   
            print_wise(f"Warning! {key} has different time-points across trials {len_timepts_list}")
        # end if len(set(len_timepts_list)) != 1:   
        for i_rep in range(len(data_dict[key])): # loops_through the repetitions of the video within the day
            data_dict[key][i_rep] = data_dict[key][i_rep][:, :min_time_pts]
        # end for i in range(len(data_dict[key])):
        data_dict[key] = np.stack(data_dict[key], axis=2)
    # end for key in data_dict.keys():
    return data_dict
# EOF
