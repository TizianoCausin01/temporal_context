import yaml
from general_utils.utils import get_upsampling_indices, print_wise
from einops import reduce
from einops.einops import EinopsError


# ---- HELPER FUNCTIONS ----
"""
bin_signal
(same as get_bins in neural_utils.preprocessing)
Computes time bins used to downsample (smooth) a time series to a new sampling frequency.
1) Computes the averaging window length based on the ratio between original and target fs
2) Generates bin edges spanning the full trial duration
3) Ensures the last bin reaches the end of the trial

INPUT:
- time_series: time_series -> time_series object containing the signal and sampling rate
- new_fs: float -> target sampling frequency (Hz)

OUTPUT:
- bins: np.ndarray -> array of integer indices defining bin edges along the time axis
"""
def bin_signal(time_series, new_fs):
    len_avg_window = time_series.fs /new_fs
    trial_duration = time_series.get_len()
    bins = np.round(np.arange(0, trial_duration, len_avg_window)).astype(int)  # bins the target trial with the required resolution, convert to int for later indexing
    if bins[-1] != trial_duration:
        bins = np.append(bins, int(trial_duration))  # adds the last time bin
    return bins
# EOF


"""
smooth_signal
(same as get_firing_rate in neural_utils.preprocessing)
Applies temporal smoothing to a time series by averaging samples within predefined bins.
1) Iterates over consecutive bin intervals
2) Extracts the corresponding time slices from the signal
3) Computes the mean activity within each bin
4) Stacks the averaged chunks along the time dimension

INPUT:
- time_series: time_series -> time_series object containing the signal array
- bins: np.ndarray -> array of integer bin edges defining smoothing windows

OUTPUT:
- smoothed_signal: np.ndarray -> time-smoothed signal with reduced temporal resolution
"""
def smooth_signal(time_series, bins):
    smoothed_signal = []
    
    for idx_bin, bin_start in enumerate(bins[:-1]): # the last el in bin is just the end of the trial, that's why the [:-1] indexing
        bin_end = bins[idx_bin + 1]
        curr_chunk = time_series.array[:,bin_start:bin_end, ...]  # slices the current chunk
        curr_avg_chunk = np.mean(curr_chunk, axis=1)  # computes the mean firing rate over the chunk
        smoothed_signal.append(curr_avg_chunk)    
    # end for idx, bin_start in enumerate(bins[:-1]):
    smoothed_signal = np.stack(smoothed_signal, axis=1) # stacks time in the columns
    return smoothed_signal
# EOF


# ---- CLASSES ----

"""
BrainAreas
Utility class for slicing neural data into predefined brain areas.
1) Loads brain-area channel indices from a YAML configuration file
2) Validates input rasters against the expected number of channels
3) Extracts and concatenates channel ranges corresponding to a given brain area

INPUT:
- monkey_name: str -> identifier used to select the correct brain-area mapping

OUTPUT (slice_brain_area):
- brain_area_response: np.ndarray -> subset of rasters corresponding to the selected brain area
"""
class BrainAreas:
    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        with open("../../brain_areas.yaml", "r") as f:
            config = yaml.safe_load(f)
        try:
            self.areas_idx = config[self.monkey_name]
        except KeyError:
            raise KeyError(f"Monkey '{self.monkey_name}' not found.", f"Supported monkeys {list(config.keys())}") from None
        # end try:
    # EOF
    def slice_brain_area(self, rasters, brain_area_name):
        if rasters.shape[0] < self.areas_idx["n_chan"]:
            raise ValueError(f"Rasters of shape {rasters.shape} doesn't match the original number of channels ({self.areas_idx["n_chan"]}).")
        # end if rasters.shape[0] < self.areas_idx["n_chan"]:

        try:
            target_brain_area = self.areas_idx[brain_area_name]
        except KeyError:
            raise KeyError(f"Brain area '{brain_area_name}' not found for monkey '{self.monkey_name}'.", f"Supported brain areas: {list(self.areas_idx.keys())}") from None
        # end try:
        brain_area_response = []
        for lims in target_brain_area:
            start, end = lims
            brain_area_response.append(rasters[start:end, ...])
        # end for lims in target_brain_area:
        brain_area_response = np.concatenate(brain_area_response)
        return brain_area_response
    # EOF
# EOC


"""
TimeSeries
Container class for neural time series data with utility methods for averaging and resampling.
1) Stores a multidimensional signal array and its sampling frequency
2) Provides duration and length accessors
3) Computes averages across trials, neurons, or both
4) Supports temporal resampling via smoothing or upsampling

INPUT:
- array: np.ndarray -> signal array with shape (neurons, time, trials, ...)
- fs: float -> sampling frequency in Hz
"""
class TimeSeries:
    def __init__(self, array: np.ndarray, fs: float):
        self.array = array
        self.fs = fs
    # EOF
    def get_fs(self):
        return self.fs
    # EOF
    def get_len(self):
        return self.array.shape[1]
    # EOF
    def get_duration_ms(self):
        return self.get_len() * 1000/self.fs 
    # EOF
    def get_duration_s(self):
        return self.get_len() / self.fs
    # EOF
    def get_trial_avg(self):
        try:
            trial_avg = reduce(self.array, 'neurons time trials -> neurons time', 'mean')
            return trial_avg
        except EinopsError:
            raise EinopsError(f"Array of size {self.array.shape} doesn't have the trial dimension (2nd dimension).") from None
            
    # EOF
    def get_neurons_avg(self):
        neurons_avg = reduce(self.array, 'neurons time ... -> 1 time ...', 'mean')
        return neurons_avg
    # EOF
    def get_overall_avg(self):
        overall_avg = reduce(self.array, 'neurons time ... -> time', 'mean')
        return overall_avg
    # EOF
    def resample(self, new_fs):
        if new_fs < self.fs: # smoothing
            bins = bin_signal(self, new_fs)
            self.array = smooth_signal(self, bins)
        elif new_fs > self.fs: # upsampling
            upsampling_indices = get_upsampling_indices(self.get_len(), self.fs, new_fs)
            self.array = self.array[:,upsampling_indices, ...]
        # end if new_fs < self.fs:
        self.fs = new_fs # updates the fs
    # EOF
# EOC
