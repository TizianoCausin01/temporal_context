import sys, os, yaml
from datetime import datetime
import numpy as np
import argparse
from scipy.spatial import cKDTree

def print_wise(mex, rank=None):
    if rank == None:
        print(datetime.now().strftime("%H:%M:%S"), f"- {mex}", flush=True)
    else:
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"- rank {rank}",
            f"{mex}",
            flush=True,
        )


# EOF

"""
get_lagplot_checks
Does the sanity checks for the autocorrelation function.
1 - If the maximum lag required is larger than the datapts available raises an index error
2 - If the maximum lag required overcomes the number of minimum datapts asked to compute the average of a diagonal prints a warning
3 - If the matrix has nans (datapts with var=0 usually with norm=0) prints a warning  
"""
def get_lagplot_checks(corr_mat, max_lag, min_datapts):
    n_timepts = corr_mat.shape[0]
    if n_timepts <= max_lag:
        raise IndexError("The maximum lag is larger than the matrix itself")
    elif n_timepts < max_lag+min_datapts:
        print_wise(f"The number of datapoints used to compute extreme offsets is < than {min_datapts}")
    # end if n_timepts < max_lag:
    nan_mask = np.isnan(corr_mat)
    if np.any(nan_mask):
        print_wise(f"There are nans in corr_mat") #{np.where(nan_mask)}")
    # end if np.any(nan_mask):
# EOF

"""
autocorr_mat
Correlates one time-series matrix to itself or one another to yield a timepts x timepts matrix.
INPUT:
    - data: np.ndarray(float) -> (features x time_pts)
    - data2: np.ndarray(float) -> (features x time_pts) in case of cross-correlation

OUTPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the matrix of cross-correlation
"""
def autocorr_mat(data, data2=None):
    if data2 is None:
        corr_mat = np.corrcoef(data, rowvar=False)
    else:
        d1_shape = data.shape
        d2_shape = data2.shape
        corr_mat = np.corrcoef(data, data2, rowvar=False) # in the future I'll create a corr function with numba
        corr_mat = corr_mat[:d1_shape[1], d2_shape[1]:]
    return corr_mat
# EOF



"""
get_lagplots
From a correlation matrix, it returns the lagplot by averaging over the diagonals.
INPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the auto-correlation or cross-correlation matrix
    - max_lag: int -> the maximum offset in tpts
    - min_datapts: int -> the minimum amount of points that a diagonal should have to be considered acceptable
    - symmetric: bool -> if we are computing a cross-correlation, corr_mat is not symmetric, otherwise it is
OUTPUT:
    - lagplot: np.ndarray -> (max_lag*2 + 1) if not symmetric, otherwise (max_lag +1), it's the correlation coefficient as a function of the lag
"""

def get_lagplot(corr_mat, max_lag=20, min_datapts=10, symmetric=False):
    # first sanity checks    
    get_lagplot_checks(corr_mat, max_lag, min_datapts)
    if not symmetric:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag*2 +1)
        lagplot[max_lag] = np.nanmean(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, -tau)
            lagplot[max_lag+tau] = np.nanmean(d) # append because the lower triangular correspond to a positive offset between data1 and data2
            d = np.diag(corr_mat, tau)
            lagplot[max_lag-tau] = np.nanmean(d) # appendleft because the upper triangular correspond to a negative offset between data1 and data2
    else:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag +1)
        lagplot[0] = np.nanmean(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, tau)
            lagplot[tau] = np.nanmean(d)
    return lagplot
# EOF

def split_integer(total: int, n: int):
    """Split total into n nearly equal integer parts."""
    base = total // n
    remainder = total % n
    # distribute the remainder (one extra for the first 'remainder' chunks)
    parts = [base + 1 if i < remainder else base for i in range(n)]
    return parts
# EOF

def make_intervals(total: int, n: int):
    parts = split_integer(total, n)
    intervals = []
    start = 0
    for p in parts:
        intervals.append((start, p))
        start = start+p
    return intervals


def get_experiment_parameters():
    with open("../experiments.yaml", "r") as f: # loads the yaml file with the experiment parameters
        experiments = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses_name", type=str) # receives the input name
    args = parser.parse_args()
    experiment_parameters = experiments[args.analyses_name]
    experiment_parameters['analyses_name'] = args.analyses_name
    return experiment_parameters
# EOF

def update_experiments_log(experiment_name):
    with open("../experiments_log.txt", "a") as f:
        f.write(f"\n{datetime.now().strftime('%H:%M:%S')} - {experiment_name}")
# EOF


"""
get_timestamps
Constructs the timestamps for a time-series 
"""
def get_timestamps(n_timepts, sampling_rate):
    timestamps = np.arange(n_timepts)/sampling_rate
    return timestamps
#EOF


"""
get_upsampling_indices
Upsamples a signals by means of nearest neighbour timept.
"""
def get_upsampling_indices(n_old_timepts, old_rate, new_rate):
    old_timestamps = get_timestamps(n_old_timepts, old_rate)
    upsample_factor = new_rate/old_rate
    n_new_timepts = int(np.round(n_old_timepts * upsample_factor))
    new_timestamps = get_timestamps(n_new_timepts, new_rate)
    tree = cKDTree(old_timestamps[:, None])
    _, indices = tree.query(new_timestamps[:, None], k=1) # Query nearest old sample for each new time (like dsearchn)
    return indices
# EOF


def delete_empty_keys(data_dict):
    new_dict = {k: v for k, v in data_dict.items() if v.shape != (0,)}
    return new_dict
