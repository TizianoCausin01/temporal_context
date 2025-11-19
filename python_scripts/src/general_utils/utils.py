import sys, os, yaml
from datetime import datetime
import numpy as np
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 


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
def autocorr_mat(data, data2=None, metric='correlation'):
    if data2 is None:
        corr_mat = np.corrcoef(data, rowvar=False)
    else:
        if metric == 'correlation':
            d1_shape = data.shape
            d2_shape = data2.shape
            corr_mat = np.corrcoef(data, data2, rowvar=False) 
            corr_mat = corr_mat[:d1_shape[1], d2_shape[1]:]
        else:
            corr_mat = pairwise_distances(data.T, data2.T, metric=metric)
        # end if metric == 'correlation':
    # end if data2 is None:
    return corr_mat
# EOF


"""
create_RDM
Creates an RDM with a specific distance metric and then indexes it with the triu method.
INPUT:
- data: np.array (features x datapoints) -> the data matrix
- metric: str or custom function -> the distance metric
OUTPUT:
- RDM_vec: np.array ([1/2 *(datapoints^2 - datapoints)],) -> the upper triangular entries of the RDM (indexed in a row-major order, excluding diagonal)
                                                            to go back to the full matrix, it's just squareform(RDM_vec)
"""
def create_RDM(data, metric='correlation'):
    if metric == 'correlation':
        RDM = 1 - np.corrcoef(data, rowvar=False)
        rows, cols = np.triu_indices(RDM.shape[0], k=1)
        RDM_vec = RDM[rows, cols]
    else:
        RDM_vec = pdist(data.T, metric=metric)
    # end if metric == 'pearson':
    return RDM_vec
# EOF


"""
spearman
Computes the spearman's rank correlation coefficient between two vectors.
the first argsort treats the position in the matrix as rank and the index as the position in the previous matrix. I.e. it gives the indices associated to each position of the sorting, as if applying it we'd get the ordered list
the second argsort translates the indices into ranks and the position goes back to the initial matrix's position.
e.g.
X = np.array([[30, 10, 20], 
              [5,   1,  9]])
argsort:
[[1 2 0]
 [1 0 2]]
argsort().argsort():
[[2 0 1]
 [1 0 2]]
"""
def spearman(x, y):
    xr = x.argsort().argsort().astype(float)
    yr = y.argsort().argsort().astype(float)
    rho = np.corrcoef(xr, yr)[0, 1]
    return rho
# EOF


"""
compute_dRSA
Starting from two data matrices (the neural data and the model) it computes the dRSA between the two.
1) It computes the RDMs time-series
2) It compares them one another through correlation or whatever other metric
INPUT:
- neural_data: np.ndarray (channels, time, trials) -> the neural data matrix already properly segmented
- model_data: np.ndarray (channels, time, trials) -> the model data matrix already properly segmented, corresponding to the exact timepoints of the neurons
- metric_RDM: str or function -> the distance metric used to compute the neural and model RDMs
- metric_RDM_model: None, str or function -> if not None, the distance metric used to compute the model RDMs
- metric_dRSA: str or function -> the similarity or dissimilarity metric used to compare the time series of RDMs

OUTPUT:
- dRSA_mat: np.ndarray (time, time) -> the matrix that compares the two time series of RDMs one another
"""

def compute_dRSA(neural_data, model_data, metric_RDM='correlation', metric_RDM_model=None, metric_dRSA=spearman):
    RDMs_neu = []
    RDMs_mod = []
    for t in range(neural_data.shape[1]):
        RDMn_t = create_RDM(neural_data[:,t,:], metric=metric_RDM)
        if metric_RDM_model is None:
            RDMm_t = create_RDM(model_data[:,t,:], metric=metric_RDM)
        else:
            RDMm_t = create_RDM(model_data[:,t,:], metric=metric_RDM_model)
        # end if metric_RDM_model is None:
        RDMs_neu.append(RDMn_t)
        RDMs_mod.append(RDMm_t)
    # end for t in range(neural_data.shape[1]):
    RDMs_neu = np.stack(RDMs_neu, axis=1)
    RDMs_mod = np.stack(RDMs_mod, axis=1)
    dRSA_mat = autocorr_mat(RDMs_neu, RDMs_mod, metric=metric_dRSA)
    return dRSA_mat
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


def binary_classification(x, y, n_splits, classification_function, *args, **kwargs):
    accuracy_list = []
    kf = KFold(n_splits=n_splits, shuffle=True) 
    for train_index, test_index in kf.split(x):
    # Split into training and testing sets
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # Initialize the model
        model = classification_function(*args, **kwargs)
    # Train the model
        model.fit(x_train, y_train)
    # Make predictions
        y_pred = model.predict(x_test)
    # Evaluate
        accuracy_list.append(accuracy_score(y_test, y_pred))
    avg_accuracy = np.mean(accuracy_list)
    return avg_accuracy


def binary_classification_over_time(condition_1, condition_2, channel_range, n_splits, classification_function, *args, **kwargs):
    min_trials = min(condition_1.shape[2], condition_2.shape[2]) # even the trials
#    condition_1, condition_2 = condition_1[:,:,:min_trials], condition_2[:,:,:min_trials]
    idx1 = np.random.choice(np.arange(0, condition_1.shape[2]), size=min_trials, replace=False)
    idx2 = np.random.choice(np.arange(0, condition_2.shape[2]), size=min_trials, replace=False)
    condition_1, condition_2 = condition_1[:,:,idx1], condition_2[:,:,idx2]
    if condition_1.shape[2] != condition_2.shape[2]:
        raise IndexError("The number of datapoints across conditions is different")
    condition_1_label = np.ones(condition_1.shape[2])
    condition_2_label = np.zeros(condition_2.shape[2])
    y = np.concatenate((condition_1_label, condition_2_label))
    x_timeseries = np.concatenate((condition_1, condition_2),axis=2)
    accuracy_over_time = []
    for i in range(x_timeseries.shape[1]):
        x = x_timeseries[channel_range[0]:channel_range[1], i, :].T
        avg_accuracy = binary_classification(x, y, n_splits, classification_function, *args, **kwargs)
        accuracy_over_time.append(avg_accuracy)
    accuracy_over_time = np.array(accuracy_over_time)
    return accuracy_over_time
