import sys, os, yaml
from datetime import datetime
import numpy as np
import torch
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
nan_check
Checks if there are NaN vals in the correlation matrix
"""
def nan_check(corr_mat: np.ndarray):
    nan_mask = np.isnan(corr_mat)
    if np.any(nan_mask):
        print_wise(f"There are nans in corr_mat") #{np.where(nan_mask)}")
    # end if np.any(nan_mask):


"""
choose_summary_stat
To choose the summary statistics to summarize the diagonals of the autocorrelation matrix
"""
def choose_summary_stat(summary_stat: str):
    if summary_stat == 'mean':
        stat = np.nanmean
    elif summary_stat == 'median':
        stat = np.nanmedian
    else:
        raise ValueError("summary_stat must be 'mean' or 'median'")
    # end if summary_stat == 'mean':
    return stat


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
    if data.shape[1] == 1:
        raise IndexError('Cannot compute RDM with only 1 trial')
    # end if data.shape[1] == 1:
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
def get_lagplot(corr_mat, max_lag=20, min_datapts=10, symmetric=False, summary_stat='mean'):
    # first sanity checks    
    get_lagplot_checks(corr_mat, max_lag, min_datapts)
    stat = choose_summary_stat(summary_stat)
    if not symmetric:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag*2 +1)
        lagplot[max_lag] = stat(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, -tau)
            lagplot[max_lag+tau] = stat(d) # append because the lower triangular correspond to a positive offset between data1 and data2
            d = np.diag(corr_mat, tau)
            lagplot[max_lag-tau] = stat(d) # appendleft because the upper triangular correspond to a negative offset between data1 and data2
    else:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag +1)
        lagplot[0] = stat(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, tau)
            lagplot[tau] = stat(d)
    return lagplot
# EOF


"""
get_lagplot_subset
Computes the lagplot (average along the diagonals) only within a specific window of the neural and model data.
INPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the auto-correlation or cross-correlation matrix
    - neural_idx: list/array/range -> time points to consider along neural axis
    - model_idx: list/array/range -> time points to consider along model axis
    - max_lag: int -> the maximum offset in tpts
    - min_datapts: int -> the minimum amount of points that a diagonal should have to be considered acceptable
    - symmetric: bool -> if we are computing a cross-correlation, corr_mat is not symmetric, otherwise it is
OUTPUT:
    - lagplot: np.ndarray -> (max_lag*2 + 1) if not symmetric, otherwise (max_lag +1), it's the correlation coefficient as a function of the lag
"""
def get_lagplot_subset(corr_mat, neural_idx, model_idx=None, max_lag=20, min_datapts=10, summary_stat='mean'):
    nan_check(corr_mat)
    corr_mat_h, corr_mat_w = corr_mat.shape # rows are neural timepts, cols are model timepts
    neural_idx = np.array(neural_idx)
    if model_idx is None: # if no model idx is specified, choose all the models
        model_idx  = np.array(range(corr_mat_w))
    else:
        model_idx  = np.array(model_idx)
    # end if model_idx is None:
    
    lagplot = np.zeros(max_lag*2 + 1) # lagplot from -max_lag to +max_lag with the zero in the middle
    center = max_lag
    stat = choose_summary_stat(summary_stat)
    
    # compute each lag
    for tau in range(-max_lag, max_lag+1):
        diag_vals = []
        # build all possible aligned pairs
        for i_neu in neural_idx:
            i_mod = i_neu + tau  # model index candidate 
            if i_mod in model_idx:
                if 0 <= i_neu < corr_mat_h and 0 <= i_mod < corr_mat_w: # check bounds bc i_mod might be out of the matrix
                    diag_vals.append(corr_mat[i_neu, i_mod])
                # end if 0 <= i_neu < corr_mat_h and 0 <= i_mod < corr_mat_w:
            # end if i_mod in model_idx:
        # end for i_neu in neural_idx:

        if not diag_vals: # if there's nothing in there (because it didn't enter the 2nd if)
            raise IndexError("The maximum lag is larger than the selected matrix itself")
        elif len(diag_vals) < min_datapts:
            print_wise(f"The number of datapoints used to compute extreme offsets is < than {min_datapts}")
        # end if n_timepts < max_lag:
        lagplot[center - tau] = stat(diag_vals)
    # end for tau in range(-max_lag, max_lag+1):
    return lagplot



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


"""
multivariate_ou
Generates a multivariate Ornstein–Uhlenbeck (OU) process with independent
dimensions, each following an exponentially correlated stochastic trajectory.

The OU process is defined by:
    x[t] = A * x[t-1] + noise
where A = exp(-dt / corr_length) controls the decay of temporal correlations.

INPUT:
    - T: float -> Total duration of the process (in arbitrary time units).
    - dim: int -> Number of independent OU dimensions to generate.
    - dt: float -> Time step used to discretize the process.
    - corr_length: float -> Correlation length (τ). Larger τ → slower decay → more temporal autocorrelation.
    - sigma: float (default 1.0) -> Noise scale governing the variance of the stochastic term.

OUTPUT:
    - x: np.ndarray (N, dim) -> Multivariate OU process, where N = int(T / dt) is the number of timepoints.
          Each column corresponds to one OU dimension.
"""
def multivariate_ou(T, dim, dt, corr_length, sigma=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    N = int(T / dt)
    x = np.zeros((N, dim))

    alpha = dt / corr_length      # decay factor
    A = np.exp(-alpha)            # autocorrelation coefficient

    for t in range(1, N):
        noise = sigma * np.sqrt(1 - A**2) * rng.standard_normal(dim)
        x[t] = A * x[t-1] + noise

    return x
# EOF


"""
Returns True if the list or np.array is empty, False otherwise.
Works for both lists and NumPy arrays.
"""
def is_empty(x):
    if x is None:
        return True
    try:
        # Works for np.array
        return x.size == 0
    except AttributeError:
        # If no size attribute, fallback to len() (lists, tuples)
        return len(x) == 0
# EOF


"""
get_layer_output_shape
Computes the output shape (excluding batch size) of a specific layer 
from a given PyTorch feature extractor when applied to a dummy input 
image of size (1, 3, 224, 224).
INPUT:
- feature_extractor: torch.nn.Module -> A PyTorch model (typically a feature extractor created via torchvision.models.feature_extraction.create_feature_extractor)
                                        which outputs a dictionary of intermediate activations.
            
- layer_name: str -> The name of the layer for which the output shape is desired. This must be one of the keys returned by the feature_extractor.

OUTPUT:
- tmp_shape: Tuple(Int) -> A tuple representing the shape of the output tensor from the specified layer, excluding the batch dimension. For example,
                          (512, 7, 7) for a convolutional layer or (768,) for a transformer block.
            
Example Usage:
    >>> from torchvision.models import resnet18
    >>> from torchvision.models.feature_extraction import create_feature_extractor
    >>> model = resnet18(pretrained=True).eval()
    >>> feat_ext = create_feature_extractor(model, return_nodes=["layer1.0.relu_1"])
    >>> shape = get_layer_out_shape(feat_ext, "layer1.0.relu_1")
    >>> print(shape)
    (64, 56, 56)
"""
def get_layer_output_shape(feature_extractor, layer_name):
    device = get_device() 
    with torch.no_grad():
        in_proxy = torch.randn(1, 3, 224, 224).to(device)
        tmp_shape = feature_extractor(in_proxy)[layer_name].shape[1:]
    return tmp_shape
# EOF 



def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print_wise(f"device being used: {device}")
    return device
# EOF


"""
get_relevant_output_layers
Returns a list of layer names from a specified deep neural network model
that are approximately aligned with regions in the primate visual cortex
— namely V1, V4, and IT (inferotemporal cortex). These layers are selected
to enable brain-model comparisons or neuroscientific analyses of model representations.

INPUT:
model_name (str): 
    The name of the model architecture. Supported models include:
    - 'resnet18'
    - 'resnet50'
    - 'vgg16'
    - 'alexnet'
    - 'vit_b_16'
OUTPUT:
    List[str]: 
        A list of strings representing layer names in the model. These layers are chosen
        based on their approximate correspondence to stages in the visual processing hierarchy
        (e.g., early visual cortex V1, intermediate V4, and higher-level IT).

Example Usage:
    >>> layers = get_relevant_output_layers('resnet18')
    >>> print(layers)
    ['conv1', 'layer1.0.relu_1', 'layer1.1.relu_1', ..., 'avgpool']

    >>> layers = get_relevant_output_layers('vit_b_16')
    >>> print(layers)
    ['conv_proj', 'encoder.layers.encoder_layer_0.add_1', ..., 'heads.head']
"""
def get_relevant_output_layers(model_name, pkg='torchvision'):
    if model_name == 'resnet18':
        return [
            'conv1',                         
            'layer1.0.relu_1',               
            'layer1.1.relu_1',               
            'layer2.0.relu_1',               
            'layer2.1.relu_1',               
            'layer3.0.relu_1',               
            'layer3.1.relu_1',               
            'layer4.0.relu_1',               
            'layer4.1.relu_1',               
            'avgpool'                        
        ]
    if model_name == 'resnet50':
        return [
            'layer1.0.conv3',
            'layer1.1.conv3',
            'layer1.0.downsample.0', 
            'layer2.0.conv3',
            'layer2.1.conv3',
            'layer2.2.conv3',
            'layer2.3.conv3',
            'layer2.0.downsample.0', 
            'layer3.0.conv3',
            'layer3.1.conv3',
            'layer3.2.conv3',
            'layer3.3.conv3',
            'layer3.4.conv3',
            'layer3.5.conv3',
            'layer3.0.downsample.0', 
            'layer4.0.conv3',
            'layer4.1.conv3',
            'layer4.2.conv3',
            'layer4.0.downsample.0', 
        ]
    if model_name == 'vgg16':
        return [
            'features.0',       # conv1_1 (V1)
            'features.2',       # conv1_2
            'features.5',       # conv2_2
            'features.10',      # conv3_3
            'features.12',      # conv4_1
            'features.16',      # conv4_3
            'features.19',      # conv5_1
            'features.23',      # conv5_3
            'features.30',      # final conv
            'classifier.0'      # first FC layer
        ]
    if model_name == 'alexnet':
        return [
            'features.0',       # conv1
            'features.4',       # conv2
            'features.7',       # conv3
            'features.9',       # conv4
            'features.11',      # conv5
            'classifier.2',     # fc6
            'classifier.5'      # fc7
        ]
    if model_name == 'vit_b_16':
        return [
            'encoder.layers.encoder_layer_0.mlp',
            'encoder.layers.encoder_layer_1.mlp',
            'encoder.layers.encoder_layer_2.mlp',
            'encoder.layers.encoder_layer_3.mlp',
            'encoder.layers.encoder_layer_4.mlp',
            'encoder.layers.encoder_layer_5.mlp',
            'encoder.layers.encoder_layer_6.mlp',
            'encoder.layers.encoder_layer_7.mlp',
            'encoder.layers.encoder_layer_8.mlp',           
            'encoder.layers.encoder_layer_9.mlp',           
            'encoder.layers.encoder_layer_10.mlp',          
            'encoder.layers.encoder_layer_11.mlp',          
            'encoder.layers.encoder_layer_12.mlp',          
            'encoder.layers.encoder_layer_13.mlp',          
            'encoder.layers.encoder_layer_14.mlp',          
            'encoder.layers.encoder_layer_15.mlp',          
            'encoder.layers.encoder_layer_16.mlp',          
            'encoder.layers.encoder_layer_17.mlp',          
            'encoder.layers.encoder_layer_18.mlp',          
            'encoder.layers.encoder_layer_19.mlp',          
            'encoder.layers.encoder_layer_20.mlp',          
            'encoder.layers.encoder_layer_21.mlp',          
            'encoder.layers.encoder_layer_22.mlp',          
            'encoder.layers.encoder_layer_23.mlp',          
        ]
    if model_name == 'vit_l_16':
        if pkg=='torchvision':
            return [
                'encoder.layers.encoder_layer_0.mlp',
                'encoder.layers.encoder_layer_1.mlp',
                'encoder.layers.encoder_layer_2.mlp',
                'encoder.layers.encoder_layer_3.mlp',
                'encoder.layers.encoder_layer_4.mlp',
                'encoder.layers.encoder_layer_5.mlp',
                'encoder.layers.encoder_layer_6.mlp',
                'encoder.layers.encoder_layer_7.mlp',
                'encoder.layers.encoder_layer_8.mlp',           
                'encoder.layers.encoder_layer_9.mlp',           
                'encoder.layers.encoder_layer_10.mlp',          
                'encoder.layers.encoder_layer_11.mlp',          
                'encoder.layers.encoder_layer_12.mlp',          
                'encoder.layers.encoder_layer_13.mlp',          
                'encoder.layers.encoder_layer_14.mlp',          
                'encoder.layers.encoder_layer_15.mlp',          
                'encoder.layers.encoder_layer_16.mlp',          
                'encoder.layers.encoder_layer_17.mlp',          
                'encoder.layers.encoder_layer_18.mlp',          
                'encoder.layers.encoder_layer_19.mlp',          
                'encoder.layers.encoder_layer_20.mlp',          
                'encoder.layers.encoder_layer_21.mlp',          
                'encoder.layers.encoder_layer_22.mlp',          
                'encoder.layers.encoder_layer_23.mlp',          
            ]
        elif pkg=='timm':
            return [
                'blocks.0.mlp.fc2',
                'blocks.1.mlp.fc2',
                'blocks.2.mlp.fc2',
                'blocks.3.mlp.fc2',
                'blocks.4.mlp.fc2',
                'blocks.5.mlp.fc2',
                'blocks.6.mlp.fc2',
                'blocks.7.mlp.fc2',
                'blocks.8.mlp.fc2',
                'blocks.9.mlp.fc2',
                'blocks.10.mlp.fc2',
                'blocks.11.mlp.fc2',
                'blocks.12.mlp.fc2',
                'blocks.13.mlp.fc2',
                'blocks.14.mlp.fc2',
                'blocks.15.mlp.fc2',
                'blocks.16.mlp.fc2',
                'blocks.17.mlp.fc2',
                'blocks.18.mlp.fc2',
                'blocks.19.mlp.fc2',
                'blocks.20.mlp.fc2',
                'blocks.21.mlp.fc2',
                'blocks.22.mlp.fc2',
                'blocks.23.mlp.fc2',
                   ]

    if 'mobilenet_v3_large' in model_name:
        return ["features.6.block.0", "features.15.block.0", "features.6.block.1", "features.15.block.1", "features.6.block.2", "features.15.block.2", "features.6.block.3", "features.15.block.3", "classifier.0", "classifier.3"]
    raise ValueError(f"Model {model_name} not supported in `get_relevant_output_layers()`.")
# EOF


"""
subsample_RDM
Starting from a full RDM, it extracts a smaller RDM using a subset of indices.
1) Selects the rows corresponding to the provided indices
2) Selects the matching columns (same indices)
3) Preserves the pairwise structure of the original RDM

INPUT:
- RDM: np.ndarray (N, N) -> full square representational dissimilarity matrix
- indices: array-like (K,) -> indices of conditions/items to keep

OUTPUT:
- subsampled_RDM: np.ndarray (K, K) -> sub-RDM restricted to the selected indices
"""
def subsample_RDM(RDM, indices):
    subsampled_RDM = RDM[np.ix_(indices, indices)]
    return subsampled_RDM
# EOF


"""
decode_matlab_strings
Decodes MATLAB strings stored in a v7.3 .mat file (HDF5 format) into Python strings.
1) Iterates over HDF5 object references pointing to MATLAB char arrays
2) Reads the corresponding uint16 character codes
3) Converts character codes to Python characters and joins them into strings

INPUT:
- h5file: h5py.File -> open HDF5 file corresponding to a MATLAB v7.3 .mat file
- ref_array: np.ndarray -> array of HDF5 object references to MATLAB char arrays

OUTPUT:
- strings: list of str -> decoded MATLAB strings
"""
def decode_matlab_strings(h5file, ref_array):
    strings = []
    for ref in ref_array.squeeze():
        chars = h5file[ref][:]
        s = ''.join(chr(c) for c in chars.flatten()) # MATLAB chars are usually stored as Nx1 uint16
        strings.append(s)
    return strings
