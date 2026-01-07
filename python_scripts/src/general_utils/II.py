import os, yaml, sys
import numpy as np
from scipy.spatial.distance import squareform
sys.path.append("..")
from general_utils.utils import RSA

"""
InformationImbalance
Computes the Information Imbalance (II) between two representational spaces
according to Glielmo et al. 2022 and Del Tatto et al. 2024.
Computes asymmetric information imbalance based on distance ranks
derived from representational dissimilarity matrices (RDMs).
About the computation of distance ranks...

compute_distance_ranks
Creates a matrix of distance ranks from an RDM. 
I.e. the element (i,j) will tell you the rank of the point i 
with respect to the point j. So the ith column tells you the
distance rank of the points with respect to the ith point.
The matrix is not symmetric anymore and only the columns are 
interpretable. The diagonal ranks are filled with N+1 vals.
We take two argsort because 
- 1st argsort gives you the row-wise (i.e. each column is treated 
independently) index that if, applied to the original vector, 
would give you a vector with the values of the original vector 
in increasing order.  
So here in each column: the position in the vector gives you the
value rank, while its value the index in the original vector.
- 2nd argsort gives you the distance rank of the element i (row) in
the neighborhood of the element j (col).
In vector terms, it yields the vector that if applied to the output
of the first argsort, would give you np.arange(0, N) (the order of 
the ranks). So the returned vector has again the value rank in the
position and the index in its value, but since it's applied to the 
output of the first argsort, it returns the distance ranks.
E.g. (This is applied to every column of RDM)
Input vector: [3,5,4,2] 
Output 1st argsort: [3,0,2,1]
Output 2nd argsort: [1,3,2,0]
2 is the 1st NN, 3 is the 2nd, etc...

OUTPUT:
- ranks: np.ndarray (N,N) -> Interpretable columns, the ith element in column j 
is the distance rank of point i in the neighborhood of j.
- kmins: nd.ndarray (k, N) -> The indices of the k nearest neighbors (rows) for each points (cols)
"""
class InformationImbalance(RSA):
    def __init__(self, 
            signal_RDM_metric: str, 
            model_RDM_metric: str, 
            k: int = 1,
            RSA_metric: str='correlation', 
            signal_RDM: np.ndarray = None, 
            model_RDM: np.ndarray = None,
            ):
        self.k = k
        super().__init__(signal_RDM_metric, model_RDM_metric, RSA_metric, signal_RDM, model_RDM)
        
    # --- GETTERS ---
    def get_distance_ranks(self, RDM_type: str):
        return getattr(self, f"{RDM_type}_distance_ranks")
    # EOF    
    def get_kmins_idx(self, RDM_type: str):
        return getattr(self, f"{RDM_type}_kmins_idx")
    # EOF    
    def get_II(self, II_type: str):
        return getattr(self, f"II_{II_type}")
    # EOF    

    # --- SETTERS ---
    def set_distance_ranks_and_kmins(self, distance_ranks: np.ndarray[int], kmins_idx: np.ndarray[int], RDM_type: str):
        self.N = distance_ranks.shape[0]
        setattr(self, "{RDM_type}_distance_ranks", distance_ranks)
        setattr(self, "{RDM_type}_kmins_idx", kmins_idx)
    # EOF

    # --- OTHER FUNCTIONS ---
    def compute_distance_ranks(self, RDM_type: str):
        RDM = squareform(self.get_RDM(RDM_type))
        self.N = RDM.shape[0] # include it also in the setter
        np.fill_diagonal(RDM, np.inf)
        order = np.argsort(RDM, axis=0)
        # stores the indices of the k mins so that we don't have to compute the argmin later
        kmins = order[:self.k, :]
        setattr(self, f"{RDM_type}_kmins_idx", kmins) 
        ranks = np.argsort(order, axis=0)
        ranks = ranks + 2*np.eye(ranks.shape[0])
        setattr(self, f"{RDM_type}_distance_ranks", ranks)
        return ranks, kmins
    # EOF
    def compute_both_distance_ranks(self):
        ranks_signal, mins_signal = self.compute_distance_ranks("signal")
        ranks_model, mins_model = self.compute_distance_ranks("model")
        return ranks_signal, mins_signal, ranks_model, mins_model
    # EOF
    def compute_II(self, II_type: str):
        if II_type == 'A2B':
            conditioning_var = 'signal'
            conditioned_var = 'model'
        elif II_type == 'B2A':
            conditioning_var = 'model'
            conditioned_var = 'signal'
        # end if II_type == 'A2B':
        conditioning_mins = getattr(self, f"{conditioning_var}_kmins_idx")
        to_be_conditioned_ranks = getattr(self, f"{conditioned_var}_distance_ranks")
        conditioned_ranks = np.take_along_axis(to_be_conditioned_ranks, conditioning_mins, axis=0)
        II = (2/(self.N**2 * self.k))*np.sum(conditioned_ranks)
        setattr(self, f"II_{II_type}", II)
        return II
    # EOF
    def compute_both_II(self):
        II_A2B = self.compute_II('A2B')
        II_B2A = self.compute_II('B2A')
        return II_A2B, II_B2A
    # EOF
# EOC



"""
compare_similarity_metrics
Compute Information Imbalance (II) between two similarity metrics to compute RDMs
INPUT
- data : np.ndarray (D, N) -> Input data used to compute both RDMs.
- metric1, metric2 : str -> Similarity / distance metrics for signal and model RDMs.
- k : int -> Number of nearest neighbors used for conditioning.

OUTPUT:
- ii_obj : InformationImbalance -> Initialized and fully computed InformationImbalance object.
- ii_A2B : float -> Information Imbalance from metric1 to metric2.
- ii_B2A : float -> Information Imbalance from metric2 to metric1.
"""
def compare_similarity_metrics(data: np.ndarray, metric1: str, metric2: str, k: int): 
    ii_obj = InformationImbalance(metric1, metric2, k)
    ii_obj.compute_RDM(data, "signal")
    ii_obj.compute_RDM(data, "model")
    ii_obj.compute_both_distance_ranks()
    ii_A2B, ii_B2A = ii_obj.compute_both_II()
    return ii_obj, ii_A2B, ii_B2A
# EOF
