import os, yaml, sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pytest
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils import print_wise, get_lagplot, autocorr_mat, create_RDM, spearman, compute_dRSA

# --- Tests for get_lagplot_checks (indirect because we don't export it) ---

def test_lagplot_checks_index_error():
    corr_mat = np.zeros((5, 5))
    with pytest.raises(IndexError):
        get_lagplot(corr_mat, max_lag=10, min_datapts=2, symmetric=False)

def test_lagplot_checks_warning_for_min_datapts(capfd): # captures file descriptors
    corr_mat = np.ones((8, 8))
    get_lagplot(corr_mat, max_lag=6, min_datapts=5, symmetric=False)
    captured = capfd.readouterr()
    assert "datapoints used to compute extreme offsets" in captured.out

def test_lagplot_checks_warning_for_nans(capfd):
    corr_mat = np.eye(5)
    corr_mat[1, 1] = np.nan
    get_lagplot(corr_mat, max_lag=2, min_datapts=2, symmetric=False)
    captured = capfd.readouterr()
    assert "There are nans" in captured.out


# --- Tests for autocorr_mat ---

def test_autocorr_mat_auto():
    data = np.random.randn(4, 10)
    corr_mat = autocorr_mat(data)
    assert corr_mat.shape == (10, 10)
    assert np.allclose(np.diag(corr_mat), 1, atol=1e-8)

def test_autocorr_mat_cross():
    data1 = np.random.randn(4, 10)
    data2 = np.random.randn(4, 10)
    corr_mat = autocorr_mat(data1, data2)
    assert corr_mat.shape == (10, 10)


# --- Tests for get_lagplot ---

def test_get_lagplot_symmetric():
    data = np.random.randn(4, 20)
    corr_mat = autocorr_mat(data)
    lagplot = get_lagplot(corr_mat, max_lag=5, min_datapts=3, symmetric=True)
    assert lagplot.shape == (6,)   # max_lag + 1
    assert not np.isnan(lagplot).all()
    assert lagplot[0] == 1

def test_get_lagplot_asymmetric():
    data1 = np.random.randn(4, 20)
    data2 = np.random.randn(4, 20)
    corr_mat = autocorr_mat(data1, data2)
    lagplot = get_lagplot(corr_mat, max_lag=5, min_datapts=3, symmetric=False)
    assert lagplot.shape == (11,)  # 2*max_lag + 1
    assert not np.isnan(lagplot).all()



# -----------------------------------------------------------
# TESTS FOR create_RDM
# -----------------------------------------------------------

def test_create_rdm_correlation_matches_pdist():
    """Check that correlation RDM matches scipy's pdist 'correlation'."""
    X = np.random.randn(5, 10)  # (features × datapoints)
    
    rdm_custom = create_RDM(X, metric='correlation')
    rdm_scipy  = pdist(X.T, metric='correlation')
    
    assert np.allclose(rdm_custom, rdm_scipy, atol=1e-12)


def test_create_rdm_output_length():
    """Check that the vectorized RDM has correct length."""
    X = np.random.randn(4, 6)
    rdm = create_RDM(X)
    
    n = X.shape[1]
    expected_len = n * (n - 1) // 2  # upper triangular
    assert rdm.shape[0] == expected_len


def test_create_rdm_custom_metric():
    """Check that passing a callable metric works."""
    def my_metric(a, b):
        return np.sum(np.abs(a - b))  # L1
    
    X = np.random.randn(3, 8)
    rdm = create_RDM(X, metric=my_metric)
    
    rdm_expected = pdist(X.T, metric=my_metric)
    assert np.allclose(rdm, rdm_expected)

# -----------------------------------------------------------
# TESTS FOR spearman
# -----------------------------------------------------------

def test_spearman_perfect_correlation():
    """Should be +1 for perfectly increasing monotonic relation."""
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 20, 30, 40])
    rho = spearman(x, y)
    assert np.isclose(rho, 1.0)


def test_spearman_perfect_negative_correlation():
    """Should be -1 for perfectly decreasing monotonic relation."""
    x = np.array([1, 2, 3])
    y = np.array([30, 20, 10])
    rho = spearman(x, y)
    assert np.isclose(rho, -1.0)


def test_spearman_against_scipy():
    """Compare against scipy.stats.spearmanr."""
    from scipy.stats import spearmanr

    x = np.random.randn(20)
    y = np.random.randn(20)

    rho_custom = spearman(x, y)
    rho_scipy, _ = spearmanr(x, y)

    assert np.isclose(rho_custom, rho_scipy, atol=1e-12)

# -----------------------------------------------------------
# TESTS FOR compute_dRSA
# -----------------------------------------------------------

def test_compute_dRSA_shape():
    """Check that output shape is (time, time)."""
    neural = np.random.randn(4, 5, 10)  # channels, time, trials
    model  = np.random.randn(4, 5, 10)
    
    dRSA = compute_dRSA(neural, model)
    
    assert dRSA.shape == (5, 5)


def test_compute_dRSA_metric_RDM_model_none_uses_same_metric():
    """Ensure metric_RDM_model=None makes neural + model use same metric."""
    neural = np.random.randn(3, 4, 6)
    model  = np.random.randn(3, 4, 6)

    dRSA1 = compute_dRSA(neural, model, metric_RDM='correlation', metric_RDM_model=None)
    dRSA2 = compute_dRSA(neural, model, metric_RDM='correlation', metric_RDM_model='correlation')

    assert np.allclose(dRSA1, dRSA2)


def test_compute_dRSA_with_custom_metric():
    """Check that custom metric for dRSA works."""
    def my_metric(a, b):
        return np.dot(a, b)  # simple similarity

    neural = np.random.randn(4, 5, 6)
    model  = np.random.randn(4, 5, 6)

    dRSA = compute_dRSA(neural, model, metric_dRSA=my_metric)

    assert dRSA.shape == (5, 5)
