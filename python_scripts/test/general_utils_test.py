import os, yaml, sys
import numpy as np
import pytest
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils import print_wise, get_lagplot, autocorr_mat

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
