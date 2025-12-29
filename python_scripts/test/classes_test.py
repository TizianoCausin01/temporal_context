import os, yaml, sys
import numpy as np
import pytest

from einops import reduce
from scipy.spatial.distance import squareform

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import TimeSeries, RSA, dRSA

def test_timeseries_init_numpy():
    x = np.random.randn(5, 10, 3)
    ts = TimeSeries(x, fs=1000)
    assert ts.type == "np"
    assert ts.get_fs() == 1000


def test_timeseries_init_list():
    x = [np.random.randn(5, 3) for _ in range(10)]
    ts = TimeSeries(x, fs=500)
    assert ts.type == "list"
    assert len(ts) == 10


def test_timeseries_init_invalid_type():
    with pytest.raises(TypeError):
        TimeSeries("not valid", fs=1000)


def test_timeseries_len_numpy():
    x = np.random.randn(4, 12, 2)
    ts = TimeSeries(x, 1000)
    assert len(ts) == 12


def test_timeseries_iter_numpy():
    x = np.random.randn(4, 6, 2)
    ts = TimeSeries(x, 1000)
    for t, frame in enumerate(ts):
        assert np.all(frame == x[:, t, :])


def test_timeseries_getitem_list():
    x = [np.random.randn(4, 2) for _ in range(5)]
    ts = TimeSeries(x, 1000)
    assert np.all(ts[2] == x[2])


def test_timeseries_to_numpy():
    x = [np.random.randn(3, 2) for _ in range(4)]
    ts = TimeSeries(x, 1000)
    ts.to_numpy()
    assert ts.type == "np"
    assert ts.array.shape == (3, 4, 2)


def test_trial_avg():
    x = np.random.randn(5, 10, 3)
    ts = TimeSeries(x, 1000)
    avg = ts.trial_avg()
    assert avg.shape == (5, 10)


def test_trial_avg_missing_trial_dim():
    x = np.random.randn(5, 10)
    ts = TimeSeries(x, 1000)
    with pytest.raises(Exception):
        ts.trial_avg()


def test_neurons_avg():
    x = np.random.randn(5, 10, 3)
    ts = TimeSeries(x, 1000)
    avg = ts.neurons_avg()
    assert avg.shape == (1, 10, 3)


def test_overall_avg():
    x = np.random.randn(5, 10, 3)
    ts = TimeSeries(x, 1000)
    avg = ts.overall_avg()
    assert avg.shape == (10,)


def test_autocorr_runs():
    with pytest.raises(Exception):
        x = np.random.randn(5, 20, 3)
        ts = TimeSeries(x, 1000)
        ac_mat, ac_line = ts.autocorr()


def test_lagged_corr_runs():
    with pytest.raises(Exception):
        x1 = np.random.randn(5, 20, 3)
        x2 = np.random.randn(5, 20, 3)
        ts1 = TimeSeries(x1, 1000)
        ts2 = TimeSeries(x2, 1000)
        ac = ts1.lagged_corr(ts2)


def test_rsa_init_defaults():
    rsa = RSA("cosine")
    assert rsa.signal_RDM_metric == "cosine"
    assert rsa.model_RDM_metric == "cosine"
    assert rsa.RSA_metric == "correlation"


def test_compute_signal_rdm():
    data = np.random.randn(10, 5)
    rsa = RSA("cosine")
    rdm = rsa.compute_RDM(data, "signal")
    assert rdm.ndim == 1
    assert rsa.signal_RDM is not None


def test_compute_both_rdms():
    x = np.random.randn(10, 5)
    y = np.random.randn(10, 5)
    rsa = RSA("cosine")
    rsa.compute_both_RDMs(x, y)
    assert rsa.signal_RDM is not None
    assert rsa.model_RDM is not None

def test_compute_rsa_correlation():
    x = np.random.randn(10, 5)
    rsa = RSA("cosine", RSA_metric="correlation")
    rsa.compute_both_RDMs(x, x)
    sim = rsa.compute_RSA()
    assert np.isclose(sim, 1.0)


def test_compute_rsa_spearman():
    x = np.random.randn(10, 5)
    rsa = RSA("cosine", RSA_metric="spearman")
    rsa.compute_both_RDMs(x, x)
    sim = rsa.compute_RSA()
    assert np.isclose(sim, 1.0)


def test_compute_rdm_timeseries():
    x = np.random.randn(5, 8, 3)
    ts = TimeSeries(x, 1000)
    drsa = dRSA("cosine")
    drsa.compute_RDM_timeseries(ts, "signal")

    rdm_ts = drsa.signal_RDM_timeseries
    assert isinstance(rdm_ts, TimeSeries)
    assert len(rdm_ts) == 8



def test_compute_drsa():
    x = np.random.randn(5, 8, 3)
    ts = TimeSeries(x, 1000)
    drsa = dRSA("cosine")

    drsa.compute_both_RDM_timeseries(ts, ts)
    mat = drsa.compute_dRSA()

    assert mat.ndim == 2


def test_compute_static_drsa():
    x = np.random.randn(5, 8, 3)
    ts = TimeSeries(x, 1000)

    drsa = dRSA("cosine", RSA_metric="correlation")
    drsa.compute_RDM_timeseries(ts, "signal")

    model_rdm = drsa.signal_RDM_timeseries[0]
    drsa.model_RDM = model_rdm

    static = drsa.compute_static_dRSA()
    assert isinstance(static, TimeSeries)
    assert len(static) == len(ts)
