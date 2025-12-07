import os, yaml, sys
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
import pytest
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import multivariate_ou
import pytest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Import your functions here
from general_utils.regression import (
    shift_xy,
    shift_concatenate_xy,
    IdentitySplit,
    choose_CV_type,
    choose_regression_type,
    evaluate_prediction,
    lagged_linear_regression
)

# ============================================================
# shift_xy
# ============================================================
def test_shift_xy_positive_lag():
    x = np.arange(10).reshape(1, -1)
    y = np.arange(10, 20).reshape(1, -1)

    x_s, y_s = shift_xy(x, y, tau=2)

    assert x_s.shape == (1, 8)
    assert y_s.shape == (1, 8)
    assert np.all(x_s == x[:, :-2])
    assert np.all(y_s == y[:, 2:])


def test_shift_xy_zero_lag():
    x = np.random.randn(3, 5)
    y = np.random.randn(3, 5)

    x_s, y_s = shift_xy(x, y, tau=0)

    assert np.all(x_s == x)
    assert np.all(y_s == y)


def test_shift_xy_negative_lag():
    x = np.arange(10).reshape(1, -1)
    y = np.arange(10, 20).reshape(1, -1)

    x_s, y_s = shift_xy(x, y, tau=-2)

    assert x_s.shape == (1, 8)
    assert y_s.shape == (1, 8)
    assert np.all(x_s == x[:, 2:])
    assert np.all(y_s == y[:, :-2])


# ============================================================
# shift_concatenate_xy
# ============================================================
def test_shift_concatenate_xy_shapes():
    x = np.random.randn(3, 10, 4)  # 4 trials
    y = np.random.randn(3, 10, 4)

    x_s, y_s = shift_concatenate_xy(x, y, tau=1)

    # Each trial contributes 9 overlapping timepoints → 4*9 rows
    assert x_s.shape == (4 * 9, 3)
    assert y_s.shape == (4 * 9, 3)


def test_shift_concatenate_xy_values():
    x = np.ones((1, 5, 2))
    y = np.zeros((1, 5, 2))

    x_s, y_s = shift_concatenate_xy(x, y, tau=1)

    # After shifting by 1, each trial yields 4 rows (transpose of shape 1x4)
    assert np.all(x_s == 1)
    assert np.all(y_s == 0)


# ============================================================
# IdentitySplit
# ============================================================
def test_identitysplit_no_shuffle():
    X = np.random.randn(10, 3)
    cv = IdentitySplit(shuffle=False)

    for train_idx, test_idx in cv.split(X):
        assert np.all(train_idx == np.arange(10))
        assert np.all(test_idx == np.arange(10))


def test_identitysplit_shuffle():
    X = np.random.randn(10, 3)
    cv = IdentitySplit(shuffle=True, random_state=0)

    for train_idx, test_idx in cv.split(X):
        assert np.all(train_idx == test_idx)
        # Must be a permutation
        assert set(train_idx) == set(range(10))


# ============================================================
# choose_CV_type
# ============================================================
def test_choose_CV_type_same():
    CV = choose_CV_type('same', shuffle=False)
    assert isinstance(CV, IdentitySplit)


def test_choose_CV_type_kf():
    CV = choose_CV_type('kf', n_splits=3, shuffle=True)
    from sklearn.model_selection import KFold
    assert isinstance(CV, KFold)


def test_choose_CV_type_loo():
    CV = choose_CV_type('loo')
    from sklearn.model_selection import LeaveOneOut
    assert isinstance(CV, LeaveOneOut)


def test_choose_CV_type_invalid():
    with pytest.raises(ValueError):
        choose_CV_type("invalid")


# ============================================================
# choose_regression_type
# ============================================================
def test_choose_regression_type_lr():
    assert isinstance(choose_regression_type('lr'), LinearRegression)


def test_choose_regression_type_ridge():
    assert isinstance(choose_regression_type('ridge', alpha=1.0), Ridge)


def test_choose_regression_type_lasso():
    assert isinstance(choose_regression_type('lasso', alpha=1.0), Lasso)


def test_choose_regression_type_en():
    assert isinstance(choose_regression_type('en', alpha=0.5), ElasticNet)


def test_choose_regression_type_invalid():
    with pytest.raises(ValueError):
        choose_regression_type("invalid")


# ============================================================
# evaluate_prediction
# ============================================================
def test_evaluate_prediction_perfect_corr():
    X = np.random.randn(20, 3)
    y = np.dot(X, np.array([[1, 2], [0, -1], [3, 1]]))  # deterministic linear mapping

    reg = LinearRegression().fit(X, y)
    avg_corr = evaluate_prediction(X, y, reg)

    assert np.isclose(avg_corr, 1.0, atol=1e-6)


def test_evaluate_prediction_worst_case():
    X = np.random.randn(20, 3)
    y = -np.random.randn(20, 2)

    reg = LinearRegression().fit(X, y)
    avg_corr = evaluate_prediction(X, y, reg)

    assert -1.0 <= avg_corr <= 1.0  # correlation still valid


# ============================================================
# lagged_linear_regression
# ============================================================
def test_lagged_linear_regression_basic():
    x = np.random.randn(2, 20, 3)
    y = x.copy()  # perfect predictability

    lr_list = lagged_linear_regression(
        x, y,
        regression_type='lr',
        cv_type='same',
        max_lag=2,
        symmetric=False
    )

    # Range -2, -1, 0, 1, 2 → 5 values
    assert len(lr_list) == 5
    # Should be high correlations
    assert np.allclose(lr_list[2], 1.0)


def test_lagged_linear_regression_symmetric():
    x = np.random.randn(1, 15, 2)
    y = x.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_list = lagged_linear_regression(
            x, y,
            symmetric=True,
            max_lag=3
        )

    # lags 0, 1, 2, 3 → 4 values
    assert len(lr_list) == 4


def test_lagged_linear_regression_invalid_regression_type():
    x = np.random.randn(1, 10, 2)
    y = np.random.randn(1, 10, 2)
    with pytest.raises(ValueError):
        lagged_linear_regression(x, y, regression_type="invalid")


def test_pick_positive_lag():
    tot_x = []
    for i in range(30):
        x = multivariate_ou(T=10.0, dim=10, dt=0.1, corr_length=0.5, random_state=42)
        tot_x.append(x.T)
    neural_data = np.stack(tot_x, axis=-1)
    model_data = np.stack(tot_x, axis=-1)
# inserts a datapoint at the beginning of the time-series and so the model is lagging behind the neural data (prediction)
    zeros = np.zeros((model_data.shape[0], 1, model_data.shape[2])) 
    model_data = np.concatenate((zeros, model_data), axis=1)[:,:-1, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_list = lagged_linear_regression(model_data, neural_data, regression_type='lr', cv_type='kf', n_splits=5, alpha=1.0, max_lag=10)
    assert np.allclose(lr_list[9], 1.0) # tests to see if the neural data is predicting the model data with 1 datapoint (at idx=max_lag we have tau=0)
