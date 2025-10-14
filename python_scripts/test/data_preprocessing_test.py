import os, yaml, sys
import numpy as np

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from data_preprocessing.preprocessing import create_bins, get_firing_rate, cut_excess_timepoints, convert_gaze_coordinates

def test_create_bins_basic():
    bins = create_bins(trial_duration=10, len_avg_window=2.5)
    assert np.array_equal(bins, np.array([0, 2, 5, 8, 10])) 
    assert bins.dtype == int


def test_create_bins_exact_division():
    bins = create_bins(10, 2)
    # 0, 2, 4, 6, 8, 10
    assert np.allclose(bins, np.arange(0, 11, 2)) 

def test_create_bins_tiny_step():
    bins = create_bins(1, 0.1)
    assert len(bins) == 11
    assert bins[0] == 0 and bins[-1] == 1


def test_get_firing_rate_shape():
    bins = np.array([0, 2, 4, 6])
    neural_signal = np.arange(12).reshape(2, 6)  # 2 neurons × 6 timepoints
    fr = get_firing_rate(bins, neural_signal)
    assert fr.shape == (2, 3)  # 3 bins (col) × 2 neurons (row)
    # check values (means)
    expected = np.array([
        [0.5, 2.5, 4.5],
        [6.5, 8.5, 10.5]
    ])
    np.testing.assert_allclose(fr, expected)


def test_get_firing_rate_single_bin():
    bins = np.array([0, 4])
    neural_signal = np.ones((3, 4))
    fr = get_firing_rate(bins, neural_signal)
    assert np.all(fr == 1)

def test_cut_excess_timepoints_basic():
    data_dict = {
        "stim1": [
            np.ones((2, 5)),
            np.ones((2, 4))
        ]
    }
    out = cut_excess_timepoints(data_dict)
    assert isinstance(out["stim1"], np.ndarray)
    assert out["stim1"].shape == (2, 4, 2)  # trimmed to shortest (4 timepoints)
    assert np.all(out["stim1"] == 1)


def test_cut_excess_timepoints_equal_lengths():
    data_dict = {
        "stim1": [np.zeros((1, 3)), np.zeros((1, 3))]
    }
    out = cut_excess_timepoints(data_dict)
    assert out["stim1"].shape == (1, 3, 2)


def test_cut_excess_timepoints_already_stacked(capsys):
    data_dict = {"stim1": np.zeros((2, 3, 4))}
    out = cut_excess_timepoints(data_dict)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert np.all(out["stim1"] == 0)


def test_convert_gaze_coordinates_center():
    gaze = np.array([[0], [0]], dtype=float)
    out = convert_gaze_coordinates(gaze.copy())
    assert np.allclose(out, np.array([[960], [540]]))  # center


def test_convert_gaze_coordinates_offsets():
    gaze = np.array([[1, -1], [1, -1]], dtype=float)
    out = convert_gaze_coordinates(gaze.copy())
    expected = np.array([
        [960 + 32, 960 - 32],
        [540 - 32, 540 + 32]
    ])
    np.testing.assert_allclose(out, expected)
