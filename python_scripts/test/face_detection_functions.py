import sys
import numpy as np
import pytest
sys.path.append('../src')
from image_processing.computational_models import (  # replace with your actual filename
    make_corners, get_height_width, compute_center,
    make_square_box, scale_box, estimate_head_box,
    smooth_face_detection
)

def test_make_corners():
    x1, y1, x2, y2 = 0, 0, 10, 10
    points = make_corners(x1, y1, x2, y2)
    expected = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
    ])
    assert np.allclose(points, expected)


def test_get_height_width():
    h, w = get_height_width(0, 0, 10, 5)
    assert h == 5
    assert w == 10 


def test_compute_center():
    cx, cy = compute_center(0, 0, 10, 10)
    assert cx == 5
    assert cy == 5


def test_make_square_box_wider_than_tall():
    # Input wider than tall → should expand height
    x1, y1, x2, y2 = 0, 0, 10, 5
    new_x1, new_y1, new_x2, new_y2 = make_square_box(x1, y1, x2, y2)
    side = 10  # max(w, h)
    assert np.isclose(new_x2 - new_x1, side)
    assert np.isclose(new_y2 - new_y1, side)
    # Center remains same
    cx, cy = compute_center(x1, y1, x2, y2)
    new_cx, new_cy = compute_center(new_x1, new_y1, new_x2, new_y2)
    assert np.isclose(cx, new_cx)
    assert np.isclose(cy, new_cy)


def test_scale_box_increases_size():
    x1, y1, x2, y2 = 0, 0, 10, 10
    scale = 1.5
    new_x1, new_y1, new_x2, new_y2 = scale_box(x1, y1, x2, y2, scale)
    assert np.isclose(new_x2 - new_x1, 15)
    assert np.isclose(new_y2 - new_y1, 15)


def test_estimate_head_box_top_and_center():
    x1, y1, x2, y2 = 0, 0, 10, 10
    new_x1, new_y1, new_x2, new_y2 = estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5)
    assert np.isclose(new_y1, 0)
    assert np.isclose(new_y2, 4)
    assert np.isclose(new_x1, 2.5)
    assert np.isclose(new_x2, 7.5)


def test_smooth_face_detection_fixes_isolated_gaps():
    # coords[0, :] represents face_presence; other rows are dummy values
    coords = np.array([
        [1, 0, 1],  # isolated zero should be fixed
        [0.9, 0.0, 0.8],
        [10, 20, 30],
        [11, 21, 31],
        [12, 22, 32],
        [13, 23, 33]
    ])
    smoothed = smooth_face_detection(coords.copy())
    assert np.all(smoothed[0, :] == [1, 1, 1])
    # middle column should copy previous coordinates
    assert np.allclose(smoothed[1:, 1], coords[1:, 0])
