import os, yaml, sys
import numpy as np

def make_corners(x1, y1, x2, y2):
    points = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
        [x1, y1]   # close
    ])
    return points

def get_height_width(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return h, w

def compute_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def make_square_box(x1, y1, x2, y2):
    h, w = get_height_width(x1, y1, x2, y2)
    cx, cy = compute_center(x1, y1, x2, y2)
    side = max(w, h) # make it square by expanding the smaller dimension
    # New coordinates (keep center fixed)
    new_x1 = cx - side / 2
    new_x2 = cx + side / 2
    new_y1 = cy - side / 2
    new_y2 = cy + side / 2
    return new_x1, new_y1, new_x2, new_y2

def scale_box(x1, y1, x2, y2, scale):
    h, w = get_height_width(x1, y1, x2, y2)
    cx, cy = compute_center(x1, y1, x2, y2)
    new_h = h * scale
    new_w = w * scale
    new_x1 = cx - new_w / 2
    new_x2 = cx + new_w / 2
    new_y1 = cy - new_h / 2
    new_y2 = cy + new_h / 2
    return new_x1, new_y1, new_x2, new_y2


def estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5):
    h, w = get_height_width(x1, y1, x2, y2)
    new_y1 = y1
    new_y2 = y1 + top_h * h
    x_center = (x1 + x2) / 2
    new_width = cnt_width * w
    new_x1 = x_center - new_width / 2
    new_x2 = x_center + new_width / 2
    return new_x1, new_y1, new_x2, new_y2

def compute_face_box(frame, face_model, scale):
    results = face_model(frame, classes=[0], verbose=False)  # Class 0 = person
    if len(results[0].boxes) == 0:
        raise IndexError("No detections found.")
    confidence = round(results[0].boxes[0].conf[0].item(), 3)
    if confidence < 0.75:
        raise IndexError
    # end if confidence < 0.75:
    box = results[0].boxes.xyxy[0] # INCREASE THESE BOUNDING BOXES
    x1, y1, x2, y2 = box
    new_x1, new_y1, new_x2, new_y2 = make_square_box(x1, y1, x2, y2)
    new_x1, new_y1, new_x2, new_y2 = scale_box(new_x1, new_y1, new_x2, new_y2, scale)
    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])
    return new_x1, new_y1, new_x2, new_y2, confidence

def compute_occluded_face_box(frame, person_model):
    results = person_model(frame, classes=[0], verbose=False)
    if len(results[0].boxes) == 0:
        raise IndexError("No detections found.")
    confidence = round(results[0].boxes[0].conf[0].item(), 3)
    if confidence < 0.7:
        raise IndexError
    # end if confidence < 0.7:
    x1, y1, x2, y2 = results[0].boxes.xyxy[0]
    # Original width and height
    new_x1, new_y1, new_x2, new_y2 = estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5)
    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])
    return new_x1, new_y1, new_x2, new_y2, confidence


def smooth_face_detection(coords):
    for i in range(1, coords.shape[1] - 1):
        if coords[0, i-1] == coords[0, i+1] and coords[0, i] != coords[0, i-1]:
            if coords[0, i] == 0: # if the model didn't detect any face but still before and after it did, we plug the previous coords
                coords[1:, i] = coords[1:, i-1]
            # end if a[0, i] == 0:
            coords[0, i] = coords[0, i-1]
        # end if a[0, i-1] == a[0, i+1] and a[0, i] != a[0, i-1]:
    # end for i in range(1, a.shape[1] - 1):
    return coords

def detect_faces(video, face_model, person_model, scale):
    coords = []
    for i in range(video.shape[0]):
        frame = video[i,:,:,:]
        try:
            x1, y1, x2, y2, confidence = compute_face_box(frame, face_model, scale=scale)
            face_presence = 1 # face present
            confidence = round(confidence, 3)
        except IndexError:
            try:
                x1, y1, x2, y2, confidence = compute_occluded_face_box(frame, person_model)
                confidence = round(confidence, 3)
                face_presence = 2 # occluded
            except IndexError:
                x1, y1, x2, y2, confidence = None, None, None, None, None
                face_presence = 0 # face absent
            # end except IndexError:
        # end except IndexError:
        coords.append(np.array([face_presence, confidence, x1, y1, x2, y2]));
    # end for i in range(video.shape[0]):
    coords = np.stack(coords, axis=1)
    coords = smooth_face_detection(coords)
    return coords
# EOF
