import os, yaml, sys
import numpy as np
import torch
import cv2
from scipy.io import savemat
from scipy.ndimage import zoom
from scipy.special import logsumexp
sys.path.append("../DeepGaze")
import deepgaze_pytorch

sys.path.append("..")
from image_processing.utils import read_video
from general_utils.utils import print_wise


"""
make_corners
For plotting the coords on a picture
"""
def make_corners(x1, y1, x2, y2):
    points = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
        [x1, y1]   # close
    ])
    return points


"""
get_height_width
To get the height and width of a box
"""
def get_height_width(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return h, w


"""
compute_center
To compute the center of a box
"""
def compute_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


"""
make_square_box
From a rectangular box, it makes it square. It's used in the context of face boxes before
scaling them.
"""
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


"""
scale_box
Increases/decreases proportionally the height and width of a box. scale is the factor with which to scale. scale > 1 , the box increases; scale < 1 the box decreases.
"""
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


"""
estimate_head_box
From a person box, it estimates the head position. The guesses are that it's gonna be at the top n% (top_h) along the y-axis and in the center (cnt_width) of the x-axis.
"""
def estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5):
    h, w = get_height_width(x1, y1, x2, y2)
    new_y1 = y1
    new_y2 = y1 + top_h * h
    x_center = (x1 + x2) / 2
    new_width = cnt_width * w
    new_x1 = x_center - new_width / 2
    new_x2 = x_center + new_width / 2
    return new_x1, new_y1, new_x2, new_y2


"""
compute_face_box
Computes the face box given a face model and a scaling factor (see scale_box). 
In case of no detections found or confidence < .75 , it will raise an IndexError that will be caught by detect_faces.
face_model is usually imported like this:

from huggingface_hub import hf_hub_download
face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(face_model_path)
"""
def compute_face_box(frame, face_model, scale):
    results = face_model(frame, classes=[0], verbose=False)  # Class 0 = person
    if len(results[0].boxes) == 0:
        raise IndexError("No detections found.")
    confidence = round(results[0].boxes[0].conf[0].item(), 3)
    if confidence < 0.75:
        raise IndexError
    # end if confidence < 0.75:
    box = results[0].boxes.xyxy[0] 
    x1, y1, x2, y2 = box
    new_x1, new_y1, new_x2, new_y2 = make_square_box(x1, y1, x2, y2)
    new_x1, new_y1, new_x2, new_y2 = scale_box(new_x1, new_y1, new_x2, new_y2, scale)
    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])
    return new_x1, new_y1, new_x2, new_y2, confidence


"""
compute_occluded_face_box
Computes a face box in the case of no face detection by the face_model. It estimates the approximate position of a face based on the person box 
"""
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


"""
smooth_face_detection
Takes care of the situations in which there is a too abrupt change in the face detection
INPUT:
    - coords: np.ndarray (6, time) -> its rows are these [face_presence, confidence, x1, y1, x2, y2] , face presence being 0 if no face was detected, 1 if a face was detected by a person model, 2 if a face was detected by the person model
"""
def smooth_face_detection(coords):
    for i in range(1, coords.shape[1] - 1): # loops over all the timepoints except from the 0th and the last, because they don't have either a previous or a successive point
        if coords[0, i-1] == coords[0, i+1] and coords[0, i] != coords[0, i-1]: # if the previous and successive points are the same, but still they are different from the point to be considered, then we gotta change the current point
            if coords[0, i] == 0: # if the model didn't detect any face but still before and after it did, we plug the previous coords
                coords[1:, i] = coords[1:, i-1]
            # end if a[0, i] == 0:
            coords[0, i] = coords[0, i-1]
        # end if a[0, i-1] == a[0, i+1] and a[0, i] != a[0, i-1]:
    # end for i in range(1, a.shape[1] - 1):
    return coords


"""
detect_faces
Loops through all the frames of a video and detects one face for each of them. If the face model doesn't reliably detect any face, it falls back to the person model, and if no person is detected with high confidence, it says that there is no face. face_model detections are normal faces, person_model detections are occluded faces. 
INPUT:
    - video: np.ndarray (t, h, w, c) -> the video, read with read_video in image_processing.utils
    - face_model -> usually imported like this:
          from huggingface_hub import hf_hub_download
          face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
          face_model = YOLO(face_model_path)
    - person_model -> usually imported like this:
          person_model_path = f'{path_to_weights}/yolov8n.pt'
          person_model = YOLO(person_model_path)
    - scale: float -> the degree to which we have to increase the face box detected by the face model

OUTPUT:
    - coords: np.ndarray (6, t) -> its rows are these [face_presence, confidence, x1, y1, x2, y2] , face presence being 0 if no face was detected, 1 if a face was detected by a person model, 2 if a face was detected by the person model

"""
def detect_faces(video, face_model, person_model, scale):
    coords = []
    for i in range(video.shape[0]):
        frame = video[i,:,:,:]
        try: # tries to detect a face with the face_model
            x1, y1, x2, y2, confidence = compute_face_box(frame, face_model, scale=scale)
            face_presence = 1 # face present
            confidence = round(confidence, 3)
        except IndexError: # if face_model detects no face
            try: # tries  to detect a face with the person_model
                x1, y1, x2, y2, confidence = compute_occluded_face_box(frame, person_model)
                confidence = round(confidence, 3)
                face_presence = 2 # occluded
            except IndexError: # if person_model detects no person
                x1, y1, x2, y2, confidence = np.nan, np.nan, np.nan, np.nan, np.nan 
                face_presence = 0 # face absent
            # end except IndexError:
        # end except IndexError:
        coords.append(np.array([face_presence, confidence, x1, y1, x2, y2])); # updates coords
    # end for i in range(video.shape[0]):
    coords = np.stack(coords, axis=1) # converts it into an array
    coords = smooth_face_detection(coords) # smooths it so that we don't have isolated points
    return coords
# EOF


"""
par_detect_faces
Another version for the function above adapted for parallel processing
"""
def par_detect_faces(paths, rank, fn, face_model, person_model, scale):
    outfn = f"{paths['livingstone_lab']}/tiziano/models/human_face_detection_{fn[:-4]}.mat" # [:-4] slice to take off the mp4 extension
    if os.path.exists(outfn):
        print_wise(f"model already exists at {outfn}", rank=rank)
        return None
    else:
        video = read_video(paths, rank, fn, vid_duration=0)
        coords = detect_faces(video, face_model, person_model, scale)
        print_wise(f"model saved at {outfn}", rank=rank)
        savemat(outfn, {"coords": coords})


"""
compute_centerbias
Creates centerbias for deepgaze by loading it from the model_weights folder
"""
def compute_centerbias(paths, h, w):
    centerbias_template = np.load(f'{paths['livingstone_lab']}/tiziano/model_weights/deep_gaze/centerbias_mit1003.npy') # downloaded at https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # rescale to match image size
    centerbias = zoom(centerbias_template, (h/centerbias_template.shape[0], w/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias = torch.from_numpy(centerbias) #.float().to(device)
    centerbias = centerbias.unsqueeze(0)
    return centerbias
#EOF


"""
prepare_dg_input
prepares the input for deep gaze such that it's [B C H W] and torch. B=1
"""
def prepare_dg_input(frame):
    input = frame.transpose(2,0,1)
    input = torch.from_numpy(input) #.float().to(device)
    input = input.unsqueeze(0)
    return input
#EOF


"""
dg_pass
Computes the deep gaze output and translates it back to probability from log probability.
"""
def dg_pass(input, model, centerbias, new_dims):
    with torch.no_grad():
        log_density_prediction = model(input, centerbias)
    log_density_prediction = cv2.resize(log_density_prediction.numpy().squeeze(), new_dims)
    d = np.exp(log_density_prediction)/np.sum(np.exp(log_density_prediction))
    d_flat = d.flatten(order="F")
    return d_flat
#EOF


"""
compute_dg_saliency
Wrapper to compute the deep-gaze visual saliency in parallel
"""
def compute_dg_saliency(paths, rank, fn, model, resize_factor):
    outfn = f"{paths['livingstone_lab']}/tiziano/models/dgIIE_{fn[:-4]}.mat" # [:-4] slice to take off the mp4 extension
    if os.path.exists(outfn):
        print_wise(f"model already exists at {outfn}", rank=rank)
        return None
    else: 
        video = read_video(paths, rank, fn)
        h, w = video.shape[1:3] 
        new_dims = (round(w*resize_factor), round(h*resize_factor))
        centerbias = compute_centerbias(paths, h, w)
        video_saliency = []
        for i_frame in range(video.shape[0]):
            current_frame = video[i_frame, :, :, :]
            input = prepare_dg_input(current_frame)
            dg_saliency = dg_pass(input, model, centerbias, new_dims)
            video_saliency.append(dg_saliency)
            if i%10 == 0: # such that every tenth frame it prints out the progression
                print_wise(f"frame {i_frame} computed", rank=rank)
        video_saliency = np.stack(video_saliency, axis=1)
        savemat(outfn, {"features" : video_saliency}) 
        print_wise(f"model saved at {outfn}", rank=rank)
#EOF
