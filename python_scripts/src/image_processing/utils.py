import sys, os
import cv2
import numpy as np

sys.path.append("..")
from general_utils.utils import print_wise

def get_video_dimensions(cap):
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return round(height), round(width), round(n_frames) # round to make them int


def read_video(paths, rank, fn, vid_duration=0):
    video_path = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos/{fn}" 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    height, width, n_frames = get_video_dimensions(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_duration == 0:
        frames_to_loop = n_frames
    else:
        frames_to_loop = round(vid_duration * fps)
    # end if vid_duration == 0:
    
    video = np.zeros((frames_to_loop, height, width, 3), dtype=np.uint8) # standard [B, H, W, C]
    counter = 0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for frame_idx in range(frames_to_loop):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {counter} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video[frame_idx, :, :, :] = frame
        counter += 1
    # end while True
    print_wise(f"{fn} read successfully", rank=rank)
    cap.release()
    return video
