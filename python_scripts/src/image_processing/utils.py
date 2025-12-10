import sys, os
import cv2
import numpy as np
from scipy.io import loadmat
sys.path.append("..")
from general_utils.utils import print_wise, get_upsampling_indices, is_empty

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



def load_stimuli_models(paths, model_name, file_names, resolution_Hz):
    all_models = {}
    models_path = f"{paths['livingstone_lab']}/tiziano/models"
    for fn in file_names:
        video_path = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos/{fn}" 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w, _ = get_video_dimensions(cap)
        if model_name=="human_face_detection":
            if (h != 1080) or (w != 1920):
                raise ValueError("The size of the movie is different to the way eye-tracking were prerpocessed")
            curr_model = loadmat(f"{models_path}/{model_name}_{fn[:-4]}.mat")['coords']
        else:
            curr_model = np.load(f"{models_path}/{model_name}_{fn[:-4]}.npz")['data']
        # end model_name=="human_face_detection":
        indices = get_upsampling_indices(curr_model.shape[1], fps, resolution_Hz)
        curr_model = curr_model[:, indices]
        all_models[fn] = curr_model
    return all_models


"""
Resize a video stored as a NumPy array of shape (n_frames, H, W, C).

INPUT:
    - video: np.ndarray -> (n_frames, H, W, C)
    - new_height: int -> desired output height
    - new_width: int -> desired output width
    - interpolation: cv2 interpolation method (default: bilinear)

OUTPUT:
    - resized_video: np.ndarray -> (n_frames, new_height, new_width, C)    
"""
def resize_video_array(video, new_height, new_width, interpolation=cv2.INTER_LINEAR, normalize=True):
    resized_video = np.stack([cv2.resize(frame, (new_width, new_height), interpolation=interpolation) for frame in video])
    if normalize:
        mean = resized_video.mean(axis=(0,1,2))
        std = resized_video.std(axis=(0,1,2)) + 1e-8
        resized_video = (resized_video - mean) / std
    # end if normalize:
    return resized_video
# EOF

"""
concatenate_frames_batch
Concatenate a batch of video frames from multiple videos, optionally processing only
the first part of long videos and keeping leftover frames from previous batches.

INPUT:
    - paths: dict -> paths to the video files
    - rank: int -> worker rank
    - frames_batch: list or np.ndarray -> leftover frames from previous batch
    - curr_video_idx: int -> current video index in fn_list
    - idx: int -> current batch index
    - batches_to_proc_togeth: int -> number of batches to process consecutively
    - batch_sizes: list/array -> sizes of each batch in frames
    - new_h, new_w: int -> target frame height and width
    - long_vids: list of bools -> flags indicating whether each video is long
    - vid_duration_lim: int (default 20) -> max duration in seconds for long videos
    - normalize: bool (default True) -> whether to normalize frames

OUTPUT:
    - frames_batch: np.ndarray -> concatenated frames for the batch
    - progression: int -> updated video index after processing

"""
def concatenate_frames_batch(paths, rank, fn_list, frames_batch, curr_video_idx, curr_batch_idx, batches_to_proc_togeth, batch_sizes, new_h, new_w, long_vids, vid_duration_lim=20, normalize=True):
    n_batches = len(batch_sizes)
    idx_tot = [curr_batch_idx + i for i in range(batches_to_proc_togeth) if curr_batch_idx + i < n_batches] # takes the next $batches_to_proc frames filtering for out of range indices 
    curr_tot_batch_size = np.sum(batch_sizes[idx_tot])
    cumulative_frames_sum = 0
    if is_empty(frames_batch):
        frames_batch = [] # otherwise we have arrays of inconsistent size to concatenate
    else:
        cumulative_frames_sum += frames_batch.shape[0]
        frames_batch = [frames_batch] # makes it a list with all the frames remained from the previous batch (ideally we read 3 batches and shuffle)
    # end if frames_batch:
    while cumulative_frames_sum < curr_tot_batch_size:
        fn = fn_list[curr_video_idx]
        if long_vids[curr_video_idx]: # if the video is marked as long
            video = read_video(paths, rank, fn, vid_duration=vid_duration_lim) # if the video is too long, we just process the beginning (vid_duration_lim is in sec)
        else:
            video = read_video(paths, rank, fn, vid_duration=0)
        # end if long_vids[progression]: 
        video = resize_video_array(video, new_h, new_w, normalize=False)
        curr_video_idx += 1
        curr_frames_n = video.shape[0] 
        cumulative_frames_sum += curr_frames_n
        frames_batch.append(video)
    # end while cumulative_frames_sum < curr_tot_batch_size:
    frames_batch = np.concatenate(frames_batch, axis=0)
    return frames_batch, curr_video_idx
# EOF


"""
shuffle_frames
Randomly shuffle the frames of a video array along the 0th dimension.

INPUT:
    - video: np.ndarray, shape (n_frames, H, W, C) -> video to shuffle

OUTPUT:
    - shuffled_video: np.ndarray, same shape as input -> frames randomly permuted
"""
def shuffle_frames(video):
    n_frames = video.shape[0] 
    indices = np.arange(n_frames)
    np.random.shuffle(indices)
    shuffled_video = video[indices, :, :, :]
    return shuffled_video
# EOF
