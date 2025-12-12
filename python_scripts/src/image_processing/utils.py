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



"""
list_videos
Creates a list with all videos with the required characteristics.
INPUT:
    - paths: dict -> paths to the video files
    - video_type: str -> the type of videos to index
OUTPUT:
    - fn_list: list{str} -> a list with the file names of the required paths (or all the videos if video_type is None)
"""
def list_videos(paths: dict, video_type: str):
    videos_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
    all_files = os.listdir(videos_dir) 
    if video_type:
        if video_type == 'YDX':
            fn_list = [f for f in all_files if "YDX" in f]
        elif video_type == 'IMG':
            fn_list = [f for f in all_files if "IMG" in f]
        elif video_type == 'faceswap':
            fn_list = [f for f in all_files if "YDX" not in f and "IMG" not in f]
        else:
            raise ValueError("video_type must be 'YDX', 'IMG' or 'faceswap'")
        return fn_list
        # end if video_type == 'YDX':
    else:
        return all_files
    # end if video_type:
# EOF

"""
get_frames_number
Computes the number of frames for each video in fn_list, marking which videos exceed a maximum duration and should be truncated.

INPUT:
    - paths: dict -> dictionary containing base paths (expects key 'livingstone_lab')
    - fn_list: list -> list of video filenames to process
    - max_duration: float -> maximum allowed duration in seconds; videos longer than
                               this are marked as long and truncated
OUTPUT:
    - frames_per_vid: list of floats -> number of frames for each video (truncated if long)
    - long_vids: list of bools -> True for long videos exceeding max_duration, else False
"""
def get_frames_number(paths: dict, fn_list: list, max_duration: float):
    videos_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
    frames_per_vid = []
    long_vids = []
    for fn in fn_list:
        video_path = os.path.join(videos_dir, fn)
        cap = cv2.VideoCapture(video_path)
        _, _, n_frames = get_video_dimensions(cap)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if n_frames/fps > max_duration:
            n_frames = fps*max_duration
            long_vids.append(True)
        else:
            long_vids.append(False)
        frames_per_vid.append(n_frames)
    # end for fn in fn_list:
    return frames_per_vid, long_vids
# EOF


"""
split_in_batches
Splits a total number of frames into batches of approximately equal size.
Useful for distributing frames across workers or for chunked processing.
PROCESS:
    1. Computes the total number of frames across all videos.
    2. Computes how many batches are needed (rounded).
    3. Splits the frame indices into n_batches approximately equal parts.
    4. Extracts and stores the size of each batch.

INPUT:
    - frames_per_vid: list/array of ints -> number of frames for each video
    - batch_size: int -> desired approximate size of each batch
OUTPUT:
    - batch_size_list: list of ints -> sizes of each batch
    - splits: list of np.ndarray -> the actual index splits (each array contains the indices for that batch)
"""
def split_in_batches(frames_per_vid, batch_size):
    tot_frame_num = round(np.sum(frames_per_vid))
    n_batches = round(tot_frame_num/batch_size)
    splits = np.array_split(np.arange(tot_frame_num), n_batches)
    batch_size_list = []
    for batch_idx in splits:
        batch_size_list.append(len(batch_idx)) # stores the current batch size
    return np.array(batch_size_list) 
