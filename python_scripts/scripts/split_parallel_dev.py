import os, yaml, sys
import numpy as np
from mpi4py import MPI
import cv2

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from parallel.parallel_funcs import parallel_setup, split_parallel
from general_utils.utils import make_intervals, print_wise


def read_send_video_chunk(start, length, comm, rank, root, paths, file_name, folder_name):
    print_wise(f"received {start}, {length}, {file_name}", rank=rank)
    p = f"{paths["livingstone_lab"]}/Stimuli/{folder_name}/{file_name}"
    cap = cv2.VideoCapture(p)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # vid_chunk = np.zeros((height*width*3, length))
    vid_chunk = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(length):
        ret, frame = cap.read()
        # vid_chunk[:,i] = frame.ravel()
        vid_chunk.append(frame.ravel())
        if ret == False:
            raise ValueError(f"Not able to read frame {i + start}")
    vid_chunk = np.stack(vid_chunk, axis=-1)  # it stacks all the 1D arrays creating another axis
    comm.send(vid_chunk, dest=root, tag=11)  # starts sending data to process with rank 1


def rec_back_video_chunk(comm, size, rank, paths, file_name, folder_name):
    tot_movie = []
    for src in range(1, size):
        d = comm.recv(source=src, tag=11)
        tot_movie.append(d)
        print_wise(f"received chunk from {src}, of size {d.shape}", rank=rank)
    tot_movie = np.concatenate(tot_movie, axis=-1)  # it concatenates them along the last axis (time) so that it doesn't matter if I have chunks of different length
    print_wise(f"tot_movie has length {tot_movie.shape}", rank=rank)


fn = "rubin2_to_girl1_10s_rev.mp4"
folder_name = "faceswap_4"
p = f"{paths["livingstone_lab"]}/Stimuli/{folder_name}/{fn}"
cap = cv2.VideoCapture(p)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)
split_parallel(n_frames, read_send_video_chunk, (fn, folder_name), paths, rec_back=True, func_merge=rec_back_video_chunk, args_merge=(fn, folder_name))
