from mpi4py import MPI
import os, yaml, sys
import torch
from torchvision import models, datasets 
from torch.utils.data import DataLoader
import argparse
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_relevant_output_layers, get_device
from image_processing.utils import list_videos, get_frames_number, split_in_batches
from parallel.parallel_funcs import master_workers_queue
from dim_reduction.pca import ipca_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_components", type=int) # number of PCA components
    parser.add_argument("--video_type", type=str) # YDX, IMG or faceswap
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--batches_to_proc", type=int) # how many frames (in batch_size units) we read and shuffle
    parser.add_argument("--max_duration", type=int, default=20)  # the maximum video duration
    args = parser.parse_args()
    task_list = get_relevant_output_layers(args.model_name)
    device = get_device() 
    model_cls = getattr(models, args.model_name)
    model = model_cls(pretrained=True).to(device).eval()
    fn_list = list_videos(paths, args.video_type)
    frames_per_vid, long_vids = get_frames_number(paths, fn_list, args.max_duration)
    batch_sizes = split_in_batches(frames_per_vid, args.batch_size)
    master_workers_queue(task_list, paths, ipca_videos, *(args.model_name, model, args.num_components, args.video_type, args.batches_to_proc, batch_sizes, fn_list, long_vids, device, args.max_duration)) 
