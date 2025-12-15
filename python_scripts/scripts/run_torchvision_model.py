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
from image_processing.utils import load_torchvision_model
from image_processing.computational_models import compute_torchvision_model
from parallel.parallel_funcs import master_workers_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_len", type=int, default=20)  # the maximum video duration
    parser.add_argument("--pca_opt", type=bool, default=True)  # whether or not projecting on the PCs 
    args = parser.parse_args()
    task_list = get_relevant_output_layers(args.model_name)
    device = get_device() 
    model = load_torchvision_model(args.model_name, device)
    kwargs = {'max_len': args.max_len, 'pca_opt': args.pca_opt}
    master_workers_queue(task_list, paths, compute_torchvision_model, *(args.model_name, model, device), **kwargs) 
