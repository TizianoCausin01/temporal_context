from mpi4py import MPI
import os, yaml, sys
import torch
from torchvision import models, datasets 
from torch.utils.data import DataLoader
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_relevant_output_layers, get_device
from image_processing.utils import load_timm_model, load_torchvision_model, get_usual_transform
from image_processing.computational_models import img_feats_extraction_pooling, map_image_order_from_ann_to_monkey
from parallel.parallel_funcs import master_workers_queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--pooling", type=str)
    
    
    args = parser.parse_args()

    device = get_device()
    transform = get_usual_transform(resize_size=args.img_size, normalize=False)
    task_list = get_relevant_output_layers(args.model_name, pkg=args.pkg)

    dataset = ImageFolder(
        root=f"{paths['livingstone_lab']}/Stimuli/{args.folder_name}/",
        transform=transform,
        is_valid_file=lambda x: not x.endswith("Thumbs.db"), 
        allow_empty=True, 
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.pkg=='torchvision':
        load_mod_function = load_torchvision_model
    elif args.pkg=='timm':
        load_mod_function = load_timm_model
    model = load_mod_function(args.model_name, device, img_size=args.img_size)
    master_workers_queue(task_list, paths, img_feats_extraction_pooling, *(args.model_name, model, dataloader, args.folder_name, args.img_size, args.pooling, device)) 
