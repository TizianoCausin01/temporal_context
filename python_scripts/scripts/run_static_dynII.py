import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import get_relevant_output_layers, compute_samples_sizes, load_img_natraster
from general_utils.II import init_static_dynII, compute_static_dynII
from parallel.parallel_funcs import master_workers_queue
from image_processing.computational_models import map_image_order_from_ann_to_monkey

# e.g. to call it:
# mpiexec -np 5 python3 run_static_dynII.py --monkey_name=three0 --date=250313 --brain_area=AIT --k=1 --folder_name=talia_20each_tizi --signal_RDM_metric=cosine --model_RDM_metric=euclidean --model_name=vit_l_16 --img_size=384 --pooling=mean --new_fs=100 --pkg=timm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--brain_area", type=str)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--signal_RDM_metric", type=str)
    parser.add_argument("--model_RDM_metric", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--new_fs", type=int) 
    parser.add_argument("--k", type=int) 
    parser.add_argument("--pkg", type=str)

    cfg = parser.parse_args()

    task_list = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    area_rasters = load_img_natraster(paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs, brain_area=cfg.brain_area)

    dataset = ImageFolder(
        root=f"{paths['livingstone_lab']}/Stimuli/{cfg.folder_name}/",
        is_valid_file=lambda x: not x.endswith("Thumbs.db"), 
        allow_empty=True, 
    )

    idx_ord = map_image_order_from_ann_to_monkey(paths, cfg.monkey_name, cfg.date, dataset)

    dyn_ii_obj = init_static_dynII(area_rasters, cfg.signal_RDM_metric, cfg.model_RDM_metric, cfg.k)

    master_workers_queue(task_list, paths, compute_static_dynII, *(dyn_ii_obj, idx_ord, cfg.monkey_name, cfg.date, cfg.brain_area, cfg.folder_name, cfg.model_name, cfg.img_size, cfg.pooling)) 
