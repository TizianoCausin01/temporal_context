import os, yaml, sys
#sys.path.append("../DeepGaze")
#import deepgaze_pytorch

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise
from image_processing.computational_models import ICF_setup, compute_ICF_saliency 
from parallel.parallel_funcs import master_workers_queue

video_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
video_fn = os.listdir(video_dir)
check_point, new_saver, input_tensor, log_density_wo_centerbias = ICF_setup(paths)
resize_factor = .1
master_workers_queue(video_fn, paths, compute_ICF_saliency, *(resize_factor, check_point, new_saver, input_tensor, log_density_wo_centerbias))
