import os, yaml, sys
from mpi4py import MPI

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise
from image_processing.utils import read_video
from parallel.parallel_funcs import master_workers_queue

initial_data = os.listdir(f"{paths['livingstone_lab']}/Stimuli/movies/faceswap_3/")
master_workers_queue(initial_data, paths, read_video) 
