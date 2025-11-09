import os, yaml, sys
import numpy as np
from ultralytics import YOLO
from scipy.io import savemat
from huggingface_hub import hf_hub_download

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise
from image_processing.computational_models import par_detect_faces
from parallel.parallel_funcs import master_workers_queue

face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(face_model_path)
person_model_path = f"{paths['livingstone_lab']}/tiziano/model_weights/yolov8n.pt"
person_model = YOLO(person_model_path)
video_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
video_fn = os.listdir(video_dir)
people_vids = [fn for fn in video_fn if ("IMG" not in fn)]
scale = 1.5
master_workers_queue(people_vids, paths, par_detect_faces, *(face_model, person_model, scale))
