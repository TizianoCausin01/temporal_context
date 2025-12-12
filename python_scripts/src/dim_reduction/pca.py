import os, yaml, sys
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torchvision.models.feature_extraction import create_feature_extractor
import torch

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise, get_layer_output_shape
from image_processing.utils import concatenate_frames_batch, shuffle_frames


def ipca_videos(paths, rank, layer_name, model_name, model, n_components, video_type, batches_to_proc, batch_sizes, fn_list, long_vids,  device, vid_duration_lim, new_h=224, new_w=224):
    save_name = (f"{video_type}_{model_name}_{layer_name}_ipca_{n_components}_PCs.pkl")
    path = os.path.join(f"{paths["livingstone_lab"]}/tiziano/models", save_name)
    if os.path.exists(path):
        print_wise(f"{path} already exists")
    else:
        print_wise(f"Fitting PCA for layer: {layer_name}", rank=rank)
        frames_batch = []
        feature_extractor = create_feature_extractor(
            model, return_nodes=[layer_name]
        ).to(device)
        tmp_shape = get_layer_output_shape(feature_extractor, layer_name)
        n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
        n_components_layer = min(n_features, n_components)  # Limit to number of features
        ipca = IncrementalPCA(n_components=n_components_layer, batch_size=batch_sizes[0])
        curr_video_idx = 0
        for idx, curr_batch_size in enumerate(batch_sizes):

            print_wise(f"starting batch {idx}", rank=rank)
            frames_batch, curr_video_idx = concatenate_frames_batch(paths, rank, fn_list, frames_batch, curr_video_idx, idx, batches_to_proc, batch_sizes, new_h, new_w, long_vids, vid_duration_lim)
            frames_batch = shuffle_frames(frames_batch)
            inputs = frames_batch[:curr_batch_size, :, :, :]
            inputs = torch.from_numpy(inputs).float().to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            frames_batch = frames_batch[curr_batch_size:, :, :, :]
            with torch.no_grad():
                feats = feature_extractor(inputs)[layer_name]
                print("feats", feats.shape)
                feats = feats.view(feats.size(0), -1).cpu().numpy()
            ipca.partial_fit(feats)
            # end with torch.no_grad():
        
        joblib.dump(ipca, path) 
        print_wise(f"Saved PCA for {layer_name} at {path}", rank=rank)

