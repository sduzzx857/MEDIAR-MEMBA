import torch
from train_tools.models import MEDIARMamba
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from core.MEDIAR.utils import compute_masks
from train_tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def tuningset_evaluation(model, dataloaders):
    cell_counts_total = []
    model.eval()

    for batch_data in tqdm(dataloaders["tuning"]):
        images = batch_data["img"].to(device)
        if images.shape[-1] > 5000:
            continue

        outputs = sliding_window_inference(
            images,
            roi_size=512,
            sw_batch_size=4,
            predictor=model,
            padding_mode="constant",
            mode="gaussian",
        )

        outputs = outputs.squeeze(0)
        outputs, _ = post_process(outputs, None)
        count = len(np.unique(outputs) - 1)
        cell_counts_total.append(count)

    cell_counts_total_sum = np.sum(cell_counts_total)
    print("Cell Counts Total: (%d)" % (cell_counts_total_sum))

    return cell_counts_total_sum

def sigmoid(z):
    """Sigmoid function for numpy arrays"""
    return 1 / (1 + np.exp(-z))

def post_process(outputs, labels=None):
    """Predict cell instances using the gradient tracking"""
    outputs = outputs.squeeze(0).cpu().numpy()
    gradflows, cellprob = outputs[:2], sigmoid(outputs[-1])
    outputs = compute_masks(gradflows, cellprob, use_gpu=True, device=device)
    outputs = outputs[0]  # (1, C, H, W) -> (C, H, W)

    if labels is not None:
        labels = labels.squeeze(0).squeeze(0).cpu().numpy()

    return outputs, labels



root = "/home/data/MEDIAR/"
mapping_file = "./train_tools/data_utils/mapping_labeled.json"
mapping_file_tuning = "./train_tools/data_utils/mapping_tuning.json"

model = MEDIARMamba().cuda().eval()
weights = torch.load('weights/pretrained/model.pth', map_location="cpu")
model.load_state_dict(weights, strict=False)
dataloaders = datasetter.get_dataloaders_labeled(root, mapping_file, mapping_file_tuning, batch_size=1)

with torch.no_grad():
    tuning_cell_counts = tuningset_evaluation(model, dataloaders)
tuning_count_dict = {"MEDIARMamba TuningSet_Cell_Count": tuning_cell_counts}


