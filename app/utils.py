import torch
import numpy as np
from model import UNetPP
def load_model(path="/app/best_model.pth", device="cpu"):
    model = UNetPP(in_channels=3)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def reconstruct_from_patches(patches, original_img_shape, stride=64):
    img_h, img_w = original_img_shape[0], original_img_shape[1]
    patch_size = patches.shape[1]

    if patches.ndim == 3:
        patches = np.expand_dims(patches, axis=-1)

    reconstructed_img = np.zeros((img_h, img_w, 1), dtype=np.float32)
    count_matrix = np.zeros((img_h, img_w, 1), dtype=np.float32)

    patch_idx = 0
    for y in range(0, img_h - patch_size + 1, stride):
        for x in range(0, img_w - patch_size + 1, stride):
            if patch_idx < len(patches):
                reconstructed_img[y:y+patch_size, x:x+patch_size] += patches[patch_idx]
                count_matrix[y:y+patch_size, x:x+patch_size] += 1
                patch_idx += 1

    count_matrix[count_matrix == 0] = 1
    reconstructed_img /= count_matrix

    return reconstructed_img


def create_error_map(true_mask, pred_mask):
    true_mask = np.squeeze(true_mask)
    pred_mask = np.squeeze(pred_mask)

    true_mask = (true_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    error_map = np.zeros((*true_mask.shape, 3), dtype=np.uint8)

    tp = (true_mask == 1) & (pred_mask == 1)
    fp = (true_mask == 0) & (pred_mask == 1)
    fn = (true_mask == 1) & (pred_mask == 0)

    error_map[tp] = [0, 255, 0]   # Зелёный — TP
    error_map[fp] = [255, 0, 0]   # Красный — FP
    error_map[fn] = [0, 0, 255]   # Синий — FN
    return error_map
