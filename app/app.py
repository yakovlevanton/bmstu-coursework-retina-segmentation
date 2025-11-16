import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from utils import (
    load_model,
    reconstruct_from_patches,
    create_error_map,
)


st.set_page_config(page_title="Сегментация сосудов (патчи)", layout="wide")


@st.cache_resource
def _load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model("/app/best_model.pth", device=device)

model = _load_model()

transform = T.Compose([
    T.ToTensor(),
])

st.title("Сегментация сосудов")

# ---- Uploaders ----
img_file = st.file_uploader("Загрузите изображение (tif)", type=["tif", "tiff"])
gt_file = st.file_uploader("Загрузите маску сосудов (gif)", type=["gif"])
fov_file = st.file_uploader("Загрузите FOV (gif)", type=["gif"])

patch_size = 128
stride = 64
batch_size = 16


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-8)

def accuracy(pred, gt):
    return (pred == gt).mean()

def sensitivity(pred, gt):
    tp = (pred * gt).sum()
    fn = ((1 - pred) * gt).sum()
    return tp / (tp + fn + 1e-8)

def specificity(pred, gt):
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    return tn / (tn + fp + 1e-8)



if img_file is not None:
    img_full = np.array(Image.open(img_file)).astype(np.float32) / 255.0
    st.image((img_full * 255).astype(np.uint8), caption="Исходное изображение", width=600)

if fov_file:
    fov_full = np.array(Image.open(fov_file).convert("L"))
    fov_full = (fov_full > 127).astype(np.uint8)

if gt_file:
    gt_full = np.array(Image.open(gt_file).convert("L"))
    gt_full = (gt_full > 127).astype(np.uint8)

if st.button("Запустить сегментацию") and img_file:

    h, w = img_full.shape[:2]

    patches_list = []
    coords = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_full[y:y+patch_size, x:x+patch_size]
            patch_t = transform(image=(patch * 255).astype(np.uint8))["image"]
            patch_t = patch_t.float() / 255.0
            patches_list.append(patch_t)
            coords.append((y, x))
    patches_tensor = torch.stack(patches_list).float().to("cpu")

    pred_patches = []

    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size]
            pred = model(batch)
            pred_bin = (pred > 0.5).float()
            pred_patches.append(pred_bin.cpu().numpy().squeeze(1))

    pred_patches_np = np.concatenate(pred_patches, axis=0)

    pred_full = reconstruct_from_patches(pred_patches_np, (h, w), stride=stride)
    pred_full = np.squeeze(pred_full)

    if fov_file:
        pred_plot = pred_full * fov_full
    else:
        pred_plot = pred_full

    st.subheader("Результаты")

    col1, col2 = st.columns(2)
    col1.image((pred_plot * 255).astype(np.uint8), caption="Предсказанная маска", width=600)

    if gt_file:
        col2.image((gt_full * 255).astype(np.uint8), caption="GT маска", width=600)


    if gt_file:
        error_map = create_error_map(pred_plot, gt_full)

        st.subheader("Карта ошибок")
        st.image(error_map, caption="TP — зелёный, FP — красный, FN — синий", width=600)


    if gt_file:
        st.subheader("Метрики")

        dice = dice_score(pred_plot, gt_full)
        iou = iou_score(pred_plot, gt_full)
        acc = accuracy(pred_plot, gt_full)
        sens = sensitivity(pred_plot, gt_full)
        spec = specificity(pred_plot, gt_full)

        st.markdown(f"""
        **Dice:** {dice:.4f}  
        **IoU:** {iou:.4f}  
        **Accuracy:** {acc:.4f}  
        **Sensitivity (Recall):** {sens:.4f}  
        **Specificity:** {spec:.4f}  
        """)

    st.subheader("Параметры изображения")

    vessel_pred = pred_plot.sum()
    vessel_gt = gt_full.sum() if gt_file else None
    fov_area = fov_full.sum() if fov_file else (h * w)

    st.markdown(f"""
    **Размер изображения:** {h} × {w}  
    **Площадь FOV:** {fov_area} px  

    **Плотность сосудов (предсказание):** {vessel_pred / fov_area:.4f}
    """)

    if gt_file:
        st.markdown(f"""
        **Плотность сосудов (GT):** {vessel_gt / fov_area:.4f}
        """)

