# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt

# ---- Model yükleme ----
@st.cache_resource
def load_model(num_classes, model_path="maskrcnn_insects.pth"):
    # Modeli başlat (önceden eğitilmiş ağırlık yok)
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Box predictor'u yeni sınıf sayısına göre değiştir
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint'i yükle
    checkpoint = torch.load(model_path, map_location=device)
    
    # Sadece uyumlu parametreleri al
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model, device

# ---- Görselleştirme ----
def visualize_prediction(img, pred, classes, threshold=0.5):
    img_vis = np.array(img).copy()
    masks = pred['masks'].cpu().detach().numpy()
    boxes = pred['boxes'].cpu().detach().numpy()
    labels = pred['labels'].cpu().detach().numpy()
    scores = pred['scores'].cpu().detach().numpy()

    for i in range(len(labels)):
        if scores[i] < threshold:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int)
        mask = masks[i, 0]
        mask = (mask > 0.5).astype(np.uint8)
        img_vis[mask==1] = img_vis[mask==1] * 0.5 + np.array([255,0,0]) * 0.5
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0,255,0), 2)
        label_name = classes[labels[i]-1]
        cv2.putText(img_vis, f"{label_name} {scores[i]:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return img_vis.astype(np.uint8)

# ---- Streamlit arayüzü ----
st.title("🐞 Insect Detection with Mask R-CNN")

uploaded_file = st.file_uploader("Resim yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    # sınıfları manuel ekle veya dataset klasöründen oku
    classes = ["Beetle", "Butterfly", "Ant", "Ladybug"]  # örnek
    num_classes = len(classes) + 1

    model, device = load_model(num_classes)

    img_tensor = transforms.ToTensor()(img_resized).to(device)
    with torch.no_grad():
        pred = model([img_tensor])[0]

    result_img = visualize_prediction(img_resized, pred, classes, threshold=0.5)
    st.image(result_img, caption="Tahmin Sonucu", use_container_width=True)
