import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
import streamlit as st
from model_loader import load_model
from utils import generate_mel_image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Sample Sound"
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary"]
valve_to_idx = {v: i for i, v in enumerate(CLASSES)}

model = load_model()

st.set_page_config(page_title="Heart Valve AI Production (Final Build)", layout="wide")
st.title("💓 Heart Valve AI Production (Final Build)")

selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)

normal_path = DATA_DIR / selected_class / "Normal"
abnormal_path = DATA_DIR / selected_class / "Abnormal"

normal_files = sorted(list(normal_path.glob("*.wav"))) if normal_path.exists() else []
abnormal_files = sorted(list(abnormal_path.glob("*.wav"))) if abnormal_path.exists() else []

all_files = [(f"Normal/{f.name}", f) for f in normal_files] + [(f"Abnormal/{f.name}", f) for f in abnormal_files]

if not all_files:
    st.warning("ไม่มีไฟล์ wav ใน class นี้")
else:
    filenames = [name for name, path in all_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)
    file_path = dict(all_files)[selected_file]
    mel_image = generate_mel_image(file_path)
    td_file = file_path.with_name(file_path.stem + "_td.png")
    td_image = Image.open(td_file) if td_file.exists() else None
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎧 Audio Playback")
        st.audio(str(file_path))
        if td_image:
            st.subheader("🩺 Time Domain Plot")
            st.image(td_image)

    with col2:
        st.subheader("🎛 Mel Spectrogram")
        st.image(mel_image)

    st.divider()
    st.subheader("AI Prediction")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(mel_image).unsqueeze(0)
    valve_tensor = torch.tensor([valve_to_idx[selected_class]], dtype=torch.long)

    if st.button("Predict Now"):
        with torch.no_grad():
            output = model(img_tensor, valve_tensor)
            prob = torch.sigmoid(output).item()

        if prob > 0.5:
            st.error("🔬 Regurgitation Detected (Abnormal)")
        else:
            st.success("Normal")

