import streamlit as st
import os
import librosa
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model_loader import load_model
from utils import generate_mel_image

# ==== CONFIG ====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Sample Sound"
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary"]
valve_to_idx = {v: i for i, v in enumerate(CLASSES)}

# Load model
model = load_model()

# Streamlit UI
st.set_page_config(page_title="Heart Valve AI Production", layout="wide")
st.title("💓 Heart Valve AI Production")

# Select Valve Class
selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)

# Load files
normal_dir = DATA_DIR / selected_class / "Normal"
abnormal_dir = DATA_DIR / selected_class / "Abnormal"
wav_files = []

for d in [normal_dir, abnormal_dir]:
    if d.exists():
        wav_files += list(d.glob("*.wav"))

if len(wav_files) == 0:
    st.warning("ไม่พบไฟล์ wav ใน class นี้")
else:
    filenames = [str(f.relative_to(DATA_DIR)) for f in wav_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)

    wav_path = DATA_DIR / selected_file
    y, sr = librosa.load(wav_path, sr=None)

    # Plot Time Domain
    st.subheader("🩺 Time Domain")
    st.line_chart(y)

    # Generate Mel Spectrogram
    mel_image = generate_mel_image(wav_path)
    st.subheader("🎛 Mel Spectrogram")
    st.image(mel_image)

    # Transform image for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(mel_image).unsqueeze(0)
    valve_idx_tensor = torch.tensor([valve_to_idx[selected_class]], dtype=torch.long)

    if st.button("Predict Now 🚀"):
        with torch.no_grad():
            output = model(img_tensor, valve_idx_tensor)
            prob = torch.sigmoid(output).item()

        if prob > 0.5:
            st.error("🔬 Abnormal (มีโอกาสเกิด Regurgitation)")
        else:
            st.success("✅ Normal (ไม่พบความผิดปกติ)")
