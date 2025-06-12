import streamlit as st
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch

from model_loader import load_model
from utils import generate_mel_image

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Sample Sound"
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary"]
valve_to_idx = {v: i for i, v in enumerate(CLASSES)}

st.set_page_config(page_title="Heart Valve AI Production", layout="wide")
st.title("💓 Heart Valve AI Production (Gdown Build)")
model = load_model()

selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)
class_path = DATA_DIR / selected_class
wav_files = sorted(list(class_path.glob("*.wav")))

if len(wav_files) == 0:
    st.warning("ไม่มีไฟล์ wav ใน class นี้")
else:
    filenames = [f.name for f in wav_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)

    wav_path = class_path / selected_file
    y, sr = librosa.load(wav_path, sr=None)

    col1, col2 = st.columns(2)

    # Time-Domain Plot
    with col1:
        st.subheader("🩺 Time Domain")
        td_path = wav_path.with_name(wav_path.stem + "_td.png")
        if td_path.exists():
            st.image(td_path)
        else:
            time = np.arange(len(y)) / sr
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time, y, color='tab:blue')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            st.pyplot(fig)

    # Mel-Spectrogram
    # Mel-Spectrogram
    with col2:
        st.subheader("🎛 Mel-Spectrogram")
        mel_path = wav_path.with_name(wav_path.stem + "_mel.png")
        if mel_path.exists():
            st.image(mel_path)
        else:
            mel_image = generate_mel_image(wav_path)
            st.image(mel_image)

    st.divider()
    st.subheader("🧪 AI Prediction")

    mel_image = generate_mel_image(wav_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(mel_image).unsqueeze(0)
    valve_idx_tensor = torch.tensor([valve_to_idx[selected_class]], dtype=torch.long)

    if st.button("Predict Now 🚀"):
        with torch.no_grad():
            output = model(img_tensor, valve_idx_tensor)
            prob = torch.sigmoid(output).item()

        st.success(f"✅ Regurgitation Probability: {prob*100:.2f}%")
        if prob > 0.5:
            st.error("🔬 Regurgitation")
        else:
            st.success("✅ Non-Regurgitation")
