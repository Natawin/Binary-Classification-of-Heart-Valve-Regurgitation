import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from model_loader import load_model
import torch
from PIL import Image
from torchvision import transforms

# ===== CONFIG =====
DATA_DIR = Path("Sample Sound")
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary"]
valve_to_idx = {"mitral": 0, "aortic": 1, "tricuspid": 2, "pulmonary": 3}

# Load model
model = load_model()

st.set_page_config(page_title="Heart Valve AI Production", layout="wide")
st.title("💓 Heart Valve AI Demo (Full Production Version)")

# Select valve class
selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)
class_path = DATA_DIR / selected_class
wav_files = sorted(list(class_path.glob("*.wav")))

if len(wav_files) == 0:
    st.warning("ไม่มีไฟล์สำหรับ class นี้")
else:
    filenames = [f.name for f in wav_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)

    wav_path = class_path / selected_file
    y, sr = librosa.load(wav_path, sr=None)

    # Plot Audio and Images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎧 Audio Playback")
        st.audio(wav_path)

        st.subheader("🩺 Time Domain Plot")
        td_path = wav_path.with_suffix("").with_name(wav_path.stem.replace(".wav", "_td.png"))
        if td_path.exists():
            st.image(td_path)

    with col2:
        st.subheader("🎛 Mel Spectrogram")
        mel_path = wav_path.with_suffix("").with_name(wav_path.stem.replace(".wav", "_mel.png"))
        if mel_path.exists():
            st.image(mel_path)

    st.divider()
    st.subheader("🧪 AI Model Prediction")

    # Add Predict button
    if mel_path.exists():
        if st.button("Predict Now 🚀"):
            img = Image.open(mel_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0)
            valve_idx_tensor = torch.tensor([valve_to_idx[selected_class]], dtype=torch.long)

            with torch.no_grad():
                output = model(img_tensor, valve_idx_tensor)
                prob = torch.sigmoid(output).item()

            st.success(f"✅ Regurgitation Probability: {prob*100:.2f}%")
            if prob > 0.5:
                st.error("🔬 Regurgitation Detected")
            else:
                st.success("✅ No Regurgitation Detected")
