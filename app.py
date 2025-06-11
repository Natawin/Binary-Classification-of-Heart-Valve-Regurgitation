import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model_class import MultiValveCNN

# ===== CONFIG =====
DATA_DIR = Path("Sample Sound")
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary", "normal"]
EXPECTED_SR = 4000

# ===== LOAD MODEL =====
model = MultiValveCNN(num_valves=5)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="Heart Valve AI Demo", layout="wide")
st.title("💓 Heart Valve AI Demo (Full Production Guarded)")

# ===== File Selector =====
selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)
class_path = DATA_DIR / selected_class
wav_files = sorted(list(class_path.glob("*.wav")))

if len(wav_files) == 0:
    st.warning("⚠ ไม่มีไฟล์ใน class นี้")
else:
    filenames = [f.name for f in wav_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)

    wav_path = class_path / selected_file

    # === Input Guard System ===
    try:
        if not wav_path.suffix.lower() == ".wav":
            st.error("❌ รองรับเฉพาะไฟล์ .wav เท่านั้น")
            st.stop()

        y, sr = librosa.load(wav_path, sr=None)
        if sr != EXPECTED_SR:
            st.error(f"⚠ Sampling Rate ต้องเป็น {EXPECTED_SR} Hz เท่านั้น")
            st.stop()

    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์เสียงได้: {e}")
        st.stop()

    # ===== Visualization =====
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎧 Audio Playback")
        st.audio(wav_path)

        st.subheader("🩺 Time Domain Plot")
        td_path = wav_path.with_suffix("").with_name(wav_path.stem.replace(".wav", "_td.png"))
        if td_path.exists():
            st.image(td_path)
        else:
            time = np.arange(len(y)) / sr
            fig_td, ax_td = plt.subplots(figsize=(10, 3))
            ax_td.plot(time, y, color='tab:blue')
            ax_td.set_xlabel("Time (s)")
            ax_td.set_ylabel("Amplitude")
            ax_td.grid(True)
            st.pyplot(fig_td)

    with col2:
        st.subheader("🎛 Mel Spectrogram")
        mel_path = wav_path.with_suffix("").with_name(wav_path.stem.replace(".wav", "_mel.png"))
        if mel_path.exists():
            st.image(mel_path)
        else:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig_mel, ax_mel = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(S_dB, sr=sr, hop_length=256, cmap='viridis', ax=ax_mel)
            fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
            st.pyplot(fig_mel)

    # ===== Prediction Section =====
    st.markdown("---")
    st.subheader("🔮 AI Prediction")
    if st.button("🚀 Run Prediction"):
        img = Image.open(mel_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(img).unsqueeze(0)

        output = model(input_tensor, torch.tensor([CLASSES.index(selected_class)]))
        prob = torch.sigmoid(output).item()

        st.success(f"✅ Prediction Confidence: {prob:.2%}")
