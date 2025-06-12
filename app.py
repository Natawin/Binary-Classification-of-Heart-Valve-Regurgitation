import streamlit as st
import os
import torch
from torchvision import transforms
from pathlib import Path
from model_loader import load_model
from utils import generate_mel_image

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Sample Sound"
CLASSES = ["mitral", "aortic", "tricuspid", "pulmonary"]
valve_to_idx = {v: i for i, v in enumerate(CLASSES)}

model = load_model()

st.set_page_config(page_title="Heart Valve AI Production (Gdown Build)", layout="wide")
st.title("❤️ Heart Valve AI Production (Gdown Build)")

selected_class = st.sidebar.selectbox("เลือก Valve Class:", CLASSES)
class_path = DATA_DIR / selected_class

#  เลือก label Normal / Abnormal
sub_labels = ["Normal", "Abnormal"]
selected_label = st.sidebar.selectbox("เลือกรูปแบบ:", sub_labels)
label_path = class_path / selected_label

wav_files = sorted([f for f in label_path.glob("*.wav")])

if len(wav_files) == 0:
    st.warning("ไม่มีไฟล์ wav ใน class นี้")
else:
    filenames = [f.name for f in wav_files]
    selected_file = st.selectbox("เลือกไฟล์:", filenames)
    wav_path = label_path / selected_file

    # Display audio
    st.audio(str(wav_path))

    # Generate Mel
    mel_image = generate_mel_image(wav_path)
    st.image(mel_image, caption="Mel-Spectrogram")

    # Prepare model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(mel_image).unsqueeze(0)
    valve_idx_tensor = torch.tensor([valve_to_idx[selected_class]], dtype=torch.long)

    if st.button("Predict Now 🚀"):
        with torch.no_grad():
            output = model(img_tensor, valve_idx_tensor)
            prob = torch.sigmoid(output).item()

        if prob > 0.5:
            st.error(f"❌ Abnormal")
        else:
            st.success(f"✅ Normal")
