import streamlit as st
import os
import gdown
import torch
from model_class import MultiValveCNN

# ✅ ใส่ File ID Google Drive
GOOGLE_DRIVE_FILE_ID = '16cImTqyHLkr07QtzDszwR5MvWrTmieMf'
MODEL_FILENAME = 'model_epoch_20.pth'

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)

def load_model():
    download_model()
    model = MultiValveCNN()
    state_dict = torch.load(MODEL_FILENAME, map_location='cpu')

    # Ultimate Debug
    debug_msg = []
    for k, v in state_dict.items():
        debug_msg.append(f"{k}: {v.shape}")

    # แสดงใน Streamlit Log (ไม่ใช่แค่ใน console)
    st.write("===== State Dict Loaded =====")
    for msg in debug_msg:
        st.write(msg)

    model.load_state_dict(state_dict)
    model.eval()
    return model


