import os
import torch
import gdown
from model_class import MultiValveCNN

# ====== CONFIG ======
GOOGLE_DRIVE_FILE_ID = 'YOUR_FILE_ID_HERE'  # <<< 🔥 ใส่ File ID ของคุณตรงนี้
MODEL_FILENAME = 'model_epoch_20.pth'

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)

def load_model():
    download_model()
    model = MultiValveCNN()
    state_dict = torch.load(MODEL_FILENAME, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model
