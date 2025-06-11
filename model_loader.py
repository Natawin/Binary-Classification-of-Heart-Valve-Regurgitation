import os
import gdown
import torch
from model_class import MultiValveCNN

# ใส่ File ID ของ Google Drive ตรงนี้
GOOGLE_DRIVE_FILE_ID = 'https://drive.google.com/file/d/107ruYjCflJ7VQG8q5pHqIYsTnu4gEer9/view?usp=sharing'
MODEL_FILENAME = 'Mel-Spectrogram_Data_Pretrain_state.pth'

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False)
    else:
        print("Model file already exists locally.")

def load_model():
    download_model()

    # สร้าง model object เปล่าก่อน
    model = MultiValveCNN()

    # โหลด state_dict เข้าไปใน model
    state_dict = torch.load(MODEL_FILENAME, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model
