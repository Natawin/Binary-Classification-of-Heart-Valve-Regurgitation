import os
import gdown
import torch

# ตั้งค่า file_id จาก Google Drive ที่เราเตรียมไว้
GOOGLE_DRIVE_FILE_ID = '107ruYjCflJ7VQG8q5pHqIYsTnu4gEer9'
MODEL_FILENAME = 'model.pt'

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?id=107ruYjCflJ7VQG8q5pHqIYsTnu4gEer9'
        gdown.download(url, MODEL_FILENAME, quiet=False)
    else:
        print("Model file already exists locally.")

def load_model():
    download_model()
    model = torch.load(MODEL_FILENAME, map_location=torch.device('cpu'))  # บน Streamlit Cloud มักใช้ CPU
    model.eval()  # ใส่ eval mode
    return model
