import librosa
import numpy as np
from PIL import Image

def generate_mel_image(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80

    # สร้างภาพ (H x W)
    image = (S_norm * 255).astype(np.uint8)
    pil_img = Image.fromarray(image).convert("RGB")

    # Resize ด้วย Pillow ไม่ง้อ cv2
    pil_img = pil_img.resize((224, 224), Image.BICUBIC)

    return pil_img
