import librosa
import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
import io

# ตัวสร้าง Input สำหรับ Model (Grayscale)
def generate_mel_tensor(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80  # Normalize to 0-1

    zoom_factor = (224 / S_norm.shape[0], 224 / S_norm.shape[1])
    resized = scipy.ndimage.zoom(S_norm, zoom_factor, order=3)
    image = (resized * 255).astype(np.uint8)

    # Convert เป็น 3 channel (RGB) สำหรับ model
    image_rgb = np.stack([resized] * 3, axis=-1)
    image_rgb = (image_rgb * 255).astype(np.uint8)
    return Image.fromarray(image_rgb), resized

# ตัวสร้างรูปสีสำหรับโชว์
def generate_mel_display(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return image
