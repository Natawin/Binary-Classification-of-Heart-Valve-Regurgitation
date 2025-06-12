import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import scipy.ndimage
import librosa.display

# ฟังก์ชันสร้าง Mel Spectrogram สำหรับ Model Input (แบบ grayscale → 3 channel)
def generate_mel_tensor(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80

    # ใช้ scipy.ndimage.zoom แทน cv2.resize
    zoom_factor = (224 / S_norm.shape[0], 224 / S_norm.shape[1])
    resized = scipy.ndimage.zoom(S_norm, zoom_factor, order=3)

    stacked = np.stack([resized]*3, axis=-1)
    stacked = (stacked * 255).astype(np.uint8)
    image = Image.fromarray(stacked)

    return image

# ฟังก์ชันสำหรับแสดง Mel-Spectrogram แบบสีบน Web
def generate_mel_display(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    display_image = Image.open(buf)
    return display_image
