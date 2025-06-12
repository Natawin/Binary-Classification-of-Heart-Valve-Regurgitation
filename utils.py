import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io

def generate_mel_tensor(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80

    # Resize & Convert → 3-channel (แค่ duplicate 3 ช่อง)
    resized = cv2.resize(S_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
    stacked = np.stack([resized]*3, axis=-1)
    stacked = (stacked * 255).astype(np.uint8)
    image = Image.fromarray(stacked)

    return image, S_norm

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
