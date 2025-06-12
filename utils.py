import librosa
import numpy as np
from PIL import Image
import cv2

def generate_mel_image(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80

    resized = cv2.resize(S_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.stack([resized, resized, resized], axis=-1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)
