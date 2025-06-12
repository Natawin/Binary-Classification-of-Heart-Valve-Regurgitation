import librosa
import numpy as np
from PIL import Image
import scipy.ndimage

def generate_mel_image(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80  # Normalize to 0-1

    # ใช้ scipy แทน cv2.resize
    zoom_factor = (224 / S_norm.shape[0], 224 / S_norm.shape[1])
    resized = scipy.ndimage.zoom(S_norm, zoom_factor, order=3)

    # Duplicate 1-channel → RGB
    image = np.stack([resized] * 3, axis=-1)
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)
