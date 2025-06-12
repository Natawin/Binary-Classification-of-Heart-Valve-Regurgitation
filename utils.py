import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_mel_image_for_model(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80
    resized = Image.fromarray((S_norm * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    image = np.stack([resized, resized, resized], axis=-1)
    image = np.array(image).astype(np.uint8)
    return Image.fromarray(image)

def plot_mel_spectrogram(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig
