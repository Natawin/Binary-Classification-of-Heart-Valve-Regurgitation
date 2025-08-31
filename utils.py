import librosa
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def generate_mel_image(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image

def generate_mel_tensor(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80  # Normalize to 0-1

    resized = transforms.functional.resize(
        Image.fromarray(np.uint8(S_norm * 255)),
        (224, 224),
        interpolation=Image.BICUBIC
    )

    image = np.array(resized) / 255.0
    image = np.stack([image, image, image], axis=-1)
    image = torch.tensor(image).permute(2, 0, 1).float()
    return image

