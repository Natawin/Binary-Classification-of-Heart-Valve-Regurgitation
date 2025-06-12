import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def generate_mel_image(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # วาดกราฟลง memory buffer
    plt.figure(figsize=(3,3))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, hop_length=256, cmap='inferno')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    image = Image.open(buf).convert("RGB")
    image = image.resize((224,224))
    return image
