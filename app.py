import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model_loader import load_model

# โหลดโมเดล
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Valve Mapping
valve_to_idx = {"mitral": 0, "aortic": 1, "tricuspid": 2, "pulmonary": 3}
idx_to_valve = {v: k for k, v in valve_to_idx.items()}

# สร้าง UI
st.title("Heart Valve AI Demo")

st.write("🎯 Upload Spectrogram Image for Prediction")

# Upload Image
uploaded_file = st.file_uploader("Upload PNG Image", type=["png", "jpg", "jpeg"])

# Valve Selector
valve_choice = st.selectbox("Select Valve", list(valve_to_idx.keys()))

# ถ้ามีไฟล์
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # ปุ่ม Predict
    if st.button("Predict"):
        # Image Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        # Valve Index
        valve_idx = torch.tensor([valve_to_idx[valve_choice]])

        # Predict
        with torch.no_grad():
            output = model(input_tensor, valve_idx)
            pred = torch.sigmoid(output).item()
            pred_label = 1 if pred >= 0.5 else 0

        # Show Result
        st.write(f"**Prediction:** {'Abnormal (Regurgitation)' if pred_label == 1 else 'Normal'}")
        st.write(f"**Confidence:** {pred:.2f}")
