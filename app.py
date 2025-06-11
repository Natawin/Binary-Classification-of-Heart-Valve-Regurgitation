import streamlit as st
from model_loader import load_model

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

st.title("Heart Valve AI Demo")
st.write("🎉 Model Loaded Successfully")

# ตัวอย่างการ infer ง่ายๆ:
# output = model(input_tensor)
# st.write(output)
