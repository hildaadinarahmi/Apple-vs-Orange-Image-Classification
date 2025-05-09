import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os

# App config
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

# Custom header
st.markdown("<h1 style='text-align: center;'>🍎🍊 Fruit Classifier: Apple vs Orange</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image, and let the model predict whether it's an <b>Apple</b> or an <b>Orange</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout using columns
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📤 Upload your fruit image", type=["jpg", "jpeg", "png"])
    
with col2:
    st.markdown("### 📝 How it works")
    st.markdown("""
    - Trained on basic visual features like **color** and **shape**.
    - Preprocessed to 150x150 pixels.
    - Uses a simple image classifier model (pickle).
    """)

st.markdown("---")

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="📷 Uploaded Image", use_container_width=True, width=200)

    try:
        # Preprocess image
        img_array = np.array(image) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Load model
try:
    with open("fruit_classifier_model.pkl", "rb") as f:
        wrapper = pickle.load(f)
        model = wrapper.model
        class_names = wrapper.class_names
except FileNotFoundError:
    st.error("🚫 Model file not found. Please ensure 'fruit_classifier_model.pkl' is in the directory.")
    st.stop()

            # Predict
            prediction = model.predict(img_batch)[0]
            classes = ["Apple", "Orange"]
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Show result
            if predicted_class == "Apple":
                st.success(f"🍎 It's an **Apple** with **{confidence:.2f}%** confidence!")
            else:
                st.warning(f"🍊 It's an **Orange** with **{confidence:.2f}%** confidence!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Tensor image (normalize & resize done in Streamlit)
import torch
img_tensor = torch.tensor(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
img_tensor = img_tensor * 2 - 1  # normalize back to [-1, 1]
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = class_names[output.argmax().item()]

