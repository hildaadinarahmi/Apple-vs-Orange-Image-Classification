
import streamlit as st
from PIL import Image
import numpy as np
import pickle

st.title("🍎 Fruit Classifier: Apple vs Orange")
st.write("""
This interactive app uses a simple image classifier to predict whether the uploaded fruit image is an **apple** or an **orange**.
""")

uploaded_file = st.file_uploader("Upload a fruit image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Load model
    with open("fruit_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(img_batch)[0]
    classes = ["Apple", "Orange"]
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}** with **{confidence:.2f}%** confidence.")

with st.expander("How does this work?"):
    st.markdown("""
    This model is trained using a basic image classification pipeline. It uses visual features like shape and color to distinguish between apples and oranges.
    
    Try uploading different images to test it out!
    """)
