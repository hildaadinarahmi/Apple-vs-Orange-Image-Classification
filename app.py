import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelWrapper:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

@st.cache_resource
def load_model():
    try:
        with open("fruit_classifier_model.pkl", "rb") as f:  # Open in read-binary mode
            wrapper = pickle.load(f)  # Load the model from the file
            return wrapper.model.eval(), wrapper.class_names
    except FileNotFoundError:
        st.error("🚫 Model file not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# App config
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

st.markdown("<h1 style='text-align: center;'>🍎🍊 Fruit Classifier: Apple vs Orange</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image, and let the model predict whether it's an <b>Apple</b> or an <b>Orange</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📤 Upload your fruit image", type=["jpg", "jpeg", "png"])
    
with col2:
    st.markdown("### 📝 How it works")
    st.markdown("""
    - Trained on basic visual features like **color** and **shape**.
    - Preprocessed to 128x128 pixels.
    - Uses a simple image classifier model (PyTorch + Pickle).
    """)

st.markdown("---")

model, class_names = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    try:
        img_array = np.array(image) / 255.0
        if img_array.shape[2] == 4:  # Handle RGBA images
            img_array = img_array[:, :, :3]
        img_tensor = torch.tensor(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
        img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            predicted_class = class_names[pred_idx]
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item() * 100

        if predicted_class.lower() == "apple":
            st.success(f"🍎 It's an **Apple** with **{confidence:.2f}%** confidence!")
        else:
            st.warning(f"🍊 It's an **Orange** with **{confidence:.2f}%** confidence!")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
