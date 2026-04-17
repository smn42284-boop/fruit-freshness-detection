
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os

# Load your trained model
@st.cache_resource
def load_my_model():
    # Note: Make sure your model file is in the same directory
    model = load_model('fruit_freshness_model.h5')
    return model

model = load_my_model()

st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")
st.title("🍎 Fruit Freshness Detector")
st.write("Upload a photo of a fruit or vegetable to check if it's fresh or rotten!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This AI model can classify fruits and vegetables as fresh or rotten.")
    st.write("**Supported produce:** Apples, Bananas, Oranges, Tomatoes, Cucumbers, Carrots")
    st.write(f"**Model Accuracy:** 96.29% on test data")

    st.header("Instructions")
    st.write("1. Upload an image of a fruit or vegetable")
    st.write("2. Wait for the AI to analyze")
    st.write("3. See the prediction and confidence score")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, width=300)

    # Preprocess and predict
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    with col2:
        st.subheader("Result")

        if prediction > 0.5:
            st.error(f"⚠️ **ROTTEN**")
            st.write(f"Confidence: {prediction*100:.1f}%")
            st.write("❌ This produce appears spoiled. Do not consume.")
        else:
            st.success(f"✅ **FRESH**")
            st.write(f"Confidence: {(1-prediction)*100:.1f}%")
            st.write("✓ This produce appears fresh and safe to eat.")

    # Additional info
    st.divider()
    st.caption(f"Model confidence: {prediction*100 if prediction > 0.5 else (1-prediction)*100:.1f}%")
