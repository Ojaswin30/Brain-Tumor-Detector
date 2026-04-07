import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model('BrainTumor10epochs.h5')

INPUT_SIZE = 64

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check for tumor")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, INPUT_SIZE, INPUT_SIZE, 3))

    # Prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Tumor Detected (Confidence: {1 - prediction:.2f})")