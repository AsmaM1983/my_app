import streamlit as st
from keras.applications import VGG16
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

# Charger le modèle Keras depuis le fichier SavedModel sur GitHub
model_url = "https://github.com/AsmaM1983/my_app/blob/main/model_VGG16.tflite"
model = tf.keras.models.load_model(model_url)

def predict_image(img):
    # Redimensionner l'image à la taille attendue par le modèle (224, 224)
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0, 0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'

st.title("Image Classifier - Streamlit")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Faire une prédiction avec le modèle
    result = predict_image(image)
    st.write(f"Prediction: {result}")
