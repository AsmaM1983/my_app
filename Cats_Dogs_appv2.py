import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

# Charger le modèle TensorFlow Lite depuis le fichier .tflite sur GitHub
model_url = "https://github.com/AsmaM1983/my_app/raw/main/model_VGG16.tflite"
model_content = requests.get(model_url).content
model = tf.lite.Interpreter(model_content=model_content)
model.allocate_tensors()

def predict_image(img):
    # Redimensionner l'image à la taille attendue par le modèle (224, 224)
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Obtenir les détails des tenseurs d'entrée et de sortie
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Remplir le tenseur d'entrée avec les données de l'image
    model.set_tensor(input_details[0]['index'], img_array)

    # Effectuer l'inférence
    model.invoke()

    # Obtenir les prédictions à partir du tenseur de sortie
    predictions = model.get_tensor(output_details[0]['index'])

    if predictions[0, 0] > 0.5:
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
