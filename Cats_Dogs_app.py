import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

# Configurer PyDrive pour accéder à Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # La première fois que vous exécutez cela, il vous redirigera vers une URL pour autoriser l'accès à Google Drive
drive = GoogleDrive(gauth)

# ID du fichier .tflite sur Google Drive
file_id = "13tTHxrhol_iaEBS0qx_FFwX7JEJo0S2f"

# Télécharger le fichier .tflite à partir de Google Drive
file = drive.CreateFile({'id': file_id})
file.GetContentFile('model_VGG16.tflite')

# Charger le modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model_VGG16.tflite')
interpreter.allocate_tensors()

def predict_image(img):
    # Redimensionner l'image à la taille attendue par le modèle (224, 224)
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Préparer l'entrée pour le modèle TensorFlow Lite
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, img_array)

    # Exécuter l'inférence
    interpreter.invoke()

    # Obtenir la sortie du modèle TensorFlow Lite
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
    prediction = output()[0]

    if prediction > 0.5:
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
