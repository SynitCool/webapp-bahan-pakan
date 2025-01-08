import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
import pandas as pd

image = None

def predict(img_inp):
    image = cv2.imread(img_inp, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    model = "VGG-16_hyperparameter.keras"

    model = tf.keras.models.load_model(model)

    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)

    predictions = model.predict(image)
    
    df = pd.read_csv("dataset_flatten.csv")
    
    weighted_data = df.drop(columns=["Bahan Kering", "Kadar Air", "Protein Kasar", "Lemak Kasar", "Serat Kasar", "Abu", "Label"]).to_numpy()
    euclidian = np.sum(np.square(weighted_data - predictions), axis=1)

    index_min = np.argmin(euclidian)
    label = df["Label"][index_min]
    bahan_kering = df["Bahan Kering"][index_min]
    kadar_air = df["Kadar Air"][index_min]
    protein_kasar = df["Protein Kasar"][index_min]
    lemak_kasar = df["Lemak Kasar"][index_min]
    serat_kasar = df["Serat Kasar"][index_min]
    abu = df["Abu"][index_min]

    return label, bahan_kering, kadar_air, protein_kasar, lemak_kasar, serat_kasar, abu

def clasifying_image():
    print("Classfying", "image")

st.title("Klasifikasi dan Estimasi Bahan Pakan Ternak")

enable_camera = st.checkbox("Aktifkan Kamera")
camera = st.camera_input("Ambil Gambar", disabled=not enable_camera)
upload_image = st.file_uploader("Unggah Gambar", accept_multiple_files=False, type=["jpg", "png", "jpeg"], disabled=enable_camera)


if upload_image:
    # To read file as bytes:
    bytes_data = upload_image.getvalue()

    # Define the output file path
    output_file_path = 'temp/image.png'  

    # Open a file in write-binary mode and save the image bytes
    with open(output_file_path, 'wb') as file:
        file.write(bytes_data)

    image = output_file_path
    
if camera:
    # To read file as bytes:
    bytes_data = camera.read()

    # Define the output file path
    output_file_path = 'temp/image.png'  

    # Open a file in write-binary mode and save the image bytes
    with open(output_file_path, 'wb') as file:
        file.write(bytes_data)

    image = output_file_path
    
st.header("Gambar yang akan diklasifikasi")
if not image:
    st.write("No Image inserted")
else:
    st.image(image)

classify_button = st.button("Klasifikasi dan estimasi gambar!", disabled=not image)

if classify_button and image != None:
    label, bahan_kering, kadar_air, protein_kasar, lemak_kasar, serat_kasar, abu = predict(image)
    st.header("Diketahui bahwa gambar tersebut adalah: " + label)
    st.header("Dengan rincian sebagai berikut:")
    st.write("Bahan Kering: ", bahan_kering)
    st.write("Kadar Air: ", kadar_air)
    st.write("Protein Kasar: ", protein_kasar)
    st.write("Lemak Kasar: ", lemak_kasar)
    st.write("Serat Kasar: ", serat_kasar)
    st.write("Abu: ", abu)
