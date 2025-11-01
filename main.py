import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
import pandas as pd

image = None

# --- SNI Reference Table ---
sni_data = {
    "Parameter": [
        "Kadar air (maks)",
        "Protein kasar (min)",
        "Lemak kasar (min)",
        "Serat kasar (maks)",
        "Abu (maks)",
    ],
    "PreStarter": [14, 22, 5, 4, 8],
    "Starter": [14, 20, 5, 5, 8],
    "Finisher": [14, 19, 5, 6, 8],
}
sni_df = pd.DataFrame(sni_data)
sni_df.index = range(1, len(sni_df) + 1)
sni_df.index.name = "No"

# --- Validation logic ---
def check_feed(sample, tipe):
    sni = sni_df.set_index("Parameter")[tipe]

    # All checks according to SNI thresholds
    if (
        sample["Kadar air (maks)"] <= sni["Kadar air (maks)"] and
        sample["Protein kasar (min)"] >= sni["Protein kasar (min)"] and
        sample["Lemak kasar (min)"] >= sni["Lemak kasar (min)"] and
        sample["Serat kasar (maks)"] <= sni["Serat kasar (maks)"] and
        sample["Abu (maks)"] <= sni["Abu (maks)"]
    ):
        return True
    else:
        return False

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

st.title("Model Klasifikasi dan Estimasi Kandungan Nutrisi Pakan Ayam Broiler")

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
    
st.header("Citra yang akan diklasifikasi dan diestimasi")
if not image:
    st.write("No Image inserted")
else:
    st.image(image)

classify_button = st.button("Klasifikasi dan estimasi", disabled=not image)

if classify_button and image != None:
    label, bahan_kering, kadar_air, protein_kasar, lemak_kasar, serat_kasar, abu = predict(image)
    st.header("Kategori pakan ayam Broiler ini adalah: " + label)
    st.header("Estimasi kandungan nutrisi (%), sebagai berikut:")
    st.write("Bahan Kering: ", "{:.2f}%".format(bahan_kering))
    st.write("Kadar Air: ", "{:.2f}%".format(kadar_air))
    st.write("Protein Kasar: ", "{:.2f}%".format(protein_kasar))
    st.write("Lemak Kasar: ", "{:.2f}%".format(lemak_kasar))
    st.write("Serat Kasar: ", "{:.2f}%".format(serat_kasar))
    st.write("Abu: ", "{:.2f}%".format(abu))

    sample = {
        "Kadar air (maks)": kadar_air,
        "Protein kasar (min)": protein_kasar,
        "Lemak kasar (min)": lemak_kasar,
        "Serat kasar (maks)": serat_kasar,
        "Abu (maks)": abu,
    }

    if (check_feed(sample, label)):
        st.subheader("Berdasarkan tabel SNI di bawah, pakan ini memenuhi syarat dan layak digunakan")
    else:
        st.subheader("Berdasarkan tabel SNI di bawah, pakan ini tidak memenuhi syarat dan layak digunakan")

    # Title and subtitle
    st.header("Persyaratan Mutu Pakan Ayam Ras Pedaging (Broiler)")
    st.subheader("Standar Nasional Indonesia (SNI), Badan Standardisasi Nasional (BSN)")

    # Create dataframe
    data = {
        "Parameter": [
            "Kadar air (maks)",
            "Protein kasar (min)",
            "Lemak kasar (min)",
            "Serat kasar (maks)",
            "Abu (maks)",
        ],
        "Pre Starter (%)": [14, 22, 5, 4, 8],
        "Starter (%)": [14, 20, 5, 5, 8],
        "Finisher (%)": [14, 19, 5, 6, 8],
    }

    df = pd.DataFrame(data)

    # Add index starting from 1
    df.index = range(1, len(df) + 1)
    df.index.name = "No"

    # Display dataframe with styling
    st.dataframe(
        df.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )

    # Optional: add a note or caption
    st.caption("Sumber: SNI - BSN (Standar Nasional Indonesia)")


