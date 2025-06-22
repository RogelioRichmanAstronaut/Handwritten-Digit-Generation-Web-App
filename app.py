import streamlit as st
import torch
import numpy as np
import os

# Importa la definición del modelo desde model.py
from model import VAE

# --- Configuración de la App ---
st.set_page_config(layout="wide")
st.title("Generador de Dígitos Manuscritos con VAE")

# --- Carga del Modelo y Vectores ---

# Constantes
LATENT_DIM = 20
MODEL_PATH = 'vae_mnist.pth'
VECTORS_PATH = 'mean_latent_vectors.pth'
DEVICE = torch.device("cpu") # Streamlit corre en CPU

@st.cache_resource
def load_model():
    """Carga el decodificador del VAE y los pesos entrenados."""
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_data
def load_mean_vectors():
    """Carga los vectores latentes promedio por dígito."""
    if not os.path.exists(VECTORS_PATH):
        return None
    
    mean_vectors = torch.load(VECTORS_PATH, map_location=DEVICE)
    return mean_vectors

model = load_model()
mean_vectors = load_mean_vectors()


# --- Interfaz de Usuario ---
if model is None or mean_vectors is None:
    st.error(
        "Error: No se encontraron los archivos del modelo. "
        "Por favor, ejecuta `training_script.py` primero para generar "
        "`vae_mnist.pth` y `mean_latent_vectors.pth`, y colócalos en la misma carpeta que `app.py`."
    )
else:
    st.success("Modelo cargado correctamente.")
    st.markdown("""
    Selecciona un dígito del 0 al 9 y presiona "Generar".
    La aplicación creará 5 imágenes nuevas de ese dígito utilizando el modelo generativo.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        digit = st.selectbox(
            label="Elige un dígito para generar:",
            options=list(range(10))
        )
        
        noise_level = st.slider(
            "Nivel de diversidad (ruido):",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1
        )

        generate_button = st.button("Generar 5 imágenes", type="primary")

    with col2:
        if generate_button:
            st.subheader(f"Imágenes generadas para el dígito: {digit}")
            
            # Obtener el vector latente promedio para el dígito seleccionado
            mean_vector = mean_vectors[digit]

            # Crear 5 columnas para mostrar las imágenes
            image_cols = st.columns(5)

            with torch.no_grad():
                for i in range(5):
                    # Añadir ruido gaussiano al vector promedio para generar diversidad
                    noise = torch.randn(LATENT_DIM) * noise_level
                    z = mean_vector + noise
                    
                    # Generar la imagen usando el decodificador
                    generated_image_tensor = model.decode(z.to(DEVICE))
                    
                    # Convertir a un formato que Streamlit pueda mostrar
                    generated_image_np = generated_image_tensor.cpu().numpy().reshape(28, 28)
                    
                    # Mostrar la imagen en su columna
                    image_cols[i].image(generated_image_np, caption=f"Imagen {i+1}", use_column_width=True)

        else:
            st.info("Esperando para generar imágenes...")