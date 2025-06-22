import streamlit as st
import torch
import numpy as np
import os

# Import the model definition from model.py
from model import VAE

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Generator with VAE")

# --- Load Model and Vectors ---

# Constants
LATENT_DIM = 20
MODEL_PATH = 'vae_mnist.pth'
VECTORS_PATH = 'mean_latent_vectors.pth'
DEVICE = torch.device("cpu") # Streamlit runs on CPU

@st.cache_resource
def load_model():
    """Loads the VAE and the trained weights."""
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_data
def load_mean_vectors():
    """Loads the mean latent vectors per digit."""
    if not os.path.exists(VECTORS_PATH):
        return None
    
    mean_vectors = torch.load(VECTORS_PATH, map_location=DEVICE)
    return mean_vectors

model = load_model()
mean_vectors = load_mean_vectors()


# --- User Interface ---
if model is None or mean_vectors is None:
    st.error(
        "Error: Model files not found. "
        "Please run `training_script.py` first to generate "
        "`vae_mnist.pth` and `mean_latent_vectors.pth`, and place them in the same folder as `app.py`."
    )
else:
    st.success("Model loaded successfully.")
    st.markdown("""
    Select a digit from 0 to 9 and press "Generate".
    The application will create 5 new images of that digit using the generative model.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        digit = st.selectbox(
            label="Choose a digit to generate:",
            options=list(range(10))
        )
        
        noise_level = st.slider(
            "Diversity level (noise):",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1
        )

        generate_button = st.button("Generate 5 images", type="primary")

    with col2:
        if generate_button:
            st.subheader(f"Generated images for digit: {digit}")
            
            # Get the mean latent vector for the selected digit
            mean_vector = mean_vectors[digit]

            # Create 5 columns to display the images
            image_cols = st.columns(5)

            with torch.no_grad():
                for i in range(5):
                    # Add Gaussian noise to the mean vector to generate diversity
                    noise = torch.randn(LATENT_DIM) * noise_level
                    z = mean_vector + noise
                    
                    # Generate the image using the decoder
                    generated_image_tensor = model.decode(z.to(DEVICE))
                    
                    # Convert to a format that Streamlit can display
                    generated_image_np = generated_image_tensor.cpu().numpy().reshape(28, 28)
                    
                    # Display the image in its column
                    image_cols[i].image(generated_image_np, caption=f"Image {i+1}", use_column_width=True)

        else:
            st.info("Waiting to generate images...")