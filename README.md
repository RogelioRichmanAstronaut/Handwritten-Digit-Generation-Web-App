# Generador de Dígitos Manuscritos con VAE y Streamlit

Este proyecto contiene el código para entrenar un Autoencoder Variacional (VAE) en el dataset MNIST y una aplicación web con Streamlit para generar nuevos dígitos.

## Cómo ejecutar el proyecto

Sigue estos pasos en orden:

### Paso 1: Entrenar el Modelo

El modelo debe ser entrenado primero para generar los archivos de pesos y los vectores latentes.

1.  **Entorno:** Se recomienda usar Google Colab con una GPU T4 para ejecutar el script de entrenamiento.
2.  **Ejecutar el script:** Sube y ejecuta el archivo `training_script.py` en tu entorno.
    ```bash
    python training_script.py
    ```
3.  **Resultados:** Después de la ejecución (tardará unos minutos), se crearán dos archivos en el mismo directorio:
    *   `vae_mnist.pth`: Los pesos del modelo VAE entrenado.
    *   `mean_latent_vectors.pth`: Los vectores latentes promedio para cada dígito (0-9).

### Paso 2: Ejecutar la Aplicación Web Localmente

Una vez que tengas los dos archivos generados, puedes ejecutar la aplicación Streamlit.

1.  **Requisitos:** Asegúrate de que los archivos `vae_mnist.pth` y `mean_latent_vectors.pth` están en la misma carpeta que `app.py`.
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecutar la app:**
    ```bash
    streamlit run app.py
    ```
    La aplicación se abrirá en tu navegador.

### Paso 3: Desplegar en Streamlit Community Cloud (Opcional)

1.  Sube todos los archivos (`app.py`, `model.py`, `requirements.txt`, `vae_mnist.pth`, `mean_latent_vectors.pth`) a un repositorio público de GitHub.
    *   **Nota:** Si los archivos de pesos son muy grandes, es posible que necesites usar Git LFS (Large File Storage).
2.  Ve a [Streamlit Community Cloud](https://share.streamlit.io/), conecta tu cuenta de GitHub y despliega la aplicación desde tu repositorio.