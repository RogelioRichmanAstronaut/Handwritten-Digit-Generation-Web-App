import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

# Importa la definición del modelo
from model import VAE

# --- Configuración ---
# Se recomienda ejecutar este script en Google Colab con una GPU T4
# Para habilitarla: Entorno de ejecución -> Cambiar tipo de entorno de ejecución -> T4 GPU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hiperparámetros
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 20 # Dimensión del espacio latente

# --- Carga de Datos (MNIST) ---
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Inicialización del Modelo y Optimizador ---
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Función de Pérdida VAE ---
def loss_function(recon_x, x, mu, logvar):
    # Pérdida de reconstrucción (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Divergencia KL: mide qué tan similar es el espacio latente a una gaussiana estándar
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Bucle de Entrenamiento ---
print("Iniciando entrenamiento del VAE...")
model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch.view(-1, 784), data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')

print("Entrenamiento finalizado.")

# --- Guardar el Modelo Entrenado ---
MODEL_PATH = 'vae_mnist.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")


# --- Calcular y Guardar Vectores Latentes Promedio ---
print("Calculando vectores latentes promedio por dígito...")
model.eval()

# Usar un dataloader sin shuffle para procesar todo el dataset en orden
full_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

latent_vectors = defaultdict(list)
with torch.no_grad():
    for data, labels in full_loader:
        data = data.to(DEVICE)
        mu, _ = model.encode(data)
        for i in range(len(labels)):
            label = labels[i].item()
            latent_vectors[label].append(mu[i].cpu().numpy())

mean_latent_vectors = torch.zeros((10, LATENT_DIM))
for digit in range(10):
    vectors = np.array(latent_vectors[digit])
    mean_vector = np.mean(vectors, axis=0)
    mean_latent_vectors[digit] = torch.from_numpy(mean_vector)

# Guardar los vectores promedio
VECTORS_PATH = 'mean_latent_vectors.pth'
torch.save(mean_latent_vectors, VECTORS_PATH)
print(f"Vectores latentes promedio guardados en: {VECTORS_PATH}")
print("¡Proceso completado!")