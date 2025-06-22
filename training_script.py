import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

# Import the model definition
from model import VAE

# --- Configuration ---
# It is recommended to run this script in Google Colab with a T4 GPU
# To enable it: Runtime -> Change runtime type -> T4 GPU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 20 # Latent space dimension

# --- Data Loading (MNIST) ---
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model and Optimizer Initialization ---
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- VAE Loss Function ---
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL divergence: measures how similar the latent space is to a standard Gaussian
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Training Loop ---
print("Starting VAE training...")
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

print("Training finished.")

# --- Save Trained Model ---
MODEL_PATH = 'vae_mnist.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")


# --- Calculate and Save Mean Latent Vectors ---
print("Calculating mean latent vectors per digit...")
model.eval()

# Use a dataloader without shuffle to process the entire dataset in order
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

# Save the mean vectors
VECTORS_PATH = 'mean_latent_vectors.pth'
torch.save(mean_latent_vectors, VECTORS_PATH)
print(f"Mean latent vectors saved to: {VECTORS_PATH}")
print("Process completed!")