import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) para el dataset MNIST.
    """
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Convierte una imagen de 28x28 a un espacio latente
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1) # -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # -> 7x7
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder: Convierte un punto del espacio latente a una imagen de 28x28
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        """Paso de codificación."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Aplanar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Truco de reparametrización para permitir el backpropagation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Paso de decodificación."""
        x = F.relu(self.decoder_fc(z))
        x = x.view(-1, 32, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x)) # Usar sigmoide para valores de píxel entre 0 y 1
        return x

    def forward(self, x):
        """Paso completo: codificar, reparametrizar y decodificar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar