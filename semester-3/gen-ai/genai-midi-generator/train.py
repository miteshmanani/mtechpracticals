# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator
from utils import load_dataset
from utils import piano_roll_to_midi

# Config
latent_dim = 100
note_dim = 128
time_steps = 32
input_dim = note_dim * time_steps
batch_size = 32
epochs = 1000

# Load data
data = load_dataset('data', max_files=500)
data = torch.tensor(data, dtype=torch.float32)

# Models
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss
adversarial_loss = nn.BCELoss()

for epoch in range(epochs):
    idx = np.random.randint(0, data.shape[0], batch_size)
    real_samples = data[idx]

    # Generate fake samples
    z = torch.randn(batch_size, latent_dim)
    fake_samples = generator(z)

    # Train Discriminator
    real_labels = torch.ones((batch_size, 1))
    fake_labels = torch.zeros((batch_size, 1))

    real_loss = adversarial_loss(discriminator(real_samples), real_labels)
    fake_loss = adversarial_loss(discriminator(
        fake_samples.detach()), fake_labels)
    d_loss = real_loss + fake_loss

    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    g_loss = adversarial_loss(discriminator(fake_samples), real_labels)

    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()
    if epoch % 100 == 0:
        generator.eval()
        with torch.no_grad():
            z = torch.randn(1, latent_dim)
            gen = generator(z).cpu().numpy()[0]
            binary = (gen > 0.5).astype(int)
            piano_roll_to_midi(
                binary, f"output/generated_epoch_{epoch:04}.mid")
        generator.train()

    if epoch % 50 == 0:
        print(
            f"[Epoch {epoch}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

generator.eval()
with torch.no_grad():
    z = torch.randn(1, latent_dim)
    generated_roll = generator(torch.randn(1, latent_dim)).detach().numpy()[0]
    # Ensure shape is correct
    if generated_roll.shape[0] != 128 * 32:
        raise ValueError(f"Expected shape (4096,), got {generated_roll.shape}")
    binary_roll = (generated_roll > 0.5).astype(int).reshape(128, 32)
    print("Generated roll shape:", generated_roll.shape)
    print("Binary roll shape:", binary_roll.shape)
    piano_roll_to_midi(binary_roll, "output/generated_001.mid")
generator.train()
