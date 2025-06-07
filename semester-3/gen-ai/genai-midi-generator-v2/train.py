# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator
from utils import load_dataset, piano_roll_to_midi
import os
import visualkeras

# Config
latent_dim = 256
note_dim = 128
time_steps = 32
num_keys = 24
batch_size = 64
epochs = 1000
n_critic = 5
lr = 0.0001
beta1, beta2 = 0.5, 0.9
gradient_penalty_weight = 10
output_dir = "C:/mtechpracticals/semester-3/gen-ai/genai-midi-generator-v2/output"
os.makedirs(output_dir, exist_ok=True)

# Load data with key labels
data, key_labels = load_dataset(
    'C:/mtechpracticals/semester-3/gen-ai/genai-midi-generator-v2/data', max_files=1000)
data = torch.tensor(data, dtype=torch.float32).reshape(-1,
                                                       note_dim, time_steps)
key_labels = torch.tensor(key_labels, dtype=torch.long)
assert data.shape[1:] == (
    note_dim, time_steps), f"Data shape mismatch: {data.shape}"

# Models
generator = Generator(latent_dim, note_dim, time_steps, num_keys)
discriminator = Discriminator(note_dim, time_steps, num_keys)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
data = data.to(device)
key_labels = key_labels.to(device)

# Optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(),
                         lr=lr, betas=(beta1, beta2))

# Validate generator output shape
with torch.no_grad():
    test_z = torch.randn(1, latent_dim, device=device)
    test_key = torch.tensor([0], device=device)
    test_output = generator(test_z, test_key)
    assert test_output.shape[1:] == (
        note_dim, time_steps), f"Generator output shape mismatch: {test_output.shape}"

# Gradient penalty for WGAN-GP


def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition, device):
    if real_samples.shape != fake_samples.shape:
        raise ValueError(
            f"Shape mismatch: real_samples {real_samples.shape}, fake_samples {fake_samples.shape}")
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand(real_samples.size(
        0), real_samples.size(1), real_samples.size(2))
    interpolates = (alpha * real_samples + (1 - alpha)
                    * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, condition)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Evaluation metric


def evaluate_piano_roll(piano_roll):
    note_density = np.mean(piano_roll)
    pitch_counts = np.sum(piano_roll, axis=1)
    pitch_probs = pitch_counts / (pitch_counts.sum() + 1e-10)
    pitch_entropy = -np.sum([p * np.log2(p + 1e-10)
                            for p in pitch_probs if p > 0])
    return {"note_density": note_density, "pitch_entropy": pitch_entropy}


# Training loop
for epoch in range(epochs):
    # Train Discriminator
    for _ in range(n_critic):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_samples = data[idx]
        key_cond = key_labels[idx]
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(z, key_cond)
        real_loss = -torch.mean(discriminator(real_samples, key_cond))
        fake_loss = torch.mean(discriminator(fake_samples.detach(), key_cond))
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_samples, fake_samples, key_cond, device)
        d_loss = real_loss + fake_loss + gradient_penalty_weight * gradient_penalty
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

    # Train Generator
    z = torch.randn(batch_size, latent_dim, device=device)
    key_cond = torch.randint(0, num_keys, (batch_size,), device=device)
    fake_samples = generator(z, key_cond)
    g_loss = -torch.mean(discriminator(fake_samples, key_cond))
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    # Log and save
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        generator.eval()
        with torch.no_grad():
            z = torch.randn(1, latent_dim, device=device)
            key_cond = torch.tensor([0], device=device)  # C major
            gen = generator(z, key_cond).cpu().numpy().squeeze(0)
            binary = (gen > np.random.uniform(0.3, 0.7, gen.shape)).astype(int)
            piano_roll_to_midi(
                binary, f"{output_dir}/generated_epoch_{epoch:04}_cmajor.mid")
        generator.train()
        torch.save(generator.state_dict(),
                   f"{output_dir}/generator_epoch_{epoch:04}.pth")

# Final output for multiple keys
generator.eval()
key_names = ["C_major", "C#_major", "D_major", "D#_major", "E_major", "F_major", "F#_major",
             "G_major", "G#_major", "A_major", "A#_major", "B_major",
             "C_minor", "C#_minor", "D_minor", "D#_minor", "E_minor", "F_minor",
             "F#_minor", "G_minor", "G#_minor", "A_minor", "A#_minor", "B_minor"]
with torch.no_grad():
    for key_idx in range(min(num_keys, 3)):
        z = torch.randn(1, latent_dim, device=device)
        key_cond = torch.tensor([key_idx], device=device)
        gen = generator(z, key_cond).cpu().numpy().squeeze(0)
        binary = (gen > 0.5).astype(int)
        piano_roll_to_midi(
            binary, f"{output_dir}/generated_final_{key_names[key_idx]}.mid")
model = generator.state_dict()
visualkeras.layered_view(model).show()  # display using your system viewer
visualkeras.layered_view(model, to_file='output.png')  # write to disk
visualkeras.layered_view(model, to_file='output.png').show()  # write and show
visualkeras.layered_view(model)
