import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator
from utils import DataLoader, MidiConverter
import os


class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(
            config.latent_dim, config.note_dim, config.time_steps, config.num_keys).to(self.device)
        self.discriminator = Discriminator(
            config.note_dim, config.time_steps, config.num_keys).to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(
        ), lr=config.lr, betas=(config.beta1, config.beta2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(
        ), lr=config.lr, betas=(config.beta1, config.beta2))
        self.data_loader = DataLoader(
            config.data_dir, max_files=config.max_files)
        self.midi_converter = MidiConverter()
        os.makedirs(config.output_dir, exist_ok=True)

    def compute_gradient_penalty(self, real_samples, fake_samples, condition):
        if real_samples.shape != fake_samples.shape:
            raise ValueError(
                f"Shape mismatch: real_samples {real_samples.shape}, fake_samples {fake_samples.shape}")
        alpha = torch.rand(real_samples.size(0), 1, 1, device=self.device)
        alpha = alpha.expand(real_samples.size(
            0), real_samples.size(1), real_samples.size(2))
        interpolates = (alpha * real_samples + (1 - alpha)
                        * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, condition)
        fake = torch.ones(real_samples.size(0), 1, device=self.device)
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

    def evaluate_piano_roll(self, piano_roll):
        note_density = np.mean(piano_roll)
        pitch_counts = np.sum(piano_roll, axis=1)
        pitch_probs = pitch_counts / (pitch_counts.sum() + 1e-10)
        pitch_entropy = -np.sum([p * np.log2(p + 1e-10)
                                for p in pitch_probs if p > 0])
        return {"note_density": note_density, "pitch_entropy": pitch_entropy}

    def validate_generator_output(self):
        with torch.no_grad():
            test_z = torch.randn(1, self.config.latent_dim, device=self.device)
            test_key = torch.tensor([0], device=self.device)
            test_output = self.generator(test_z, test_key)
            assert test_output.shape[1:] == (self.config.note_dim, self.config.time_steps), \
                f"Generator output shape mismatch: {test_output.shape}"

    def train(self):
        self.validate_generator_output()
        data, key_labels = self.data_loader.load_data()
        data = data.to(self.device)
        key_labels = key_labels.to(self.device)

        for epoch in range(self.config.epochs):
            # Train Discriminator
            for _ in range(self.config.n_critic):
                idx = np.random.randint(
                    0, data.shape[0], self.config.batch_size)
                real_samples = data[idx]
                key_cond = key_labels[idx]
                z = torch.randn(self.config.batch_size,
                                self.config.latent_dim, device=self.device)
                fake_samples = self.generator(z, key_cond)
                real_loss = - \
                    torch.mean(self.discriminator(real_samples, key_cond))
                fake_loss = torch.mean(self.discriminator(
                    fake_samples.detach(), key_cond))
                gradient_penalty = self.compute_gradient_penalty(
                    real_samples, fake_samples, key_cond)
                d_loss = real_loss + fake_loss + \
                    self.config.gradient_penalty_weight * gradient_penalty
                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

            # Train Generator
            z = torch.randn(self.config.batch_size,
                            self.config.latent_dim, device=self.device)
            key_cond = torch.randint(
                0, self.config.num_keys, (self.config.batch_size,), device=self.device)
            fake_samples = self.generator(z, key_cond)
            g_loss = -torch.mean(self.discriminator(fake_samples, key_cond))
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # Log and save
            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                self.generator.eval()
                with torch.no_grad():
                    z = torch.randn(1, self.config.latent_dim,
                                    device=self.device)
                    key_cond = torch.tensor([0], device=self.device)  # C major
                    gen = self.generator(z, key_cond).cpu().numpy().squeeze(0)
                    binary = (gen > np.random.uniform(
                        0.3, 0.7, gen.shape)).astype(int)
                    self.midi_converter.piano_roll_to_midi(
                        binary, f"{self.config.output_dir}/generated_epoch_{epoch:04}_cmajor.mid")
                self.generator.train()
                torch.save(self.generator.state_dict(
                ), f"{self.config.output_dir}/generator_epoch_{epoch:04}.pth")

        # Final output for multiple keys
        self.generate_final_outputs()

    def generate_final_outputs(self):
        self.generator.eval()
        key_names = ["C_major", "C#_major", "D_major", "D#_major", "E_major", "F_major", "F#_major",
                     "G_major", "G#_major", "A_major", "A#_major", "B_major",
                     "C_minor", "C#_minor", "D_minor", "D#_minor", "E_minor", "F_minor",
                     "F#_minor", "G_minor", "G#_minor", "A_minor", "A#_minor", "B_minor"]
        with torch.no_grad():
            for key_idx in range(min(self.config.num_keys, 3)):
                z = torch.randn(1, self.config.latent_dim, device=self.device)
                key_cond = torch.tensor([key_idx], device=self.device)
                gen = self.generator(z, key_cond).cpu().numpy().squeeze(0)
                binary = (gen > 0.5).astype(int)
                self.midi_converter.piano_roll_to_midi(
                    binary, f"{self.config.output_dir}/generated_final_{key_names[key_idx]}.mid")
