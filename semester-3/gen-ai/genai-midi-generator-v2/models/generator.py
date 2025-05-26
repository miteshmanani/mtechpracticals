# models/generator.py
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, note_dim, time_steps, num_conditions):
        super(Generator, self).__init__()
        self.note_dim = note_dim  # 128
        self.time_steps = time_steps  # 32
        self.condition_dim = num_conditions

        # Condition embedding
        self.condition_embed = nn.Embedding(num_conditions, 32)

        # Project latent vector + condition
        # Start with 8x4 feature map
        self.project = nn.Linear(latent_dim + 32, 256 * 8 * 4)

        # Convolutional layers
        self.model = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),  # (16, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),   # (32, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(
                2, 2), padding=(1, 1)),    # (64, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(2, 1), stride=(
                2, 1), padding=(0, 0)),     # (128, 32)
            nn.Sigmoid()
        )

    def forward(self, z, condition):
        c = self.condition_embed(condition)  # (batch_size, 32)
        x = torch.cat([z, c], dim=1)  # (batch_size, latent_dim + 32)
        x = self.project(x).view(-1, 256, 8, 4)  # (batch_size, 256, 8, 4)
        return self.model(x).squeeze(1)  # (batch_size, 128, 32)
