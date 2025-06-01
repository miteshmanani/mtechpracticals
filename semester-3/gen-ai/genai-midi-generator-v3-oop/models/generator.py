import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, note_dim, time_steps, num_conditions):
        super(Generator, self).__init__()
        self.note_dim = note_dim
        self.time_steps = time_steps
        self.condition_dim = num_conditions
        self.condition_embed = nn.Embedding(num_conditions, 32)
        self.project = nn.Linear(latent_dim + 32, 256 * 8 * 4)
        self.model = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(
                4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=(
                2, 1), stride=(2, 1), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, z, condition):
        c = self.condition_embed(condition)
        x = torch.cat([z, c], dim=1)
        x = self.project(x).view(-1, 256, 8, 4)
        return self.model(x).squeeze(1)
