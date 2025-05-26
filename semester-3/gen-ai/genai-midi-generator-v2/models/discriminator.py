# models/discriminator.py
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, note_dim, time_steps, num_conditions):
        super(Discriminator, self).__init__()
        self.condition_dim = num_conditions
        self.condition_embed = nn.Embedding(num_conditions, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(
                2, 2), padding=(1, 1)),  # (64, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(
                2, 2), padding=(1, 1)),  # (32, 8)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(
                2, 2), padding=(1, 1)),  # (16, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(
                2, 2), padding=(1, 1)),  # (8, 2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512 * 8 * 2 + 32, 1)

    def forward(self, x, condition):
        if x.dim() != 4 or x.size(1) != 1:
            x = x.unsqueeze(1)  # [batch_size, 1, 128, 32]
        c = self.condition_embed(condition)  # [batch_size, 32]
        x = self.conv(x)  # [batch_size, 512 * 8 * 2]
        x = torch.cat([x, c], dim=1)  # [batch_size, 512 * 8 * 2 + 32]
        return self.fc(x)  # [batch_size, 1]
