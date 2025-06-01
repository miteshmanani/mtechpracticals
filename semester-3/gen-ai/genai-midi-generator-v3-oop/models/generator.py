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
        self.project = nn.Linear(latent_dim + 32, 256 * 8 * 4)

        # Convolutional layers
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(True)
        self.conv_transpose1 = nn.ConvTranspose2d(
            256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # (16, 8)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))  # (32, 16)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(True)
        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))  # (64, 32)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(True)
        self.conv_transpose4 = nn.ConvTranspose2d(32, 1, kernel_size=(
            2, 1), stride=(2, 1), padding=(0, 0))  # (128, 32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, condition):
        # Embed condition
        c = self.condition_embed(condition)  # (batch_size, 32)
        x = torch.cat([z, c], dim=1)  # (batch_size, latent_dim + 32)
        x = self.project(x).view(-1, 256, 8, 4)  # (batch_size, 256, 8, 4)

        # Apply layers explicitly
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv_transpose1(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv_transpose2(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.conv_transpose3(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.conv_transpose4(x)
        x = self.sigmoid(x)

        return x.squeeze(1)  # (batch_size, 128, 32)
