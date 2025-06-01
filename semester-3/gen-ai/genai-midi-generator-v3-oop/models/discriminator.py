import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, note_dim, time_steps, num_conditions):
        super(Discriminator, self).__init__()
        self.condition_dim = num_conditions

        # Condition embedding
        self.condition_embed = nn.Embedding(num_conditions, 32)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(
            2, 2), padding=(1, 1))  # Output: (64, 16)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))  # Output: (32, 8)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))  # Output: (16, 4)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(
            4, 4), stride=(2, 2), padding=(1, 1))  # Output: (8, 2)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(512 * 8 * 2 + 32, 1)

    def forward(self, x, condition):
        if x.dim() != 4 or x.size(1) != 1:
            x = x.unsqueeze(1)  # [batch_size, 1, 128, 32]

        # Embed condition
        c = self.condition_embed(condition)  # [batch_size, 32]

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.flatten(x)  # [batch_size, 512 * 8 * 2]

        # Concatenate condition embedding and apply fully connected layer
        x = torch.cat([x, c], dim=1)  # [batch_size, 512 * 8 * 2 + 32]
        return self.fc(x)  # [batch_size, 1]
