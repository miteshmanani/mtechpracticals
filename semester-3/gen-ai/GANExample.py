import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
image_channels = 3
image_size = 128  # Increased to 128x128
epochs = 500  # Increased to 500
batch_size = 32
lr = 0.0002
beta1 = 0.5

# Create directories
input_dir = r"C:\mtechpracticalsdatasets\janu"
dataset_dir = r"C:\mtechpracticalsdatasets\janu\dataset"
generated_dir = r"C:\mtechpracticalsdatasets\janu\generated"
checkpoint_dir = r"C:\mtechpracticalsdatasets\janu\checkpoints"

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Preprocess and standardize images


def preprocess_images(input_dir, output_dir):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            # Denormalize and save
            img = (img * 0.5 + 0.5) * 255
            img = img.byte()
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, filename), 'JPEG')


# Preprocess all images
print("Preprocessing images...")
preprocess_images(input_dir, dataset_dir)

# Custom Dataset


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.images = [os.path.join(directory, f) for f in os.listdir(
            directory) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = transforms.Resize((image_size, image_size))(img)
        img = self.transform(img)
        return img

# Generator (adjusted for 128x128)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 64 x 64
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 128 x 128
        )

    def forward(self, x):
        return self.main(x)

# Discriminator (adjusted for 128x128)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 8 x 8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 1024 x 4 x 4
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Initialize networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(),
                         lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Data loading
dataset = ImageDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop with checkpointing
print("Starting training...")
best_g_loss = float('inf')
patience = 50  # Early stopping patience
patience_counter = 0

for epoch in range(epochs):
    for i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Labels
        real_label = torch.ones(batch_size, 1, 1, 1).to(device)
        fake_label = torch.zeros(batch_size, 1, 1, 1).to(device)

        # Train Discriminator
        discriminator.zero_grad()
        output = discriminator(real_images)
        d_loss_real = criterion(output, real_label)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output = discriminator(fake_images.detach())
        d_loss_fake = criterion(output, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        generator.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_label)
        g_loss.backward()
        g_optimizer.step()

    # Save generated images
    with torch.no_grad():
        fake_images = generator(torch.randn(
            16, latent_dim, 1, 1, device=device))
        fake_images = (fake_images + 1) / 2  # Denormalize
        # Still output 100x100 as requested
        resize_transform = transforms.Resize((100, 100))
        fake_images = torch.stack([resize_transform(img)
                                  for img in fake_images])
        save_image(fake_images, f"{generated_dir}/epoch_{epoch+1}.jpg")

    # Checkpointing and overfitting monitoring
    print(
        f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    if g_loss.item() < best_g_loss:
        best_g_loss = g_loss.item()
        patience_counter = 0
        # Save best model
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'epoch': epoch,
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }, f"{checkpoint_dir}/best_model.pth")
    else:
        patience_counter += 1

    # Early stopping
    # if patience_counter >= patience:
    #    print(f"Early stopping triggered at epoch {epoch+1}")
    #    break

print("Training completed!")
