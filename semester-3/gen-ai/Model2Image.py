import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from datetime import datetime

# Define the Generator class (make sure it matches the one used during training)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0,
                               bias=False),   # 100 → 1024
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1,
                               bias=False),  # 1024 → 512
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1,
                               bias=False),  # 512 → 256
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1,
                               bias=False),      # 256 → 128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf, 64, 4, 2, 1, bias=False),           # 128 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, nc, 4, 2, 1, bias=False),            # 64 → 3
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Configs
nz = 100  # size of latent vector
ngf = 128  # generator feature map size
nc = 3    # output image channels
num_images = 5
model_path = r"C:\mtechpracticalsdatasets\janu\checkpoints\best_model.pth"
output_dir = "./generated_images"

os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator and load weights
generator = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
checkpoint = torch.load(model_path, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])  # ✅ correct key
generator.eval()

# Generate random noise and produce images
with torch.no_grad():
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    fake_images = generator(noise).detach().cpu()

# Save generated images
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
vutils.save_image(fake_images, os.path.join(
    output_dir, f"generated_{timestamp}.png"), normalize=True, nrow=num_images)

print(
    f"✅ Successfully generated {num_images} images. Check '{output_dir}' folder.")
