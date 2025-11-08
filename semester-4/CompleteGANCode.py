# Section 1: Imports and Environment Setup
import cv2
import numpy as np
import csv
import os
import time
import torch
import matplotlib.pyplot as plt
import warnings

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg16
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision import transforms

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage.color import lab2rgb

warnings.filterwarnings("ignore", message="Failed to initialize NumPy: _ARRAY_API not found")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##############   TEST CELL ##########################################
import torch
checkpoint_path = '/home/mmanani/ImageColourizationDataSet/checkpoints/gan_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))  # Load on CPU first
    print("Checkpoint loaded successfully!")
    print("Epoch:", checkpoint['epoch'])
    print("Loss:", checkpoint['loss'])
else:
    print("Checkpoint not found!")
##############   TEST CELL ##########################################

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import os

# Base directory
base_dir = r'/home/mmanani/ImageColourizationDataSet/ILSVRC/Data/DET/train'

# Get all subdirectories
sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Count images in each subdirectory
for folder in sub_dirs:
    folder_path = os.path.join(base_dir, folder)
    image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"{folder}: {image_count} images")

# Total count
total_images = sum(len([f for f in os.listdir(os.path.join(base_dir, d)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) 
                  for d in sub_dirs if os.path.isdir(os.path.join(base_dir, d)))
print(f"Total images: {total_images}")


# Hyperparameters
batch_size = 16              # ✅ Keep — good balance for stability vs GPU memory
pretrain_epochs = 15         # ⬆️ From 10 → 15 (lets generator learn color mapping better before adversarial training)
num_epochs = 300             # ⬆️ Slightly higher to allow slower GAN convergence
learning_rate_g = 1.2e-4     # ⬆️ From 1e-4 → 2e-4 (G learns slightly faster during pretraining & GAN phase)
learning_rate_d = 6e-5       # ⬇️ From 3e-4 → 1e-4 (slows D to prevent early dominance)
beta1, beta2 = 0.5, 0.999    # Adam betas — good default for GAN stability
lambda_L1 = 60              # Weight for L1 loss (keep)
lambda_perceptual = 1.0      # Weight for perceptual loss (tune if too smooth or too noisy)
lambda_gan = 1.2             # new — scale GAN loss a bit lower to prevent hallucinated hues

# Section 2: Improved Dataset Class for U-Net GAN
from torch.utils.data import Dataset
import torch
import os
import cv2
import numpy as np
import random
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, image_dir,transform_prob=0.5, max_size=7*1024*1024*1024):
        """
        image_dir: Root folder containing training images
        transform_prob: Probability for applying augmentations
        max_size: Cap dataset size (default: 5 GB)
        """
        self.image_paths = []
        total_size = 0

        # Walk directory and collect images up to max_size
        for root, _, files in os.walk(image_dir):
            random.shuffle(files)  # Helps distribute diversity
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    if total_size + size <= max_size:
                        self.image_paths.append(path)
                        total_size += size
                    else:
                        break
            if total_size >= max_size:
                break

        if not self.image_paths:
            raise ValueError(f"No valid images found in {image_dir}.")
        self.transform_prob = transform_prob
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError
        except Exception:
            # fallback to another random image
            return self.__getitem__(random.randint(0, len(self.image_paths)-1))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Occasionally apply augmentations
        if random.random() < self.transform_prob:
            img_rgb = np.array(self.augment(transforms.ToPILImage()(img_rgb)))

        # Convert to LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # Normalize L to [-1, 1] and ab to [-1, 1]
        l_channel = (img_lab[:, :, 0] / 50.0) - 1.0        # L in [-1, 1]
        ab_channels = (img_lab[:, :, 1:] - 128) / 128.0    # ab in [-1, 1]

        # Convert to tensors
        l_channel = torch.from_numpy(l_channel).float().unsqueeze(0)
        ab_channels = torch.from_numpy(ab_channels.transpose((2, 0, 1))).float()

        return {'L': l_channel, 'ab': ab_channels}
    
    # Path to dataset
image_dir = r'/home/mmanani/ImageColourizationDataSet/ILSVRC/Data/DET/train'
# Load dataset with correct transform
dataset = ColorizationDataset(image_dir)
# Split into train/val/test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Section 3: Model Definition
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, feature_maps=64):
        super(Generator, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = self._block(input_channels, feature_maps, normalize=False)  # 1 → 64
        self.enc2 = self._block(feature_maps, feature_maps * 2)
        self.enc3 = self._block(feature_maps * 2, feature_maps * 4)
        self.enc4 = self._block(feature_maps * 4, feature_maps * 8)
        self.enc5 = self._block(feature_maps * 8, feature_maps * 8)
        self.enc6 = self._block(feature_maps * 8, feature_maps * 8)
        self.enc7 = self._block(feature_maps * 8, feature_maps * 8)
        self.enc8 = self._block(feature_maps * 8, feature_maps * 8, normalize=False)

        # Decoder (Upsampling path with skip connections)
        self.dec1 = self._up_block(feature_maps * 8, feature_maps * 8, dropout=True)
        self.dec2 = self._up_block(feature_maps * 16, feature_maps * 8, dropout=True)
        self.dec3 = self._up_block(feature_maps * 16, feature_maps * 8, dropout=True)
        self.dec4 = self._up_block(feature_maps * 16, feature_maps * 8)
        self.dec5 = self._up_block(feature_maps * 16, feature_maps * 4)
        self.dec6 = self._up_block(feature_maps * 8, feature_maps * 2)
        self.dec7 = self._up_block(feature_maps * 4, feature_maps)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder with skip connections
        d1 = self.dec1(e8);  d1 = torch.cat([d1, e7], 1)
        d2 = self.dec2(d1);  d2 = torch.cat([d2, e6], 1)
        d3 = self.dec3(d2);  d3 = torch.cat([d3, e5], 1)
        d4 = self.dec4(d3);  d4 = torch.cat([d4, e4], 1)
        d5 = self.dec5(d4);  d5 = torch.cat([d5, e3], 1)
        d6 = self.dec6(d5);  d6 = torch.cat([d6, e2], 1)
        d7 = self.dec7(d6);  d7 = torch.cat([d7, e1], 1)
        d8 = self.dec8(d7)
        return d8


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input: (3, 256, 256)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),            # (64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),          # (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),         # (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),         # (512, 31, 31)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),           # (1, 30, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Apply weight initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

G.apply(weights_init_normal)
D.apply(weights_init_normal)

# Load pretrained VGG for perceptual loss
from torchvision.models import vgg16, VGG16_Weights
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

def perceptual_loss(pred, target):
    pred_norm = (pred - vgg_mean) / vgg_std
    target_norm = (target - vgg_mean) / vgg_std
    pred_features = vgg(pred_norm)
    target_features = vgg(target_norm)
    return nn.functional.mse_loss(pred_features, target_features)

# Loss functions
criterion_GAN = nn.BCELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=learning_rate_g, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=learning_rate_d, betas=(beta1, beta2))

# Section 5: Training Loop
# Assuming these are defined earlier: device, pretrain_epochs, num_epochs, train_loader, G, D, optimizer_G, optimizer_D, criterion_L1, perceptual_loss
# train_step definition with error fix
def train_step(G, D, optimizer_G, optimizer_D, real_ab, L):
    # Generator
    fake_ab = G(L)
    fake_img = torch.cat([L, fake_ab], dim=1)
    # Discriminator (real)
    real_img = torch.cat([L, real_ab], dim=1)
    real_pred = D(real_img)
    real_label = torch.clamp(1.0 - torch.rand_like(real_pred) * 0.1, 0.9, 1.0)
    # Discriminator (fake)
    fake_pred = D(fake_img.detach())
    fake_label = torch.clamp(torch.rand_like(fake_pred) * 0.1, 0.0, 0.1)
    # D loss
    d_loss_real = criterion_GAN(real_pred, real_label)
    d_loss_fake = criterion_GAN(fake_pred, fake_label)
    d_loss = (d_loss_real + d_loss_fake) / 2
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()
    # G loss
    fake_pred = D(fake_img)
    g_gan_loss = criterion_GAN(fake_pred, real_label)
    g_l1_loss = criterion_L1(fake_ab, real_ab) * 100
    g_perceptual_loss = perceptual_loss(fake_img, real_img)
    g_loss = g_gan_loss + g_l1_loss + g_perceptual_loss
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()
    return g_loss.item(), d_loss.item()
checkpoint_dir = '/home/mmanani/ImageColourizationDataSet/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to load checkpoint (updated for GAN to load both G and D)
def load_checkpoint(checkpoint_path, models, optimizers):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Detect format automatically
        if isinstance(checkpoint['model_state_dict'], dict) and 'G' in checkpoint['model_state_dict']:
            # Multi-model (GAN) checkpoint
            for key in models:
                models[key].load_state_dict(checkpoint['model_state_dict'][key])
            for key in optimizers:
                optimizers[key].load_state_dict(checkpoint['optimizer_state_dict'][key])
        else:
            # Single-model (Pretraining) checkpoint
            list(models.values())[0].load_state_dict(checkpoint['model_state_dict'])
            list(optimizers.values())[0].load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        start_batch = checkpoint.get('batch', 0)
        loss = checkpoint.get('loss', None)
        print(f"✅ Resumed from checkpoint at epoch {start_epoch}, batch {start_batch}, loss {loss}")
        return start_epoch, start_batch

    print("⚠️ No checkpoint found, starting fresh.")
    return 0, 0


# -----------------------------
# 2. PER-EPOCH COPY (MAIN CHECKPOINT UNCHANGED)
# -----------------------------
def save_per_epoch_copy(epoch, checkpoint_dir="checkpoints"):
    main_path = os.path.join(checkpoint_dir, 'gan_checkpoint.pth')
    if not os.path.exists(main_path):
        return
    copy_path = os.path.join(checkpoint_dir, f'gan_checkpoint_epoch_{epoch+1}.pth')
    import shutil
    shutil.copy2(main_path, copy_path)
    print(f"Epoch copy saved: {copy_path}")

# ==========================================
# Pretraining Generator (L1 + Perceptual)
# ==========================================
print("Pretraining Generator...")
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/colorization_gan')

# Load checkpoint if exists
start_epoch, start_batch = load_checkpoint(
    os.path.join(checkpoint_dir, 'pretrain_checkpoint.pth'),     
    {"G": G},
    {"optimizer_G": optimizer_G}
)

for epoch in range(start_epoch, pretrain_epochs):
    G.train()
    total_g_loss = 0.0
    start_time = time.time()

    # Handle resumption from a specific batch
    if epoch == start_epoch and start_batch > 0:
        print(f"⏩ Resuming from batch {start_batch} of epoch {start_epoch}")
        iterator = iter(train_loader)
        for _ in range(start_batch):
            next(iterator)
        batches = iterator
    else:
        batches = iter(train_loader)

    for batch_idx, batch in enumerate(
        tqdm(batches, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}", total=len(train_loader))
    ):
        # Skip batches before resumption point
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        L = batch['L'].to(device)
        real_ab = batch['ab'].to(device)

        # Forward pass
        fake_ab = G(L)

        # Compute losses
        g_l1_loss = criterion_L1(fake_ab, real_ab) * lambda_L1
        g_perceptual_loss = perceptual_loss(
            torch.cat([L, fake_ab], dim=1),
            torch.cat([L, real_ab], dim=1)
        ) * lambda_perceptual
        g_loss = g_l1_loss + g_perceptual_loss

        # Backpropagation
        optimizer_G.zero_grad(set_to_none=True)
        g_loss.backward()
        optimizer_G.step()

        total_g_loss += g_loss.item()

        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Pretrain/L1_Loss', g_l1_loss.item(), global_step)
        writer.add_scalar('Pretrain/Perceptual_Loss', g_perceptual_loss.item(), global_step)
        writer.add_scalar('Pretrain/Total_G_Loss', g_loss.item(), global_step)


        # Auto checkpoint every N batches
        if (batch_idx + 1) % 500 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'pretrain_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'loss': g_loss.item()
            }, checkpoint_path)
            print(f"💾 Saved intermediate checkpoint at batch {batch_idx + 1}")
    # End of epoch logging
    avg_g_loss = total_g_loss / len(train_loader)
    print(f"✅ Pretrain Epoch [{epoch+1}/{pretrain_epochs}] | Avg G Loss: {avg_g_loss:.4f}")

    # Save epoch checkpoint
    torch.save({
        'epoch': epoch + 1,
        'batch': 0,
        'model_state_dict': G.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'loss': avg_g_loss
    }, os.path.join(checkpoint_dir, 'pretrain_checkpoint.pth'))
writer.close()
print("🎯 Generator pretraining complete!")


# =========================================================
#  Section 6: Inference and Visualization (Final Version)
# =========================================================
import torch
import numpy as np
import cv2
from skimage.color import lab2rgb

# -----------------------------
# Colorization Inference Function
# -----------------------------
def colorize_image(image_path, G, device='cuda'):
    """
    Takes a grayscale image path and returns a colorized RGB image.
    """
    G.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Convert to LAB (float32)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    # --- Match training normalization ---
    l_resized = cv2.resize(l_channel, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    if l_resized.max() > 100.0:              # normalize to 0–100 if image is 0–255
        l_resized = (l_resized / 255.0) * 100.0
    l_input = torch.from_numpy((l_resized / 50.0 - 1.0)).unsqueeze(0).unsqueeze(0).to(device)

    # --- Generator prediction ---
    with torch.no_grad():
        ab_pred = G(l_input)
        ab_pred = torch.clamp(ab_pred, -1.0, 1.0)
        ab_pred = ab_pred.squeeze(0).cpu().numpy() * 128.0  # Denormalize to [-128, 127]

    # --- Resize and combine ---
    ab_resized = cv2.resize(ab_pred.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_CUBIC)
    np.clip(ab_resized, -128, 127, out=ab_resized)

    l_original = cv2.resize(l_channel, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if l_original.max() > 100.0:
        l_original = (l_original / 255.0) * 100.0

    lab_full = np.dstack((l_original, ab_resized)).astype(np.float64)
    rgb = lab2rgb(lab_full)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


# =========================================================
#  Smoothness Loss Function (Minor Stability Tweak)
# =========================================================
def smoothness_loss(ab):
    """
    Encourages smooth spatial variation in the ab color channels.
    """
    dx = torch.abs(ab[:, :, :, 1:] - ab[:, :, :, :-1])
    dy = torch.abs(ab[:, :, 1:, :] - ab[:, :, :-1, :])
    loss = (dx.mean() + dy.mean()) * 0.1  # tuned to 0.1 for mild smoothing
    return loss

# =========================================================
#  Differentiable LAB → RGB (GPU-optimized)
# =========================================================
def lab_to_rgb_torch(L, ab):
    """Convert LAB ([-1,1]) tensors to RGB [0,1] on GPU (differentiable)."""
    L = (L + 1) * 50.0
    a = ab[:, 0:1] * 128.0
    b = ab[:, 1:2] * 128.0
    lab = torch.cat([L, a, b], dim=1).permute(0, 2, 3, 1)

    eps, kappa = 0.008856, 903.3
    fY = (lab[..., 0] + 16.0) / 116.0
    fX = fY + (lab[..., 1] / 500.0)
    fZ = fY - (lab[..., 2] / 200.0)

    f3 = lambda f: torch.where(f**3 > eps, f**3, (116 * f - 16) / kappa)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X, Y, Z = f3(fX) * Xn, f3(fY) * Yn, f3(fZ) * Zn
    xyz = torch.stack([X, Y, Z], dim=-1)

    M = torch.tensor([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]],
                      device=lab.device, dtype=lab.dtype)
    rgb = torch.matmul(xyz.view(-1, 3), M.T).view_as(xyz)
    rgb = torch.clamp(rgb, 0.0, None)

    threshold, a = 0.0031308, 0.055
    c_pow = rgb.clamp(min=0.0).pow(1.0 / 2.4)
    rgb = torch.where(rgb > threshold, 1.055 * c_pow - 0.055, 12.92 * rgb)
    rgb = rgb.permute(0, 3, 1, 2).clamp(0, 1)
    return rgb.contiguous()

from torch.utils.tensorboard import SummaryWriter

# =========================================================
#  TRAINING INITIALIZATION
# =========================================================
fid_metric = FrechetInceptionDistance(feature=64).to(device)
tbwriter = SummaryWriter(log_dir='runs/colorization_gan')

# VGG for perceptual loss
vgg = vgg16(weights='IMAGENET1K_V1').features[:9].to(device).eval()
for p in vgg.parameters(): p.requires_grad_(False)
vgg_loss_fn = torch.nn.L1Loss()

lambda_vgg, lambda_L1, lambda_gan = 0.2, 150.0, 1.0  # per your setup
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# CSV init
csv_file = 'training_metrics.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Batch', 'G_Loss', 'D_Loss', 'PSNR', 'SSIM'])
print("Starting GAN Training...")

# Load checkpoints
models = {'G': G, 'D': D}
optimizers = {'G': optimizer_G, 'D': optimizer_D}
start_epoch, start_batch = load_checkpoint(os.path.join(checkpoint_dir, 'gan_checkpoint.pth'), models, optimizers)
if start_epoch > 0:
    start_epoch += 1
    print(f"⏩ Resuming training from Epoch {start_epoch}")
else:
    pretrain_path = os.path.join(checkpoint_dir, 'pretrain_checkpoint.pth')
    if os.path.exists(pretrain_path):
        G.load_state_dict(torch.load(pretrain_path)['model_state_dict'])
        print("✅ Loaded pretrain weights for Generator")

print(f"Discriminator LR: {learning_rate_d} | Generator LR: {learning_rate_g}")

# =========================================================
#  TRAINING LOOP
# =========================================================
scaler = torch.cuda.amp.GradScaler()
prev_d_losses = []

for epoch in range(start_epoch, num_epochs):
    G.train(); D.train()
    total_g_loss = total_d_loss = 0.0
    metrics_list = []

    pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch}/{num_epochs}", total=len(train_loader))

    for batch_idx, batch in enumerate(pbar):
        L = batch['L'].to(device, non_blocking=True)
        real_ab = batch['ab'].to(device, non_blocking=True)

        # ==================================================
        #  DISCRIMINATOR UPDATE
        # ==================================================
        optimizer_D.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_ab_detached = G(L).detach()
        real_pred = D(torch.cat([L, real_ab], dim=1))
        fake_pred = D(torch.cat([L, fake_ab_detached], dim=1))

        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
            F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        )
        d_loss.backward()
        optimizer_D.step()

        # ==================================================
        #  GENERATOR UPDATE
        # ==================================================
        optimizer_G.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            fake_ab = torch.clamp(G(L), -1.0, 1.0)
            g_gan = F.binary_cross_entropy_with_logits(D(torch.cat([L, fake_ab], dim=1)), torch.ones_like(real_pred))
            g_l1 = F.l1_loss(fake_ab, real_ab) * lambda_L1
            g_smooth = smoothness_loss(fake_ab)
            color_energy = torch.mean(torch.sqrt(fake_ab[:, 0]**2 + fake_ab[:, 1]**2))
            color_reg = F.mse_loss(color_energy, torch.tensor(0.6, device=device)) * 5.0
            ab_mean = torch.mean(fake_ab, dim=[0, 2, 3])
            chroma_balance = torch.abs(ab_mean[0] - ab_mean[1]) * 5.0

            fake_rgb = lab_to_rgb_torch(L, fake_ab)
            real_rgb = lab_to_rgb_torch(L, real_ab)
            f_vgg = (fake_rgb - mean) / std
            r_vgg = (real_rgb - mean) / std
            perceptual_loss = vgg_loss_fn(vgg(f_vgg), vgg(r_vgg)) * lambda_vgg

            g_loss = lambda_gan * g_gan + g_l1 + g_smooth + color_reg + chroma_balance + perceptual_loss

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # ==================================================
        #  METRICS + LOGGING
        # ==================================================
        g_val, d_val = g_loss.item(), d_loss.item()
        total_g_loss += g_val
        total_d_loss += d_val

        # Compute PSNR/SSIM for one sample per batch (CPU)
        with torch.no_grad():
            fake_rgb_np = fake_rgb[0].permute(1, 2, 0).cpu().numpy() * 255
            real_rgb_np = real_rgb[0].permute(1, 2, 0).cpu().numpy() * 255
            psnr_val = psnr(real_rgb_np, fake_rgb_np, data_range=255)
            ssim_val = ssim(real_rgb_np, fake_rgb_np, channel_axis=-1, data_range=255, win_size=3)
        metrics_list.append([epoch, batch_idx, g_val, d_val, psnr_val, ssim_val])

        tbwriter.add_scalar('GAN/Generator_Loss', g_val, epoch * len(train_loader) + batch_idx)
        tbwriter.add_scalar('GAN/Discriminator_Loss', d_val, epoch * len(train_loader) + batch_idx)

        pbar.set_postfix({'G': f'{g_val:.3f}', 'D': f'{d_val:.3f}'})

    # ==================================================
    #  EPOCH END: SAVE METRICS, LOGS & FID
    # ==================================================
    with open(csv_file, 'a', newline='') as f:
        csv.writer(f).writerows(metrics_list)

    avg_g = total_g_loss / len(train_loader)
    avg_d = total_d_loss / len(train_loader)
    print(f"✅ Epoch [{epoch}/{num_epochs}] | G: {avg_g:.4f} | D: {avg_d:.4f}")

    # ---- Adaptive Discriminator Wake-Up ----
    prev_d_losses.append(avg_d)
    if len(prev_d_losses) > 3:
        diff1, diff2 = abs(prev_d_losses[-1] - prev_d_losses[-2]), abs(prev_d_losses[-2] - prev_d_losses[-3])
        if diff1 < 5e-4 and diff2 < 5e-4:
            for pg in optimizer_D.param_groups:
                old_lr = pg['lr']
                pg['lr'] *= 1.2
                print(f"⚡ Discriminator stagnant → boosted LR {old_lr:.6f} → {pg['lr']:.6f}")
            prev_d_losses.clear()

    # ---- Generate and Save Example Images ----
    G.eval()
    with torch.no_grad():
        img1 = '/home/mmanani/ImageColourizationDataSet/sourceimages/63775aec-17ca-4b4b-a147-fc0d3d426f91.jpg'
        img2 = '/home/mmanani/ImageColourizationDataSet/sourceimages/Grayscaleimage43180.jpg'
        colorized1 = colorize_image(img1, G)
        colorized2 = colorize_image(img2, G)

        tbwriter.add_image(f'Generated/test_image1_epoch_{epoch+1}', torch.tensor(colorized1).permute(2, 0, 1), epoch+1)
        tbwriter.add_image(f'Generated/test_image2_epoch_{epoch+1}', torch.tensor(colorized2).permute(2, 0, 1), epoch+1)

        plt.imsave(f"/home/mmanani/ImageColourizationDataSet/generatedcolouredimages/test_epoch_{epoch+1}.jpg", colorized1)
        plt.imsave(f"/home/mmanani/ImageColourizationDataSet/generatedcolouredimages/test2_epoch_{epoch+1}.jpg", colorized2)

    # ---- Compute FID ----
    with torch.no_grad():
        sample = next(iter(train_loader))
        L = sample['L'].to(device)
        real_ab = sample['ab'].to(device)
        fake_ab = torch.clamp(G(L), -1, 1)
        real_imgs = ((torch.cat([L, real_ab], dim=1) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fake_imgs = ((torch.cat([L, fake_ab], dim=1) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fid_metric.update(fake_imgs, real=False)
        fid_metric.update(real_imgs, real=True)
        fid_val = fid_metric.compute().item()
        fid_metric.reset()

    print(f"[Epoch {epoch}] FID: {fid_val:.4f}")
    tbwriter.add_scalar("FID", fid_val, epoch)

    torch.save({
        'epoch': epoch, 'batch': 0,
        'model_state_dict': {'G': G.state_dict(), 'D': D.state_dict()},
        'optimizer_state_dict': {'G': optimizer_G.state_dict(), 'D': optimizer_D.state_dict()},
        'loss': {'G': avg_g, 'D': avg_d}
    }, os.path.join(checkpoint_dir, 'gan_checkpoint.pth'))
    save_per_epoch_copy(epoch, checkpoint_dir)

tbwriter.close()
torch.save(G.state_dict(), f"{checkpoint_dir}/generator_final.pth")
torch.save(D.state_dict(), f"{checkpoint_dir}/discriminator_final.pth")
print("🎯 GAN Training completed successfully!")

