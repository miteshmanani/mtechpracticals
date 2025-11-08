# =========================================================
#  GAN Training (GPU Optimized + Full Metrics + Perceptual Regularization)
# =========================================================
import os, csv, time, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

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
