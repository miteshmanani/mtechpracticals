# =========================================================
#  GAN Training (Improved for Stability + Color Regularization)
#  (Final patched block - includes GPU-friendly LAB->RGB for VGG)
# =========================================================
from torch.utils.tensorboard import SummaryWriter

fid_metric = FrechetInceptionDistance(feature=64).to(device)
tbwriter = SummaryWriter(log_dir='runs/colorization_gan')

# -----------------------------
#  Perceptual (VGG) feature extractor INIT
# -----------------------------
vgg = vgg16(weights='IMAGENET1K_V1').features[:9].to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False
vgg_loss_fn = nn.L1Loss()
lambda_vgg = 0.2   # weight of perceptual loss


# -----------------------------
#  CSV INIT (if missing)
# -----------------------------
csv_file = 'training_metrics.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'G_Loss', 'D_Loss', 'PSNR', 'SSIM'])
print("Starting GAN Training...")

# -----------------------------
#  LOAD CHECKPOINT
# -----------------------------
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

# -----------------------------
#  TRAINING LOOP
# -----------------------------
print(f"Discriminator Learning Rate:{learning_rate_d}")
print(f"Generator Learning Rate:{learning_rate_g}")

for epoch in range(start_epoch, num_epochs):
    G.train()
    D.train()
    total_g_loss = total_d_loss = 0.0
    metrics_list = []
    start_time = time.time()

    if epoch == start_epoch and start_batch > 0:
        iterator = iter(train_loader)
        for _ in range(start_batch):
            next(iterator)
        batches = iterator
    else:
        batches = train_loader

    pbar = tqdm(batches, desc=f"GAN Epoch {epoch}/{num_epochs}", total=len(train_loader))

    for batch_idx, batch in enumerate(pbar):
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        L = batch['L'].to(device, non_blocking=True)          # [-1,1]
        real_ab = batch['ab'].to(device, non_blocking=True)   # [-1,1]

        # ==================================================
        #  DISCRIMINATOR UPDATE
        # ==================================================
        optimizer_D.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_ab_detached = G(L).detach()
        real_input = torch.cat([L, real_ab], dim=1)
        fake_input = torch.cat([L, fake_ab_detached], dim=1)

        real_pred = D(real_input)
        fake_pred = D(fake_input)

        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        optimizer_D.step()

        # ==================================================
        #  GENERATOR UPDATE (with color regularization + perceptual)
        # ==================================================
        optimizer_G.zero_grad(set_to_none=True)
        fake_ab = G(L)
        fake_ab_clamped = torch.clamp(fake_ab, -1.0, 1.0)

        fake_input = torch.cat([L, fake_ab_clamped], dim=1)
        g_gan = F.binary_cross_entropy_with_logits(D(fake_input), torch.ones_like(real_pred))
        g_l1 = F.l1_loss(fake_ab_clamped, real_ab) * lambda_L1
        g_smooth = smoothness_loss(fake_ab_clamped)

        # --- 🧩 Color regularization ---
        color_energy = torch.mean(torch.sqrt(fake_ab_clamped[:, 0]**2 + fake_ab_clamped[:, 1]**2))
        color_energy_loss = F.mse_loss(color_energy, torch.tensor(0.6, device=device)) * 5.0

        ab_mean = torch.mean(fake_ab_clamped, dim=[0, 2, 3])
        chroma_balance_loss = torch.abs(ab_mean[0] - ab_mean[1]) * 5.0

        # --- Perceptual Loss (use differentiable conversion)
        # Convert current LAB tensors to RGB in [0,1] using torch ops (keeps gradients)
        fake_rgb_vgg = lab_to_rgb_torch(L, fake_ab_clamped)   # (B,3,H,W) in [0,1]
        real_rgb_vgg = lab_to_rgb_torch(L, real_ab)           # reuse L + real_ab -> (B,3,H,W)

        # VGG expects inputs normalized like imagenet: mean/std
        # Normalize to VGG input: multiply by 255 then normalize channels
        # But torchvision's pretrained VGG expects inputs in range [0,1] then normalized with imagenet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        fake_vgg_in = (fake_rgb_vgg - mean) / std
        real_vgg_in = (real_rgb_vgg - mean) / std

        fake_feats = vgg(fake_vgg_in)
        real_feats = vgg(real_vgg_in)
        perceptual_loss = vgg_loss_fn(fake_feats, real_feats) * lambda_vgg

        # --- Combined Generator Loss ---
        g_loss = (
            lambda_gan * g_gan +
            g_l1 +
            g_smooth +
            color_energy_loss +
            chroma_balance_loss +
            perceptual_loss
        )

        g_loss.backward()
        optimizer_G.step()

        # ==================================================
        #  METRICS
        # ==================================================
        with torch.no_grad():
            fake_rgb = torch.cat([L, fake_ab_clamped], dim=1).cpu().numpy()[0].transpose(1, 2, 0) * 255
            real_rgb = torch.cat([L, real_ab], dim=1).cpu().numpy()[0].transpose(1, 2, 0) * 255
            psnr_score = psnr(real_rgb, fake_rgb, data_range=255)
            ssim_score = ssim(real_rgb, fake_rgb, channel_axis=-1, data_range=255, win_size=3)

        g_val, d_val = g_loss.item(), d_loss.item()
        metrics_list.append([epoch, batch_idx, g_val, d_val, psnr_score, ssim_score])
        total_g_loss += g_val
        total_d_loss += d_val

        global_step = epoch * len(train_loader) + batch_idx
        tbwriter.add_scalar('GAN/Generator_Loss', g_loss.item(), global_step)
        tbwriter.add_scalar('GAN/Discriminator_Loss', d_loss.item(), global_step)
        tbwriter.add_scalar('GAN/L1_Loss', g_l1.item(), global_step)
        tbwriter.add_scalar('GAN/Smoothness_Loss', g_smooth.item(), global_step)
        tbwriter.add_scalar('GAN/Color_Energy_Loss', color_energy_loss.item(), global_step)
        tbwriter.add_scalar('GAN/Chroma_Balance_Loss', chroma_balance_loss.item(), global_step)
        tbwriter.add_scalar('GAN/Perceptual_Loss', perceptual_loss.item(), global_step)

        pbar.set_postfix({'G': f'{g_val:.4f}', 'D': f'{d_val:.4f}'})

        # Auto checkpoint every 500 batches
        if (batch_idx + 1) % 500 == 0:
            torch.save({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': {'G': G.state_dict(), 'D': D.state_dict()},
                'optimizer_state_dict': {'G': optimizer_G.state_dict(), 'D': optimizer_D.state_dict()},
                'loss': {'G': g_val, 'D': d_val}
            }, os.path.join(checkpoint_dir, 'gan_checkpoint.pth'))

    # ==================================================
    #  EPOCH END: LOG + TEST + SAVE
    # ==================================================
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics_list)

    avg_g = total_g_loss / len(train_loader)
    avg_d = total_d_loss / len(train_loader)
    print(f"✅ GAN Epoch [{epoch}/{num_epochs}] | G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")


    # =====================================
    #  🔁 Adaptive Discriminator Wake-Up Logic
    # =====================================
    if not hasattr(globals(), 'prev_d_losses'):
        prev_d_losses = []

    prev_d_losses.append(avg_d)
    print(f"Length of prev_d_losses:{len(prev_d_losses)}")
    # Keep only last 2 epochs to monitor stability
    if len(prev_d_losses) > 2:
        prev_d_losses.pop(0)
    print(f"Prior to Equalto Check")
    # Check if D is flatlined (same loss over 2 epochs)
    if len(prev_d_losses) == 2:
        print("If Passed")
        diff1 = abs(prev_d_losses[-1] - prev_d_losses[-2])
        diff2 = abs(prev_d_losses[-2] - prev_d_losses[-3])
        print(f"Diff-1:{diff1} & Diff-2:{diff2}")
        if diff1==0 or (diff1 < 0.0005 and diff2 < 0.0005):
            for pg in optimizer_D.param_groups:
                old_lr = pg['lr']
                pg['lr'] *= 1.2
                print(f"⚡ Discriminator seems stagnant — boosted LR from {old_lr:.6f} → {pg['lr']:.6f}")
            prev_d_losses = []  # reset tracking window

    # --- Sample generation (uses your inference function)
    G.eval()
    with torch.no_grad():
        test_img1 = '/home/mmanani/ImageColourizationDataSet/sourceimages/63775aec-17ca-4b4b-a147-fc0d3d426f91.jpg'
        test_img2 = '/home/mmanani/ImageColourizationDataSet/sourceimages/Grayscaleimage43180.jpg'
        colorized1 = colorize_image(test_img1, G)
        colorized2 = colorize_image(test_img2, G)

    tbwriter.add_image(f'Generated/test_image1_epoch_{epoch+1}', torch.tensor(colorized1).permute(2, 0, 1), epoch+1)
    tbwriter.add_image(f'Generated/test_image2_epoch_{epoch+1}', torch.tensor(colorized2).permute(2, 0, 1), epoch+1)

    plt.imsave(f"/home/mmanani/ImageColourizationDataSet/generatedcolouredimages/test_epoch_{epoch+1}.jpg", colorized1)
    plt.imsave(f"/home/mmanani/ImageColourizationDataSet/generatedcolouredimages/test2_epoch_{epoch+1}.jpg", colorized2)

    with torch.no_grad():
        try:
            real_imgs = torch.cat([L, real_ab], dim=1)
            fake_imgs = torch.cat([L, fake_ab_clamped], dim=1)
        except NameError:
            # fallback: use one batch from train_loader if variables not defined
            print("In Name Error Block")
            sample_batch = next(iter(train_loader))
            L = sample_batch['L'].to(device)
            real_ab = sample_batch['ab'].to(device)
            fake_ab_clamped = torch.clamp(G(L), -1.0, 1.0)
            real_imgs = torch.cat([L, real_ab], dim=1)
            fake_imgs = torch.cat([L, fake_ab_clamped], dim=1)
        real_imgs = ((real_imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fake_imgs = ((fake_imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fid_metric.update(fake_imgs, real=False)
        fid_metric.update(real_imgs, real=True)
        fid_value = fid_metric.compute().item()
        fid_metric.reset()

    tbwriter.add_scalar("FID", fid_value, epoch)
    print(f"[Epoch {epoch}] FID: {fid_value:.4f}")

    torch.save({
        'epoch': epoch,
        'batch': 0,
        'model_state_dict': {'G': G.state_dict(), 'D': D.state_dict()},
        'optimizer_state_dict': {'G': optimizer_G.state_dict(), 'D': optimizer_D.state_dict()},
        'loss': {'G': avg_g, 'D': avg_d}
    }, os.path.join(checkpoint_dir, 'gan_checkpoint.pth'))
    save_per_epoch_copy(epoch, checkpoint_dir)

tbwriter.close()
torch.save(G.state_dict(), '/home/mmanani/ImageColourizationDataSet/generator_final.pth')
torch.save(D.state_dict(), '/home/mmanani/ImageColourizationDataSet/discriminator_final.pth')
print("🎯 GAN Training completed successfully!")