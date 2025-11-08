# GAN Training
# Initialize CSV file
csv_file = 'training_metrics.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Batch', 'G_Loss', 'D_Loss', 'PSNR', 'SSIM'])        
print("Starting GAN Training...")
models = {'G': G, 'D': D}
optimizers = {'G': optimizer_G, 'D': optimizer_D}
start_epoch, start_batch = load_checkpoint(os.path.join(checkpoint_dir, 'gan_checkpoint.pth'), models, optimizers)
if start_epoch == 0:
    g_checkpoint = torch.load(os.path.join(checkpoint_dir, 'pretrain_checkpoint.pth'))
    G.load_state_dict(g_checkpoint['model_state_dict'])
    print("Loaded pretrain weights for G")

for epoch in range(start_epoch, num_epochs):
    total_g_loss, total_d_loss = 0, 0
    metrics_list = []
    start_time = time.time()
    if epoch == start_epoch and start_batch > 0:
        iterator = iter(train_loader)
        for _ in range(start_batch):
            next(iterator)
        batches = iterator
    else:
        batches = train_loader

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"GAN Epoch {epoch+1}/{num_epochs}", total=len(train_loader))):
        if epoch == start_epoch and batch_idx < start_batch:
            continue
        L = batch['L'].to(device)
        real_ab = batch['ab'].to(device)
        g_loss, d_loss = train_step(G, D, optimizer_G, optimizer_D, real_ab, L)
        psnr_score, ssim_score = None, None
        if real_ab is not None:
            fake_ab = G(L)
            fake_rgb = torch.cat((L, fake_ab), dim=1).detach()
            real_rgb = torch.cat((L, real_ab), dim=1).detach()
            fake_rgb = fake_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255
            real_rgb = real_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255
            psnr_score = psnr(real_rgb[0], fake_rgb[0], data_range=255)
            ssim_score = ssim(real_rgb[0], fake_rgb[0], multichannel=True, data_range=255)  # Fixed SSIM
            g_loss_val = g_loss.item() if torch.is_tensor(g_loss) else g_loss
            d_loss_val = d_loss.item() if torch.is_tensor(d_loss) else d_loss
            metrics_list.append([epoch, batch_idx, g_loss_val, d_loss_val, psnr_score, ssim_score])
        
        total_g_loss += g_loss.item() if torch.is_tensor(g_loss) else g_loss
        total_d_loss += d_loss.item() if torch.is_tensor(d_loss) else d_loss
        elapsed_time = time.time() - start_time
        if elapsed_time >= 600:
            print(f"Pausing for 1 min at {elapsed_time:.1f}s...")
            time.sleep(60)
            start_time = time.time()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics_list)
    
    avg_g = total_g_loss / len(train_loader)
    avg_d = total_d_loss / len(train_loader)
    print(f"GAN Epoch [{epoch+1}/{num_epochs}], G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")
    torch.save({
        'epoch': epoch,
        'batch': 0,
        'model_state_dict': {'G': G.state_dict(), 'D': D.state_dict()},
        'optimizer_state_dict': {'G': optimizer_G.state_dict(), 'D': optimizer_D.state_dict()},
        'loss': {'G': avg_g, 'D': avg_d}
    }, os.path.join(checkpoint_dir, 'gan_checkpoint.pth'))

torch.save(G.state_dict(), '/home/mmanani/ImageColourizationDataSet/generator_final.pth')
torch.save(D.state_dict(), '/home/mmanani/ImageColourizationDataSet/discriminator_final.pth')
print("GAN Training completed!")