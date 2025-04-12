import torch
from torch import Tensor, nn
F = nn.functional

from model import VQVAE model_config_teacher, model_config_student

from os.path import expanduser
from tqdm import tqdm
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from accelerate import Accelerator
from PIL import Image
import wandb

# from torchmetrics.functional.image import structural_similarity_index_measure

def codebook_diversity_loss(min_encodings: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes a diversity loss on the codebook usage. This loss encourages the network to use a wide range of codebook embeddings.
    
    Args:
        min_encodings (Tensor): One-hot encoded assignments from your quantizer,
                                shape (N, n_e) where N is the total number of latent vectors.
        eps (float): A small constant to avoid log(0) issues.
        
    Returns:
        diversity_loss (Tensor): A scalar tensor representing the diversity loss.
    """
    # Compute the average usage of each code across the batch (or flattened latent space)
    avg_usage = min_encodings.mean(dim=0)  # shape: (n_e,)
    
    # Calculate the entropy of this average distribution: -Σ(p * log(p))
    entropy = -torch.sum(avg_usage * torch.log(avg_usage + eps))
    
    # Maximum entropy occurs when usage is completely uniform:
    max_entropy = torch.log(torch.tensor(min_encodings.size(1), device=min_encodings.device, dtype=torch.float))
    
    # We can define the loss as the gap from maximum entropy:
    diversity_loss = max_entropy - entropy
    
    return diversity_loss
    
def gradient_loss_fn(pred:Tensor, target:Tensor) -> Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return nn.functional.l1_loss(pred_dx, target_dx) + nn.functional.l1_loss(pred_dy, target_dy)

def color_mean_std_loss(pred:Tensor, target:Tensor) -> Tensor:
    pool_kernel = 4
    # Use average pooling to reduce spatial dimensions
    pred_pool = F.avg_pool2d(pred, pool_kernel)
    target_pool = F.avg_pool2d(target, pool_kernel)
    
    pred_mean = pred_pool.mean(dim=[2, 3])
    target_mean = target_pool.mean(dim=[2, 3])
    
    pred_std = pred_pool.std(dim=[2, 3])
    target_std = target_pool.std(dim=[2, 3])
    
    mean_loss = F.l1_loss(pred_mean, target_mean)
    std_loss = F.l1_loss(pred_std, target_std)
    return mean_loss + std_loss

def save_validation_images(batch_idx, original, recon, embeddings, save_dir="validation_images", ret=False):
        """
        Save a stacked visualization for a batch of images.
        
        For each image in the batch, display:
        - Original image
        - Reconstructed image
        - TSNE visualization of the image's embeddings
        - Embedding visualization (argmax indices reshaped if possible)
        
        All rows are stacked vertically.
        
        Args:
            batch_idx (int): The current batch index (for filename).
            original (Tensor): Original images of shape (B, C, H, W).
            recon (Tensor): Reconstructed images of shape (B, C, H, W).
            embeddings (Tensor): One-hot embeddings from the VQ layer.
                                Expected shape: (B*L, n_e), where L is the number of latent vectors per image.
            save_dir (str): Directory to save the image.
        """ #pip install umap-learn
        import math
        from matplotlib import pyplot as plt
        import io
        
        B = original.shape[0]
        # Determine number of latent vectors per image.
        L_total = embeddings.shape[0]
        L = L_total // B  # latent vectors per image
        n_e = embeddings.shape[1]
        
        # Reshape embeddings to (B, L, n_e)
        embeddings = embeddings.view(B, L, n_e)
        
        # Create a subplot grid: one row per image, 4 columns per row.
        fig, ax = plt.subplots(B, 3, figsize=(16, 4 * B))
        # Ensure ax is 2D (if B==1, matplotlib returns 1D array)
        if B == 1:
            ax = ax[None, :]
        
        for i in range(B):
            # --- Original Image ---
            img_orig = original[i].cpu().numpy().transpose(1, 2, 0)
            # If grayscale, squeeze and use gray colormap
            if img_orig.shape[2] == 1:
                img_orig = img_orig.squeeze(-1)
                cmap = 'gray'
            else:
                cmap = None
            ax[i, 0].imshow(img_orig, cmap=cmap)
            ax[i, 0].set_title("Original")
            ax[i, 0].axis('off')
            
            # --- Reconstructed Image ---
            img_recon = recon[i].cpu().numpy().transpose(1, 2, 0)
            if img_recon.shape[2] == 1:
                img_recon = img_recon.squeeze(-1)
                cmap = 'gray'
            else:
                cmap = None
            ax[i, 1].imshow(img_recon, cmap=cmap)
            ax[i, 1].set_title("Reconstructed")
            ax[i, 1].axis('off')
            
            # --- Embedding Visualization ---
            # Convert one-hot to indices.
            indices = embeddings[i].argmax(dim=1).cpu().numpy()  # shape: (L,)
            # Try to reshape indices into a square if possible.
            H_lat = int(math.sqrt(L))
            W_lat = H_lat
            if H_lat * W_lat != L:
                H_lat, W_lat = 1, L  # fallback: display as a row
            emb_img = indices.reshape(H_lat, W_lat)
            ax[i, 2].imshow(emb_img, cmap='viridis')
            ax[i, 2].set_title("Embedding Codes")
            ax[i, 2].axis('off')
        
        plt.tight_layout()
        # plt.savefig(f"batch_{batch_idx}_validation.png")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")  # Save as PNG in memory
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        pil_image.save(f"batch_{batch_idx}_validation.png")
        if ret:
            return pil_image
        

def main(): #Error noti: RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED CAN MEAN OUT OF VRAM
    BATCH = 2
    LR = 4e-5 #5e-5
    STEPS = 40000
    EVAL = 500
    GRAD_S =17
    SAVE_S = 5000

    model_config = model_config_teacher
    
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=GRAD_S)
    accelerator.init_trackers(
        project_name="VQVAE",
        config={'LR':LR,'STEPS':STEPS,'BATCH':BATCH,'MODEL':model_config,'EVAL':EVAL, 'GRAD_S':GRAD_S},
        init_kwargs={"wandb": {"entity": "wandb org name"}}
    )

    DEVICE = accelerator.device

    model = VQVAE(**model_config).to(DEVICE)
    # model.load_state_dict(torch.load('', weights_only=True))
    if accelerator.is_main_process:
        params = sum([i.numel() for i in model.parameters()])
        print('Params:',params, ' Size:',params*32/8/1000/1000,'MBs')
    opt = optim.AdamW(model.parameters(), LR, weight_decay=1e-1)

    if accelerator.is_main_process:
        test_in = torch.rand(1,3,256,256).to(DEVICE)
        enc_test = model.encoder(test_in)
        enc_vq_test = model.vector_quantization.encode(enc_test)#model.pre_quantization_conv(enc_test))
        dec_vq_test = model.vector_quantization.decode(enc_vq_test)
        dec_test = model.decoder(dec_vq_test)
        print('Input:',test_in.shape,'\nEncoded:',enc_test.shape,'\nEncoded VQ:',enc_vq_test.shape, '\nDecoded VQ:',dec_vq_test.shape,'\nDecoded:',dec_test.shape)

    class InfiniteDataLoader:
        def __init__(self, data_loader):
            self.data_loader = data_loader
            self.data_iter = iter(data_loader)

        def __iter__(self):
            return self

        def __next__(self):
            try:
                data = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.data_loader)  # Reset the data loader
                data = next(self.data_iter)
            return data
    #,v2.RandomRotation((0,180)), v2.RandomErasing(0.2)
    trans = v2.Compose([v2.Resize((512,512)),v2.RandomHorizontalFlip(),v2.RandomVerticalFlip(),v2.RandomInvert(0.2),v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    ds = ImageFolder(expanduser('~/Desktop/hugeData/combinedDatasetSymlink/'), trans)
    if accelerator.is_main_process:
        exampel_item =ds.__getitem__(0)
        print(exampel_item[0].min(), exampel_item[0].max())
        print('Dataset Size:',len(ds))
        print('Number of processes:', accelerator.num_processes)
        print('Batch per device:',BATCH//accelerator.num_processes)

    dl = DataLoader(
        ds,
        batch_size=BATCH // accelerator.num_processes,
        shuffle=True,
        num_workers=2,           # Increase as needed
        pin_memory=True,
        persistent_workers=True, # Keeps workers alive between epochs
        timeout=3600             # Optional: increase timeout if necessary
    )
    dl = InfiniteDataLoader(dl)
    
    model, opt, dl = accelerator.prepare(model, opt, dl)

    def trainingStep(step):
        img, _ = next(dl)
        img = img.to(DEVICE)
        with accelerator.autocast():
            emb_loss, recon, perplex, min_encodings = model(img)
            recon_loss = F.mse_loss(recon, img)
            l1_loss = F.l1_loss(recon, img)
            diversity_loss = codebook_diversity_loss(min_encodings)
            loss = emb_loss + l1_loss + recon_loss + diversity_loss
            loss = loss / GRAD_S
        accelerator.backward(loss)

        if step % GRAD_S == 0:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        return loss.item(), perplex.item(), emb_loss.item(), recon_loss.item(), l1_loss.item(), diversity_loss.item()

    model.train()
    for step in (TT:=tqdm(range(STEPS+1), disable=not accelerator.is_main_process)):
        loss, perplex, emb_loss, recon_loss, l1_loss, diversity_loss = trainingStep(step)
        accelerator.log({"loss":loss, "perplex":perplex, "emb_loss":emb_loss, "recon_loss":recon_loss, "l1_loss":l1_loss, "diversity_loss":diversity_loss}, step=step) #"color_loss":color_loss
        TT.set_description(f'Loss:{loss:.4} | Perplex:{perplex:.4}')

        if step % EVAL == 0:
            if accelerator.is_main_process:
                r_model = accelerator.unwrap_model(model)
                with torch.no_grad():
                    r_model.eval()
                    test_img = next(dl)[0].to(DEVICE)
                    _,recon,_,_ = r_model(test_img)
                    accelerator.print(f'Min:{recon.min()} Max:{recon.max()}')
                    _, _, _, min_encodings, _ = r_model.vector_quantization(r_model.encoder(test_img))
                    test_img = ((test_img + 1) / 2).clip(0,1)
                    recon = ((recon + 1) / 2).clip(0,1)
                    plotted = save_validation_images(0, test_img.cpu(), recon.cpu(), min_encodings.cpu(), ret=True)
                    accelerator.log({'validation_image':wandb.Image(plotted)}, step=step)
                r_model.train()

        if step % SAVE_S == 0:
            if accelerator.is_main_process:
                accelerator.save(accelerator.unwrap_model(model).state_dict(), f'Training_VQVAE_{step}_d.pt')

    accelerator.end_training()
    if accelerator.is_main_process:
        accelerator.save(accelerator.unwrap_model(model).state_dict(), 'Trained_VQVAE_d.pt')


if __name__ == '__main__':
    main()
