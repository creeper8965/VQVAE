import torch
from torch import nn, Tensor
F = nn.functional

def GetAct():
    return nn.Hardswish(inplace=True)

def linear_attention(q, k, v, eps=1e-6):
    """
    Compute linear attention using the kernel feature map φ(x)=elu(x)+1.
    
    Given q, k, v of shape (B, num_heads, N, d), this function computes:
    
      output[i] = φ(q_i) * (∑_j φ(k_j)^T v_j) / (φ(q_i) * (∑_j φ(k_j)))
    
    Args:
        q (Tensor): Queries of shape (B, num_heads, N, d).
        k (Tensor): Keys of shape (B, num_heads, N, d).
        v (Tensor): Values of shape (B, num_heads, N, d).
        eps (float): Small constant to prevent division by zero.
    
    Returns:
        Tensor: The attention output of shape (B, num_heads, N, d).
    """
    # Apply non-linearity for the kernel feature map.
    q = F.elu(q) + 1  # Now all elements are positive.
    k = F.elu(k) + 1

    # Compute the KV product.
    # kv: shape (B, num_heads, d, d)
    kv = torch.einsum("b h n d, b h n e -> b h d e", k, v)

    # Compute the unnormalized output for each query vector.
    # out: shape (B, num_heads, N, d)
    out = torch.einsum("b h n d, b h d e -> b h n e", q, kv)

    # Compute the normalization factor:
    # k_sum: shape (B, num_heads, d)
    k_sum = k.sum(dim=2)
    # For each token in q, compute its dot product with the sum of key features.
    # denom: shape (B, num_heads, N)
    denom = torch.einsum("b h n d, b h d -> b h n", q, k_sum)

    # Normalize the output token-wise.
    out = out / (denom.unsqueeze(-1) + eps)
    return out

class MultiHeadAttn(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.09):
        """
        A multi-head attention module that uses linear attention instead of quadratic
        softmax attention. All computations are self-contained.

        Args:
            in_channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for the attention weights (applied to the projection).
        """
        super().__init__()
        # Ensure that in_channels is divisible by num_heads.
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # 1x1 convolution to project the input into queries, keys, and values.
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        # Final projection after attention.
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization over channels (GroupNorm here).
        self.norm = nn.GroupNorm(32, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the attention module.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) with residual connection.
        """
        B, C, H, W = x.shape

        # Apply normalization.
        x_norm = self.norm(x)

        # Compute queries, keys, and values together.
        qkv = self.qkv_conv(x_norm)  # Shape: (B, 3*C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # Each has shape (B, C, H, W)

        # Reshape to separate heads:
        # Reshape: (B, C, H, W) -> (B, num_heads, head_dim, H*W) and then transpose to (B, num_heads, H*W, head_dim)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Now q, k, v are of shape (B, num_heads, N, head_dim) with N = H*W.
        attn = linear_attention(q, k, v)  # Shape: (B, num_heads, N, head_dim)

        # Reconstruct the tensor: transpose and reshape back to (B, C, H, W)
        attn = attn.transpose(2, 3).reshape(B, C, H, W)

        # Final projection and dropout.
        out = self.proj(attn)
        out = self.dropout(out)
        return x + out  # Residual connection

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, dropout=0.1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.GroupNorm(32, in_dim),
            GetAct(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout2d(dropout),
            nn.GroupNorm(32, res_h_dim),
            GetAct(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),  # Pointwise
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, n_heads):
        super().__init__()
        blocks = []
        for i in range(n_res_layers):
            blocks.append(ResidualLayer(in_dim, h_dim, res_h_dim))
            blocks.append(MultiHeadAttn(h_dim, n_heads))
        blocks.append(GetAct())
        self.stack = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x)

class Residual1by1(nn.Module):
    def __init__(self, dim:int, act:bool=True):
        super().__init__()
        self.act = GetAct() if act else nn.Identity()
        self.c1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.c2(self.act(self.c1(x))))

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_dim, n_heads, compression=8):
        super().__init__()
        kernel = 4
        stride = 2

        layers = [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            GetAct(),
            Residual1by1(h_dim//2),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            GetAct(),
            Residual1by1(h_dim),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            GetAct(),
        ]
        
        if compression >= 8:
            layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1))
            layers.append(GetAct())
            layers.append(Residual1by1(h_dim))

        if compression >= 16:
            layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1))
            layers.append(GetAct())
            layers.append(Residual1by1(h_dim))

        layers.append(ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, n_heads))
        layers.append(nn.Conv2d(h_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_dim, n_heads, compression=8):
        super().__init__()
        kernel = 3  # Reduced kernel size to match feature map sizes
        padding = 1

        layers = [
            nn.Conv2d(in_dim, h_dim, kernel_size=kernel, stride=1, padding=padding),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, n_heads),
        ]

        if compression >= 16:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))  # Upsample x2
            layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=1, padding=padding))
            layers.append(GetAct())
            layers.append(ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers//4, n_heads))

        if compression >= 8:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))  # Upsample x2
            layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=1, padding=padding))
            layers.append(GetAct())
            layers.append(ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers//4, n_heads))

        layers.extend([
            nn.Upsample(scale_factor=2, mode="nearest"),  # Upsample x2
            nn.Conv2d(h_dim, h_dim // 2, kernel_size=kernel, stride=1, padding=padding),
            GetAct(),
            ResidualStack(h_dim//2, h_dim//2, h_dim//2,n_res_layers//4,n_heads),

            nn.Upsample(scale_factor=2, mode="nearest"),  # Upsample x2
            nn.Conv2d(h_dim // 2, out_dim, kernel_size=kernel, stride=1, padding=padding),
        ])

        self.inverse_conv_stack = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.inverse_conv_stack(x)


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.uniform_(self.embedding.weight, a=-1.0/self.n_e, b=1.0/self.n_e)

    def forward(self, z:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = z_flattened.pow(2).sum(1, True) + self.embedding.weight.pow(2).sum(1) - 2 * z_flattened.matmul(self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = d.argmin(1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings = min_encodings.scatter(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = min_encodings.matmul(self.embedding.weight).view(*z.shape)

        # compute loss for embedding
        loss = (z_q.detach() - z).pow(2).mean() + self.beta * (z_q - z.detach()).pow(2).mean()

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = min_encodings.mean(0)
        perplexity = (-(e_mean * (e_mean + 1e-10).log()).sum()).exp()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    def encode(self, z: Tensor) -> Tensor:
        """
        Encodes the input image latent representations into discrete indices.
        
        Args:
            z (Tensor): Input tensor of shape (batch, channel, height, width).
            
        Returns:
            encoding_indices (Tensor): Discrete indices of shape (batch, height, width),
                                    representing the codebook entries for each latent vector.
        """
        # Rearrange to shape (batch, height, width, channel) and flatten to (B*H*W, e_dim)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Compute squared distances between z and each embedding in the codebook
        d = z_flattened.pow(2).sum(1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(1) - \
            2 * z_flattened.matmul(self.embedding.weight.t())
        
        # Get the indices of the nearest embeddings
        min_encoding_indices = d.argmin(dim=1)  # shape: (B*H*W,)
        
        # Reshape the indices back to (batch, height, width)
        batch_size, height, width, _ = z.view(z.size(0), -1, self.e_dim).shape[0], z.shape[1], z.shape[2], z.shape[3]
        encoding_indices = min_encoding_indices.view(z.size(0), z.shape[1], z.shape[2])
        
        return encoding_indices

    def decode(self, encoding_indices: Tensor) -> Tensor:
        """
        Decodes discrete encoding indices into quantized latent vectors.

        Args:
            encoding_indices (Tensor): Tensor of shape (batch, height, width) containing the 
                                    indices from the codebook for each latent vector.

        Returns:
            Tensor: The quantized latent representation with shape (batch, e_dim, height, width)
                    that can be passed to the decoder.
        """
        # Look up the embeddings for each index.
        # This will produce a tensor of shape (batch, height, width, e_dim)
        z_q = self.embedding(encoding_indices)
        
        # Permute the dimensions to match the expected input of the decoder: (batch, e_dim, height, width)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, channels=3, compress=8, n_heads=8):
        super().__init__()
        self.encoder = Encoder(channels, h_dim, n_res_layers, res_h_dim, embedding_dim, n_heads=n_heads, compression=compress)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, out_dim=channels, n_heads=n_heads, compression=compress)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, min_encodings, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, perplexity, min_encodings

model_config_teacher = {
    "h_dim": 192,               # Wider base encoder/decoder → better low-level feature capture
    "res_h_dim": 256,           # Big jump in residual path → richer representations
    "n_res_layers": 12,         # Slightly shallower to balance param cost
    "n_embeddings": 510,       # Big expressive codebook = more nuanced recon
    "embedding_dim": 512,       # Keep this large, matches codebook richness
    "beta": 0.35,               # Still balanced #0.25 in 128*128
    "channels": 3,
    "compress": 4,              # Still okay — good tradeoff for latent cost
    "n_heads": 8               # h_dim / 8 — perfect match for attention if used
}

model_config_student = {
    "h_dim": 64,               # Wider base encoder/decoder → better low-level feature capture
    "res_h_dim": 256,           # Big jump in residual path → richer representations
    "n_res_layers": 12,         # Slightly shallower to balance param cost
    "n_embeddings": 510,       # Big expressive codebook = more nuanced recon
    "embedding_dim": 512,       # Keep this large, matches codebook richness
    "beta": 0.25,               # Still balanced #0.25 in 128*128
    "channels": 3,
    "compress": 4,              # Still okay — good tradeoff for latent cost
    "n_heads": 8               # h_dim / 8 — perfect match for attention if used
}

