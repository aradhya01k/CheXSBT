import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# ---------------------
# Patch Embedding Module
# ---------------------
class PatchEmbedding(nn.Module):
    """
    Splits the input image into non-overlapping patches and projects them to a desired embedding dimension.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        # Convolution layer to extract patch embeddings
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Project image to patch embeddings
        x = self.proj(x)  # Shape: (B, C, H/patch, W/patch)
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten spatial dimensions
        x = self.norm(x)  # Apply LayerNorm
        return x

# ---------------------
# MLP Block
# ---------------------
class MLP(nn.Module):
    """
    Standard Feed-Forward Network with GELU activation.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.ff(x)

# ---------------------
# Shifted Window Attention (W-MSA & SW-MSA)
# ---------------------
class ShiftedWindowMSA(nn.Module):
    """
    Implements Window-based Multi-head Self-Attention (W-MSA) and Shifted Window MSA (SW-MSA).
    """
    def __init__(self, embed_dim, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.emb_size = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted

        # Linear projection to obtain Q, K, V
        self.linear1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

        # Relative positional encoding for attention
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))

        # Compute relative indices for positional embeddings
        coords = np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        self.register_buffer('relative_indices', torch.tensor(coords[None, :, :] - coords[:, None, :] + window_size - 1))

    def forward(self, x):
        B, N, C = x.shape
        h_dim = C // self.num_heads

        # Calculate height and width
        H = W = int(np.sqrt(N))
        if H * W != N:
            H = int(np.sqrt(N))
            W = N // H
        assert H * W == N, f"Cannot reshape {N} tokens into 2D grid"

        # Linear projection to get Q, K, V
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=H, w=W, k=3, c=C)

        # Pad if image size is not divisible by window size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, 0, 0, pad_r, 0, pad_b))  # Pad bottom and right
        Hp, Wp = x.shape[1], x.shape[2]

        # Apply shift if using shifted window attention
        if self.shifted:
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        # Partition into windows and reshape for multi-head attention
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)

        # Separate Q, K, V
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # Scaled dot-product attention
        wei = (Q @ K.transpose(-2, -1)) / np.sqrt(h_dim)

        # Add relative positional encoding
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding.to(x.device)

        # Attention mask for shifted windows to prevent information leakage
        if self.shifted:
            mask = torch.zeros((self.window_size**2, self.window_size**2), device=x.device)
            shift = self.window_size * (self.window_size // 2)
            mask[-shift:, :-shift] = float('-inf')
            mask[:-shift, -shift:] = float('-inf')
            mask = rearrange(mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += mask
            wei[:, :, :, -1] += mask

        # Apply attention weights
        wei = F.softmax(wei, dim=-1) @ V

        # Merge windows back and remove padding
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = x[:, :H, :W, :].contiguous()
        x = rearrange(x, 'b h w c -> b (h w) c')

        return self.linear2(x)  # Final projection

# ---------------------
# Dual Attention Swin Block (DAST Block)
# ---------------------
class SwinBlock(nn.Module):
    """
    Swin Block consisting of alternating W-MSA, SW-MSA and MLPs, with residual connections.
    """
    def __init__(self, embed_dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        # First attention cycle
        self.norm1 = nn.LayerNorm(embed_dim)
        self.w_msa1 = ShiftedWindowMSA(embed_dim, num_heads, window_size, shifted=False)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.sw_msa1 = ShiftedWindowMSA(embed_dim, num_heads, window_size, shifted=True)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp1 = MLP(embed_dim, hidden_dim)

        # Second attention cycle
        self.norm4 = nn.LayerNorm(embed_dim)
        self.w_msa2 = ShiftedWindowMSA(embed_dim, num_heads, window_size, shifted=False)

        self.norm5 = nn.LayerNorm(embed_dim)
        self.sw_msa2 = ShiftedWindowMSA(embed_dim, num_heads, window_size, shifted=True)

        self.norm6 = nn.LayerNorm(embed_dim)
        self.mlp2 = MLP(embed_dim, hidden_dim)

        # Final MLP block
        self.norm7 = nn.LayerNorm(embed_dim)
        self.mlp3 = MLP(embed_dim, hidden_dim)

    def forward(self, x):
        # Apply two cycles of attention + MLP
        residual = x
        x = residual + self.w_msa1(self.norm1(x))

        residual = x
        x = residual + self.sw_msa1(self.norm2(x))

        residual = x
        x = self.mlp1(self.norm3(x)) + residual

        residual = x
        x = residual + self.w_msa2(self.norm4(x))

        residual = x
        x = residual + self.sw_msa2(self.norm5(x))

        residual = x
        x = self.mlp2(self.norm6(x)) + residual

        residual = x
        x = self.mlp3(self.norm7(x)) + residual

        return x

# ---------------------
# Swin Stage
# ---------------------
class SwinStage(nn.Module):
    """
    A stage in the Swin Transformer, consisting of multiple Swin blocks.
    """
    def __init__(self, depth, embed_dim, num_heads, window_size=7):
        super().__init__()
        blocks = [SwinBlock(embed_dim, num_heads, window_size) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

# ---------------------
# Swin Transformer Backbone
# ---------------------
class SwinTransformer(nn.Module):
    """
    Swin transformer using shifted window attention.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.stages = nn.ModuleList()
        in_dim = embed_dim

        for i in range(len(depths)):
            # Append a stage with Swin blocks
            self.stages.append(SwinStage(depths[i], in_dim, num_heads[i]))
            if i < len(depths) - 1:
                # Linear projection between stages (dimension doubling)
                self.stages.append(nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, in_dim * 2)
                ))
                in_dim *= 2

        self.output_dim = in_dim

    def forward(self, x):
        x = self.patch_embed(x)  # Initial patch embedding
        for stage in self.stages:
            x = stage(x)  # Pass through each stage
        return x

# ---------------------
# Vision Encoder (DAST-based)
# ---------------------
class VisionEncoder(nn.Module):
    """
    DAST-based Vision Transformer encoder.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_blocks=4, window_size=7):
        super().__init__()
        # Swin Transformer backbone
        self.swin = SwinTransformer(
            img_size=224, patch_size=4, in_chans=3,
            embed_dim=embed_dim, depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 32)
        )

        # Swin blocks
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size) for _ in range(num_blocks)
        ])

        # Final projection layers
        self.fc1 = nn.Linear(embed_dim, 1024)
        self.fc2 = nn.Linear(1024, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        features = self.swin(images)  # Extract patch features

        # Apply extra transformer blocks
        for block in self.blocks:
            features = block(features)

        # Global average pooling across patches
        features = features.mean(dim=1)

        # Project to final embedding space
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        embeddings = self.fc2(x)

        return embeddings
