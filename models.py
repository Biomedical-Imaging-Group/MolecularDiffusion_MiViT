import torch
import torch.nn as nn

# --------------- Transformer Core --------------- #
class Transformer(nn.Module):
    """A modular Transformer encoder that processes input sequences."""
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0, use_pos_encoding=False, activation_fct ='relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_encoding = use_pos_encoding

        # Positional encoding (optional)
        if self.use_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))  # Max 100 patches

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,activation=activation_fct, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if self.use_pos_encoding:
            x = x + self.pos_embedding[:, :x.shape[1], :]
        for layer in self.encoder_layers:
            x = layer(x)
        return self.norm(x)

# --------------- Linear Projection Embedding --------------- #
class LinearProjectionEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        if x.dim() == 3:  # Case: (num_images, patch_size, patch_size)
            num_images, h, w = x.shape
            assert h == w == self.patch_size, "Patch size mismatch"
            x = x.view(num_images, -1)  # [num_image, patch_size^2]
            return self.proj(x).unsqueeze(0)  # [1, num_images, embed_dim] (Adding batch dim)

        elif x.dim() == 4:  # Case: (batch_size, num_images, patch_size, patch_size)
            batch_size, num_images, h, w = x.shape
            assert h == w == self.patch_size, "Patch size mismatch"
            x = x.view(batch_size, num_images, -1)  # [batch_size, num_images, patch_size^2]
            return self.proj(x)  # [batch_size, num_images, embed_dim]

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected (num_images, C, H, W) or (B, num_images, C, H, W).")

# --------------- Specialized Transformer Class --------------- #
class LinearTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0):
        super().__init__()
        self.patch_embedding = LinearProjectionEmbedding(patch_size, embed_dim)
        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, dropout)

    def forward(self, x):
        x = self.patch_embedding(x)  # Shape: [batch_size, num_images, embed_dim]
        return self.transformer(x)

    
