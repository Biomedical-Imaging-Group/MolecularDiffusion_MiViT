import torch
import torch.nn as nn

MAX_TOKENS = 128  # Assume that we will use maximally 100 tokens

# --------------- Transformer Core --------------- #
class Transformer(nn.Module):
    """A modular Transformer encoder that processes input sequences."""
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding, activation_fct):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_encoding = use_pos_encoding

        # Positional encoding (optional)
        if self.use_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, MAX_TOKENS, embed_dim))  

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

# Different Embeddings:

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
        

class CNNEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # A single 2D convolution layer applied independently per image
        self.conv = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(patch_size, patch_size))

    def forward(self, x):
        """
        Input: x of shape (B, num_images, patch_size, patch_size)
        Output: Tensor of shape (B, num_images, embed_dim)
        """
        batch_size, num_images, h, w = x.shape
        assert h == w == self.patch_size, "Patch size mismatch"

        # Reshape to apply Conv2d independently per image:
        x = x.view(batch_size * num_images, 1, h, w)  # Shape: (B * num_images, 1, patch_size, patch_size)
        
        # Apply convolution (removing spatial dimensions)
        x = self.conv(x)  # Shape: (B * num_images, embed_dim, 1, 1)

        # Remove last two dimensions (1,1) to get (B * num_images, embed_dim)
        x = x.squeeze(-1).squeeze(-1)

        # Reshape back to (B, num_images, embed_dim)
        x = x.view(batch_size, num_images, self.embed_dim)

        return x


class DeepCNNEmbedding(nn.Module):
    def __init__(self, patch_size=7, embed_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, num_images, 7, 7) -> (B, num_images, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to reduce spatial dims
        )
        self.fc = nn.Linear(128, embed_dim)  # Project to embed_dim

    def forward(self, x):
        batch_size, num_images, h, w = x.shape  # Expect (B, num_images, 7, 7)
        x = x.view(batch_size * num_images, 1, h, w)  # Flatten num_images into batch
        x = self.conv_layers(x)  # (B * num_images, 128, 1, 1)
        x = x.view(batch_size, num_images, -1)  # Reshape back: (B, num_images, 128)
        return self.fc(x)  # Final projection to embed_dim
    

# --------------- General Transformer Class --------------- #
class GeneralTransformer(nn.Module):
    def __init__(self, 
                 embedding_cls,  # Class for embedding (e.g., LinearProjectionEmbedding)
                 embed_kwargs,   # Arguments for embedding class
                 embed_dim, 
                 num_heads, 
                 hidden_dim, 
                 num_layers, 
                 mlp_head,       # MLP head module
                 dropout=0, 
                 use_pos_encoding=False, 
                 tr_activation_fct='relu', 
                 use_regression_token=True):
        """
        Generic Transformer class allowing flexible embedding and head customization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = embedding_cls(**embed_kwargs)  # Instantiate embedding

        self.use_regression_token = use_regression_token
        if use_regression_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable regression token

        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, 
                                       dropout, use_pos_encoding=use_pos_encoding, activation_fct=tr_activation_fct)
        self.mlp_head = mlp_head  # Pass any MLP head architecture

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, num_images, embed_dim]

        if self.use_regression_token:
            batch_size = x.shape[0]
            reg_token = self.reg_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
            x = torch.cat([reg_token, x], dim=1)  # [batch_size, num_images + 1, embed_dim]

        x = self.transformer(x)

        if self.use_regression_token:
            reg_out = x[:, 0, :]  # Extract regression token
            return self.mlp_head(reg_out)  # Output shape: [batch_size, 1]
        else:
            # When no regression token is used, average the tokens across the sequence
            avg_tokens = x.mean(dim=1)  # [batch_size, embed_dim], average over the sequence
            return self.mlp_head(avg_tokens)  # Pass averaged output through MLP head




