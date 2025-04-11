import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

MAX_TOKENS = 128  # Assume that we will use maximally 100 tokens

# --------------- Custom Multi-Head Attention --------------- #
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters with Glorot / fan_avg
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, activation_fct, dropout=0.0, ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if not callable(activation_fct):
            raise ValueError("activation_fct must be a callable function from torch.nn.functional or a custom function.")
        self.activation = activation_fct
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --------------- Transformer Encoder Layer with Skip Connections --------------- #
class TransformerEncoderLayerWithSkip(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, activation_fct, dropout=0.0, ):
        super().__init__()
        # Multi-head attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, hidden_dim, activation_fct, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # First sub-layer: Multi-head attention with skip connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)  # Skip connection
        x = self.norm1(x)  # Post-norm architecture
        
        # Second sub-layer: Feed-forward with skip connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Skip connection
        x = self.norm2(x)  # Post-norm architecture
        
        return x

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

        # Custom encoder layers with skip connections
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithSkip(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                hidden_dim=hidden_dim,
                activation_fct=activation_fct,
                dropout=dropout,
            )
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection projection if dimensions mismatch
        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip connection
        return self.relu(out)

class DeepResNetEmbedding(nn.Module):
    def __init__(self, patch_size=7, embed_dim=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to reduce spatial dims
        self.fc = nn.Linear(128, embed_dim)  # Project to embed_dim

    def forward(self, x):
        batch_size, num_images, h, w = x.shape
        x = x.reshape(batch_size * num_images, 1, h, w)  # Flatten num_images into batch

        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.global_pool(x)  # (B * num_images, 128, 1, 1)
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
                 tr_activation_fct,
                 dropout=0, 
                 use_pos_encoding=False,
                 use_regression_token=False,
                 single_prediction=True):
        """
        Generic Transformer class allowing flexible embedding and head customization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = embedding_cls(**embed_kwargs)  # Instantiate embedding
        self.norm = nn.LayerNorm(embed_dim)
        self.use_regression_token = use_regression_token
        self.single_prediction = single_prediction
        if use_regression_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable regression token

        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, 
                                       dropout, use_pos_encoding=use_pos_encoding, activation_fct=tr_activation_fct)
        self.mlp_head = mlp_head  # Pass any MLP head architecture

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, num_images, embed_dim]
        
        x = self.norm(x)

        if self.use_regression_token:
            batch_size = x.shape[0]
            reg_token = self.reg_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
            x = torch.cat([reg_token, x], dim=1)  # [batch_size, num_images + 1, embed_dim]

        x = self.transformer(x)

        if self.use_regression_token:
            reg_out = x[:, 0, :]  # Extract regression token
            return self.mlp_head(reg_out)  # Output shape: [batch_size, 1]
        else:
            if self.single_prediction:
                # When no regression token is used, average the tokens across the sequence
                avg_tokens = x.mean(dim=1)  # [batch_size, embed_dim], average over the sequence
                return self.mlp_head(avg_tokens)  # Pass averaged output through MLP head
            else:
                # Predict one value per input image in the sequence
                return self.mlp_head(x)  # Shape: [batch_size, num_images, 1]


# Resnet Model:

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU):
        super().__init__()
        self.activation = activation(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = activation(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)

        return out


class LightResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, feature_size=64, activation=nn.ReLU):
        super().__init__()
        self.in_channels = 32
        self.activation = activation

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act = activation(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_size = feature_size
        self.fc1 = nn.Linear(128 * block.expansion, self.feature_size)
        self.fc_act = activation(inplace=True)
        self.fc2 = nn.Linear(self.feature_size, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, activation=self.activation))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc_act(out)
        out = self.fc2(out)

        return out


class MultiImageLightResNet(nn.Module):
    def __init__(self, image_size, num_classes=1, single_prediction=True, activation=nn.ReLU):
        super().__init__()
        self.single_prediction = single_prediction
        self.resnet = LightResNet(BasicBlock, [1, 1, 1], num_classes, activation=activation)

    def forward(self, x):
        batch_size, num_images, height, width = x.shape
        x = x.reshape(batch_size * num_images, 1, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, num_images, 1)

        if self.single_prediction:
            x = torch.mean(x, dim=1, keepdim=False)

        return x





# Helpers used for training:  


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Shape (N, C, H, W)
        self.labels = labels  # Shape (N, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]



