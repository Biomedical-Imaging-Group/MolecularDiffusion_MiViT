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

# --------------- Feed Forward Network --------------- #
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, activation_fct='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation_fct == 'gelu':
            self.activation = F.gelu
        elif activation_fct == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation_fct}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --------------- Transformer Encoder Layer with Skip Connections --------------- #
class TransformerEncoderLayerWithSkip(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0, activation_fct='gelu'):
        super().__init__()
        # Multi-head attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout, activation_fct)
        
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
                dropout=dropout,
                activation_fct=activation_fct
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
        x = x.view(batch_size * num_images, 1, h, w)  # Flatten num_images into batch

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
                 dropout=0, 
                 use_pos_encoding=False, 
                 tr_activation_fct='gelu', 
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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
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
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu2(out)
        
        return out

class LightResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, feature_size=64):
        super().__init__()
        self.in_channels = 32  # Reduced from 64
        
        # Always use 1 input channel for grayscale
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)  # Smaller kernel
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # 32 channels
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)   # 64 channels
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # Max of 128 channels
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define the fully connected layers
        self.feature_size = feature_size
        self.fc1 = nn.Linear(128 * block.expansion, self.feature_size)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.feature_size, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Print input shape for debugging
        # print(f"Input shape: {x.shape}")
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        # print(f"Shape after avgpool: {out.shape}")
        
        out = torch.flatten(out, 1)
        # print(f"Shape after flatten: {out.shape}")
        
        # Two-layer output with ReLU
        out = self.fc1(out)
        out = self.fc_relu(out)
        out = self.fc2(out)
        
        return out

class MultiImageLightResNet(nn.Module):
    def __init__(self, image_size, num_classes=1, single_prediction=True):
        super().__init__()
        self.single_prediction = single_prediction
        # Create a lightweight ResNet backbone with fewer blocks
        self.resnet = LightResNet(BasicBlock, [1, 1, 1], num_classes)  # Just 1 block per layer
        
    def forward(self, x):
        # Input shape: [B, num_images, H, W]
        batch_size, num_images, height, width = x.shape
        
        # Reshape to process each image independently
        # Add channel dimension for grayscale images
        x = x.view(batch_size * num_images, 1, height, width)  # [B*num_images, 1, H, W]
        
        # Process through ResNet
        x = self.resnet(x)  # [B*num_images, 1]
        
        # Reshape back to separate batch and num_images
        x = x.view(batch_size, num_images, 1) # [B, num_images, 1]
        
        if self.single_prediction:
            # Average over num_images dimension
            x = torch.mean(x, dim=1, keepdim=False)  # [B, 1]
        
        return x



# Define model hyperparameters
patch_size = 7
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 6
dropout = 0.0
# Define MLP heads

# Define model instances

def get_transformer_models(patch_size = patch_size, embed_dim = embed_dim, num_heads = num_heads, hidden_dim = hidden_dim, num_layers = num_layers,
                            dropout= dropout, use_pos_encoding = False, tr_activation_fct='gelu', use_regression_token=True , single_prediction = True, name_suffix = ''):
    """
    Returns different variants of the GeneralTransformer model.
    """
    embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
    twoLayerMLP = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)  # Output a single scalar value
    )

    oneLayerMLP = nn.Sequential(
        nn.Linear(embed_dim, 1)  # Output a single scalar value
    )
    # Define model instances
    models = {
        "linear_2layer" + name_suffix: GeneralTransformer(
            embedding_cls=LinearProjectionEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=twoLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "linear_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=LinearProjectionEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "cnn_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=CNNEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "deepcnn_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=DeepResNetEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "cnn_2layer"+ name_suffix: GeneralTransformer(
            embedding_cls=CNNEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=twoLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "deepcnn_2layer"+ name_suffix: GeneralTransformer(
            embedding_cls=DeepResNetEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=twoLayerMLP,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            tr_activation_fct=tr_activation_fct,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        )
    }
    
    return models





# Helpers used for training:  


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Shape (N, C, H, W)
        self.labels = labels  # Shape (N, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]



