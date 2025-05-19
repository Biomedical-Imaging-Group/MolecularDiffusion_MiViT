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
        x = x.reshape(batch_size * num_images, 1, h, w)  # Shape: (B * num_images, 1, patch_size, patch_size)
        
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
    

class MLPHead(nn.Module):
    def __init__(self, 
                 input_dim,       # input dimension (e.g., embed_dim or 2*embed_dim)
                 hidden_dim=128,  # size of the hidden layer
                 output_dim=1,    # size of the output (default: scalar for regression)
                 dropout=0.0,     # optional dropout
                 activation=nn.ReLU):  # activation function (default: ReLU)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GeneralTransformer(nn.Module):
    def __init__(self, 
                 embedding_cls,
                 embed_kwargs,
                 embed_dim, 
                 num_heads, 
                 hidden_dim, 
                 num_layers, 
                 mlp_head,
                 tr_activation_fct,
                 dropout=0, 
                 use_pos_encoding=False,
                 use_regression_token=False,
                 single_prediction=True,
                 use_global_features=False,
                 fusion_type='early',  # 'early' or 'late'
                 global_feature_dim=None):
        """
        Generic Transformer class with optional fusion of global features.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = embedding_cls(**embed_kwargs)
        self.norm = nn.LayerNorm(embed_dim)
        self.use_regression_token = use_regression_token
        self.single_prediction = single_prediction
        self.use_global_features = use_global_features
        self.fusion_type = fusion_type

        if use_regression_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers, 
                                       dropout, use_pos_encoding=use_pos_encoding, activation_fct=tr_activation_fct)

        # Optional MLP to project global features to embed_dim
        if use_global_features:
            assert global_feature_dim is not None, "Must provide global_feature_dim if using global features"
            self.feature_projector = nn.Sequential(
                nn.Linear(global_feature_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # Output head
        if fusion_type == 'late' and use_global_features:
            self.mlp_head = mlp_head(input_dim=embed_dim * 2)  # concatenate token + features
        else:
            self.mlp_head = mlp_head(input_dim=embed_dim)

    def forward(self, x, features=None):
        """
        x: [batch_size, num_images, image_size, image_size]
        features: [batch_size, num_features] or None
        """
        x = self.embedding(x)  # [batch_size, num_images, embed_dim]
        x = self.norm(x)

        batch_size = x.shape[0]

        if self.use_regression_token:
            reg_token = self.reg_token.expand(batch_size, 1, -1)  # [batch_size, 1, embed_dim]

            if self.use_global_features and self.fusion_type == 'early':
                assert features is not None, "Global features required for early fusion"
                projected = self.feature_projector(features)  # [batch_size, embed_dim]
                projected = projected.unsqueeze(1)  # [batch_size, 1, embed_dim]
                reg_token = reg_token + projected  # inject info into CLS token

            x = torch.cat([reg_token, x], dim=1)  # prepend CLS/regression token

        x = self.transformer(x)  # [batch_size, seq_len, embed_dim]

        if self.use_regression_token:
            reg_out = x[:, 0, :]  # [batch_size, embed_dim]
        else:
            reg_out = x.mean(dim=1)  # fallback: average all tokens

        if self.use_global_features and self.fusion_type == 'late':
            assert features is not None, "Global features required for late fusion"
            projected = self.feature_projector(features)  # [batch_size, embed_dim]
            reg_out = torch.cat([reg_out, projected], dim=-1)  # [batch_size, embed_dim * 2]

        return self.mlp_head(reg_out)




class ModularTransformer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 mlp_head,
                 tr_activation_fct,
                 dropout=0,
                 use_pos_encoding=False,
                 use_regression_token=False,
                 single_prediction=True,
                 # New modular parameters
                 mode='images_only',  # 'images_only', 'features_only', or 'both'
                 image_embedding_cls=None,   # Image embedding class (optional)
                 image_embed_kwargs=None,    # Image embedding arguments (optional)
                 features_dim=None,          # Number of feature dimensions (optional)
                 feature_embedding_type='linear',  # How to embed features: 'linear', 'mlp'
                 fusion_method='add'         # Method to combine: 'add', 'concat_proj', 'concat_features'
                ):
        """
        Modular Transformer that can process images, features, or both, optimized for PyTorch efficiency.
        
        Parameters:
        -----------
        embed_dim: Dimension of token embeddings in the transformer
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feed forward network
        num_layers: Number of transformer layers
        mlp_head: MLP head module for final predictions
        tr_activation_fct: Activation function for transformer
        dropout: Dropout rate
        use_pos_encoding: Whether to use positional encoding
        use_regression_token: Whether to use a learnable regression token
        single_prediction: If True, return a single prediction per sequence
        mode: Input mode - 'images_only', 'features_only', or 'both'
        image_embedding_cls: Class for image embedding
        image_embed_kwargs: Arguments for image embedding class
        features_dim: Dimension of input features
        feature_embedding_type: How to embed features ('linear' or 'mlp')
        fusion_method: Method to fuse image and feature embeddings:
                      - 'add': Simple addition
                      - 'concat_proj': Concatenate and project to embed_dim
                      - 'concat_features': Keep raw features and concat with adjusted image embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        self.use_regression_token = use_regression_token
        self.single_prediction = single_prediction
        self.fusion_method = fusion_method
        self.features_dim = features_dim
        
        # Validate configuration
        if mode not in ['images_only', 'features_only', 'both']:
            raise ValueError("mode must be one of: 'images_only', 'features_only', 'both'")
            
        if mode == 'both' and fusion_method not in ['add', 'concat_proj', 'concat_features']:
            raise ValueError("fusion_method must be one of: 'add', 'concat_proj', 'concat_features'")
        
        # Handle the special case of concat_features fusion method
        if mode == 'both' and fusion_method == 'concat_features':
            # For concat_features, we need to adjust the image embedding dimension
            # so that when concatenated with raw features, it equals embed_dim
            image_embed_dim = embed_dim - features_dim
            
            # Make sure it's valid
            if image_embed_dim <= 0:
                raise ValueError(f"embed_dim ({embed_dim}) must be greater than features_dim ({features_dim}) when using 'concat_features' fusion")
                
            # Update image_embed_kwargs to use the adjusted embed_dim
            if image_embed_kwargs is None:
                image_embed_kwargs = {}
            image_embed_kwargs['embed_dim'] = image_embed_dim
        else:
            # For other modes, use the full embed_dim for image embedding
            image_embed_dim = embed_dim
        
        # Set up image embedding if needed
        self.image_embedding = None
        if mode in ['images_only', 'both']:
            if image_embedding_cls is None:
                raise ValueError("image_embedding_cls must be provided when using images")
            if image_embed_kwargs is None:
                image_embed_kwargs = {}
            self.image_embedding = image_embedding_cls(**image_embed_kwargs)
        
        # Set up feature embedding if needed
        self.feature_embedding = None
        if mode in ['features_only', 'both'] and fusion_method != 'concat_features':
            if features_dim is None:
                raise ValueError("features_dim must be provided when using features")
                
            if feature_embedding_type == 'linear':
                self.feature_embedding = nn.Linear(features_dim, embed_dim)
            elif feature_embedding_type == 'mlp':
                self.feature_embedding = nn.Sequential(
                    nn.Linear(features_dim, embed_dim * 2),
                    nn.LayerNorm(embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            else:
                raise ValueError(f"Unknown feature_embedding_type: {feature_embedding_type}")
        
        # Fusion layer if using both modalities with concatenative projection
        self.fusion_layer = None
        if mode == 'both' and fusion_method == 'concat_proj':
            self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
        # Layer normalization for embeddings
        self.norm = nn.LayerNorm(embed_dim)
        
        # Regression token if needed
        if use_regression_token:
            self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
        # Transformer backbone
        self.transformer = Transformer(
            embed_dim, 
            num_heads, 
            hidden_dim, 
            num_layers,
            dropout, 
            use_pos_encoding=use_pos_encoding, 
            activation_fct=tr_activation_fct
        )
        
        # Output head
        self.mlp_head = mlp_head
    
    def forward(self, images=None, features=None):
        """
        Forward pass for the modular transformer using direct tensor inputs instead of dictionaries.
        
        Parameters:
        -----------
        images: tensor of shape [batch_size, num_images, image_size, image_size] or None
                Required for 'images_only' and 'both' modes
        features: tensor of shape [batch_size, num_images, features_dim] or None
                  Required for 'features_only' and 'both' modes
            
        Returns:
        --------
        Tensor of predictions with shape depending on configuration:
            - With use_regression_token=True: [batch_size, output_dim]
            - With single_prediction=True: [batch_size, output_dim]
            - Otherwise: [batch_size, num_images, output_dim]
        """
        # Input validation based on mode
        if self.mode == 'images_only' and images is None:
            raise ValueError("Images are required for 'images_only' mode")
        if self.mode == 'features_only' and features is None:
            raise ValueError("Features are required for 'features_only' mode")
        if self.mode == 'both' and (images is None or features is None):
            raise ValueError("Both images and features are required for 'both' mode")
        
        batch_size = None
        
        # Process inputs based on mode
        if self.mode == 'images_only':
            batch_size = images.shape[0]
            x = self.image_embedding(images)
            
        elif self.mode == 'features_only':
            batch_size = features.shape[0]
            
            # Replace NaNs with zeros
            features = torch.nan_to_num(features, nan=0.0)
            
            x = self.feature_embedding(features)
            
        else:  # 'both' mode
            batch_size = images.shape[0]
            
            # Validate that both inputs have the same batch size and sequence length
            if (images.shape[0] != features.shape[0] or 
                images.shape[1] != features.shape[1]):
                raise ValueError("Images and features must have the same batch size and sequence length")
            
            # Replace NaNs in features with zeros
            features = torch.nan_to_num(features, nan=0.0)
            
            # Get image embeddings
            image_embeddings = self.image_embedding(images)
            
            # Fuse embeddings according to the selected method
            if self.fusion_method == 'add':
                # Project features to embed_dim and add to image embeddings
                feature_embeddings = self.feature_embedding(features)
                x = image_embeddings + feature_embeddings
                
            elif self.fusion_method == 'concat_proj':
                # Project features to embed_dim, concat with image embeddings, then project back
                feature_embeddings = self.feature_embedding(features)
                concat_embeddings = torch.cat([image_embeddings, feature_embeddings], dim=-1)
                x = self.fusion_layer(concat_embeddings)
                
            elif self.fusion_method == 'concat_features':
                # Directly concatenate raw features with image embeddings
                # (image_embeddings already have dim = embed_dim - features_dim)
                x = torch.cat([image_embeddings, features], dim=-1)
                # The resulting x now has dimension embed_dim
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Add regression token if needed
        if self.use_regression_token:
            reg_token = self.reg_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
            x = torch.cat([reg_token, x], dim=1)  # [batch_size, sequence_length + 1, embed_dim]
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Extract appropriate tokens for prediction
        if self.use_regression_token:
            # Use the first token (regression token) for prediction
            token_out = x[:, 0, :]  # [batch_size, embed_dim]
        elif self.single_prediction:
            # Average tokens across sequence dimension
            token_out = x.mean(dim=1)  # [batch_size, embed_dim]
        else:
            # Use all tokens
            token_out = x  # [batch_size, sequence_length, embed_dim]
        
        # Pass through MLP head
        return self.mlp_head(token_out)
        



# --------------- ResNet Model --------------- #

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


class MultiImageResNet(nn.Module):
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


class LightImagesFeaturesResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=64, activation=nn.ReLU):
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

        return out  # Now returns features, not predictions

class MultiImageFeatureResNet(nn.Module):
    def __init__(self, image_size, external_dim, feature_size=64, hidden_size=128, activation=nn.ReLU):
        super().__init__()
        self.resnet = LightImagesFeaturesResNet(BasicBlock, [1, 1, 1], feature_size, activation=activation)

        self.feature_size = feature_size
        self.external_dim = external_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_size + external_dim, hidden_size),
            activation(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, external_features):
        batch_size, num_images, height, width = x.shape
        x = x.reshape(batch_size * num_images, 1, height, width)
        features = self.resnet(x)
        features = features.view(batch_size, num_images, -1)
        features = torch.mean(features, dim=1)  # mean over time

        combined = torch.cat([features, external_features], dim=1)
        out = self.mlp(combined)
        return out





# Helpers used for training:  


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Shape (N, C, H, W)
        self.labels = labels  # Shape (N, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ImageFeatureDataset(Dataset):
    def __init__(self, images, features, labels):
        self.images = images  # Shape (N, C, H, W)
        self.features = features  # Shape (N, C, N_features)
        self.labels = labels  # Shape (N, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.features[idx], self.labels[idx]



