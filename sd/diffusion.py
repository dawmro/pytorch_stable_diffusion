import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention



class TimeEmbedding(nn.Module):
    """
    Time embedding module for diffusion models.
    
    This module encodes time steps (noise levels) into high-dimensional embeddings
    that condition the denoising process. It follows a standard architecture:
    
    1. Initial projection to a higher-dimensional space (4x expansion)
    2. Non-linear activation (SiLU/Swish)
    3. Final projection in the expanded space
    
    The expanded embedding dimension provides richer representation capacity for
    encoding the noise level information, which helps the model better condition
    its denoising process based on the current time step.
    
    Attributes:
        linear_1 (nn.Linear): First projection layer that expands the embedding dimension
        linear_2 (nn.Linear): Second projection layer in the expanded space
    """
    def __init__(self, n_embed: int):
        """
        Initialize the time embedding module.
        
        Args:
            n_embed (int): Base embedding dimension, which will be expanded to 4*n_embed
                internally for richer representation
        """
        super().__init__()
        # First projection layer that expands the embedding dimension by a factor of 4
        # This provides richer representation capacity for encoding time information
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        
        # Second projection layer that operates in the expanded space
        # This allows for more complex transformations of the time information
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x):
        """
        Forward pass through the time embedding module.
        
        This method processes time step information through a series of
        linear projections and non-linear activations to produce a rich
        embedding that conditions the denoising process.
        
        Args:
            x (torch.Tensor): Time step tensor of shape (Batch_Size, n_embed)
                representing the noise level to be encoded
                
        Returns:
            torch.Tensor: Time embedding of shape (Batch_Size, 4*n_embed)
                that provides rich conditioning information for the denoising process
        """
        # Input tensor shape: (Batch_Size, n_embed)
        # First projection expands the embedding dimension by a factor of 4
        # From: (Batch_Size, n_embed)
        # To:   (Batch_Size, 4*n_embed)
        x = self.linear_1(x)
        
        # Apply SiLU/Swish activation function
        # This non-linearity helps capture complex relationships in the time information
        # From: (Batch_Size, 4*n_embed)
        # To:   (Batch_Size, 4*n_embed)
        x = F.silu(x)
        
        # Second projection in the expanded space
        # This allows for more complex transformations of the time information
        # From: (Batch_Size, 4*n_embed)
        # To:   (Batch_Size, 4*n_embed)
        x = self.linear_2(x)
        
        # Return the final time embedding
        # Shape: (Batch_Size, 4*n_embed)
        return x


class UNET_ResidualBlock(nn.Module):
    """
    Residual block for U-Net architecture with time conditioning.
    
    This module implements a residual block that processes feature maps while being
    conditioned on time information. It consists of two main branches:
    
    1. Feature Branch: Processes the input features through normalization, activation,
       and convolution to extract meaningful patterns.
    
    2. Time Branch: Processes the time embedding through activation and linear projection
       to condition the feature processing based on the current noise level.
    
    The outputs of these branches are merged, processed further, and combined with a
    residual connection to the input. This design allows the network to learn both
    feature-specific transformations and time-dependent adjustments.
    
    The residual connection is implemented as either an identity mapping (when input
    and output channels match) or a 1x1 convolution (when channel dimensions differ).
    
    Attributes:
        groupnorm_feature (nn.GroupNorm): Normalization layer for input features
        conv_feature (nn.Conv2d): Convolution layer for feature processing
        linear_time (nn.Linear): Linear projection for time embedding
        groupnorm_merged (nn.GroupNorm): Normalization layer for merged features
        conv_merged (nn.Conv2d): Convolution layer for merged features
        residual_layer (nn.Module): Either identity or 1x1 convolution for residual connection
    """
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        """
        Initialize the residual block with time conditioning.
        
        Args:
            in_channels (int): Number of input channels for the feature maps
            out_channels (int): Number of output channels for the feature maps
            n_time (int, optional): Dimension of the time embedding. Defaults to 1280.
        """
        super().__init__()
        # Feature processing branch
        # Group normalization with 32 groups for feature normalization
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        # 3x3 convolution for feature processing
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time processing branch
        # Linear projection to match the feature channel dimension
        self.linear_time = nn.Linear(n_time, out_channels)
        
        # Merged feature processing
        # Group normalization for the merged features
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        # 3x3 convolution for the merged features
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection
        if in_channels == out_channels:
            # When input and output channels match, use identity mapping
            self.residual_layer = nn.Identity()
        else:
            # When channels differ, use 1x1 convolution to match dimensions
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        """
        Forward pass through the residual block with time conditioning.
        
        This method processes the input features while conditioning on time information.
        It applies a series of transformations to both the features and time embedding,
        merges them, and adds a residual connection to the input.
        
        Args:
            feature (torch.Tensor): Input feature maps of shape 
                (Batch_Size, In_Channels, Height, Width)
            time (torch.Tensor): Time embedding of shape (1, n_time)
                representing the current noise level
                
        Returns:
            torch.Tensor: Processed feature maps of shape 
                (Batch_Size, Out_Channels, Height, Width)
        """

        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        # Store the input for the residual connection
        residual = feature
        
        # Process the feature branch
        # Normalize the features
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        # Apply SiLU activation for non-linearity
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        # Project to the output channel dimension
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # Process the time branch
        # Apply SiLU activation to the time embedding
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        # Project to match the feature channel dimension
        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Merge the feature and time branches
        # Time is a 1D tensor, doesn't have batch and channel dimension, so we need to add 2 dimensions to it
        # Add spatial dimensions to the time tensor for broadcasting
        # This allows the time information to condition each spatial location
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # Process the merged features
        # Normalize the merged features
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        # Apply SiLU activation for non-linearity
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        # Apply final convolution
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # Add the residual connection and return
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    """
    Attention block for U-Net architecture that processes spatial features with self-attention and cross-attention.
    
    This module implements a sophisticated attention mechanism that combines self-attention
    (to capture relationships within the feature map) and cross-attention (to condition
    on external context, such as text embeddings). The architecture follows a transformer-like
    design with residual connections and normalization layers.
    
    The block consists of three main stages:
    1. Self-attention: Captures relationships between different spatial positions in the feature map
    2. Cross-attention: Conditions the features on external context (e.g., text embeddings)
    3. Feed-forward network: Processes the attended features with a GeGLU activation
    
    Each stage includes normalization layers and residual connections to stabilize training
    and preserve information flow. The spatial dimensions are temporarily reshaped to treat
    spatial positions as a sequence, allowing the attention mechanisms to capture long-range
    dependencies across the image.
    
    Attributes:
        groupnorm (nn.GroupNorm): Initial normalization layer for the input features
        conv_input (nn.Conv2d): 1x1 convolution for initial feature processing
        layernorm_1 (nn.LayerNorm): Normalization before self-attention
        attention_1 (SelfAttention): Self-attention mechanism for capturing spatial relationships
        layernorm_2 (nn.LayerNorm): Normalization before cross-attention
        attention_2 (CrossAttention): Cross-attention mechanism for conditioning on context
        layernorm_3 (nn.LayerNorm): Normalization before feed-forward network
        linear_geglu_1 (nn.Linear): First linear layer for GeGLU activation
        linear_geglu_2 (nn.Linear): Second linear layer for GeGLU activation
        conv_output (nn.Conv2d): 1x1 convolution for final feature processing
    """
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        """
        Initialize the U-Net attention block.
        
        Args:
            n_head (int): Number of attention heads
            n_embed (int): Embedding dimension per head
            d_context (int, optional): Dimension of the context embeddings. Defaults to 768.
        """
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        """
        Forward pass through the attention block.
        
        This method processes the input features through a series of attention mechanisms
        and transformations, with residual connections at each stage. The spatial dimensions
        are temporarily reshaped to treat spatial positions as a sequence, allowing the
        attention mechanisms to capture long-range dependencies.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Features, Height, Width)
            context (torch.Tensor): Context tensor of shape (Batch_Size, Sequence_Length, Dimension)
                used for cross-attention conditioning
                
        Returns:
            torch.Tensor: Processed tensor of shape (Batch_Size, Features, Height, Width)
        """
        # Store the input for the final residual connection
        residual_long = x

        # Stage 1: Initial feature normalization and processing
        # Apply group normalization to stabilize the features
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # Apply 1x1 convolution for initial feature processing
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)

        # Extract tensor dimensions for reshaping operations
        n, c, h, w = x.shape

        # Reshape the tensor to treat spatial positions as a sequence
        # This allows the attention mechanisms to capture relationships between different spatial positions
        # Transpose to be able to apply the self-attention
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        # Stage 2: Self-attention with residual connection
        # Store the input for the residual connection
        # (Batch_Size, Height * Width, Features)
        residual_short = x
        
        # Apply layer normalization before self-attention
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # Apply self-attention to capture relationships between spatial positions
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # Add the residual connection to preserve information flow
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residual_short 

        # Stage 3: Cross-attention with residual connection
        # Store the input for the residual connection
        # (Batch_Size, Height * Width, Features)
        residual_short = x
        
        # Apply layer normalization before cross-attention
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # Apply cross-attention to condition on the external context
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # Add the residual connection to preserve information flow
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residual_short

        # Stage 4: Feed-forward network with GeGLU activation and residual connection
        # Store the input for the residual connection
        # (Batch_Size, Height * Width, Features)
        residual_short = x
        
        # Apply layer normalization before the feed-forward network
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)

        # Apply GeGLU activation (Gated GELU)
        # This splits the output of the first linear layer into two parts:
        # one for the main path and one for the gating mechanism
        # GeGLU as implemented in the original code: 
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Apply the second linear layer to project back to the original dimension
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # Add the residual connection to preserve information flow
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residual_short
        
        # Restore the original spatial dimensions
        # Transpose back to the original shape
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # Reshape back to the original 4D tensor
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Apply the final 1x1 convolution and add the long residual connection
        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residual_long


class Upsample(nn.Module):
    """
    Upsampling module that doubles the spatial dimensions of feature maps.
    
    This module performs upsampling by a factor of 2 using nearest neighbor interpolation,
    followed by a convolution layer to refine the upsampled features. This approach is
    commonly used in U-Net architectures to gradually increase spatial resolution during
    the decoding phase.
    
    The nearest neighbor interpolation provides a simple and efficient way to increase
    spatial dimensions, while the subsequent convolution helps to refine the upsampled
    features and reduce artifacts that may be introduced during interpolation.
    
    Attributes:
        conv (nn.Conv2d): Convolution layer that refines the upsampled features
    """
    def __init__(self, channels):
        """
        Initialize the upsampling module.
        
        Args:
            channels (int): Number of input and output channels for the convolution layer
        """
        super().__init__()
        # Create a 3x3 convolution layer that preserves the number of channels
        # This convolution helps refine the upsampled features and reduce artifacts
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass through the upsampling module.
        
        This method first doubles the spatial dimensions of the input tensor using
        nearest neighbor interpolation, then applies a convolution to refine the
        upsampled features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width)
            
        Returns:
            torch.Tensor: Upsampled tensor of shape (Batch_Size, Channels, Height*2, Width*2)
        """
        # First, double the spatial dimensions using nearest neighbor interpolation
        # This preserves the number of channels while increasing height and width by a factor of 2
        # From: (Batch_Size, Channels, Height, Width)
        # To:   (Batch_Size, Channels, Height*2, Width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        
        # Apply a convolution to refine the upsampled features
        # This helps reduce artifacts and improve feature quality
        # Shape remains: (Batch_Size, Channels, Height*2, Width*2)
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    A specialized sequential container for U-Net layers with conditional processing.
    
    This class extends PyTorch's Sequential container to handle different types of
    layers in a U-Net architecture. It intelligently routes inputs to each layer
    based on the layer type, passing the appropriate additional context:
    
    1. For attention blocks: Passes the latent representation and text embeddings
    2. For residual blocks: Passes the latent representation and time embeddings
    3. For standard layers: Passes only the latent representation
    
    This design allows for a clean, modular U-Net architecture where different
    types of layers can be easily combined while ensuring each receives the
    appropriate inputs for its operation.
    
    This pattern is commonly used in transformer and U-Net architectures to
    handle the different types of conditioning information that various layers
    need to process.
    """
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequential layers with conditional routing.
        
        This method processes the input through each layer in sequence, but
        intelligently routes the inputs based on the layer type. This allows
        different types of layers to receive the appropriate additional context
        they need for their operation.
        
        Args:
            x (torch.Tensor): The latent representation to be processed
            context (torch.Tensor): Text embeddings from CLIP for attention layers
            time (torch.Tensor): Time embeddings for residual layers
            
        Returns:
            torch.Tensor: The processed latent representation after passing through
                all layers with appropriate conditioning
        """
        # Process each layer in the sequence in order
        for layer in self:
            # Route inputs based on layer type
            if isinstance(layer, UNET_AttentionBlock):
                # Attention blocks need both the latent and text embeddings
                # to compute cross-attention between them
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # Residual blocks need both the latent and time embeddings
                # to condition the residual connection on the time step
                x = layer(x, time)
            else:
                # Standard layers (like convolutions) only need the latent
                # representation without additional context
                x = layer(x)
        
        return x



class UNET(nn.Module):
    """
    U-Net architecture for Stable Diffusion's noise prediction network.
    
    This class implements a U-Net architecture with skip connections, which is a key
    component in Stable Diffusion for predicting and removing noise from latent representations.
    The architecture consists of three main parts:
    
    1. Encoder Path: A series of downsampling blocks that progressively reduce spatial
       dimensions while increasing feature channels. Each block includes residual blocks
       and attention mechanisms to capture both local and global dependencies.
    
    2. Bottleneck: The middle section that processes features at the lowest spatial
       resolution, typically including residual blocks and attention mechanisms.
    
    3. Decoder Path: A series of upsampling blocks that progressively increase spatial
       dimensions while decreasing feature channels. Each block concatenates features
       from the corresponding encoder block via skip connections, allowing the network
       to preserve fine-grained details.
    
    The U-Net architecture is particularly effective for image generation tasks because
    it combines multi-scale feature processing with skip connections that help preserve
    spatial details during the generation process.
    
    Attributes:
        encoders (nn.ModuleList): List of encoder blocks for downsampling
        bottleneck (SwitchSequential): Middle section processing features at lowest resolution
        decoders (nn.ModuleList): List of decoder blocks for upsampling
    """
    def __init__(self):
        """
        Initialize the U-Net architecture with encoder, bottleneck, and decoder paths.
        
        The architecture follows a standard U-Net design with:
        - Encoder path: 4x downsampling with increasing feature channels (320->640->1280)
        - Bottleneck: Processing at lowest resolution with 1280 channels
        - Decoder path: 4x upsampling with decreasing feature channels (1280->640->320)
        
        Each block in the encoder and decoder includes residual blocks and attention
        mechanisms to capture both local and global dependencies.
        """
        super().__init__()

        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            # 8: head, 40: embedding size
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            # reduce size of image by 2
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1)),
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            # 8: head, 80: embedding size
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            # reduce size of image by 2
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1)),
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            # 8: head, 160: embedding size
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            # reduce size of image by 2
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1)),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_AttentionBlock(8, 160)),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            # there is double amount of input features because of the skip connection
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            # upsample to increase size of image
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

             # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            # upsample to increase size of image
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            # upsample to increase size of image
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        """
        Forward pass through the U-Net architecture.
        
        This method processes the input through the encoder path, bottleneck, and decoder path,
        utilizing skip connections to preserve fine-grained details. The U-Net architecture
        progressively downsamples the input in the encoder path, processes it at the lowest
        resolution in the bottleneck, and then progressively upsamples it in the decoder path.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 4, Height/8, Width/8)
                representing the noisy latent representation
            context (torch.Tensor): Text embeddings of shape (Batch_Size, Seq_Len, Dim)
                used for conditioning the generation process
            time (torch.Tensor): Time embedding of shape (1, 1280) representing the
                noise level for the current denoising step
                
        Returns:
            torch.Tensor: Processed tensor of shape (Batch_Size, 320, Height/8, Width/8)
                representing the predicted noise to be removed
        """
        # Input tensor shapes:
        # x: (Batch_Size, 4, Height/8, Width/8) - Noisy latent representation
        # context: (Batch_Size, Seq_Len, Dim) - Text embeddings for conditioning
        # time: (1, 1280) - Time embedding representing noise level

        # Store intermediate outputs from encoder path for skip connections
        skip_connections = []
        
        # Process through encoder path (downsampling)
        for layers in self.encoders:
            # Apply the current encoder block
            x = layers(x, context, time)
            # Store the output for skip connections
            skip_connections.append(x)

        # Process through bottleneck (lowest resolution)
        x = self.bottleneck(x, context, time)

        # Process through decoder path (upsampling)
        for layers in self.decoders:
            # Concatenate with the corresponding skip connection from the encoder
            # This doubles the number of features before processing
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            # Apply the current decoder block
            x = layers(x, context, time)
        
        # Return the final output
        return x


class UNET_OutputLayer(nn.Module):
    """
    Final output layer for the U-Net architecture in Stable Diffusion.
    
    This module processes the U-Net's output features and projects them back to the
    original latent space dimension. It consists of three main components:
    
    1. Group Normalization: Normalizes the features across groups to improve training stability
    2. SiLU Activation: Applies a non-linear activation function to introduce non-linearity
    3. Convolution: Projects the features to the desired output dimension
    
    This layer is crucial for ensuring the U-Net's output matches the expected dimensions
    of the latent space, allowing for proper integration with the rest of the diffusion model.
    
    Attributes:
        groupnorm (nn.GroupNorm): Group normalization layer for feature normalization
        conv (nn.Conv2d): Convolution layer for final projection to output channels
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the U-Net output layer.
        
        Args:
            in_channels (int): Number of input channels from the U-Net
            out_channels (int): Number of output channels for the final projection
        """
        super().__init__()
        # Group normalization with 32 groups for feature normalization
        # This helps stabilize training by normalizing features across groups
        self.groupnorm = nn.GroupNorm(32, in_channels)
        
        # Final convolution layer that projects to the desired output dimension
        # This maps the U-Net's output features back to the latent space
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass through the U-Net output layer.
        
        This method processes the U-Net's output features through normalization,
        activation, and final projection to produce the output in the latent space.
        
        Args:
            x (torch.Tensor): Input tensor from the U-Net of shape 
                (Batch_Size, in_channels, Height/8, Width/8)
            
        Returns:
            torch.Tensor: Processed tensor of shape 
                (Batch_Size, out_channels, Height/8, Width/8)
        """
        # Input tensor shape: (Batch_Size, in_channels, Height/8, Width/8)
        
        # Apply group normalization to stabilize features
        # Shape remains: (Batch_Size, in_channels, Height/8, Width/8)
        x = self.groupnorm(x)
        
        # Apply SiLU (Swish) activation for non-linearity
        # This helps capture complex patterns in the features
        # Shape remains: (Batch_Size, in_channels, Height/8, Width/8)
        x = F.silu(x)
        
        # Final convolution to project to the output dimension
        # This maps the features back to the latent space
        # From: (Batch_Size, in_channels, Height/8, Width/8)
        # To:   (Batch_Size, out_channels, Height/8, Width/8)
        x = self.conv(x)
        
        # Return the final output tensor
        # Shape: (Batch_Size, out_channels, Height/8, Width/8)
        return x


class Diffusion(nn.Module):
    """
    Diffusion model for image generation in Stable Diffusion.
    
    This class implements the core diffusion process that gradually denoises
    a latent representation to generate an image. It consists of three main components:
    
    1. Time Embedding: Encodes the noise level (time step) into a high-dimensional
       representation that conditions the denoising process.
    
    2. U-Net: A convolutional network with skip connections that processes the
       noisy latent representation and text embeddings to predict the noise.
    
    3. Output Layer: Projects the U-Net's output back to the original latent space
       dimension for the next denoising step.
    
    The model follows the standard diffusion architecture where noise is gradually
    removed from a latent representation conditioned on text embeddings and time
    information, eventually producing a clean image.
    
    Attributes:
        time_embedding (TimeEmbedding): Module that encodes time steps into embeddings
        unet (UNET): U-Net architecture for noise prediction
        final (UNET_OutputLayer): Final projection layer to match input dimensions
    """
    def __init__(self):
        """
        Initialize the Diffusion model components.
        
        Sets up the time embedding module, U-Net architecture, and final output layer.
        The time embedding dimension is 320, which is expanded internally to 1280
        for better representation capacity.
        """
        super().__init__()
        # Time embedding module that encodes the noise level (time step)
        # The embedding dimension is 320, which is expanded to 1280 internally
        self.time_embedding = TimeEmbedding(320)
        
        # U-Net architecture for processing the noisy latent representation
        # and predicting the noise to be removed
        self.unet = UNET()
        
        # Final projection layer that maps the U-Net output back to the
        # original latent space dimension (4 channels)
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        """
        Forward pass through the diffusion model.
        
        This method processes a noisy latent representation through the U-Net,
        conditioned on text embeddings and time information, to predict the noise
        that should be removed in the next denoising step.
        
        Args:
            latent (torch.Tensor): Noisy latent representation of shape 
                (Batch_Size, 4, Height/8, Width/8)
            context (torch.Tensor): Text embeddings from CLIP of shape 
                (Batch_Size, Sequence_Length, Embedding_Dimension)
            time (torch.Tensor): Time step tensor of shape (Batch_Size, 320)
                indicating the noise level
                
        Returns:
            torch.Tensor: Predicted noise of shape (Batch_Size, 4, Height/8, Width/8)
                that can be subtracted from the latent to move toward a clean image
        """
        # Encode the time step into a higher-dimensional embedding
        # This provides information about the current noise level to the U-Net
        # From: (Batch_Size, 320)
        # To:   (Batch_Size, 1280)
        time = self.time_embedding(time)

        # Process the noisy latent through the U-Net, conditioned on
        # text embeddings and time information
        # From: (Batch_Size, 4, Height/8, Width/8)
        # To:   (Batch_Size, 320, Height/8, Width/8)
        output = self.unet(latent, context, time)

        # Project the U-Net output back to the original latent space dimension
        # This ensures the predicted noise has the same shape as the input latent
        # From: (Batch_Size, 320, Height/8, Width/8)
        # To:   (Batch_Size, 4, Height/8, Width/8)
        output = self.final(output)

        # Return the predicted noise that can be subtracted from the latent
        # to move toward a clean image in the next denoising step
        return output

