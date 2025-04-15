import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """
    Attention block for the VAE decoder that captures global dependencies in the image.
    
    This module implements a self-attention mechanism that allows each spatial position
    in the feature map to attend to all other positions. This helps the model capture
    long-range dependencies and global context that may be important for image reconstruction.
    
    The attention block consists of:
    1. Group normalization for stable feature distributions
    2. Self-attention mechanism that computes attention weights between all spatial positions
    3. Residual connection to preserve original features
    
    The self-attention mechanism reshapes the input tensor to treat spatial positions
    as a sequence, applies attention, and then reshapes back to the original format.
    This allows the model to consider relationships between distant parts of the image
    that might be difficult to capture with standard convolutions alone.
    
    Attributes:
        groupnorm (nn.GroupNorm): Normalizes features across groups of 32 channels
        attention (SelfAttention): Self-attention mechanism that computes attention weights
    """
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store the original input for the residual connection
        residual = x
        
        # Extract dimensions from input tensor
        # Shape: (Batch_Size, Channels, Height, Width)
        n, c, h, w = x.shape
        
        # Reshape for self-attention:
        # 1. Flatten spatial dimensions (Height, Width) into a single dimension
        # From: (Batch_Size, Channels, Height, Width)
        # To:   (Batch_Size, Channels, Height*Width)
        x = x.view((n, c, h * w))
        
        # 2. Transpose to prepare for self-attention
        # Self-attention expects shape: (Batch_Size, Sequence_Length, Features)
        # From: (Batch_Size, Channels, Height*Width)
        # To:   (Batch_Size, Height*Width, Channels)
        x = x.transpose(-1, -2)
        
        # Apply self-attention to capture global dependencies
        # This allows each spatial position to attend to all other positions
        # Shape remains: (Batch_Size, Height*Width, Channels)
        x = self.attention(x)
        
        # Reverse the transposition to restore channel-first format
        # From: (Batch_Size, Height*Width, Channels)
        # To:   (Batch_Size, Channels, Height*Width)
        x = x.transpose(-1, 2)
        
        # Reshape back to original 4D tensor format
        # From: (Batch_Size, Channels, Height*Width)
        # To:   (Batch_Size, Channels, Height, Width)
        x = x.view((n, c, h, w))
        
        # Add residual connection to help with gradient flow
        # This preserves original features while adding attention-enhanced features
        x += residual
        
        return x


class VAE_ResidualBlock(nn.Module):
    """
    Residual block for the VAE decoder that refines features through skip connections.
    
    This module implements a residual learning approach where the input is added back
    to the output of a series of transformations. This helps with gradient flow in
    deep networks and allows the model to learn incremental refinements to the features.
    
    The residual block consists of:
    1. Two group normalization layers for stable feature distributions
    2. Two convolution layers for feature transformation
    3. SiLU (Swish) activation functions for non-linearity
    4. A residual connection that adds the input to the output
    
    If the input and output channels differ, a 1x1 convolution is used to transform
    the input channels to match the output channels before the residual addition.
    
    Residual blocks are a key component of modern deep learning architectures as they
    help address the vanishing gradient problem and allow for training of very deep networks.
    
    Attributes:
        groupnorm_1 (nn.GroupNorm): First group normalization layer
        conv_1 (nn.Conv2d): First convolution layer
        groupnorm_2 (nn.GroupNorm): Second group normalization layer
        conv_2 (nn.Conv2d): Second convolution layer
        residual_layer (nn.Module): Layer for transforming input channels if needed
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First group normalization layer - normalizes across groups of 32 channels
        # This helps with training stability and reduces internal covariate shift
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        
        # First convolution layer - transforms input channels to output channels
        # kernel_size=3 with padding=1 maintains spatial dimensions
        # This is the main feature transformation layer
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second group normalization layer - normalizes the output of first convolution
        # Uses same group size (32) but for out_channels
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        
        # Second convolution layer - maintains the number of channels
        # This layer refines the features further
        # Also uses 3x3 kernel with padding=1 to maintain spatial dimensions
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Handle residual connection based on channel dimensions
        # If input and output channels differ, we need to transform the input
        # to match the output channels for the residual addition
        if in_channels != out_channels:
            # Use 1x1 convolution to transform channels without affecting spatial dimensions
            # This is a common practice in residual networks for channel matching
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            # If channels match, use identity mapping (just pass through the input)
            # This is the standard residual connection
            self.residual_layer = nn.Identity()   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        This method applies a series of transformations to the input tensor and adds
        the original input back to the result (possibly transformed if channels differ).
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, In_Channels, Height, Width)
            
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Out_Channels, Height, Width)
                with refined features and residual connection
        """
        # Input tensor shape: (Batch_Size, In_Channels, Height, Width)
        # Store input for residual connection - this will be added back later
        # This is the key to residual learning - helps with gradient flow
        residual = x

        # First transformation block:
        # 1. Group Normalization - normalizes across groups of 32 channels
        #    This helps with training stability and reduces internal covariate shift
        x = self.groupnorm_1(x)
        
        # 2. SiLU (Swish) activation - f(x) = x * sigmoid(x)
        #    This is a smooth, non-linear activation that helps with training
        #    Better than ReLU in deep networks due to its smoothness
        x = F.silu(x)
        
        # 3. First convolution - transforms features while maintaining spatial dimensions
        #    Uses 3x3 kernel with padding=1 to preserve spatial size
        x = self.conv_1(x)      

        # Second transformation block:
        # 1. Group Normalization - normalizes the features after first convolution
        #    Helps maintain stable feature distributions
        x = self.groupnorm_2(x)
        
        # 2. SiLU activation - applies non-linearity again
        #    Helps the network learn more complex patterns
        x = F.silu(x)
        
        # 3. Second convolution - further refines the features
        #    Maintains same number of channels as output
        x = self.conv_2(x)

        # Residual connection:
        # Add the original input (possibly transformed if channels don't match)
        # This helps with:
        # - Gradient flow in deep networks
        # - Feature preservation
        # - Training stability
        x += self.residual_layer(residual)

        return x
    

class VAE_Decoder(nn.Sequential):
    """
    Variational Autoencoder (VAE) Decoder for Stable Diffusion.
    
    This decoder reconstructs images from latent representations produced by the VAE encoder.
    It implements a progressive upsampling architecture that transforms the compressed
    latent space back into the original image space through a series of operations:
    
    1. Initial projection and feature expansion
    2. Multiple residual blocks for feature refinement
    3. Attention mechanisms to capture global dependencies
    4. Progressive upsampling to restore spatial dimensions
    5. Channel reduction as spatial dimensions increase
    6. Final projection to RGB image space
    
    The architecture follows a U-Net-like structure with skip connections (residual blocks)
    that help maintain fine details during reconstruction. The decoder is designed to
    work in conjunction with the VAE encoder to enable efficient image compression
    and reconstruction in the Stable Diffusion pipeline.
    
    The scaling factor of 0.18215 is used to normalize the latent space and is derived
    from the KL divergence term in the VAE loss function. This helps with training
    stability and ensures the latent space follows a standard normal distribution.
    
    Attributes:
        All layers are defined in the Sequential constructor and include:
        - Convolutional layers for feature transformation
        - Residual blocks for feature refinement
        - Attention blocks for capturing global dependencies
        - Upsampling layers for spatial dimension restoration
        - Group normalization for training stability
        - SiLU activation for non-linearity
    """
    def __init__(self):
        """
        Initialize the VAE Decoder architecture.
        
        The decoder progressively transforms a latent representation into an image
        through a series of upsampling and feature refinement operations.
        """
        super().__init__(
            # Initial projection from latent space (4 channels)
            # Shape: (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8)
            nn.Conv2d(4, 4, kernel_size=1),
            
            # Expand to higher-dimensional feature space
            # Shape: (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # First residual block for feature refinement
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # Attention block to capture global dependencies
            # Allows the model to consider relationships between distant parts of the image
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            
            # Second residual block for further feature refinement
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # Third residual block for additional feature refinement
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # Fourth residual block for further feature refinement
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # Fifth residual block for additional feature refinement
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # First upsampling: Double spatial dimensions
            # Repeats each pixel in both height and width directions
            # Shape: (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),
            
            # Convolution after upsampling to refine features
            # Shape: (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # Residual block after first upsampling
            # Shape: (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            # Second residual block after first upsampling
            # Shape: (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            # Third residual block after first upsampling
            # Shape: (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            # Second upsampling: Double spatial dimensions again
            # Shape: (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),
            
            # Convolution after second upsampling
            # Shape: (Batch_Size, 512, Height/2, Width/2) -> (Batch_Size, 512, Height/2, Width/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # Reduce channels while maintaining spatial dimensions
            # Shape: (Batch_Size, 512, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(512, 256),
            
            # Residual block with reduced channels
            # Shape: (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            
            # Second residual block with reduced channels
            # Shape: (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            
            # Third upsampling: Double spatial dimensions to full resolution
            # Shape: (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            # Convolution at full resolution
            # Shape: (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            # Further reduce channels at full resolution
            # Shape: (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128),
            
            # Residual block with further reduced channels
            # Shape: (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # Second residual block with further reduced channels
            # Shape: (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # Normalize features across groups of 32 channels
            # Helps with training stability
            # Shape: (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128),
            
            # Apply SiLU activation for non-linearity
            # Better gradient flow than ReLU in deep networks
            # Shape: (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(),
            
            # Final projection to RGB image space
            # Shape: (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through the VAE Decoder.
        
        This method takes a latent representation from the encoder and reconstructs
        the original image by passing it through a series of upsampling and feature
        refinement operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 4, Height/8, Width/8)
                representing the latent space encoding from the VAE encoder.
                
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (Batch_Size, 3, Height, Width)
                representing the RGB image in the original resolution.
        """
        # Input shape: (Batch_Size, 4, Height/8, Width/8)
        # This is the compressed latent representation from the encoder
        
        # Reverse the scaling factor applied by the encoder (0.18215)
        # This scaling helps with numerical stability during training
        # The factor 0.18215 is derived from the KL divergence term in the VAE loss
        x /= 0.18215

        # Sequentially apply each module in the decoder
        # This includes convolutions, residual blocks, attention mechanisms, and upsampling
        # The data progressively transforms from latent space to image space
        for module in self:
            x = module(x)

        # Output shape: (Batch_Size, 3, Height, Width)
        # The final tensor represents the reconstructed RGB image
        return x