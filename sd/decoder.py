import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First group normalization layer - normalizes across groups of 32 channels
        # This helps with training stability and reduces internal covariate shift
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        
        # First convolution layer - transforms input channels to output channels
        # kernel_size=3 with padding=1 maintains spatial dimensions
        # This is the main feature transformation layer
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second group normalization layer - normalizes the output of first convolution
        # Uses same group size (32) but for out_channels
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        
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
        # Input tensor shape: (Batch_Size, In_Channels, Height, Width)
        # Store input for residual connection - this will be added back later
        # This is the key to residual learning - helps with gradient flow
        residual = x

        # First transformation block:
        # 1. Group Normalization - normalizes across groups of 32 channels
        #    This helps with training stability and reduces internal covariate shift
        x = self.groupnorm1(x)
        
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
        x = self.groupnorm2(x)
        
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
    

        
        

