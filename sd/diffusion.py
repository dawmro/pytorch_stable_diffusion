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
        unet (UNet): U-Net architecture for noise prediction
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
        self.unet = UNet()
        
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

