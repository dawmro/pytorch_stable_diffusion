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

