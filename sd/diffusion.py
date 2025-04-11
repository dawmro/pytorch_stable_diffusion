import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


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

