import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# https://ezyang.github.io/convolution-visualizer/

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # sequence of submodules, each reduces dimensions of data, but increases depth
        super().__init__(
            # (Batch size, Channel, Height, Width) -> (Batch size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # (Batch size, 128, Height/2, Width/2) -> (Batch size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            # (Batch size, 256, Height/2, Width/2) -> (Batch size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            # (Batch size, 256, Height/2, Width/2) -> (Batch size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            # (Batch size, 256, Height/4, Width/4) -> (Batch size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            # (Batch size, 512, Height/4, Width/4) -> (Batch size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            # (Batch size, 512, Height/4, Width/4) -> (Batch size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8
            VAE_AttentionBlock(512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            nn.SiLU(),
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Channel, Height, Width)
        # noise: (Batch size, Out Channel, Height/8, Width/8)

        # run modules sequentially
        for module in self:
            # To be verified
            # Padding at downsampling should be asymmetric for layers with stride=2 to 
            # Avoid Checkerboard Artifacts and for Better Feature Alignment
            if getattr(module, 'stride', None) == (2, 2):  
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        # splits the tensor along dimension 1 (the channel dimension), first 4 channels: mean, last 4 channels log_variance
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8 to help Numerical Stability. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # Convert log variance back to the actual variance, since it's needed for many operations in the VAE, 
        # such as sampling from the latent space during generation.
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()

        # Convert variance back to standard deviation, which is needed for sampling from the latent space.
        # standard deviation is more practical for many operations and interpretations of the model.
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev) 
        # sample from a standard normal distribution N(0,1) and transform it
        # Allows the model to learn both the mean and variance of the latent distribution
        # The reparameterization trick is what makes VAEs trainable while maintaining their probabilistic nature.
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise

        # Scale by a constant
        # Constant taken from original repo: 
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215


        return x


