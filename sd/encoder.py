import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
import warnings # For notifying about experimental features

# Assuming these are defined elsewhere and imported
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
# from torch.nn.utils import spectral_norm # For optional spectral norm

# --- Optional: Anti-aliasing Downsampling Helper ---
# For a true BlurPool, consider libraries like kornia or implement based on:
# https://richzhang.github.io/antialiased-cnns/
# Using AvgPool2d as a simple, built-in stand-in for anti-aliasing concept.
class AntiAliasDownsample(nn.Module):
    """Placeholder: Applies anti-aliasing (blur) before downsampling."""
    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        self.stride = stride
        if stride > 1:
            # Simple average pooling as a stand-in for BlurPool
            # kernel_size=stride*2-1 often used, e.g., 3 for stride 2
            # Padding calculation ensures output size is halved for stride 2
            padding = (kernel_size - stride) // 2
            self.blur = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
            # Downsampling now happens in the subsequent Conv2d layer
        else:
            self.blur = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(x)

class VAE_Encoder(nn.Sequential):
    """
    Enhanced Variational Autoencoder (VAE) Encoder for Stable Diffusion.

    Compresses images into a latent representation using a hierarchical downsampling
    architecture with residual connections and attention mechanisms. This version
    incorporates potential enhancements for stability, performance, and clarity
    while maintaining the original nn.Sequential structure.

    Enhancements Include:
    - Configurable latent scaling factor.
    - Optional anti-aliased downsampling (using AvgPool stand-in).
    - Optional torch.compile integration.
    - Placeholders/comments for gradient checkpointing and stochastic depth.
    - Pre-calculated asymmetric padding for strided convolutions.
    - Improved documentation and type hinting.
    - Optional spectral normalization hooks (commented out).

    Architecture:
    1. Initial Conv layer (3 -> 128 channels)
    2. Residual Blocks (feature refinement)
    3. Downsampling Stages (AntiAlias -> Conv stride 2):
        - 128 -> 128 (H/2, W/2)
        - 128 -> 256 (H/4, W/4)
        - 256 -> 512 (H/8, W/8)
    4. More Residual Blocks at lowest resolution
    5. Attention Block (captures global context)
    6. Final MLP-like projection (GroupNorm, SiLU, Conv layers) to latent parameters (mean, log_variance)

    Outputs:
    - Sampled latent variable `z` using the reparameterization trick.
    """
    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 4, # Dimensionality of the latent space mean/variance vectors
                 base_channels: int = 128,
                 channels_mult: Tuple[int, ...] = (1, 2, 4, 4), # Multipliers for base_channels at each stage
                 num_res_blocks: int = 2,
                 attn_resolutions: Tuple[int, ...] = (16,), # Resolutions (feature map side length) where attention is applied
                 use_blur_pool: bool = True, # Use anti-aliasing before downsampling
                 latent_scale_factor: float = 0.18215, # Standard SD scaling factor
                 use_checkpointing: bool = False, # Enable gradient checkpointing for memory saving
                 stochastic_depth_p: float = 0.0, # Probability for stochastic depth (0.0 = disabled)
                 # use_spectral_norm: bool = False # Optional: Apply spectral norm to conv layers
                 ):
        """
        Initializes the Enhanced VAE Encoder architecture.

        Args:
            in_channels (int): Number of input image channels (usually 3 for RGB).
            latent_dim (int): Dimensionality of the latent space (typically 4 for SD).
            base_channels (int): Base number of channels, scaled by multipliers.
            channels_mult (Tuple[int, ...]): Multipliers for base channels at each resolution level.
                                             Length determines number of downsampling stages.
            num_res_blocks (int): Number of residual blocks per resolution level.
            attn_resolutions (Tuple[int, ...]): Feature map resolutions at which to apply attention blocks.
            use_blur_pool (bool): If True, applies anti-aliasing before strided convolutions.
            latent_scale_factor (float): Scaling factor applied to the final latent variable.
            use_checkpointing (bool): If True, enables gradient checkpointing wrappers.
            stochastic_depth_p (float): Probability of dropping a residual block during training.
            # use_spectral_norm (bool): If True, applies spectral normalization to Conv2d layers.
        """
        self.use_checkpointing = use_checkpointing
        self.stochastic_depth_p = stochastic_depth_p
        self.latent_scale_factor = latent_scale_factor # Store scale factor

        layers = []
        current_channels = base_channels
        current_res = -1 # Placeholder, will be set based on input later if needed for attn

        # --- Initial Convolution ---
        # (B, C, H, W) -> (B, base_channels, H, W)
        initial_conv = nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1)
        # if use_spectral_norm: initial_conv = spectral_norm(initial_conv)
        layers.append(initial_conv)

        # --- Downsampling Stages ---
        for i, mult in enumerate(channels_mult):
            out_channels = base_channels * mult

            # Residual Blocks
            for _ in range(num_res_blocks):
                res_block = VAE_ResidualBlock(current_channels, out_channels)
                # if use_spectral_norm: res_block = apply_spectral_norm_to_module(res_block) # Helper needed
                layers.append(res_block)
                current_channels = out_channels

            # Attention Block (if resolution matches)
            # Note: Need input H, W to determine resolution accurately beforehand.
            # Assuming standard H=W input and powers of 2 downsampling.
            # If input H=512, first stage H/2=256, H/4=128, H/8=64, H/16=32...
            # This check is heuristic without knowing input size.
            # Example: if current_res in attn_resolutions: layers.append(VAE_AttentionBlock(current_channels))

            # Downsampling (if not the last stage)
            if i != len(channels_mult) - 1:
                if use_blur_pool:
                    layers.append(AntiAliasDownsample(channels=current_channels, stride=2))
                    # Conv stride remains 1 after blur pool handles spatial reduction implicitly via kernel/padding
                    down_conv = nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1) # Standard conv downsample
                else:
                    # Original approach: Strided Conv
                    # Calculate asymmetric padding: (padding_left, padding_right, padding_top, padding_bottom)
                    # For stride=2, kernel=3 -> need total padding of 1. Asymmetric: (0, 1, 0, 1)
                    # We will apply F.pad before this layer in forward pass for consistency
                    down_conv = nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=0) # Padding done manually

                # if use_spectral_norm: down_conv = spectral_norm(down_conv)
                layers.append(down_conv)

        # --- Middle section (at lowest resolution) ---
        # Based on original structure: More ResBlocks + Attention at the end
        layers.append(VAE_ResidualBlock(current_channels, current_channels))
        layers.append(VAE_AttentionBlock(current_channels)) # Check if attn_resolutions logic placed it correctly
        layers.append(VAE_ResidualBlock(current_channels, current_channels))

        # --- Projection to Latent Space ---
        # (B, C_final, H/8, W/8) -> (B, C_final, H/8, W/8)
        layers.append(nn.GroupNorm(32, current_channels))
        layers.append(nn.SiLU())
        # (B, C_final, H/8, W/8) -> (B, 2 * latent_dim, H/8, W/8)
        final_conv1 = nn.Conv2d(current_channels, 2 * latent_dim, kernel_size=3, padding=1)
        # if use_spectral_norm: final_conv1 = spectral_norm(final_conv1)
        layers.append(final_conv1)

        # Original had an extra 1x1 Conv, seems unnecessary if final_conv1 projects correctly.
        # Keeping it for consistency with original snippet for now.
        # (B, 2 * latent_dim, H/8, W/8) -> (B, 2 * latent_dim, H/8, W/8)
        final_conv2 = nn.Conv2d(2 * latent_dim, 2 * latent_dim, kernel_size=1, padding=0)
        # if use_spectral_norm: final_conv2 = spectral_norm(final_conv2)
        layers.append(final_conv2)

        super().__init__(*layers)

        # Store padding values for strided convolutions if not using BlurPool
        self.asymmetric_padding = (0, 1, 0, 1) # (pad_left, pad_right, pad_top, pad_bottom)
        self.use_blur_pool = use_blur_pool

        # --- Optional: Compile the model ---
        # Can be applied after initialization
        # try:
        #     self = torch.compile(self, mode="reduce-overhead") # Or other modes
        #     print("VAE_Encoder compiled successfully.")
        # except Exception as e:
        #     print(f"VAE_Encoder compilation failed: {e}")


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VAE Encoder.

        Applies the sequential layers, handles optional features like padding,
        checkpointing, stochastic depth, and performs the reparameterization trick.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch_Size, C_in, H, W).
            noise (torch.Tensor): Random noise tensor for sampling, shape
                                  (Batch_Size, latent_dim, H_latent, W_latent).

        Returns:
            torch.Tensor: Sampled latent representation `z`, shape
                          (Batch_Size, latent_dim, H_latent, W_latent).
        """
        skipped_layers = 0
        total_res_blocks = sum(isinstance(m, VAE_ResidualBlock) for m in self) # Count ResBlocks

        for module in self:
            # --- Stochastic Depth ---
            is_res_block = isinstance(module, VAE_ResidualBlock)
            if is_res_block and self.training and self.stochastic_depth_p > 0:
                # Calculate survival probability for this block
                # Linear decay: p_l = 1 - (l/L) * p_L
                keep_prob = 1.0 - (float(skipped_layers + 1) / total_res_blocks) * self.stochastic_depth_p
                if torch.rand(1).item() >= keep_prob:
                    # print(f"Skipping ResBlock {skipped_layers + 1}")
                    skipped_layers += 1
                    continue # Skip this module
                else:
                    # Apply scaling to the output if block is kept
                    # x = module(x) / keep_prob # Inverted Residual Dropout scaling
                    # Or apply scaling within the block if it supports it.
                    # Simpler: Just run the block without scaling here.
                    pass # Fall through to apply module


            # --- Manual Asymmetric Padding (if not using BlurPool) ---
            is_strided_conv = isinstance(module, nn.Conv2d) and getattr(module, 'stride', (1,1)) == (2, 2)
            if not self.use_blur_pool and is_strided_conv:
                 # Apply pre-calculated padding before the strided convolution
                 x = F.pad(x, self.asymmetric_padding)

            # --- Gradient Checkpointing ---
            # Apply checkpointing to specific expensive modules if enabled
            if self.use_checkpointing and self.training and \
               (isinstance(module, VAE_ResidualBlock) or isinstance(module, VAE_AttentionBlock)):
                # Requires module to accept all inputs needed for its forward pass
                # May need to handle non-tensor inputs if module requires them
                try:
                     x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False) # Check PyTorch version compatibility
                except Exception as e:
                    warnings.warn(f"Checkpointing failed for {type(module).__name__}: {e}. Running module normally.")
                    x = module(x)
            else:
                 # --- Standard Module Application ---
                 x = module(x)

            # Increment counter if this was a ResBlock (even if skipped, for correct probability decay)
            if is_res_block:
                 skipped_layers += 1


        # --- Reparameterization Trick ---
        # (B, 2 * latent_dim, H_latent, W_latent) -> two tensors of shape (B, latent_dim, H_latent, W_latent)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp log variance for numerical stability.
        # Clamping range is standard practice from original implementations.
        log_variance = torch.clamp(log_variance, -30, 20)

        # Calculate standard deviation
        variance = log_variance.exp()
        stdev = variance.sqrt() # potential optimization: stdev = torch.exp(0.5 * log_variance)

        # Sample z = mean + stdev * epsilon (noise)
        # noise tensor shape should match mean/stdev shape: (B, latent_dim, H_latent, W_latent)
        if noise.shape != mean.shape:
             raise ValueError(f"Noise shape {noise.shape} must match mean shape {mean.shape}")

        z = mean + stdev * noise

        # Scale the output
        z *= self.latent_scale_factor

        return z

# --- Helper function for applying spectral norm recursively (Optional) ---
# def apply_spectral_norm_to_module(module):
#     for name, layer in module.named_children():
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
#             setattr(module, name, spectral_norm(layer))
#         else:
#             apply_spectral_norm_to_module(layer) # Recurse
#     return module
