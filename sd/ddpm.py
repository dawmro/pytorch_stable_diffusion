import torch
import numpy as np


class DDPMSampler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) Sampler.

    This class implements the sampling process for DDPMs, which are used in generative models
    to iteratively denoise a sample from a Gaussian distribution. The class initializes the
    noise schedule parameters and provides methods for sampling during the diffusion process.

    Attributes:
        betas (torch.Tensor): Variance of noise added at each step, computed from beta_start to beta_end.
        alphas (torch.Tensor): Complement of betas, representing the amount of signal retained.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas, used to compute the overall signal retention.
        one (torch.Tensor): A tensor representing the value one, used in calculations.
        generator (torch.Generator): Random number generator for reproducibility.
        num_train_timesteps (int): Total number of training steps in the diffusion process.
        timesteps (torch.Tensor): Timesteps for the diffusion process, generated in reverse order.
    
    Args:
        generator (torch.Generator): Random number generator for reproducibility.
        num_training_steps (int, optional): Number of steps in the diffusion process. Defaults to 1000.
        beta_start (float, optional): Starting value for the noise variance. Defaults to 0.000085.
        beta_end (float, optional): Ending value for the noise variance. Defaults to 0.012.
    """

    # beta: variance of noise added with each of steps
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.000085, beta_end: float = 0.012):
        """
        Initialize the DDPMSampler with the specified parameters.

        Args:
            generator (torch.Generator): Random number generator for reproducibility.
            num_training_steps (int, optional): Number of steps in the diffusion process. Defaults to 1000.
            beta_start (float, optional): Starting value for the noise variance. Defaults to 0.000085.
            beta_end (float, optional): Ending value for the noise variance. Defaults to 0.012.
        """
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        # num_training_steps: into how many pieces linear space will be divided
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # alpha: variance of noise added with each of steps
        self.alphas = 1.0 - self.betas
        # cumprod: cumulative product of the elements of the input tensor along a given dimension
        # [a1, a2, a3, a4, ...] -> [a1, a1*a2, a1*a2*a3, a1*a2*a3*a4, ...]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())


    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # 999, 998, 997, ... 0 = 1000 steps
        # 999, 979, 959, ... 0 = 50 steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # redefine timesteps according how many are needed
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    # build mean and variance and sample from it
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add diffusion noise to samples at specified timesteps
        
        Implements equation (4) from DDPM paper: q(x_t | x_0) = N(sqrt(alpha) x₀, (1 - alpha) I)
        
        Args:
            original_samples (torch.FloatTensor): Batch of images at time 0 (shape [batch, channels, ...])
            timesteps (torch.LongTensor): Tensor of t values for each sample (shape [batch])
        
        Returns:
            torch.FloatTensor: Noisy samples at specified diffusion steps
            
        Shape Examples:
            original_samples: (32, 3, 256, 256)  → Batch of 32 images
            timesteps: (32,) → Each element from 0-999 indicating their noise step
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # alphas_cumprod has only one dimmension, to combine it with latents and do broadcasting more dimmensions must be the same
        # add dimmension until number of dimmensions is the same as in latents
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # standard deviation
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # add dimmension until number of dimmensions is the same
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples



        
