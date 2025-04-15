import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    """
    Generate an image from a text prompt using Stable Diffusion.
    
    This function implements the core generation pipeline for Stable Diffusion, which consists
    of several key steps:
    
    1. Text Conditioning: Convert the text prompt into embeddings using the CLIP model
    2. Latent Initialization: Either start from random noise or encode an input image
    3. Diffusion Process: Iteratively denoise the latents using the U-Net model
    4. Image Decoding: Convert the final latents into a visible image using the VAE decoder
    
    The function supports both text-to-image generation and image-to-image generation (with
    the input_image parameter). It also implements classifier-free guidance (CFG) for improved
    prompt adherence, which works by comparing the predictions with and without the prompt.
    
    Args:
        prompt (list or tuple): The text prompt to condition the generation on
        uncond_prompt (list or tuple, optional): The unconditional prompt for classifier-free guidance.
            Defaults to None.
        input_image (PIL.Image, optional): Input image for image-to-image generation.
            Defaults to None.
        strength (float, optional): How much to transform the input image (0-1).
            Only used when input_image is provided. Defaults to 0.8.
        do_cfg (bool, optional): Whether to use classifier-free guidance. Defaults to True.
        cfg_scale (float, optional): Scale factor for classifier-free guidance. Defaults to 7.5.
        sampler_name (str, optional): Name of the sampler to use. Defaults to "ddpm".
        n_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
        models (dict, optional): Dictionary of models to use. Must contain 'clip', 'diffusion', and 'decoder'.
            'encoder' is required if input_image is provided. Defaults to {}.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        device (torch.device, optional): Device to run the models on. Defaults to None (auto-detect).
        idle_device (torch.device, optional): Device to move models to when not in use.
            Defaults to None (keep on same device).
        tokenizer (object, optional): Tokenizer for converting text to tokens. Defaults to None.
        
    Returns:
        numpy.ndarray: Generated image as a numpy array with shape (Height, Width, Channels)
        
    Raises:
        ValueError: If prompt is not a non-empty list or tuple
        ValueError: If uncond_prompt is not a non-empty list or tuple when provided
        ValueError: If strength is not between 0 and 1
        ValueError: If sampler_name is not supported
    """
    with torch.no_grad():

        # check if strength is between 0 and 1
        if not 0 < strength <= 1:
            # if strength is not between 0 and 1, raise an error
            raise ValueError(f"Strength must be between 0 and 1, got {strength}")
             
        # Set up device management
        if idle_device: 
            # Move models to idle_device when not in use to save memory
            to_idle = lambda x: x.to(idle_device)
        else:   
            # Keep models on the same device if no idle_device specified
            to_idle = lambda x: x

        # Auto-detect device if not specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up random number generator for reproducibility
        generator = torch.Generator(device=device)
        if seed is None:
            # Use random seed if none provided
            generator.seed()
        else:
            # Use specified seed for reproducibility
            generator.manual_seed(seed)

        # Load and prepare CLIP model for text encoding
        clip = models["clip"]
        clip.to(device)

        # Process text prompts and generate embeddings
        if do_cfg:
            # Classifier-free guidance: process both conditional and unconditional prompts
            
            # Encode the conditional prompt (the actual prompt) into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Generate embeddings for the conditional prompt
            cond_context = clip(cond_tokens)

            # Encode the unconditional prompt (typically empty or negative prompt)
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # Generate embeddings for the unconditional prompt
            uncond_context = clip(uncond_tokens)
            
            # Combine conditional and unconditional embeddings into single tensor to create input batch for unet 
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Standard generation: process only the conditional prompt            
            # # Do one step through unet only with prompt, without combining cond and uncond context
            # Encode the prompt
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
            # Generate embeddings for the prompt
            context = clip(tokens)
        
        # Move CLIP model to idle device to free up memory
        to_idle(clip)

        # Set up the diffusion sampler
        if sampler_name == "ddpm":
            # Initialize DDPM (Denoising Diffusion Probabilistic Models) sampler
            sampler = DDPMSampler(generator)
            # Configure the number of inference steps
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not found")
        
        # Define the shape of the latent representation
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # Handle image-to-image generation
        if input_image:
            # Load and prepare VAE encoder
            encoder = models["encoder"]
            encoder.to(device)

            # Preprocess the input image
            # Resize to the expected dimensions
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # Convert to numpy array
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # Convert to PyTorch tensor
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            # Rescale pixel values from [0, 255] to [-1, 1]
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # Add batch dimension
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            
            # Rearrange dimensions beacause VAE_Encoder expects (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Generate random noise for the encoder
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # Encode the input image to latent space
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Configure the sampler with the specified strength
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)

            # Add noise to the latents based on the strength parameter
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move encoder to idle device to free up memory
            to_idle(encoder)

        # Handle text-to-image generation
        else:
            # Start with random noise in latent space, bacuase there is no input image
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Load and prepare the diffusion model (U-Net)
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Set up the denoising schedule
        # During training, 1000 steps are used, but for inference we use fewer steps
        # The timesteps are spaced evenly across the full range
        timesteps = tqdm(sampler.timesteps)
        
        # Iterative denoising process
        for i, timestep in enumerate(timesteps):
            # Generate time embedding for the current timestep
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare the input for the diffusion model
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            # For classifier-free guidance, duplicate the latents
            if do_cfg:
                # Repeat latents for both conditional and unconditional paths
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise using the U-Net model
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            # Apply classifier-free guidance if enabled
            if do_cfg:
                # Split the output into conditional and unconditional predictions
                output_cond, output_uncond = model_output.chunk(2)
                # Combine predictions using the CFG scale
                # Higher scale = stronger adherence to the prompt
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Update the latents by removing the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        # Move diffusion model to idle device to free up memory
        to_idle(diffusion)

        # Load and prepare the VAE decoder
        decoder = models["decoder"]
        decoder.to(device)

        # Decode the final latents into an image by running them through decoder
        image = decoder(latents)

        # Move decoder to idle device to free up memory
        to_idle(decoder)

        # Rescale pixel values from [-1, 1] to [0, 255]
        image = rescale(image, (-1, 1), (0, 255), clamp=True)

        # Rearrange dimensions to (Batch, Height, Width, Channels) for image saving on cpu, channel dimension should be last
        images = images.permute(0, 2, 3, 1)
        
        # Convert to CPU, uint8, and numpy array for output
        images = images.to("cpu", torch.uint8).numpy()
        
        # Return the first (and only) image
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    """
    Rescale tensor values from one range to another.
    
    This utility function performs linear rescaling of tensor values from one range to another.
    It's commonly used in image processing to convert between different value ranges, such as
    from [0, 255] to [-1, 1] for neural network inputs, or from [-1, 1] to [0, 255] for image outputs.
    
    The rescaling is performed using the formula:
        x_new = (x_old - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    
    Args:
        x (torch.Tensor): Input tensor to be rescaled
        old_range (tuple): Tuple of (old_min, old_max) defining the current range of values
        new_range (tuple): Tuple of (new_min, new_max) defining the desired range of values
        clamp (bool, optional): Whether to clamp the output to the new range. Defaults to False.
        
    Returns:
        torch.Tensor: Rescaled tensor with values in the new range
        
    Example:
        >>> x = torch.tensor([0, 128, 255])
        >>> rescale(x, (0, 255), (-1, 1))
        tensor([-1.0000,  0.0000,  1.0000])
    """
    # Extract the minimum and maximum values from the ranges
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    # Shift the values to start from zero
    x -= old_min
    
    # Scale the values to the new range
    x *= (new_max - new_min) / (old_max - old_min)
    
    # Shift the values to the new minimum
    x += new_min
    
    # Optionally clamp the values to ensure they stay within the new range
    if clamp:
        x = x.clamp(new_min, new_max)
        
    return x


def get_time_embedding(timestep):
    """
    Generate sinusoidal positional embeddings for diffusion timesteps.
    
    This function creates a time embedding vector for a given diffusion timestep using
    sinusoidal positional encoding, similar to the positional embeddings used in transformers.
    The embedding captures the timestep information in a way that preserves relative
    distances between timesteps and allows the model to learn temporal dependencies.
    
    The function follows these steps:
    1. Generate frequency bands using an exponential decay
    2. Multiply the timestep by these frequencies
    3. Apply sine and cosine functions to create the embedding
    4. Concatenate the sine and cosine components
    
    This approach allows the model to better understand the noise level at each timestep
    and adjust its denoising strategy accordingly.
    
    Args:
        timestep (int): The diffusion timestep to generate embeddings for
        
    Returns:
        torch.Tensor: A tensor of shape (1, 320) containing the time embedding
        
    Note:
        The output dimension is 320 because it concatenates 160 sine and 160 cosine values.
    """
    # Generate frequency bands using exponential decay
    # These frequencies determine how quickly the sine/cosine waves oscillate
    # Shape: (160,) - 160 different frequency bands
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    
    # Multiply the timestep by each frequency to create phase-shifted values
    # Shape: (1, 160) - One timestep × 160 frequencies
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # Apply sine and cosine functions and concatenate the results
    # This creates a rich representation of the timestep that preserves relative distances
    # Shape: (1, 320) - Concatenated sine and cosine values (160 + 160)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        
        
        
        
            














            




            
        
        
                
            
            
