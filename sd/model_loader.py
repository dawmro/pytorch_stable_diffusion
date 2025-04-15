from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter  # Module for converting checkpoint data to model-usable format


def preload_models_from_standard_weights(ckpt_path, device):
    """Load PyTorch models from a pre-trained checkpoint file.
    
    This function loads a pre-trained CLIP, VAE, and Diffusion model
    from a standard checkpoint format file (.ckpt), converting and 
    mapping weights to the respective model architectures.
    
    Args:
        ckpt_path (str): Path to the checkpoint file containing model weights
        device (torch.device): Device to load models onto (e.g., "cuda" or "cpu")
        
    Returns:
        dict: Dictionary with keys 'clip', 'encoder', 'decoder', 'diffusion',
              each containing the loaded model instances on the specified device
    """
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    # Converts standard checkpoint format into a Python dictionary of tensors
    # containing model weights split into their respective model components

    encoder = VAE_Encoder().to(device)
    # Initialize the VAE Encoder model and move to specified device
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    # Load encoder weights from checkpoint; strict=True requires exact key matching

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    # Initialize and load VAE Decoder with weights from checkpoint

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    # Initialize and load Diffusion model with its weights from checkpoint

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)
    # Initialize and load CLIP model with textual/image encoding weights from checkpoint

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion
    }
    # Return dictionary of all loaded models, ready for inference/prediction
