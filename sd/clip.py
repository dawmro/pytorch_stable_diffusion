import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) model for text encoding.
    
    CLIP is a neural network trained on a variety of image-text pairs to learn
    a joint embedding space where similar concepts in different modalities (text
    and images) are mapped to nearby points. This implementation focuses on the
    text encoding branch of CLIP.
    
    The architecture consists of:
    1. Token embedding layer that converts text tokens to dense vectors
    2. Multiple transformer layers for contextual understanding
    3. Layer normalization for stable feature distributions
    
    The model processes text tokens through a series of transformer layers,
    allowing it to capture complex semantic relationships and generate rich
    text embeddings that can be used for various downstream tasks in the
    Stable Diffusion pipeline, such as conditioning the image generation process.
    
    Attributes:
        embedding (CLIPEmbedding): Converts token indices to dense embeddings
        layers (nn.ModuleList): List of transformer layers for processing embeddings
        layernorm (nn.LayerNorm): Final normalization layer
    """
    def __init__(self):
        """
        Initialize the CLIP text encoder.
        
        Sets up the embedding layer, transformer layers, and final normalization.
        The model processes text tokens through a series of transformer layers
        to generate rich text embeddings.
        """
        super().__init__()
        # Embedding layer converts token indices to dense vectors
        # Parameters: vocabulary size (49408), embedding dimension (768), max sequence length (77)
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        # Create 12 transformer layers for processing the embeddings
        # Each layer applies self-attention and feed-forward networks
        # Parameters: number of attention heads (12), embedding dimension (768)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        # Final layer normalization for stable feature distributions
        # Normalizes across the embedding dimension (768)
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass through the CLIP text encoder.
        
        This method processes text tokens through the embedding layer and
        transformer layers to generate rich text embeddings.
        
        Args:
            tokens (torch.LongTensor): Input token indices of shape (Batch_Size, Sequence_Length)
                representing the text to be encoded
                
        Returns:
            torch.FloatTensor: Text embeddings of shape (Batch_Size, Sequence_Length, Embedding_Dimension)
                that capture the semantic meaning of the input text
        """
        # Ensure tokens are of the correct type (long integers)
        tokens = tokens.type(torch.long)

        # Convert token indices to dense embeddings
        # From: (Batch_Size, Sequence_Length)
        # To:   (Batch_Size, Sequence_Length, Embedding_Dimension)
        state = self.embedding(tokens)

        # Process embeddings through transformer layers
        # Each layer applies self-attention and feed-forward networks
        # Shape remains: (Batch_Size, Sequence_Length, Embedding_Dimension)
        for layer in self.layers:
            state = layer(state)
        
        # Apply final layer normalization for stable feature distributions
        # From: (Batch_Size, Sequence_Length, Embedding_Dimension)
        # To:   (Batch_Size, Sequence_Length, Embedding_Dimension)
        output = self.layernorm(state)

        return output
    
