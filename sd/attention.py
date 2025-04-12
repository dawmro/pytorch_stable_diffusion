import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        # Combined projection for query, key, and value matrices
        # This is more efficient than separate projections
        # Shape: (d_embed, d_embed * 3)
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        
        # Output projection after attention computation
        # Shape: (d_embed, d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # Number of attention heads for multi-head attention
        # Each head can focus on different aspects of the input
        self.n_heads = n_heads
        
        # Dimension of each attention head
        # Total embedding dimension is divided among the heads
        # Shape per head: (d_embed // n_heads)
        self.d_embed = d_embed // n_heads

    def forward(self, x, causal_mask: bool = False):
        # Input shape: (Batch_Size, Sequence_Length, Embedding_Dimension)
        # causal_mask: If True, prevents attending to future tokens (for autoregressive models)
        
        # Store original shape for later reshaping
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Shape for multi-head attention
        # (Batch_Size, Sequence_Length, Num_Heads, Embedding_Dimension/Num_Heads)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_embed)

        # Project input to query, key, and value matrices
        # 1. Linear projection: (Batch_Size, Sequence_Length, Embedding_Dimension) -> (Batch_Size, Sequence_Length, Embedding_Dimension*3)
        # 2. Split into three tensors: (Batch_Size, Sequence_Length, Embedding_Dimension) each
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # From: (Batch_Size, Sequence_Length, Embedding_Dimension)
        # To:   (Batch_Size, Num_Heads, Sequence_Length, Embedding_Dimension/Num_Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention scores (scaled dot-product attention)
        # Matrix multiplication: Q @ K^T
        # From: (Batch_Size, Num_Heads, Sequence_Length, Embedding_Dimension/Num_Heads) @ (Batch_Size, Num_Heads, Embedding_Dimension/Num_Heads, Sequence_Length)
        # To:   (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length)
        weight = q @ k.transpose(-1, -2)

        # Apply causal masking if needed (for autoregressive models)
        if causal_mask:
            # Create upper triangular mask (1s above diagonal, 0s on and below diagonal)
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Set masked positions to negative infinity (will become 0 after softmax)
            weight.masked_fill_(mask, -torch.inf) 
                 
        # Scale attention scores by sqrt(d_k) to prevent extremely small gradients
        # This is the "scaled" part of scaled dot-product attention
        weight /= math.sqrt(self.d_head) 

        # Apply softmax to get attention probabilities
        # From: (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length)
        # To:   (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length) with probabilities summing to 1
        weight = F.softmax(weight, dim=-1)

        # Apply attention weights to values
        # From: (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length) @ (Batch_Size, Num_Heads, Sequence_Length, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Num_Heads, Sequence_Length, Embedding_Dimension/Num_Heads)
        out = weight @ v

        # Reshape back to original format
        # From: (Batch_Size, Num_Heads, Sequence_Length, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Sequence_Length, Num_Heads, Embedding_Dimension/Num_Heads)
        output = out.transpose(1, 2) 

        # Flatten heads and embedding dimensions
        # From: (Batch_Size, Sequence_Length, Num_Heads, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Sequence_Length, Embedding_Dimension)
        output = output.reshape(input_shape) 

        # Final linear projection
        # From: (Batch_Size, Sequence_Length, Embedding_Dimension)
        # To:   (Batch_Size, Sequence_Length, Embedding_Dimension)
        output = self.out_proj(output) 

        return output


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism that allows a sequence to attend to a different sequence.
    
    This module implements a cross-attention mechanism where the query comes from one sequence
    (typically the latent representation) and the key/value pairs come from another sequence
    (typically the context embeddings, such as text embeddings from CLIP). This allows the
    model to condition the generation process on external information.
    
    The architecture follows the standard attention mechanism with separate projections for
    query, key, and value, followed by multi-head attention computation. The module includes:
    
    1. Query projection: Projects the input sequence to the embedding dimension
    2. Key projection: Projects the context sequence to the embedding dimension
    3. Value projection: Projects the context sequence to the embedding dimension
    4. Output projection: Projects the attended features back to the original dimension
    
    The attention computation involves:
    1. Computing attention scores between query and key
    2. Scaling the scores to prevent extremely small gradients
    3. Applying softmax to obtain attention probabilities
    4. Weighting the values by the attention probabilities
    
    This mechanism is crucial for conditioning the generation process on text prompts,
    allowing the model to generate images that match the semantic content of the prompt.
    
    Attributes:
        q_proj (nn.Linear): Linear projection for query sequence
        k_proj (nn.Linear): Linear projection for key sequence
        v_proj (nn.Linear): Linear projection for value sequence
        out_proj (nn.Linear): Linear projection for output sequence
        n_heads (int): Number of attention heads
        d_head (int): Dimension of each attention head
    """
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        """
        Initialize the cross-attention module.
        
        Args:
            n_heads (int): Number of attention heads
            d_embed (int): Embedding dimension for the query sequence
            d_cross (int): Embedding dimension for the key/value sequence
            in_proj_bias (bool, optional): Whether to include bias in input projections. Defaults to True.
            out_proj_bias (bool, optional): Whether to include bias in output projection. Defaults to True.
        """
        super().__init__()
        # Query projection: Projects the input sequence to the embedding dimension
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        
        # Key projection: Projects the context sequence to the embedding dimension
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        # Value projection: Projects the context sequence to the embedding dimension
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        # Output projection: Projects the attended features back to the original dimension
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        # Number of attention heads for multi-head attention
        self.n_heads = n_heads
        
        # Dimension of each attention head
        # Total embedding dimension is divided among the heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, y):
        """
        Forward pass through the cross-attention mechanism.
        
        This method computes attention between two sequences, where x is the query sequence
        (typically the latent representation) and y is the key/value sequence (typically
        the context embeddings). The attention mechanism allows each position in the query
        sequence to attend to all positions in the key/value sequence.
        
        Args:
            x (torch.Tensor): Query tensor of shape (Batch_Size, Sequence_Length_Q, Embedding_Dimension_Q)
                representing the sequence to be conditioned (e.g., latent representation)
            y (torch.Tensor): Key/value tensor of shape (Batch_Size, Sequence_Length_KV, Embedding_Dimension_KV)
                representing the conditioning sequence (e.g., text embeddings)
                
        Returns:
            torch.Tensor: Attended tensor of shape (Batch_Size, Sequence_Length_Q, Embedding_Dimension_Q)
                representing the query sequence conditioned on the key/value sequence
        """
        # Store the original shape for later reshaping
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Define the shape for multi-head attention
        # This reshapes the embeddings to separate the heads
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project the query sequence to the embedding dimension
        # This preserves the original dimension but allows for transformation
        q = self.q_proj(x)
        
        # Project the key sequence to the embedding dimension
        # This maps the context embeddings to the same dimension as the query
        k = self.k_proj(y)
        
        # Project the value sequence to the embedding dimension
        # This maps the context embeddings to the same dimension as the query
        v = self.v_proj(y)

        # Reshape the query for multi-head attention
        # From: (Batch_Size, Sequence_Length_Q, Embedding_Dimension)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_Q, Embedding_Dimension/Num_Heads)
        q = q.view(interim_shape).transpose(1, 2)
        
        # Reshape the key for multi-head attention
        # From: (Batch_Size, Sequence_Length_KV, Embedding_Dimension)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_KV, Embedding_Dimension/Num_Heads)
        k = k.view(interim_shape).transpose(1, 2)
        
        # Reshape the value for multi-head attention
        # From: (Batch_Size, Sequence_Length_KV, Embedding_Dimension)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_KV, Embedding_Dimension/Num_Heads)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention scores between query and key
        # Matrix multiplication: Q @ K^T
        # From: (Batch_Size, Num_Heads, Sequence_Length_Q, Embedding_Dimension/Num_Heads) @ (Batch_Size, Num_Heads, Embedding_Dimension/Num_Heads, Sequence_Length_KV)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_Q, Sequence_Length_KV)
        weight = q @ k.transpose(-1, -2)

        # Scale the attention scores to prevent extremely small gradients
        # This is the "scaled" part of scaled dot-product attention
        weight /= math.sqrt(self.d_head)
        
        # Apply softmax to obtain attention probabilities
        # This ensures the attention weights sum to 1 for each query position
        # From: (Batch_Size, Num_Heads, Sequence_Length_Q, Sequence_Length_KV)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_Q, Sequence_Length_KV) with probabilities summing to 1
        weight = F.softmax(weight, dim=-1)
        
        # Apply attention weights to values
        # From: (Batch_Size, Num_Heads, Sequence_Length_Q, Sequence_Length_KV) @ (Batch_Size, Num_Heads, Sequence_Length_KV, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Num_Heads, Sequence_Length_Q, Embedding_Dimension/Num_Heads)
        output = weight @ v
        
        # Reshape back to the original format
        # From: (Batch_Size, Num_Heads, Sequence_Length_Q, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Sequence_Length_Q, Num_Heads, Embedding_Dimension/Num_Heads)
        output = output.transpose(1, 2).contiguous()
        
        # Flatten heads and embedding dimensions
        # From: (Batch_Size, Sequence_Length_Q, Num_Heads, Embedding_Dimension/Num_Heads)
        # To:   (Batch_Size, Sequence_Length_Q, Embedding_Dimension)
        output = output.view(input_shape)
        
        # Final linear projection
        # From: (Batch_Size, Sequence_Length_Q, Embedding_Dimension)
        # To:   (Batch_Size, Sequence_Length_Q, Embedding_Dimension)
        output = self.out_proj(output)

        # Return the attended tensor
        # Shape: (Batch_Size, Sequence_Length_Q, Embedding_Dimension)
        return output
