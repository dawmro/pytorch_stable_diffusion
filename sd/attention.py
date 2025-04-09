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


