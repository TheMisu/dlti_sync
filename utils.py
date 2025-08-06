"""
This module contains utility functions for the processing pipelines.
Included functions:
1. GPU memory cleanup
2. Cosine similarity between embeddings
"""

import numpy as np
import torch
import gc


def clear_memory():
    """
    This function clears the GPU memory and colelcts the garbage
    """
    torch.cuda.empty_cache()
    gc.collect()


def cosine_similarity(a, b):
    """
    This function computes the cosine similarity between two vectors a and b

    Keyword parameters:
    a -- first vector (numpy array or torch tensor)
    b -- second vector (numpy array or torch tensor)

    Returns:
    float: cosine similarity score
           or np.nan if inputs are invalid (NaN, div by 0 etc)
    """
    try:
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        if isinstance(b, torch.Tensor):
            b = b.numpy()

        a = a.flatten()
        b = b.flatten()

        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            print("Cosine similarity input contains Nan")
            return np.nan

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            print("Cosine similarity input is a zero vector")
            return np.nan

        dot_product = np.dot(a, b)
        similarity = dot_product / (norm_a * norm_b)

        return similarity
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return np.nan
