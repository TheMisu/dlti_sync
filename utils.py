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
    float: cosine similarity score (0 on error)
    """
    try:
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        if isinstance(b, torch.Tensor):
            b = b.numpy()

        a = a.flatten()
        b = b.flatten()

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0
