"""
Utilidades para asegurar reproducibilidad completa en experimentos ML.
Fija semillas aleatorias para numpy, torch, sklearn y configura determinismo en CUDA.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fija todas las semillas aleatorias para reproducibilidad completa.
    
    Args:
        seed: Semilla aleatoria a usar (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA determinístico (puede reducir performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Semillas fijadas (seed={seed}) para reproducibilidad completa")
    print(f"  - Python random, NumPy, PyTorch, CUDA")
    print(f"  - CuDNN determinístico activado")
