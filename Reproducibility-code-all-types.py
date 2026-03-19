import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """
    Reproducibility checklist (what happens first, second, third...):

    1) Seed Python's built-in RNG (random)
       - Affects: random.random(), random.randint(), random.choice(), etc.

    2) Seed NumPy's RNG
       - Affects: np.random.* functions (rand, randint, choice, etc.)

    3) Seed PyTorch CPU RNG
       - Affects: model weight initialisation, torch.rand(), torch.randn(), data shuffles, etc.

    4) If a CUDA GPU is available, seed PyTorch CUDA RNG(s)
       - Affects: random numbers generated on the GPU (and some GPU ops that use RNG).

    5) Configure cuDNN for determinism (GPU backend used by many PyTorch ops)
       - deterministic=True encourages repeatable algorithm choices
       - benchmark=False disables autotuning that can introduce variability

    The typical reproducibility pair

    For experiments/papers:

    deterministic = True
    benchmark = False

    For performance/training at scale (when exact repeatability is not critical):
    
    deterministic = False
    benchmark = True
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#line to be called at the start of thee script, before data generation and model creation
set_seed(42)