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

    Notes:
    - This makes runs *as reproducible as practical*, but some GPU operations may still
      be non-deterministic depending on the specific layers/ops used.

    The typical reproducibility pair

    For experiments/papers:

    deterministic = True
    benchmark = False

    For performance/training at scale (when exact repeatability is not critical):
    
    deterministic = False
    benchmark = True
    """

    # 1) Seed Python's built-in random module (used by many simple random utilities)
    random.seed(seed)

    # 2) Seed NumPy's random generator (if you use np.random anywhere)
    np.random.seed(seed)

    # 3) Seed PyTorch random generator for CPU computations
    torch.manual_seed(seed)

    # 4) If running on GPU, also seed CUDA random generators
    if torch.cuda.is_available():
        # Seed the RNG for the currently selected GPU
        torch.cuda.manual_seed(seed)

        # Seed the RNG for all GPUs (important if you ever use multiple GPUs)
        torch.cuda.manual_seed_all(seed)

    # 5) Make cuDNN use deterministic algorithms where possible (more reproducible, can be slower)
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN performance autotuner (autotuner can pick different algorithms run-to-run)
    torch.backends.cudnn.benchmark = False


# Call this once at the very start of your script (before data generation and model creation)
set_seed(42)