from .registry import register_dataset
from .npy_cached import NPYCachedDataset

# SIMULATED DATASETS

@register_dataset("rand_sop")
class SumOfPeaksDataset(NPYCachedDataset):
    def __init__(self, root, seed=42, transform=None):
        super().__init__(root, name="rand_sop", seed=seed, transform=transform)

@register_dataset("rand_sop_823")
class SumOfPeaksDataset(NPYCachedDataset):
    def __init__(self, root, seed=42, transform=None):
        super().__init__(root, name="rand_sop_823", seed=seed, transform=transform)

# REAL DATASETS

@register_dataset("NIST")
class SumOfPeaksDataset(NPYCachedDataset):
    def __init__(self, root, seed=None, transform=None):
        super().__init__(root, name="NIST", seed=None, transform=transform)

@register_dataset("linear") # example, not actual dataset
class LinearDataset(NPYCachedDataset):
    def __init__(self, root, seed=None, transform=None):
        super().__init__(root, name="linear", seed=None, transform=transform)