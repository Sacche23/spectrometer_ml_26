import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYCachedDataset(Dataset):
    '''
    Creates I.npy and S.npy pairs under
    {root}/spectra_data/processed/{name}/I[_seed{seed}].npy
    {root}/spectra_data/processed/{name}/S[_seed{seed}].npy
    '''
    def __init__(self,
                 root: str,
                 name: str,
                 seed: int = None,
                 transform = None):
        proc = os.path.join(root, "processed", name)
        suffix = f"_seed{seed}" if seed is not None else ""
        fi = os.path.join(proc, f"I{suffix}.npy")
        fs = os.path.join(proc, f"S{suffix}.npy")
        if not (os.path.exists(fi) and os.path.exists(fs)):
            raise FileNotFoundError(f"Missing cache files: {fi}, {fs}")
        
        I = np.load(fi, mmap_mode="r")
        S = np.load(fs, mmap_mode="r")
        self.I = torch.from_numpy(I)
        self.S = torch.from_numpy(S)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        i, s = self.I[idx], self.S[idx]
        return (self.transform(i), s) if self.transform else (i, s)