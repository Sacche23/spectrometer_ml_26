import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset

class NPYCachedDataset(Dataset):
    '''
    Creates I.npy and S.npy pairs under
    {root}/spectra_data/processed/{name}/I[_s{seed}].npy
    {root}/spectra_data/processed/{name}/S[_s{seed}].npy
    '''
    def __init__(self,
                 root: str,
                 name: str,
                 seed: int = None,
                 transform = None):
        proc = os.path.join(root, "processed", name)

        fi = os.path.join(proc, f"I.npy")
        fs = os.path.join(proc, f"S.npy")
        if not (os.path.exists(fi) and os.path.exists(fs)):
            raise FileNotFoundError(
                f"Dataset '{name}' is incomplete.\n"
                f"Expected:\n"
                f"    {fi}\n"
                f"    {fs}\n"
                f"Have you run generate_spectra.py?"
            )
        
        # ----------------------------
        # Load dataset metadata
        # ----------------------------
        meta = os.path.join(proc, "dataset.json")

        if os.path.exists(meta):
            with open(meta, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # ----------------------------
        # Load arrays
        # ----------------------------    
        I = np.load(fi, mmap_mode="r").astype(np.float32)
        S = np.load(fs, mmap_mode="r").astype(np.float32)
        self.I = torch.from_numpy(I)
        self.S = torch.from_numpy(S)
        self.transform = transform

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        i, s = self.I[idx], self.S[idx]
        return (self.transform(i), s) if self.transform else (i, s)
