import os, sys
import argparse
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


import numpy as np
import torch

# adjust this import if your project layout differs
from src.models.model import get_model  

def main():
    p = argparse.ArgumentParser(
        description="Run cnn2 on one rand_sop sample and dump input/output to .bin files"
    )
    p.add_argument("--data-dir", "-d",
                   default="data/spectra_data/processed/rand_sop",
                   help="Path to processed rand_sop folder containing I_s42.npy")
    p.add_argument("--model-path", "-m", required=True,
                   help="Path to your trained cnn2 checkpoint (.pth)")
    p.add_argument("--index", "-i", type=int, default=0,
                   help="Index into I_s42.npy to use as input")
    p.add_argument("--output-dir", "-o", default="inference_bin",
                   help="Directory to write input.bin and output.bin")
    args = p.parse_args()

    # 1) Load the raw input array
    I = np.load(os.path.join(args.data_dir, "I_s42.npy"))  # shape (N, 41)
    if args.index < 0 or args.index >= len(I):
        raise IndexError(f"Index {args.index} out of range (0..{len(I)-1})")
    x = I[args.index].astype(np.float32)  # shape (41,)

    # 2) Instantiate model and load weights
    ModelCls = get_model("cnn2")
    model = ModelCls(input_dim=41, output_dim=1000)
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # 3) Run inference
    with torch.no_grad():
        inp = torch.from_numpy(x).unsqueeze(0)        # (1, 41)
        out = model(inp).squeeze(0).cpu().numpy()     # (1000,)

    # 4) Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 5) Dump to .bin files
    in_path  = os.path.join(args.output_dir, "input.bin")
    out_path = os.path.join(args.output_dir, "output.bin")
    x.tofile(in_path)
    out.astype(np.float32).tofile(out_path)

    print(f"Wrote {in_path} ({x.size} floats)")
    print(f"Wrote {out_path} ({out.size} floats)")

    # Plot the output
    plt.figure()
    plt.plot(out)
    plt.title(f"Model output for sample {args.index}")
    plt.xlabel("Output Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()