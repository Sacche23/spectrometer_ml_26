#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 <input.csv> <displacements.npy> <wavelengths.npy> <responsivity.npy>"
  exit 1
fi

input_csv="$1"
disp_out="$2"
wave_out="$3"
resp_out="$4"

# ensure output directories exist
mkdir -p "$(dirname "$disp_out")" \
         "$(dirname "$wave_out")" \
         "$(dirname "$resp_out")"

python3 - <<EOF
import numpy as np
import csv

# read just the first line via csv.reader, skip the first empty cell
with open("$input_csv", newline='') as f:
    reader = csv.reader(f)
    header = next(reader)            # e.g. ["", "d1", "d2", ...]
displacements = np.array(header[1:], dtype=float)
np.save("$disp_out", displacements)

# load the rest of the file as floats, skipping the header line
data = np.loadtxt("$input_csv", delimiter=",", skiprows=1)
wavelengths  = data[:, 0]
responsivity = data[:, 1:]

np.save("$wave_out", wavelengths)
np.save("$resp_out", responsivity)
EOF

echo "Saved displacements --> $disp_out"
echo "Saved wavelengths --> $wave_out"
echo "Saved responsivity --> $resp_out"

