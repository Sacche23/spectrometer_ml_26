# Machine Learning for Black Phosphorus Spectrometer
Training repository for CNN used for spectrometer inference for [wavelength-scale black phosphorus spectrometer](https://www.nature.com/articles/s41566-021-00787-x). Includes training and evaluation scripts, as well as scripts for computing inference from Tikhonov and Lasso regularized regression models in original paper. Can be used for any spectrometer which measures a vector of datapoints (for example photocurrent at N different displacements) as a method of inference.

## Theory



## Network Architecture

## Installation
1. Clone the repo
```bash
git clone https://github.com/lwylonis/spectrometer_ml.git
cd spectrometer_ml
```
2. Create a virtual-env and activate
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Install Jupyter (if you plan to run notebooks)
```
pip install jupyterlab
```

## Usage

1. **Process responsivity data**
```bash
bash scripts/csv_to_npy.sh \
    data/responsivity_data/raw/PhotoResponseSheet1.csv \
    data/responsivity_data/processed/displacements.npy \
    data/responsivity_data/processed/wavelengths.npy \
    data/responsivity_data/processed/responsivity.npy
```
Outputs .npy array for responsivity matrix, as well as wavelength and
displacement values.

2. **Generate spectra**
```bash
bash scripts/generate_spectra.sh
```
NOTE: edit generate_spectra.sh to contain choice of dataset, seed, sample size, etc.
Will either generate random spectra + outputs as .npy files, or pull data from
specified dataset.

3. **Train model**
```bash
bash scripts/train.sh
```
Again, you may change the hyperparameters in train.sh

4. **Evaluate model**
```bash
bash scripts/evaluate.sh
```
NOTE: You MUST keep same parameters for seed and validation size as training, otherwise
the validation split will be different from training.

5. **Compare model to Lasso and Tikhonov regularized regression**
```bash
bash scripts/compare_experiment
```
For this, also keep same parameters for seed and validation size as training! Additionally,
make sure to change alpha parameter to optimize Lasso/Tikhonov for your dataset. This will
print a summary of the mean squared error of the different models vs input over validation
dataset, as well as plotting images of the predicted spectra vs. ground truth.

## Customization

### Adding custom randomly simulated spectra method:

1. **Open `scripts/generate_spectra.py`**
2. **Add your method name to `VALID_METHODS = {"rand_sop", "uniform", "<your_method>"}`**
3. **Add your method to `generate_S`**
```python
    def generate_S(rng: np.random.Generator, n_samples: int, s_dim : int, method : str):
        if method == "uniform":
            return rng.uniform(size=(n_samples, s_dim)).astype(np.float32)
        elif method == "rand_sop":
            return sum_of_peaks(rng, n_samples, s_dim)
        elif method == "<your_method>:
            return <your_function>(rng, n_samples, s_dim)
```
4. **Implement your function under `Custom Generation Functions` section**
5. **Create a dataset in `src/datasets/datasets.py` under `SIMULATED DATASETS`**
```python
@register_dataset("<your_method>")
class SumOfPeaksDataset(NPYCachedDataset):
    def __init__(self, root, seed=42, transform=None):
        super().__init__(root, name="<your_method>", seed=seed, transform=transform)
```

Now you can run `python3 scripts/generate_random_spectra.py --method <your_method>` and
then train the model on this dataset!

### Adding custom dataset:

1. **Follow steps 1-3 for adding randomly simulated spectra method**
2. **Implement your function under `Custom Data Sampling Functions` section**
3. **Add your method to list of real functions (path is generated w/o random seed)**
```python
if method in {"custom1", }: # Add every real dataset here
    np.save(out_dir / f"S.npy", S)
    np.save(out_dir / f"I.npy", I)
```

Similarly to above, simply run same script to produce training data.

### Adding custom deep learning network architecture:

1. **Implent your class in `model.py` under `Model Classes` section**
2. **Follow below structure**
```python
@register_model("<model_name>")
class <YourModel>(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO: Instantiate layers
    def __init__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return model(x)
        return x
```

Now you can pass in `<model_name>` as your model parameter in your training script
to train your custom architecture.