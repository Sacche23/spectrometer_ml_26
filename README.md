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

##