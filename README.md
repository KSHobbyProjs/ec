# ec
A toolkit for running eigenvector continuation (EC) on eigenvalue data read from a file.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/ec.git
cd ec
pip install -r requirements.txt
```
Dependencies include `numpy`, `h5py`, `scipy`

---

## Usage
Run EC using `python run_ec.py`.  

```bash
python run_pmm.py \
      input_file.h5 \
    --model gaussian.Gaussian1d:N=128,V0=-4.0,R=2.0 \
    --parameters 5.0,20.0:50 \
    --vectors
    --out ec_predictions.h5
```

---

## Input

The program requires an `.h5` input file containing parameters and eigenpair data given as a full HDF5 dataset.

- `.h5` files — full dataset:
  - `parameters` — 1D NumPy array of parameter values (`len(parameters)`)  
  - `energies` — 2D NumPy array of shape (`len(parameters)`, `knum`)  
    (*knum* is the number of eigenpairs per parameter, as set by `--knum`)  
  - `eigenvectors` — 3D NumPy array of shape (`len(parameters)`, `knum`, `vector dimension`)

### Notes on Data Shapes

- Preferred shapes: `parameters` as a 1D array, `energies` as a 2D `(len(parameters), knum)` array, and `eigenvectors` as a 3D `(len(parameters), knum, vector dimension)` array.  
- The program will attempt to **infer the correct structure** if the input does not exactly match this layout.

---

## Output

The program writes results to a file specified by the user.  
- If the output filename ends with `.h5`, the program writes a full HDF5 dataset.  
- Any other file extension will produce a `.dat`-style column-format summary.

- `.h5` files — full dataset
  - `parameters` — 1D NumPy array of parameter values (`len(parameters)`)  
  - `energies` — 2D NumPy array of shape (`len(parameters)`, `knum`) containing eigenvalues  
    (*knum* is the number of eigenpairs per parameter, as set by `--knum`)  
  - `eigenvectors` — 3D NumPy array of shape (`len(parameters)`, `knum`, `vector dimension`) containing eigenvectors

- `.dat` files — summary dataset
  - Column 1: `parameters` — parameter values  
  - Columns 2..(knum+1): `energies` — one column per eigenvalue  
  - Note: `eigenvectors` are **not** included in `.dat` files, and columns separated by `\t` delimiter.

---

## Key Arguments
- `input_file`          : Input file with eigenpair data.
- `--model`             : Name of the physical model to use (will attempt to find this in the metadata of the input file, but `--model` flag takes precendence.)
- `--parameters`        : Parameter values in `start,end:len`, `start,end:len,exp` or `val1,val2,val3` format.
  Example: `5.0,6.0,7.0` or `5.0,20.0:150`.
- `--out`               : Output filename for predicted energies.
- `--vectors`           : Output eigenvectors as well as eigenvalues.
- `--verbose`           : Increase verbosity.

---
