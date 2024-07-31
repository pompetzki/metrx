# metriX
A library containing a collection of distance and similarity measures to compare time series data.

## Installation & Usage
```bash
conda env create -f environment.yml
conda activate metriX_env
pip install -e .
```
To test, run example:
```bash
python examples/example.py
```

## ToDo's
- [ ] Combine run() and init_state() for the distance measures 
- [ ] Add Wasserstein distance of batch-wise particles
- [ ] Add tests
    