# metriX
A library containing a collection of distance and similarity measures to compare time series data.

## Installation & Usage
```bash
conda env create -f environment.yml
conda activate metriX_env
pip install -e .
```
To test, there are two examples:
Either compare batches of particles
```bash
python examples/example_particle_data.py
```
or batches of time series data
```bash
python examples/example_time_series_data.py
```

## ToDo's
- [x] Combine run() and init_state() for the distance measures 
- [x] Add examples for time-series data and for particles
- [ ] Add 1-Wasserstein distance and Sliced Wasserstein distance as statistical measures. (Theo)
- [ ] Add tests (Firas)
    