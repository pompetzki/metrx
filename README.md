![metrix_loogv3](https://github.com/user-attachments/assets/51f66576-866f-4a89-a17b-8154bc7a8381)

![continous integration](https://github.com/pompetzki/metrx/actions/workflows/continuous_integration.yml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/metrx)](https://pypi.org/project/metrx/)



A lightweight **JAX**-based library offering a collection of distance and similarity measures for data analysis. Designed for
scalability and accelerator support, it includes high-performance, parallelizable implementations of a wide range of commonly
used metrics.

## Installation 
You have the choice to install the latest release via PyPI by running
```
pip install metrx
```
or you do an editable installation by cloning this repository and then running

```bash
pip install -e .
```

## Implemented Metrics
This library is still in development and more metrics will be added over time.
The following metrics are currently implemented.
### Distance Measures
- [Minkowski Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L171)
- [Euclidean Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L277)
- [Cosine Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L438)
- [Mahalanobis Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L490)
- [Dynamic Time Warping](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L754)
- [Discrete Frechet Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L897)
- [Sinkhorn Distance](https://github.com/pompetzki/metrx/blob/main/metrx/distance_measures.py#L1136)

### Statistical Measures
- [Relative Entropy (Kullback-Leibler Divergence)](https://github.com/pompetzki/metrx/blob/main/metrx/statistical_measures.py#L174)
- [Frechet Inception Distance](https://github.com/pompetzki/metrx/blob/main/metrx/statistical_measures.py#L295)
- [Maximum Mean Discrepancy](https://github.com/pompetzki/metrx/blob/main/metrx/statistical_measures.py#L425)
- [Wassersteim Distance](https://github.com/pompetzki/metrx/blob/main/metrx/statistical_measures.py#L605)
- [Gromov-Wassersteim Distance](https://github.com/pompetzki/metrx/blob/main/metrx/statistical_measures.py#L690)


## Examples
To test, there are two examples:
Either compare batches of particles
```bash
python examples/example_particle_data.py
```
or batches of time series data
```bash
python examples/example_time_series_data.py
```
    
## Citation
If you use this libarary in your work, please consider citing it as follows:
```
@software{metrx2024github,
  author = {Pompetzki, Kay and Gruner, Theo and Al-Hafez, Firas, and Le, An T., and Peters, Jan},
  title = {MetrX: A JAX-Based Collection of Similarity and Statistical Measures for Accelerated Data Analysis.},
  url = {https://github.com/pompetzki/metrx},
  year = {2024},
}
```
