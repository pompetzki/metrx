import functools
import jax
import jax.numpy as jnp

from typing import Sequence


@functools.partial(jax.jit, static_argnames=["reg"])
def fit_gaussian2data(x: jax.Array, reg: float = 1e-5) -> Sequence[jax.Array]:
    """
    Fit a Gaussian model to each time step n given the samples from x as array of shape (b, n, d).

    Parameters
    ----------
    x: chex.Array
        Samples from x as array of shape (b, n, d).
    reg:
        Regularization term to avoid singular covariance matrices.

    Returns
    -------
    mean: chex.Array
        Mean of the Gaussian model as array of shape (n, d).
    tri_lower_matrix: chex.Array
        Lower triangular matrix of the Cholesky decomposition of the covariance matrix as array of shape (n, d, d).
    """
    """ Fit a Gaussian model to each time step given the samples from x """
    b, t, d = x.shape

    mean = jnp.mean(x, axis=0, keepdims=True)

    x_centered = x - mean
    covariance_matrix = jnp.einsum("...btn, ...btm->...tnm", x_centered, x_centered) / (
        b - 1 + 1e-5
    )
    covariance_matrix = covariance_matrix + reg * jnp.eye(d)

    tri_lower_matrix = jax.vmap(jnp.linalg.cholesky)(covariance_matrix)

    return mean[0], tri_lower_matrix
