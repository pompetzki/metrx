import chex
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from typing import Dict, Sequence
from metrix import DistanceMeasures, StatisticalMeasures


def get_samples(rng_key: chex.PRNGKey, **kwargs) -> chex.Array:
    """
    Generate batches of particles sampled from a multivariate normal distribution.

    Parameters
    ----------
    rng_key : chex.PRNGKey
        Random number generator key.
    **kwargs : Dict
        Dictionary of parameters

    Returns
    -------
    chex.Array
        Samples from a specific multivariate normal distribution.
    """
    return jax.random.multivariate_normal(
        key=rng_key,
        mean=kwargs["mean"],
        cov=kwargs["cov"],
        shape=(kwargs["batch_size"],)
    )


def visualize(xy: Sequence, costs: Dict) -> None:
    """
    Visualize the generated data and the results.

    Parameters
    ----------
    xy: Sequence
        Tuple of generated particles.
    costs: Dict
        Dictionary of distance measures.

    Returns
    -------
    None

    """
    # Set backend
    matplotlib.use("TkAgg")

    # Create figure
    fig, ax = plt.subplots(2,1, figsize=(12, 4))
    ax.flatten()

    # Visualize trajectories
    for particles, color, marker in zip(xy, ("blue", "orange"), ("x", "o")):
        ax[0].scatter(particles[..., 0], particles[..., 1], color=color, marker=marker)
    ax[0].set(xlabel="x", ylabel="y", title="Example: Particle-based Data")
    ax[0].grid()

    # Visualize costs
    ax[1].axis("off")
    cell_text = []
    for key, val in costs.items():
        cell_text.append([key, val["mean"], val["std"], val["median"]])
    table = ax[1].table(cellText=cell_text, colLabels=["Distance Measure", "Mean", "Std", "Median"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.show()


def main(**kwargs: Dict) -> None:
    """
    Main function to generate samples and calculate distance measures between two batches of particles.

    Parameters
    ----------
    kwargs: Dict
        Dictionary of configuration parameters.

    Returns
    -------
    None

    """
    rng_key = jax.random.PRNGKey(kwargs["seed"])

    rng_key, rng_key_x, rng_key_y = jax.random.split(rng_key, num=3)
    x = get_samples(rng_key_x, **kwargs["x"])
    y = get_samples(rng_key_y, **kwargs["y"])

    cost_dict = {}
    for _Measure in [DistanceMeasures, StatisticalMeasures]:
        for _name in _Measure.list_all_names():
            _measure = _Measure.create_instance(_name)
            if isinstance(_measure, DistanceMeasures):
                costs = jax.vmap(jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None))(x, y)
            else:
                costs = _measure(x, y)
            cost_dict.update(
                {
                    f"{_name}": {
                        "mean": jnp.mean(costs),
                        "std": jnp.std(costs),
                        "median": jnp.median(costs)
                    }
                }
            )
    visualize((x, y), cost_dict)


if __name__ == "__main__":
    config = {
        "seed": 0,
        "dim": 2,
        "x": {
            "batch_size": 64,
            "mean": jnp.array([0.5, 0.5]),
            "cov": jnp.array([[0.2, 0.01], [0.01, 0.2]]),
        },
        "y": {
            "batch_size": 64,
            "mean": jnp.array([-0.5, -0.5]),
            "cov": jnp.array([[0.2, -0.01], [-0.01, 0.2]]),
        }
    }
    main(**config)
