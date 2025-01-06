import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from typing import Dict, Sequence
from metrx import DistanceMeasures, StatisticalMeasures
from metrx.statistical_measures import GromovWassersteinDistance


def _generate_trajectory(
    time: jax.Array,
    amplitude: jax.Array,
    frequency: jax.Array,
    offset_x: jax.Array,
    offset_y: jax.Array,
    phase_shift: jax.Array,
    angle: float = 0.0,
) -> jax.Array:
    """
    Generate a trajectory given the parameters.

    Parameters
    ----------
    time: jax.Array
        Time steps as (1, t) array
    amplitude: jax.Array
        Amplitude of the trajectory as (b, 1) array
    frequency: jax.Array
        Frequency of the trajectory as (b, 1) array
    offset_x: jax.Array
        Offset in x-direction as (b, 1) array
    offset_y: jax.Array
        Offset in y-direction as (b, 1) array
    phase_shift: jax.Array
        Phase shift of the trajectory as (b, 1) array
    angle:
        Rotation angle in radians

    Returns
    -------
    jax.Array
        Trajectory as (b, t, 2) array

    """

    def _rotate(array: jax.Array) -> jax.Array:
        rotation_matrix = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        return jnp.dot(array, rotation_matrix)

    tau = jnp.stack(
        (
            (time - jnp.max(time) / 2.0) + offset_x,
            amplitude
            * jnp.sin(
                2 * jnp.pi * frequency * (time - jnp.max(time) / 2.0) + phase_shift
            )
            + offset_y,
        ),
        axis=-1,
    )

    return _rotate(tau)


def get_samples(rng_key: jax.Array, **kwargs) -> jax.Array:
    """
    Generate batches of trajectory samples from a common base distribution.

    Parameters
    ----------
    rng_key : jax.Array
        Random number generator key.
    **kwargs : Dict
        Dictionary of trajectory parameters

    Returns
    -------
    jax.Array
        Samples from a specific trajectory distribution.
    """
    time_steps = jnp.linspace(0.0, 1, kwargs["time_steps"])[jnp.newaxis, :]

    rng_key, *rng_samples = jax.random.split(rng_key, num=6)
    amplitudes = (
        kwargs["amplitude"]
        + jax.random.normal(rng_samples[0], shape=(kwargs["batch_size"], 1))
        * kwargs["sigma"]
    )
    frequencies = (
        kwargs["frequency"]
        + jax.random.normal(rng_samples[1], shape=(kwargs["batch_size"], 1))
        * kwargs["sigma"]
    )
    offsets_x = (
        kwargs["offset_x"]
        + jax.random.normal(rng_samples[2], shape=(kwargs["batch_size"], 1))
        * kwargs["sigma"]
    )
    offsets_y = (
        kwargs["offset_y"]
        + jax.random.normal(rng_samples[3], shape=(kwargs["batch_size"], 1))
        * kwargs["sigma"]
    )
    phase_shift = (
        kwargs["phase_shift"]
        + jax.random.normal(rng_samples[4], shape=(kwargs["batch_size"], 1))
        * kwargs["sigma"]
    )

    return _generate_trajectory(
        time_steps,
        amplitudes,
        frequencies,
        offsets_x,
        offsets_y,
        phase_shift,
        kwargs["rotation"],
    )


def visualize(xy: Sequence, costs: Dict) -> None:
    """
    Visualize the generated data and the results.

    Parameters
    ----------
    xy: Sequence
        Tuple of generated trajectories.
    costs: Dict
        Dictionary of distance measures.

    Returns
    -------
    None

    """
    # Set backend
    matplotlib.use("TkAgg")

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 4))
    ax.flatten()

    # Visualize trajectories
    for taus, color in zip(xy, ("blue", "orange")):
        for sample in taus:
            ax[0].plot(sample[..., 0], sample[..., 1], color=color, alpha=0.5)
    ax[0].set(xlabel="x", ylabel="fy", title="Example: time-series data")
    ax[0].grid()

    # Visualize costs
    ax[1].axis("off")
    cell_text = []
    for key, val in costs.items():
        cell_text.append([key, val["mean"], val["std"], val["median"]])
    table = ax[1].table(
        cellText=cell_text,
        colLabels=["Distance Measure", "Mean", "Std", "Median"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.show()


def main(**kwargs: Dict) -> None:
    """
    Main function to generate samples and calculate distance measures between two batches of trajectories.

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
            try:
                if isinstance(_measure, DistanceMeasures):
                    costs = jax.vmap(
                        jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None)
                    )(x, y)
                else:
                    costs = _measure(x, y)
                cost_dict.update(
                    {
                        f"{_name}": {
                            "mean": jnp.mean(costs),
                            "std": jnp.std(costs),
                            "median": jnp.median(costs),
                        }
                    }
                )
            except AssertionError as e:
                print(f"AssertionError occurred: {e}")

    visualize((x, y), cost_dict)


if __name__ == "__main__":
    config = {
        "seed": 0,
        "dim": 2,
        "x": {
            "batch_size": 32,
            "time_steps": 64,
            "amplitude": 1.0,
            "frequency": 2.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "phase_shift": 0.0,
            "sigma": 0.05,
            "rotation": 0.0,
        },
        "y": {
            "batch_size": 32,
            "time_steps": 64,
            "amplitude": 1.0,
            "frequency": 2.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "phase_shift": 1.0,
            "sigma": 0.05,
            "rotation": 0.0,
        },
    }
    main(**config)
