from typing import Dict
from pathlib import Path

import chex
import jax
import numpy as np
import jax.numpy as jnp

import metriX
from metriX import DistanceMeasures, StatisticalMeasures


def _generate_trajectory(
    time: chex.Array,
    amplitude: chex.Array,
    frequency: chex.Array,
    offset_x: chex.Array,
    offset_y: chex.Array,
    phase_shift: chex.Array,
    angle: float = 0.0,
) -> chex.Array:
    """
    Generate a trajectory given the parameters.

    Parameters
    ----------
    time: chex.Array
        Time steps as (1, t) array
    amplitude: chex.Array
        Amplitude of the trajectory as (b, 1) array
    frequency: chex.Array
        Frequency of the trajectory as (b, 1) array
    offset_x: chex.Array
        Offset in x-direction as (b, 1) array
    offset_y: chex.Array
        Offset in y-direction as (b, 1) array
    phase_shift: chex.Array
        Phase shift of the trajectory as (b, 1) array
    angle:
        Rotation angle in radians

    Returns
    -------
    chex.Array
        Trajectory as (b, t, 2) array

    """

    def _rotate(array: chex.Array) -> chex.Array:
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


def get_samples(rng_key: chex.PRNGKey, **kwargs) -> chex.Array:
    """
    Generate batches of trajectory samples from a common base distribution.

    Parameters
    ----------
    rng_key : chex.PRNGKey
        Random number generator key.
    **kwargs : Dict
        Dictionary of trajectory parameters

    Returns
    -------
    chex.Array
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


def test_trajectories() -> None:
    """
    Test all measures on trajectory data.

    """

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

    # set Jax-backend to CPU
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Jax backend device: {jax.default_backend()} \n")

    test_dir_path = Path(metriX.__file__).parent.parent / "tests"

    rng_key = jax.random.PRNGKey(config["seed"])

    rng_key, rng_key_x, rng_key_y = jax.random.split(rng_key, num=3)
    x = get_samples(rng_key_x, **config["x"])
    y = get_samples(rng_key_y, **config["y"])
    x_1D, y_1D = jnp.squeeze(x[:, 1, 1]), jnp.squeeze(y[:, 1, 1])

    for _Measure in [DistanceMeasures, StatisticalMeasures]:
        for _name in _Measure.list_all_names():
            _measure = _Measure.create_instance(_name)
            inputs = (x_1D, y_1D) if _name == "CosineDistance" else (x, y)
            if isinstance(_measure, DistanceMeasures):
                costs = jax.vmap(
                    jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None)
                )(*inputs)
                costs_jitted = jax.jit(jax.vmap(
                    jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None)
                ))(*inputs)
            else:
                if _name != "MaximumMeanDiscrepancy":
                    costs = _measure(*inputs)
                    costs_jitted = jax.jit(_measure)(*inputs)
                else:
                    costs = _measure(*inputs)
                    costs_jitted = costs    # todo: MMD not yet jitable!

            data = dict(mean=np.mean(costs), std=np.std(costs), median=np.median(costs))
            data_jitted = dict(mean=np.mean(costs_jitted), std=np.std(costs_jitted), median=np.median(costs_jitted))

            # save the results (can be used to update the test datasets
            # np.savez(test_dir_path / f'test_datasets/{_name}.npz', **data)

            # load the results
            loaded = np.load(test_dir_path / f'test_datasets/{_name}.npz')

            # assert close non-jitted
            assert np.allclose(data["mean"], loaded["mean"]), f"{_name} failed: Mean not close"
            assert np.allclose(data["std"], loaded["std"]), f"{_name} failed: Std not close"
            assert np.allclose(data["median"], loaded["median"]), f"{_name} failed: Median not close"

            # assert close jitted
            assert np.allclose(data_jitted["mean"], loaded["mean"]), f"{_name} failed: Mean not close"
            assert np.allclose(data_jitted["std"], loaded["std"]), f"{_name} failed: Std not close"
            assert np.allclose(data_jitted["median"], loaded["median"]), f"{_name} failed: Median not"
