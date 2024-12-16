from typing import Dict
from pathlib import Path

import pytest
import jax
import numpy as np
import jax.numpy as jnp

import metrx
from metrx import DistanceMeasures, StatisticalMeasures


CONFIG = {
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
TEST_DIR_PATH = Path(metrx.__file__).parent.parent / "tests"


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


@pytest.mark.parametrize(
    "dist_type,name",
    [
        (dist_type, name)
        for dist_type, name_list in [
            (DistanceMeasures, DistanceMeasures.list_all_names()),
            (StatisticalMeasures, StatisticalMeasures.list_all_names()),
        ]
        for name in name_list
    ],
)
def test_distances(dist_type: DistanceMeasures | StatisticalMeasures, name: str):

    # set Jax-backend to CPU
    jax.config.update("jax_platform_name", "cpu")
    print(f"Jax backend device: {jax.default_backend()} \n")

    rng_key = jax.random.PRNGKey(CONFIG["seed"])

    rng_key, rng_key_x, rng_key_y = jax.random.split(rng_key, num=3)
    x = get_samples(rng_key_x, **CONFIG["x"])
    y = get_samples(rng_key_y, **CONFIG["y"])
    x_1D, y_1D = jnp.squeeze(x[:, 1, 1]), jnp.squeeze(y[:, 1, 1])

    _measure = dist_type.create_instance(name)
    inputs = (x_1D, y_1D) if name == "CosineDistance" else (x, y)

    if isinstance(_measure, DistanceMeasures):
        costs = jax.vmap(jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None))(
            *inputs
        )
        costs_jitted = jax.jit(
            jax.vmap(jax.vmap(_measure, in_axes=(None, 0)), in_axes=(0, None))
        )(*inputs)
    else:
        costs = _measure(*inputs)
        costs_jitted = jax.jit(_measure)(*inputs)

    data = dict(mean=np.mean(costs), std=np.std(costs), median=np.median(costs))
    data_jitted = dict(
        mean=np.mean(costs_jitted),
        std=np.std(costs_jitted),
        median=np.median(costs_jitted),
    )

    # load the results
    loaded = np.load(TEST_DIR_PATH / f"test_datasets/{name}.npz")

    # assert close non-jitted
    assert np.allclose(data["mean"], loaded["mean"]), f"{name} failed: Mean not close"
    assert np.allclose(data["std"], loaded["std"]), f"{name} failed: Std not close"
    assert np.allclose(
        data["median"], loaded["median"]
    ), f"{name} failed: Median not close"

    if name != "WassersteinDistance":
        # todo: find a way to verify closeness for WassersteinDistance as well.
        # assert close jitted
        assert np.allclose(
            data_jitted["mean"], loaded["mean"]
        ), f"{name} failed: Mean not close"
        assert np.allclose(
            data_jitted["std"], loaded["std"]
        ), f"{name} failed: Std not close"
        assert np.allclose(
            data_jitted["median"], loaded["median"]
        ), f"{name} failed: Median not"
