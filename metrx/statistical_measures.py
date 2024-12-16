from abc import ABC, abstractmethod
from typing import Sequence, Optional, Any, Dict, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
from ott.geometry.costs import CostFn

from metrx.distance_measures import DistanceMeasures
from metrx.utils import fit_gaussian2data


@struct.dataclass
class StatisticalMeasures(ABC):
    """
    Base class for all statistical measures. This class is used to create a registry of all statistical measures.
    To create an instance of a distance measure, use the `create_instance` method.

    Parameters
    ----------
    _registry: `dict`, class attribute, default = {}.
        A dictionary to store all the statistical measures.

    Returns
    -------
    `DistanceMeasures`
        An instance of the base class for all statistical measures.
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs: Dict) -> None:
        """
        Register all the statistical measures in the registry.

        Parameters
        ----------
        kwargs: `dict`
            A dictionary of keyword arguments.

        Returns
        -------
        'None'
        """
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def create_instance(
        cls, name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> "StatisticalMeasures":
        """
        Create an instance of the statistical measure using the name of the statistical measure.

        Parameters
        ----------
        name: `str`, default = None
            The name of the statistical measure.
        args: `Any`
            Arguments passed to the constructor of the statistical measure.
        kwargs: `Any`
            Keyword arguments passed to the constructor of the statistical measure.

        Returns
        -------
        `StatisticalMeasures`
            An instance of the statistical measure.

        """
        if name in cls._registry:
            return cls._registry[name].create(*args, **kwargs)
        else:
            registered = ", ".join([key for key in cls._registry.keys()])
            raise ValueError(
                f"Unknown class name: {name}. Registered measures: {registered}"
            )

    @classmethod
    @abstractmethod
    def construct(cls, *args, **kwargs) -> "StatisticalMeasures":
        """Create an Instance of the statistical distance measure"""

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "StatisticalMeasures":
        """
        Create an instance of the the statistical distance measure.

        Parameters
        ----------
        args: `Any`
            Arguments passed to the constructor of the statistical distance measure.
        kwargs: `Any`
            Keyword arguments passed to the constructor of the statistical distance measure.

        Returns
        -------
        `Statistical distance`
            An instance of the relative entropy divergence measure.
        """
        return cls.construct(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
        """
        Call the statistical measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the run method.
        kwargs: `Any`
            The keyword arguments to pass to the run method.

        Returns
        -------
        `jax.Array`
            The estimated statistical measure.
        """
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> jax.Array:
        """
        Estimate the statistical measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the method.
        kwargs: `Any`
            The keyword arguments to pass to the method.

        Returns
        -------
        `jax.Array`
            The estimated statistical measure.
        """
        raise NotImplementedError

    @classmethod
    def list_all_names(cls):
        """
        Get the names of the registered distance measures.

        Returns
        -------
        `List[str]`
            The names of the registered distance measures.
        """
        return list(cls._registry.keys())

    @classmethod
    def list_all(cls):
        """
        Get the classes of all the registered distance measures.

        Returns
        -------
        `List[str]`
            The classes of the registered distance measures.

        """
        return list(cls._registry.values())


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Relative Entropy Divergence -------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class RelativeEntropy(StatisticalMeasures):
    """
    Compute the relative entropy divergence between two time series data both represented as Gaussian distributions
    for each time step n. The relative entropy divergence is computed as the Kullback-Leibler divergence between the
    two Gaussian distributions for each time step n. Thus, the relative entropy divergence is a measure of time series
    data with equal length.

    Parameters
    ----------
    reverse: `bool`, default = False
        A boolean flag to compute the reverse KL divergence.
    reg: `float`, default = 1e-5
        A float value to control the regularization term to avoid singular covariance matrices.
    mean: `bool`, default = False
        A boolean flag to compute the mean of the relative entropy divergence over all time steps.
    median: `bool`, default = False
        A boolean flag to compute the median of the relative entropy divergence over all time steps.
    total_sum: `bool`, default = False
        A boolean flag to compute the total sum of the relative entropy divergence over all time steps.

    Returns
    -------
    `RelativeEntropy`
        An instance of the relative entropy divergence measure.
    """

    reverse: Optional[bool] = struct.field(default=None, pytree_node=False)
    alpha: Optional[float] = struct.field(default=None, pytree_node=False)
    reg: Optional[float] = struct.field(default=None, pytree_node=False)
    mean: Optional[bool] = struct.field(default=None, pytree_node=False)
    median: Optional[bool] = struct.field(default=None, pytree_node=False)
    total_sum: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(
        cls,
        reverse: bool = False,
        reg: float = 1e-5,
        mean: bool = False,
        median: bool = False,
        total_sum: bool = False,
    ) -> "RelativeEntropy":
        """
        Construct an instance of the relative entropy divergence measure.

        Parameters
        ----------
        reverse: `bool`, default = False
            A boolean flag to compute the reverse KL divergence.
        reg: `float`, default = 1e-5
            A float value to control the regularization term to avoid singular covariance matrices.
        mean: `bool`, default = False
            A boolean flag to compute the mean of the relative entropy divergence over all time steps.
        median: `bool`, default = False
            A boolean flag to compute the median of the relative entropy divergence over all time steps.
        total_sum: `bool`, default = True
            A boolean flag to compute the total sum of the relative entropy divergence over all time steps.

        Returns
        -------
        `RelativeEntropy`
            An instance of the relative entropy divergence measure.
        """
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(
            reverse=reverse, reg=reg, mean=mean, median=median, total_sum=total_sum
        )

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Run the relative entropy divergence measure.

        Parameters
        ----------
        x: `jax.Array`
            Empirical data of shape (b_x, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n, d).
        y: `jax.Array`
            Empirical data of shape (b_y, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n, d).

        Returns
        -------
        jax.Array
            The relative entropy divergence measure of shape (1, ).
        """
        assert (
            x.shape[1:] == y.shape[1:]
        ), f"The objects inside both batches need to have the same shape. Got {x.shape[1:]} and {y.shape[1:]}."
        assert 3 >= x.ndim >= 1 and 3 >= y.ndim >= 1, (
            f"The two batches need to be of shape (b, d, ) if particles, or (b, n, d) if time series data. "
            f"Got x = {x.shape} and y = {y.shape}."
        )
        if x.ndim == 2:
            x = x[..., jnp.newaxis, :]
        if y.ndim == 2:
            y = y[..., jnp.newaxis, :]

        mean_x, tri_lower_x = fit_gaussian2data(x, self.reg)
        mean_y, tri_lower_y = fit_gaussian2data(y, self.reg)

        mvnrml_x = distrax.MultivariateNormalTri(loc=mean_x, scale_tri=tri_lower_x)
        mvnrml_y = distrax.MultivariateNormalTri(loc=mean_y, scale_tri=tri_lower_y)

        if self.reverse:
            relative_entropy = mvnrml_y.kl_divergence(mvnrml_x)[jnp.newaxis, ...]
        else:
            relative_entropy = mvnrml_x.kl_divergence(mvnrml_y)[jnp.newaxis, ...]

        if self.mean:
            return jnp.mean(relative_entropy, axis=-1)
        elif self.median:
            return jnp.median(relative_entropy, axis=-1)
        return jnp.sum(relative_entropy, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Frechet Inception Distance --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class FrechetInceptionDistance(StatisticalMeasures):
    """
    Compute the Frechet Inception Distance (FID) between two time series data both represented as Gaussian distributions
    for each time step n. The Frechet Inception Distance is a measure of the similarity between two time series data
    distributions. The FID corresponds to the 2-Wasserstein distance between the two Gaussian distributions for each
    time step n.

    Parameters
    ----------
    alpha: `float`, default = 1.0
        A float value to control the proportion of the Covariance matrices. If 0.0, the resulting distance is equivalent
        to the the Euclidean distance between the means of the two Gaussian distributions. If 1.0, the resulting
        distance corresponds to the FID.
    reg: `float`, default = 1e-5
        A float value to control the regularization term to avoid singular covariance matrices.
    mean: `bool`, default = False
        A boolean flag to compute the mean of the FID over all time steps.
    median: `bool`, default = False
        A boolean flag to compute the median of the FID over all time steps.
    total_sum: `bool`, default = True
        A boolean flag to compute the total sum of the FID over all time steps.

    """

    alpha: Optional[float] = struct.field(default=None, pytree_node=False)
    reg: Optional[float] = struct.field(default=None, pytree_node=False)
    mean: Optional[bool] = struct.field(default=None, pytree_node=False)
    median: Optional[bool] = struct.field(default=None, pytree_node=False)
    total_sum: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(
        cls,
        alpha: float = 1.0,
        reg: float = 1e-5,
        mean: bool = False,
        median: bool = False,
        total_sum: bool = False,
    ) -> "FrechetInceptionDistance":
        """
        Construct an instance of the Frechet Inception Distance measure.

        Parameters
        ----------
        alpha: `float`, default = 1.0
            A float value to control the proportion of the Covariance matrices. If 0.0, the resulting distance is
            equivalent to the the Euclidean distance between the means of the two Gaussian distributions. If 1.0, the
            resulting distance corresponds to the FID.
        reg: `float`, default = 1e-5
            A float value to control the regularization term to avoid singular covariance matrices.
        mean: `bool`, default = False
            A boolean flag to compute the mean of the FID over all time steps.
        median: `bool`, default = False
            A boolean flag to compute the median of the FID over all time steps.
        total_sum: `bool`, default = True
            A boolean flag to compute the total sum of the FID over all time steps.

        Returns
        -------
        `FrechetInceptionDistance`
            An instance of the Frechet Inception Distance measure.
        """
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(alpha=alpha, reg=reg, mean=mean, median=median, total_sum=total_sum)

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Run the total Frechet Inception Distance measure.

        Parameters
        ----------
        x: `jax.Array`
            Empirical data of shape (b_x, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n, d).
        y: `jax.Array`
            Empirical data of shape (b_y, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n, d).

        Returns
        -------
        `jax.Array`
            The Frechet Inception Distance measure of shape (1, ).
        """
        assert (
            x.shape[1:] == y.shape[1:]
        ), f"The objects inside both batches need to have the same shape. Got {x.shape[1:]} and {y.shape[1:]}."
        assert 3 >= x.ndim >= 1 and 3 >= y.ndim >= 1, (
            f"The two batches need to be of shape (b, d, ) if particles, or (b, n, d) if time series data. "
            f"Got x = {x.shape} and y = {y.shape}."
        )
        if x.ndim == 2:
            x = x[..., jnp.newaxis, :]
        if y.ndim == 2:
            y = y[..., jnp.newaxis, :]

        mean_x, tri_lower_x = fit_gaussian2data(x, self.reg)
        mean_y, tri_lower_y = fit_gaussian2data(y, self.reg)

        delta_mean = mean_x - mean_y
        mean_diff = jnp.linalg.norm(delta_mean, axis=-1) ** 2

        cov_x = jnp.einsum("...mi, ...ni -> ...mn", tri_lower_x, tri_lower_x)
        trace_cov_x = jax.vmap(jnp.trace)(cov_x)

        cov_y = jnp.einsum("...mi, ...ni -> ...mn", tri_lower_y, tri_lower_y)
        trace_cov_y = jax.vmap(jnp.trace)(cov_y)

        cov_product = jnp.einsum("...mi, ...ni -> ...mn", cov_x, cov_y)
        eig_vals, eig_vecs = jax.vmap(jnp.linalg.eigh)(cov_product)
        sqrt_cov_product = jnp.einsum(
            "...mj,...j, ...nj->...mn", eig_vecs, jnp.sqrt(eig_vals), eig_vecs
        )
        trace_sqrt_product = jax.vmap(jnp.trace)(sqrt_cov_product)

        distances = mean_diff + self.alpha * (
            trace_cov_x + trace_cov_y - 2 * trace_sqrt_product
        )

        if self.mean:
            return jnp.mean(distances, axis=-1)
        elif self.median:
            return jnp.median(distances, axis=-1)
        return jnp.sum(distances, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Maximum Mean Discrepancy ---------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class MaximumMeanDiscrepancy(StatisticalMeasures):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two time series data. Depending on the selected distance
    the two time series have to be of the same length (N_x = N_y), .e.g., the squared Euclidean distance. However,
    using dynamic time warping as distance measure, the two time series can have different lengths (N_x != N_y).

    Parameters
    ----------
    distance: `DistanceMeasures`, default = SquaredEuclideanDistance
        A distance measure to compute the distance between the two time series data.
    bandwidths: `Sequence[float]`, default = [1.0, 10.0, 20.0, 40.0, 80.0, 100.0, 130.0, 200.0, 400.0, 800.0, 1000.0]
        A sequence of bandwidths to compute the MMD.
    unbiased: `bool`, default = True
        A boolean flag to compute the unbiased MMD.

    Returns
    -------
    `MaximumMeanDiscrepancy`
        An instance of the Maximum Mean Discrepancy measure
    """

    distance: DistanceMeasures = struct.field(pytree_node=False)

    bandwidths: Tuple = struct.field(pytree_node=False)
    unbiased: Optional[bool] = struct.field(default=True, pytree_node=False)

    @classmethod
    def construct(
        cls,
        distance_measure: Union[DistanceMeasures, str] = "EuclideanDistance",
        distance_kwargs: Dict = {},
        bandwidths: Optional[Union[Sequence[float], Tuple]] = None,
        unbiased: bool = True,
    ) -> "MaximumMeanDiscrepancy":
        """
        Construct an instance of the Maximum Mean Discrepancy measure.

        Parameters
        ----------
        distance: `DistanceMeasures`, default = None
            A distance measure to compute the distance between the two time series data. If None, the Squared Euclidean
            distance is used.
        bandwidths: `Sequence[float]`, default = None
            A sequence of bandwidths to compute the MMD. If None, the default bandwidths are used.
        unbiased
            A boolean flag to compute the unbiased MMD. If None, the unbiased MMD is computed.

        Returns
        -------
        `MaximumMeanDiscrepancy`
            An instance of the Maximum Mean Discrepancy measure
        """
        if isinstance(distance_measure, str):
            distance_measure = DistanceMeasures.create_instance(
                distance_measure, **distance_kwargs
            )

        if bandwidths is None:
            bandwidths = (
                1.0,
                10.0,
                20.0,
                40.0,
                80.0,
                100.0,
                130.0,
                200.0,
                400.0,
                800.0,
                1000.0,
            )

        elif isinstance(bandwidths, Sequence):
            bandwidths = tuple(bandwidths)

        return cls(distance=distance_measure, bandwidths=bandwidths, unbiased=unbiased)

    def _mmd_kernel(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Compute the Maximum Mean Discrepancy kernel between two time series data. Depending on the selected distance
        the two time series have to be of the same length (N_x = N_y), .e.g., the squared Euclidean distance. However,
        using dynamic time warping as distance measure, the two time series can have different lengths (N_x != N_y).

        Parameters
        ----------
        x: `jax.Array`
            Empirical data of shape (B_x, N_x, D).
        y: `jax.Array`
            Empirical data of shape (B_y, N_y, D).

        Returns
        -------
        `jax.Array`
            The Maximum Mean Discrepancy kernel of shape (B_x, B_y).
        """
        distance_matrix = jax.vmap(
            jax.vmap(self.distance, in_axes=(None, 0)), in_axes=(0, None)
        )(x, y)

        def rbf_kernel(kernelized_dist: jax.Array, bandwidth: float) -> Any:
            return kernelized_dist + jnp.exp(-0.5 / bandwidth * distance_matrix), None

        kernelized_distance_matrix = jnp.zeros((x.shape[0], y.shape[0]))
        kernelized_distance_matrix, _ = jax.lax.scan(
            rbf_kernel, kernelized_distance_matrix, jnp.array(self.bandwidths)
        )
        return kernelized_distance_matrix

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Run the Maximum Mean Discrepancy measure.

        Parameters
        ----------
        x: `jax.Array`
            Empirical data of shape (b_x, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n_x, d).
        y: `jax.Array`
            Empirical data of shape (b_y, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n_y, d).

        Returns
        -------
        `jax.Array`
            The Maximum Mean Discrepancy measure of shape ().
        """
        assert x.shape[-1] == y.shape[-1], (
            f"The objects inside both batches need to have the same dimensionality. "
            f"Got {x.shape[1:]} and {y.shape[1:]}."
        )
        assert 3 >= x.ndim >= 1 and 3 >= y.ndim >= 1, (
            f"The two batches need to be of shape (b, d, ) if particles, or (b, n, d) if time series data. "
            f"Got x = {x.shape} and y = {y.shape}."
        )
        if x.ndim == 2:
            x = x[..., jnp.newaxis, :]
        if y.ndim == 2:
            y = y[..., jnp.newaxis, :]

        kxx = self._mmd_kernel(x, x)
        i, j = jnp.diag_indices(kxx.shape[-1])
        kxx = kxx.at[..., i, j].set(0.0)

        kyy = self._mmd_kernel(y, y)
        i, j = jnp.diag_indices(kyy.shape[-1])
        kyy = kyy.at[..., i, j].set(0.0)

        kxy = self._mmd_kernel(x, y)

        b_x, b_y = x.shape[0], y.shape[0]
        c_xy = 2.0 / (b_x * b_y)
        if self.unbiased:
            c_xx = 1 / (b_x * (b_x - 1)) if b_x > 1 else 1 / b_x
            c_yy = 1 / (b_y * (b_y - 1)) if b_y > 1 else 1 / b_y
        else:
            c_xx = 1 / b_x
            c_yy = 1 / b_y

        return c_xx * jnp.sum(kxx) - c_xy * jnp.sum(kxy) + c_yy * jnp.sum(kyy)


# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- 1-Wasserstein distance ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


@struct.dataclass
class OTTCostWrapper(CostFn):
    distance_measure: DistanceMeasures = struct.field(pytree_node=False)

    @classmethod
    def construct(cls, distance_measure: DistanceMeasures):
        return cls(distance_measure)

    def __call__(self, x: jax.Array, y: jax.Array) -> float:
        distance = self.distance_measure(x, y)
        return distance.squeeze()


@struct.dataclass
class WassersteinDistance(StatisticalMeasures):
    """
    Approximation of the 1-Wasserstein distance using the Sinkhorn iterations.
    Depending on the selected distance the two time series have to be of the same length (N_x = N_y), .e.g., the squared Euclidean distance. However,
    using dynamic time warping as distance measure, the two time series can have different lengths (N_x != N_y).

    Parameters
    ----------
    distance: `DistanceMeasures`, default = SquaredEuclideanDistance
        A distance measure to compute the distance between the two time series data.
    distance_kwargs: `Dict`, containing the configurations to intitialize the `DistanceMeasures`, if `DistanceMeasurs` is a str.
    epsilon: `float`, default = None
        The regularization parameter for the regularized optimal transport problem
    return_regularized_cost: `bool`, optional, default = None.
        If True, the regularized cost is returned. If False, the unregularized cost is returned.

    Returns
    -------
    `WassersteinDistance`
        An instance of the Wasserstein distance measure
    """

    cost_fn: OTTCostWrapper = struct.field(pytree_node=False)
    epsilon: Optional[float] = struct.field(default=None, pytree_node=False)
    return_regularized_cost: bool = struct.field(default=False, pytree_node=False)

    @classmethod
    def construct(
        cls,
        distance_measure: Union[DistanceMeasures, str] = "EuclideanDistance",
        distance_kwargs: Dict = {},
        epsilon: Optional[float] = None,
        return_regularized_cost: bool = False,
    ) -> StatisticalMeasures:

        if isinstance(distance_measure, str):
            distance_measure = DistanceMeasures.create_instance(
                distance_measure, **distance_kwargs
            )

        # Build cost function wrapper for the ott-jax library
        cost_fn = OTTCostWrapper.construct(distance_measure)

        return cls(cost_fn, epsilon, return_regularized_cost)

    @property
    def distance_measure(self):
        return self.cost_fn.distance_measure

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Run the 1-Wasserstein distance.

        Parameters
        ----------
        x: `jax.Array`
            Empirical data of shape (b_x, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n_x, d).
        y: `jax.Array`
            Empirical data of shape (b_y, d) if particles are considered. If time series data is considered, the shape
            is (b_x, n_y, d).

        Returns
        -------
        `jax.Array`
            The Maximum Mean Discrepancy measure of shape ().
        """
        geom = pointcloud.PointCloud(x, y, cost_fn=self.cost_fn, epsilon=self.epsilon)
        ot_prob = linear_problem.LinearProblem(geom)
        solver = sinkhorn.Sinkhorn()
        out = solver(ot_prob)

        if self.return_regularized_cost:
            return out.reg_ot_cost

        return jnp.sum(out.matrix * out.geom.cost_matrix)
