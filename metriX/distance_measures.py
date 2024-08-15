import chex
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from flax import struct
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
from ott.geometry.costs import CostFn
from typing import Sequence, Optional, Any, Dict


@struct.dataclass
class DistanceMeasures(ABC):
    """
    Base class for all distance measures. The distance measures are used to calculate the similarity between two time
    series data. The base class provides a registry to store all the implemented distance measures.
    To create an instance of a distance measure, use the `create_instance` method.

    Parameters
    ----------
    _registry: `dict`, class attribute, default = {}.
        A dictionary to store all the distance measures.

    Returns
    -------
    `DistanceMeasures`
        An instance of the base class for all distance measures.
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs: Dict) -> None:
        """
        Initialize the subclass of the DistanceMeasures class.

        Parameters
        ----------
        kwargs: `Dict`
            The keyword arguments to pass to the superclass.

        Returns
        -------
        `None`
        """
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def create_instance(cls, name: Optional[str] = None, *args: Sequence, **kwargs: Dict) -> "DistanceMeasures":
        """
        Create an instance of the distance measure.

        Parameters
        ----------
        name: `str`, optional, default = None.
            The name of the distance measure to create.
        args: `Sequence`
            The arguments to pass to the constructor of the distance measure.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor of the distance measure.

        Returns
        -------
        `DistanceMeasures`
            An instance of the distance measure.
        """
        if name in cls._registry:
            return cls._registry[name].create(*args, **kwargs)
        else:
            registered = ", ".join([key for key in cls._registry.keys()])
            raise ValueError(f"Unknown class name: {name}. Registered measures: {registered}")

    def __call__(self, *args: Any, **kwargs: Any) -> chex.Array:
        """
        Call the distance measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the run method.
        kwargs: `Any`
            The keyword arguments to pass to the run method.

        Returns
        -------
        `chex.Array`
            The estimated distance measure.
        """
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> chex.Array:
        """
        Estimate the distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the method.
        kwargs: `Dict`
            The keyword arguments to pass to the method.

        Returns
        -------
        `chex.Array`
            The estimated distance measure.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Minkowski Distance ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class MinkowskiDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the Minkowski distance between the individual pairs of points.
    This implies that the series must be of equal length. The Minkowski distance is a metric in a normed vector space.
    It is the L_p norm of the difference between two vectors. The Minkowski distance is calculated between the two time
    series data of shape (b_x, n, d) and (b_y, n, d). The result is of shape (b_x, b_y).

    Parameters
    ----------
    p: `int`, optional, default = 2.
        The order of the Minkowski distance.
    mean: `bool`, optional, default = False.
        If True, the mean of the distances is returned.
    median: `bool`, optional, default = False.
        If True, the median of the distances is returned.
    total_sum: `bool`, optional, default = True.
        If True, the total sum of the distances is returned.

    Returns
    -------
    `MinkowskiDistance`
        An instance of the Minkowski distance measure.
    """
    p: Optional[float] = struct.field(default=None, pytree_node=False)
    mean: Optional[bool] = struct.field(default=None, pytree_node=False)
    median: Optional[bool] = struct.field(default=None, pytree_node=False)
    total_sum: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(
            cls,
            p: float = 2,
            mean: bool = False,
            median: bool = False,
            total_sum: bool = False) -> "MinkowskiDistance":
        """
        Construct the Minkowski distance measure.

        Parameters
        ----------
        p: `int`, default = 2.
            The order of the Minkowski distance.
        mean: `bool`, default = False.
            If True, the mean of the distances is returned.
        median: `bool`, default = False.
            If True, the median of the distances is returned.
        total_sum: `bool`, default = True.
            If True, the total sum of the distances is returned.

        Returns
        -------
        `MinkowskiDistance`
            An instance of the Minkowski distance measure.
        """
        assert p >= 1, "The order p of the Minkowski distance should be in [0, inf]. Got {p}"
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(p=p, mean=mean, median=median, total_sum=total_sum)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "MinkowskiDistance":
        """
        Create an instance of the Minkowski distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `MinkowskiDistance`:
            An instance of the Minkowski distance measure.
        """
        return cls.construct(*args, **kwargs)

    def run(self, x: chex.Array, y: Optional[chex.Array] = None) -> chex.Array:
        """
        Estimate the Minkowski distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        y: `chex.Array`, optional
            The second input data of shape = (b_y, N, D). If not provided, y = 0.

        Returns
        -------
        `chex.Array`:
            The estimated Minkowski distance of shape (b_x, b_y).
        """
        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but d doesn't match. "
            f"Got x={x.shape} and y={y.shape}."
        )
        assert x.shape[1] == y.shape[1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but n doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}.")

        x = jnp.expand_dims(x, axis=1)

        if y is None:
            y = jnp.zeros_like(x)
        else:
            y = jnp.expand_dims(y, axis=0)

        x_centered = x - y
        distance = jnp.linalg.norm(x_centered, axis=-1, ord=self.p)
        if self.mean:
            return jnp.mean(distance, axis=-1)
        elif self.median:
            return jnp.median(distance, axis=-1)
        return jnp.sum(distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Euclidean Distance ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class EuclideanDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the Euclidean distance between the individual pairs of points.
    This implies that the series must be of equal length. The Euclidean distance is a metric in a normed vector space.
    It is the L_2 norm of the difference between two vectors. The Euclidean distance is calculated between the two time
    series data of shape (b_x, n, d) and (b_y, n, d). The result is of shape (b_x, b_y).

    Parameters
    ----------
    mean: `bool`, optional, default = False.
        If True, the mean of the distances is returned.
    median: `bool`, optional, default = False.
        If True, the median of the distances is returned.
    total_sum: `bool`, optional, default = True.
        If True, the total sum of the distances is returned.

    Returns
    -------
    `EuclideanDistance`
        An instance of the Euclidean distance measure.
    """
    mean: Optional[bool] = struct.field(default=None, pytree_node=False)
    median: Optional[bool] = struct.field(default=None, pytree_node=False)
    total_sum: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(cls, mean: bool = False, median: bool = False, total_sum: bool = False) -> "EuclideanDistance":
        """
        Construct the Euclidean distance measure.

        Parameters
        ----------
        mean: `bool`, default = False.
            If True, the mean of the distances is returned.
        median: `bool`, default = False.
            If True, the median of the distances is returned.
        total_sum: `bool`, default = True.
            If True, the total sum of the distances is returned.

        Returns
        -------
        `EuclideanDistance`
            An instance of the Euclidean distance measure.
        """
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(mean=mean, median=median, total_sum=total_sum)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "EuclideanDistance":
        """
        Create an instance of the Euclidean distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `EuclideanDistance`:
            An instance of the Euclidean distance measure.
        """
        return cls.construct(*args, **kwargs)

    def run(self, x: chex.Array, y: Optional[chex.Array] = None) -> chex.Array:
        """
        Estimate the Euclidean distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        y: `chex.Array`, optional
            The second input data of shape = (b_y, N, D). If not provided, y = 0.

        Returns
        -------
        `chex.Array`:
            The estimated Euclidean distance of shape (b_x, b_y).
        """
        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but d doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}."
        )
        assert x.shape[1] == y.shape[1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but n doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}.")

        x = jnp.expand_dims(x, axis=1)

        if y is None:
            y = jnp.zeros_like(x)
        else:
            y = jnp.expand_dims(y, axis=0)

        x_centered = x - y  # (b_x, b_mu, N, D)
        squared_distance = jnp.linalg.norm(x_centered, axis=-1) # (b_x, b_mu, N)
        if self.mean:
            return jnp.mean(squared_distance, axis=-1)
        elif self.median:
            return jnp.median(squared_distance, axis=-1)
        return jnp.sum(squared_distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Squared Euclidean Distance --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class SquaredEuclideanDistance(EuclideanDistance):
    """
    Similarity measure between time series data using the squared Euclidean distance between the individual pairs of
    points. This implies that the series must be of equal length. The squared Euclidean distance is not metric in a
    normed vector space as it does not satisfy the triangle inequality. The distance is calculated between the two time
    series data of shape (b_x, n, d) and (b_y, n, d). The result is of shape (b_x, b_y).

    Parameters
    ----------
    mean: `bool`, optional, default = False.
        If True, the mean of the distances is returned.
    median: `bool`, optional, default = False.
        If True, the median of the distances is returned.
    total_sum: `bool`, optional, default = True.
        If True, the total sum of the distances is returned.

    Returns
    -------
    `SquaredEuclideanDistance`
        An instance of the squared Euclidean distance measure.
    """
    def run(self, x: chex.Array, y: Optional[chex.Array] = None) -> chex.Array:
        """
        Estimate the squared Euclidean distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        y: `chex.Array`, optional
            The second input data of shape = (b_y, N, D). If not provided, y = 0.

        Returns
        -------
        `chex.Array`:
            The estimated squared Euclidean distance of shape (b_x, b_y).

        """
        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but d doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}."
        )
        assert x.shape[1] == y.shape[1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but n doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}.")

        x = jnp.expand_dims(x, axis=1)

        if y is None:
            y = jnp.zeros_like(x)
        else:
            y = jnp.expand_dims(y, axis=0)

        x_centered = x - y
        squared_distance = jnp.linalg.norm(x_centered, axis=-1) ** 2
        if self.mean:
            return jnp.mean(squared_distance, axis=-1)
        elif self.median:
            return jnp.median(squared_distance, axis=-1)
        return jnp.sum(squared_distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Mahalanobis Distance -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class MahalanobisDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the Mahalanobis distance between the individual pairs of points.
    This implies that the series must be of equal length. The distance is calculated between a time
    series data of shape (b_x, n, d) and a sequence of multivariate Normal distributions with mean of shape
    (b_mu, n, d) and covariance of shape (b_cov, n, d, d). The result is of shape (b_x, b_mu).

    Parameters
    ----------
    mean: `bool`, optional, default = False.
        If True, the mean of the distances is returned.
    median: `bool`, optional, default = False.
        If True, the median of the distances is returned.
    total_sum: `bool`, optional, default = True.
        If True, the total sum of the distances is returned.

    Returns
    -------
    `MahalanobisDistance`
        An instance of the Mahalanobis distance measure.
    """
    mean: Optional[bool] = struct.field(default=None, pytree_node=False)
    median: Optional[bool] = struct.field(default=None, pytree_node=False)
    total_sum: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(cls, mean: bool = False, median: bool = False, total_sum: bool = False) -> "MahalanobisDistance":
        """
        Construct the Mahalanobis distance measure.

        Parameters
        ----------
        mean: `bool`, default = False.
            If True, the mean of the distances is returned.
        median: `bool`, default = False.
            If True, the median of the distances is returned.
        total_sum: `bool`, default = True.
            If True, the total sum of the distances is returned.

        Returns
        -------
        `MahalanobisDistance`
            An instance of the Mahalanobis distance measure.
        """
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(mean=mean, median=median, total_sum=total_sum)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Any) -> "MahalanobisDistance":
        """
        Create an instance of the Mahalanobis distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `MahalanobisDistance`:
            An instance of the Mahalanobis distance measure.
        """
        return cls.construct(*args, **kwargs)

    def init_stats(
        self,
        x: chex.Array,
        mu: Optional[chex.Array] = None,
        covariance_matrix: Optional[chex.Array] = None,
        precision_matrix: Optional[chex.Array] = None
        ) -> Sequence:
        """
        Initialize the state for the Mahalanobis distance measure. If the covariance matrix is provided, its lower
        triangular matrix is calculated. If the precision matrix is provided, its lower triangular matrix is calculated.
        We use the identity matrix if neither covariance nor precision matrices are provided.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        mu: `chex.Array`, optional
            The mean of the input data of shape = (b_mu, N, D). If not provided, mu = 0.
        covariance_matrix: `chex.Array`, optional
            The covariance matrix of the input data of shape = (b_cov, N, D, D). If not provided, covariance_matrix = 0.
        precision_matrix: `chex.Array`, optional
            The precision matrix of the input data of shape = (b_prec, N, D, D). If not provided, precision_matrix = 0.

        Returns
        -------
        chex.Array :
            The mean of the input data.
        chex.Array :
            The lower triangular matrix of the covariance matrix.
        chex.Array :
            The lower triangular matrix of the precision matrix
        """
        if mu is None:
            mu = jnp.zeros_like(x)
        else:
            mu = jnp.expand_dims(mu, axis=0)

        lower_tri_covariance = None
        if covariance_matrix is not None:
            if covariance_matrix.ndim == 2:
                covariance_matrix = covariance_matrix[jnp.newaxis, ...]
            lower_tri_covariance = jax.vmap(jnp.linalg.cholesky)(covariance_matrix)

        lower_tri_precision = None
        if precision_matrix is not None:
            if precision_matrix.ndim == 2:
                precision_matrix = precision_matrix[jnp.newaxis, ...]
            lower_tri_precision = jax.vmap(jnp.linalg.cholesky)(precision_matrix)

        if covariance_matrix is None and precision_matrix is None:
            b_2, n, d = mu.shape[-3:]
            lower_tri_precision = jnp.expand_dims(jnp.eye(d), [0, 1])
            lower_tri_precision = lower_tri_precision.repeat(n, axis=1).repeat(b_2, axis=0)

        return mu, lower_tri_covariance, lower_tri_precision

    def run(
            self,
            x: chex.Array,
            mu: Optional[chex.Array] = None,
            covariance_matrix: Optional[chex.Array] = None,
            precision_matrix: Optional[chex.Array] = None
    ) -> chex.Array:
        """
        Estimate the Mahalanobis distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        mu: `chex.Array`, optional
            The mean of the input data of shape = (b_mu, N, D). If not provided, mu = 0.
        covariance_matrix: `chex.Array`, optional
            The covariance matrix of the input data of shape = (b_cov, N, D, D). If not provided, covariance_matrix = 0.
        precision_matrix: `chex.Array`, optional
            The precision matrix of the input data of shape = (b_prec, N, D, D). If not provided, precision_matrix = 0.

        Returns
        -------
        `chex.Array`:
            The estimated Mahalanobis distance of shape (b_x, b_mu).
        """
        assert x.shape[-1] == mu.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and mu = (b_y, n, d) but d doesn't match. "
            f"Got x = {x.shape} and mu = {mu.shape}."
        )
        assert x.shape[1] == mu.shape[1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and mu = (b_y, n, d) but n doesn't match. "
            f"Got x = {x.shape} and mu = {mu.shape}.")
        x = jnp.expand_dims(x, axis=1)

        mu, lower_tri_covariance, lower_tri_precision = self.init_stats(x, mu, covariance_matrix, precision_matrix)

        x_centered = x - mu
        if lower_tri_covariance is not None:
            lower_tri = lower_tri_covariance
            _solve_fn = jax.vmap(jax.vmap(jnp.linalg.solve))
            x_transformed = jax.vmap(_solve_fn, in_axes=[None, 0])(lower_tri, x_centered)
        elif lower_tri_precision is not None:
            lower_tri = lower_tri_precision
            x_transformed = jnp.einsum("...mn, ...n->...m", lower_tri, x_centered)
        else:
            raise ValueError("Neither covariance nor precision marices were given.")

        distance = jnp.linalg.norm(x_transformed, axis=-1)

        if self.mean:
            return jnp.mean(distance, axis=-1)
        elif self.median:
            return jnp.median(distance, axis=-1)
        return jnp.sum(distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Squared Mahalanobis Distance -------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class SquaredMahalanobisDistance(MahalanobisDistance):
    """
    Similarity measure between time series data using the squared Mahalanobis distance between the individual pairs of
    points. This implies that the series must be of equal length. The distance is calculated between a time
    series data of shape (b_x, n, d) and a sequence of multivariate Normal distributions with mean of shape
    (b_mu, n, d) and covariance of shape (b_cov, n, d, d). The result is of shape (b_x, b_mu).

    Parameters
    ----------
    mean: `bool`, optional, default = False.
        If True, the mean of the distances is returned.
    median: `bool`, optional, default = False.
        If True, the median of the distances is returned.
    total_sum: `bool`, optional, default = True.
        If True, the total sum of the distances is returned.

    Returns
    -------
    `SquaredMahalanobisDistance`
        An instance of the squared Mahalanobis distance measure.
    """

    def run(
            self,
            x: chex.Array,
            mu: Optional[chex.Array] = None,
            covariance_matrix: Optional[chex.Array] = None,
            precision_matrix: Optional[chex.Array] = None
    ) -> chex.Array:
        """
        Estimate the Mahalanobis distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, N, D).
        mu: `chex.Array`, optional
            The mean of the input data of shape = (b_mu, N, D). If not provided, mu = 0.
        covariance_matrix: `chex.Array`, optional
            The covariance matrix of the input data of shape = (b_cov, N, D, D). If not provided, covariance_matrix = 0.
        precision_matrix: `chex.Array`, optional
            The precision matrix of the input data of shape = (b_prec, N, D, D). If not provided, precision_matrix = 0.

        Returns
        -------
        `chex.Array`:
            The estimated Mahalanobis distance of shape (b_x, b_mu).
        """
        assert x.shape[-1] == mu.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and mu = (b_y, n, d) but d doesn't match. "
            f"Got x = {x.shape} and mu = {mu.shape}."
        )
        assert x.shape[1] == mu.shape[1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and mu = (b_y, n, d) but n doesn't match. "
            f"Got x = {x.shape} and mu = {mu.shape}.")
        x = jnp.expand_dims(x, axis=1)

        mu, lower_tri_covariance, lower_tri_precision = self.init_stats(x, mu, covariance_matrix, precision_matrix)

        x_centered = x - mu

        if lower_tri_covariance is not None:
            lower_tri = lower_tri_covariance
            _solve_fn = jax.vmap(jax.vmap(jnp.linalg.solve))
            x_transformed = jax.vmap(_solve_fn, in_axes=[None, 0])(lower_tri, x_centered)
        elif lower_tri_precision is not None:
            lower_tri = lower_tri_precision
            x_transformed = jnp.einsum("...mn, ...n->...m", lower_tri, x_centered)
        else:
            raise ValueError("Neither covariance nor precision marices were given.")

        squared_distance = jnp.linalg.norm(x_transformed, axis=-1) ** 2

        if self.mean:
            return jnp.mean(squared_distance, axis=-1)
        elif self.median:
            return jnp.median(squared_distance, axis=-1)
        return jnp.sum(squared_distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Dynamic Time Warping -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class DynamicTimeWarping(DistanceMeasures):
    """
    Similarity measure between time series data using the Dynamic Time Warping (DTW) [1, 2]. The DTW is an algorithm
    used to measure the similarity between two sequences that may vary in time or speed. This implies that the series
    can be of different lengths. The resulting DTW distance is just a similarity measure and not a metric as it does
    not satisfy the triangle inequality. The DTW distance is calculated between the two time series data of shape
    (b_x, n_x, d) and (b_y, n_y, d). The result is of shape (b_x, b_y).

    The algorithm follows a dynamic programming approach to find the optimal alignment between the two sequences. We
    used the DTW implementation from [3].

    Parameters
    ----------
    distance: `DistanceMeasures`, optional, default = None.
        The distance measure to use for the DTW. If None, the Euclidean distance is used.

    Returns
    -------
    `DynamicTimeWarping`
        An instance of the Dynamic Time Warping distance measure.

    References
    ----------
    [1] T. K. Vintsyuk. Speech discrimination by dynamic programming. Cybernetics, 4(1):52–57,1968.
        Available: https://link.springer.com/article/10.1007/BF01074755
    [2] H. Sakoe and S. Chiba. Dynamic programming algorithm optimization for spoken word recognition.
        IEEE transactions on acoustics, speech, and signal processing, 26(1):43–49, 1978.
        Available: https://ieeexplore.ieee.org/document/1163055
    [3] K. Heidler. (Soft-)DTW for JAX, Github, https://github.com/khdlr/softdtw_jax
    """
    distance: Optional[DistanceMeasures] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(cls, distance: Optional[DistanceMeasures] = None) -> "DynamicTimeWarping":
        """
        Construct the Dynamic Time Warping distance measure.

        Parameters
        ----------
        distance: `DistanceMeasures`, optional, default = None.
            The distance measure to use for the DTW. If None, the Euclidean distance is used.

        Returns
        -------
        `DynamicTimeWarping`
            An instance of the Dynamic Time Warping distance measure.
        """
        if distance is None:
            distance = EuclideanDistance.construct()
        return cls(distance=distance)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "DynamicTimeWarping":
        """
        Create an instance of the Dynamic Time Warping distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `DynamicTimeWarping`
            An instance of the Dynamic Time Warping distance measure.
        """
        return cls.construct(*args, **kwargs)

    def init_model_matrix(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Initialize the state for the Dynamic Time Warping distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        chex.Array:
            The model matrix for the dynamice time warping measure of shape (b_x, b_y, n_x + n_y - 1, n_y).
        """

        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n_x, d) and y = (b_y, n_y, d) but d doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}.")

        def _construct_model_matrix(_x: chex.Array, _y: chex.Array) -> chex.Array:
            _x = jnp.expand_dims(_x, axis=1)
            _y = jnp.expand_dims(_y, axis=1)
            _distance_matrix = self.distance(x=_x, y=_y)

            _h, _ = _distance_matrix.shape
            _rows = []
            for _row in range(_h):
              _rows.append(jnp.pad(_distance_matrix[_row], (_row, _h-_row-1), constant_values=jnp.inf))
            return jnp.stack(_rows, axis=1)

        return jax.vmap(jax.vmap(_construct_model_matrix, in_axes=(None, 0)), in_axes=(0, None))(x, y)

    def run(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Estimate the Dynamic Time Warping distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        `chex.Array`:
            The estimated Dynamic Time Warping distance of shape (b_x, b_y).
        """
        def _body_fn(carry: Sequence, anti_diagonal: chex.Array) -> Any:
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

            next_row = best + anti_diagonal
            next_row = jnp.pad(next_row, (1, 0), constant_values=jnp.inf)

            return (one_ago, next_row), next_row

        def _run(model_matrix: chex.Array) -> chex.Array:
            init = (
            jnp.pad(model_matrix[0], (1, 0), constant_values=jnp.inf),
            jnp.pad(model_matrix[1] + model_matrix[0, 0], (1, 0), constant_values=jnp.inf)
            )
            carry, ys = jax.lax.scan(_body_fn, init, model_matrix[2:], unroll=2)
            return carry[1][-1]

        model_matrix = self.init_model_matrix(x, y)
        return jax.vmap(jax.vmap(_run))(model_matrix)


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Discrete Frechet Distance --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class DiscreteFrechetDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the discrete Frechet distance [1. 2]. It is the minimum length of
    a leash required for a dog and its owner to walk along their respective curves without backtracking. This implies
    that the series can be of different lengths. The discrete Frechet distance is a measure of similarity between two
    curves. The discrete Frechet distance is calculated between the two time series data of shape (b_x, n_x, d) and
    (b_y, n_y, d). The result is of shape (b_x, b_y).

    The algorithm follows a dynamic programming approach to find the optimal alignment between the two sequences. We
    followed the implementation of DTW from [3] and replaced the sum operation with the max operation resulting in
    the Discrete Frechet distance.

    Parameters
    ----------
    distance: `DistanceMeasures`, optional, default = None.
        The distance measure to use for the discrete Frechet distance. If None, the Euclidean distance is used.

    Returns
    -------
    `DiscreteFrechetDistance`
        An instance of the discrete Frechet distance measure.

    References
    ----------
    [1] M. Fr ́echet. Sur quelques points du calcul fonctionnel. 1906.
      Available: https://link.springer.com/article/10.1007/BF03018603
    [2] T. Eiter and H. Mannila. Computing discrete fr ́echet distance. 1994.
      Available: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    [3] K. Heidler. (Soft-)DTW for JAX, Github, https://github.com/khdlr/softdtw_jax
    """
    distance: Optional[DistanceMeasures] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(cls, distance: Optional[DistanceMeasures] = None) -> "DiscreteFrechetDistance":
        """
        Construct the discrete Frechet distance measure.

        Parameters
        ----------
        distance: `DistanceMeasures`, optional, default = None.
            The distance measure to use for the discrete Frechet distance. If None, the Euclidean distance is used.

        Returns
        -------
        `DiscreteFrechetDistance`
            An instance of the discrete Frechet distance measure.
        """
        if distance is None:
            distance = EuclideanDistance.construct()
        return cls(distance=distance)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "DiscreteFrechetDistance":
        """
        Create an instance of the discrete Frechet distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `DiscreteFrechetDistance`
            An instance of the discrete Frechet distance measure.
        """
        return cls.construct(*args, **kwargs)

    def init_model_matrix(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Initialize the state for the discrete Frechet distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        chex.Array:
            The model matrix for the discrete Frechet distance measure of shape (b_x, b_y, n_x + n_y - 1, n_y).
        """
        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n_x, d) and y = (b_y, n_y, d) but d doesn't match. "
            f"Got x = {x.shape} and y = {y.shape}.")

        def _construct_model_matrix(_x: chex.Array, _y: chex.Array) -> chex.Array:
            _x = jnp.expand_dims(_x, axis=1)
            _y = jnp.expand_dims(_y, axis=1)
            _distance_matrix = self.distance(x=_x, y=_y)

            _h, _ = _distance_matrix.shape

            _rows = []
            for _row in range(_h):
              _rows.append(jnp.pad(_distance_matrix[_row], (_row, _h-_row-1), constant_values=jnp.inf))
            return jnp.stack(_rows, axis=1)

        return jax.vmap(jax.vmap(_construct_model_matrix, in_axes=(None, 0)), in_axes=(0, None))(x, y)

    def run(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Estimate the discrete Frechet distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        `chex.Array`:
            The estimated discrete Frechet distance of shape (b_x, b_y).
        """
        def _body_fn(carry: Sequence, anti_diagonal: chex.Array) -> Any:
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

            next_row = jnp.maximum(best, anti_diagonal)
            next_row = jnp.pad(next_row, (1, 0), constant_values=jnp.inf)

            return (one_ago, next_row), next_row

        def _run(model_matrix: chex.Array) -> chex.Array:
            init = (
            jnp.pad(model_matrix[0], (1, 0), constant_values=jnp.inf),
            jnp.pad(
              jnp.maximum(model_matrix[1], model_matrix[0]),
              (1, 0),
              constant_values=jnp.inf
              )
            )
            carry, ys = jax.lax.scan(_body_fn, init, model_matrix[2:], unroll=2)
            return carry[1][-1]

        model_matrix = self.init_model_matrix(x, y)
        return jax.vmap(jax.vmap(_run))(model_matrix)


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Sinkhorn Distance ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class BaseCost(CostFn):
    """
    The base cost function for the Sinkhorn distance measure. The cost function is a weighted sum of the spatial and
    temporal distances.

    Parameters
    ----------
    weights: `Sequence[float]`, optional, default = None.
        The weights for the spatial and temporal distances. If None, the weights are set to [1., 1.].
    distances: `Sequence[DistanceMeasures]`, optional, default = None.
        The spatial and temporal distance measures. If None, the spatial distance is the squared Euclidean distance

    Returns
    -------
    `BaseCost`
        An instance of the base cost function.
    """
    weights: Optional[Sequence[float]] = None
    distances: Optional[Sequence[DistanceMeasures]] = None

    @classmethod
    def construct(cls) -> "BaseCost":
        """
        Construct the base cost function for the Sinkhorn distance measure.

        Returns
        -------
        `BaseCost`
            An instance of the base cost function.
        """
        weights = [1., 1.]
        distances = [
            SquaredEuclideanDistance.construct(),
            MinkowskiDistance.construct(p=1)
        ]
        return cls(weights=weights, distances=distances)

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "BaseCost":
        """
        Create an instance of the base cost function for the Sinkhorn distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs:
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `BaseCost`
            An instance of the base cost function.
        """
        return cls.construct(*args, **kwargs)

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Calculate the pairwise cost between two points.

        Parameters
        ----------
        x: `jnp.ndarray`
            The first point of shape (d, ).
        y: `jnp.ndarray`
            The second point of shape (d, ).

        Returns
        -------
        `float`
            The pairwise cost between the two points.
        """
        total_cost = 0
        spatio_weights, temporal_weights = self.weights
        spatio_dist, temporal_dist = self.distances
        x, y = x[jnp.newaxis, jnp.newaxis, ...], y[jnp.newaxis, jnp.newaxis, ...]

        # Spatial coordinate related cost
        total_cost += spatio_weights * spatio_dist(x[..., :-1], y[..., :-1]).squeeze()

        # Temporal coordinate related cost
        total_cost += temporal_weights * temporal_dist(x[..., -1:], y[..., -1:]).squeeze()
        return total_cost


@struct.dataclass
class SinkhornDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the Sinkhorn distance measure [1, 2]. The Sinkhorn distance is a
    regularized optimal transport distance that is a measure of similarity between two probability distributions. Here,
    we consider each point in the time series as a weighted particle. Thus, the time series represent a discrete
    probability measure. The Sinkhorn distance is calculated between the two time series data of shape (b_x, n_x, d) and
    (b_y, n_y, d). The result is of shape (b_x, b_y).

    The algorithm follows a linear programming approach to find the optimal transport plan between the two sequences.
    We used the implementation of the Sinkhorn distance from [3].

    Parameters
    ----------
    solver: `Any`, optional, default = None.
        The solver to use for the Sinkhorn distance. If None, the Sinkhorn solver is used.
    cost_fn: `CostFn`, optional, default = None.
        The cost function to use for the Sinkhorn distance. If None, the base cost function is used.
    epsilon: `float`, optional, default = None.
        The regularization parameter for the Sinkhorn distance. If None, the default value is used.
    return_regularized_cost: `bool`, optional, default = False.
        If True, the regularized cost is returned. If False, the unregularized cost is returned.

    Returns
    -------
    `SinkhornDistance`
        An instance of the Sinkhorn distance measure.

    References
    ----------
    [1] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. 2013.
        Available: https://arxiv.org/abs/1306.0895
    [2] G. Peyré and M. Cuturi. Computational Optimal Transport. 2019.
        Available: https://arxiv.org/abs/1803.00567
    [3] M. Cuturi. Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein, arXiv, 2022.
        Available: arXiv preprint arXiv:2201.12324
        Github: https://github.com/ott-jax/ott
        Docs: https://ott-jax.readthedocs.io/en/latest/
    """
    solver: Optional[Any] = None
    cost_fn: Optional[CostFn] = struct.field(default=None, pytree_node=False)
    epsilon: Optional[float] = struct.field(default=None, pytree_node=False)
    return_regularized_cost: Optional[bool] = struct.field(default=None, pytree_node=False)

    @classmethod
    def construct(
        cls,
        epsilon: Optional[float]=None,
        cost_fn: Optional[CostFn]=None,
        return_regularized_cost: Optional[bool]=None,
        ) -> "SinkhornDistance":
        """
        Construct the Sinkhorn distance measure.

        Parameters
        ----------
        epsilon: `float`, optional, default = None.
            The regularization parameter for the Sinkhorn distance.
        cost_fn: `CostFn`, optional, default = None.
            The cost function to use for the Sinkhorn distance.
        return_regularized_cost: `bool`, optional, default = None.
            If True, the regularized cost is returned. If False, the unregularized cost is returned.

        Returns
        -------
        `SinkhornDistance`
            An instance of the Sinkhorn distance measure.
        """
        if cost_fn is None:
          cost_fn = BaseCost.construct()

        if return_regularized_cost is None:
          return_regularized_cost = False

        return cls(
            solver=sinkhorn.Sinkhorn(),
            epsilon=epsilon,
            cost_fn = cost_fn,
            return_regularized_cost=return_regularized_cost
        )

    @classmethod
    def create(cls, *args: Sequence, **kwargs: Dict) -> "SinkhornDistance":
        """
        Create an instance of the Sinkhorn distance measure.

        Parameters
        ----------
        args: `Sequence`
            The arguments to pass to the constructor.
        kwargs: `Dict`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `SinkhornDistance`
            An instance of the Sinkhorn distance measure.
        """
        return cls.construct(*args, **kwargs)

    def init_geometry(self, x: chex.Array, y: chex.Array) -> Any:
        """
        Initialize the geometry for the Sinkhorn distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        `Any`
            The geometry of the Sinkhorn distance measure.
        """
        assert x.shape[-1] == y.shape[-1], print(
            f"The two inputs need to be of the shape x = (b_x, n, d) and y = (b_y, n, d) but d doesn't match. "
            f"Got x = { x.shape} and y = {y.shape}.")

        def _construct_geom(_x: chex.Array, _y: chex.Array) -> Any:
            # Add time to given arrays based on a linear interpolation
            t_x, _ = _x.shape
            _x = jnp.concatenate((_x, jnp.linspace(0, 1, t_x)[:, jnp.newaxis]), axis=-1)
            t_y, _ = _y.shape
            _y = jnp.concatenate((_y, jnp.linspace(0, 1, t_y)[:, jnp.newaxis]), axis=-1)

            # Generat and return a geometry for a Linear OT problem
            if self.epsilon is not None:
                geometry = pointcloud.PointCloud(_x, _y, cost_fn=self.cost_fn, epsilon=self.epsilon)
            else:
                geometry = pointcloud.PointCloud(_x, _y, cost_fn=self.cost_fn)
            return geometry

        return jax.vmap(jax.vmap(_construct_geom, in_axes=(None, 0)), in_axes=(0, None))(x, y)

    def run(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """
        Estimate the Sinkhorn distance measure.

        Parameters
        ----------
        x: `chex.Array`
            The input data of shape = (b_x, n_x, d).
        y: `chex.Array`
            The second input data of shape = (b_y, n_y, d).

        Returns
        -------
        `chex.Array`
            The estimated Sinkhorn distance of shape (b_x, b_y).
        """
        def _run(geometry: Any) -> chex.Array:
            ot_problem = linear_problem.LinearProblem(geometry)
            solution = self.solver(ot_problem)
            if self.return_regularized_cost:
                return solution.reg_ot_cost
            return jnp.sum(solution.matrix * solution.geom.cost_matrix)

        geometry = self.init_geometry(x, y)
        return jax.vmap(jax.vmap(_run))(geometry)
