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
    Base class for all distance measures. The distance measures are used to calculate the similarity between two data
    points. This points could be either one dimensional points or time-series data. The base class provides a registry
    to store all the implemented distance measures. To create an instance of a distance measure, use
    the `create_instance` method.

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
    def create_instance(
        cls, name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> "DistanceMeasures":
        """
        Create an instance of the distance measure.

        Parameters
        ----------
        name: `str`, optional, default = None.
            The name of the distance measure to create.
        args: `Any`
            The arguments to pass to the constructor of the distance measure.
        kwargs: `Any`
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
            raise ValueError(
                f"Unknown class name: {name}. Registered measures: {registered}"
            )

    @classmethod
    @abstractmethod
    def construct(cls, *args, **kwargs) -> "DistanceMeasures":
        """Create an Instance of the Distance Measure"""

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "DistanceMeasures":
        """
        Create an instance of the Minkowski distance measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the constructor.
        kwargs: `Any`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `DistanceMeasures`:
            An instance of the distance measure.
        """
        return cls.construct(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
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
        `jax.Array`
            The estimated distance measure.
        """
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> jax.Array:
        """
        Estimate the distance measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the method.
        kwargs: `Any`
            The keyword arguments to pass to the method.

        Returns
        -------
        `jax.Array`
            The estimated distance measure.
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
# ------------------------------------------------ Minkowski Distance ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class MinkowskiDistance(DistanceMeasures):
    """
    Similarity measure between data points using the Minkowski distance. The Minkowski distance is a metric in a normed
    vector space. It is the L_p norm of the difference between two vectors. If data point corresponds to a time series,
    the distance is calculated between the individual pairs of points. This implies that the series must be of equal
    length.

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
        total_sum: bool = False,
    ) -> "MinkowskiDistance":
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
        assert (
            p >= 1
        ), "The order p of the Minkowski distance should be in [0, inf]. Got {p}"
        if not mean and not median and not total_sum:
            total_sum = True
        return cls(p=p, mean=mean, median=median, total_sum=total_sum)

    def run(self, x: jax.Array, y: Optional[jax.Array] = None) -> jax.Array:
        """
        Estimate the Minkowski distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data of shape (d, ) if particle, or (n, d) if time series.
        y: `jax.Array`, optional
            The second input data of shape (d, ) if particle, or (n, d) if time series. If not provided, y = 0.

        Returns
        -------
        `jax.Array`:
            The estimated Minkowski distance of shape ().
        """
        if y is None:
            y = jnp.zeros_like(x)

        assert (
            x.ndim == y.ndim and x.shape == y.shape
        ), f"The two data points need to be of the same shape. Got x={x.shape} and y={y.shape}."
        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)  # (n = 1, d)
            y = jnp.expand_dims(y, axis=0)  # (n = 1, d)

        distance = jnp.linalg.norm(x - y, axis=-1, ord=self.p)  # (n, )
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
    Similarity measure between data points using the Euclidean distance. The Euclidean distance is a metric in a normed
    vector space. It is the L_p norm of the difference between two vectors. If data point corresponds to a time series,
    the distance is calculated between the individual pairs of points. This implies that the series must be of equal
    length.

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
    def construct(
        cls, mean: bool = False, median: bool = False, total_sum: bool = False
    ) -> "EuclideanDistance":
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

    def run(self, x: jax.Array, y: Optional[jax.Array] = None) -> jax.Array:
        """
        Estimate the Euclidean distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n, d) if time series.
        y: `jax.Array`, optional
            The second input data of shape (d, ) if particle, or (n, d) if time series. If not provided, y = 0.

        Returns
        -------
        `jax.Array`:
            The estimated Euclidean distance of shape ().
        """
        if y is None:
            y = jnp.zeros_like(x)

        assert (
            x.ndim == y.ndim and x.shape == y.shape
        ), f"The two data points need to be of the same shape. Got x={x.shape} and y={y.shape}."
        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)  # (n = 1, d)
            y = jnp.expand_dims(y, axis=0)  # (n = 1, d)

        distance = jnp.linalg.norm(x - y, axis=-1)  # (n, )
        if self.mean:
            return jnp.mean(distance, axis=-1)
        elif self.median:
            return jnp.median(distance, axis=-1)
        return jnp.sum(distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Squared Euclidean Distance --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class SquaredEuclideanDistance(EuclideanDistance):
    """
    Similarity measure between a data point using the squared Euclidean distance. The distance is not metric in a normed
    vector space as it does not satisfy the triangle inequality. If data point corresponds to a time series,
    the distance is calculated between the individual pairs of points. This implies that the series must be of equal
    length.

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

    def run(self, x: jax.Array, y: Optional[jax.Array] = None) -> jax.Array:
        """
        Estimate the squared Euclidean distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n, d) if time series.
        y: `jax.Array`, optional
            The second input data of shape (d, ) if particle, or (n, d) if time series. If not provided, y = 0.

        Returns
        -------
        `jax.Array`:
            The estimated squared Euclidean distance of shape ().

        """
        if y is None:
            y = jnp.zeros_like(x)

        assert (
            x.ndim == y.ndim and x.shape == y.shape
        ), f"The two data points need to be of the same shape. Got x={x.shape} and y={y.shape}."

        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series."
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)  # (n = 1, d)
            y = jnp.expand_dims(y, axis=0)  # (n = 1, d)

        squared_distance = jnp.linalg.norm(x - y, axis=-1) ** 2  # (n, )
        if self.mean:
            return jnp.mean(squared_distance, axis=-1)
        elif self.median:
            return jnp.median(squared_distance, axis=-1)
        return jnp.sum(squared_distance, axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Cosine Distance --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class CosineDistance(DistanceMeasures):
    eps: float = struct.field(default=1e-8, pytree_node=False)

    @classmethod
    def construct(cls, eps: float = 1e-8) -> "CosineDistance":
        return cls(eps)

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Estimate the Cosine distance measure d = 1 - 'cosine similarity'.
        Note that the cosine distance is only a quasi-metric.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ). Time series data is not supported at this point
        y: `jax.Array`, optional
            The second input data of shape (d, ).

        Returns
        -------
        `jax.Array`:
            The estimated cosine distance of shape ().

        """
        assert (
            x.shape == y.shape
        ), f"The two data points need to be of the same shape. Got x={x.shape} and y={y.shape}."

        # Deal with pseudo-time series data, i.e., data of shape (1, d)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y.squeeze(0)

        assert (
            x.ndim == y.ndim <= 1
        ), f"The cosine distance only supports data of shape (d,), but received {x.shape}"

        norm_x = jnp.maximum(jnp.linalg.norm(x), self.eps)
        norm_y = jnp.maximum(jnp.linalg.norm(y), self.eps)

        dot_product = jnp.dot(x, y)
        cosine_similarity = dot_product / (norm_x * norm_y)
        # cosine_similarity = jnp.clip(cosine_similarity, -1.0, 1.0)
        return jnp.clip(1 - cosine_similarity, 0.0, 2.0)


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Mahalanobis Distance -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class MahalanobisDistance(DistanceMeasures):
    """
    Similarity measure between a data point and a multivariate Normal using the Mahalanobis distance.
    If data point corresponds to a time series, the distance is calculated between the individual pairs.
    This implies that the series must be of equal length.

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
    def construct(
        cls, mean: bool = False, median: bool = False, total_sum: bool = False
    ) -> "MahalanobisDistance":
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

    def run(
        self,
        x: jax.Array,
        mu: Optional[jax.Array] = None,
        covariance_matrix: Optional[jax.Array] = None,
        precision_matrix: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Estimate the Mahalanobis distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n, d) if time series.
        mu: `jax.Array`, optional
            The mean data point of shape (d, ) if particle, or (n, d) if time series. If not provided, mu = 0.
        covariance_matrix: `jax.Array`, optional
            The covariance matrix of shape (d, d, ) if particle, or (n, d, d) if time series. If not provided,
            covariance_matrix = eye(d).
        precision_matrix: `jax.Array`, optional
            The precision matrix of shape (d, d, ) if particle, or (n, d, d) if time series. If not provided,
            precision_matrix = eye(d).

        Returns
        -------
        `jax.Array`:
            The estimated Mahalanobis distance of shape ().
        """
        if mu is None:
            mu = jnp.zeros_like(x)

        assert (
            x.ndim == mu.ndim and x.shape == mu.shape
        ), f"The data point and the mean need to be of the same shape. Got x={x.shape}."
        assert x.ndim <= 2 and mu.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {mu.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            mu = jnp.expand_dims(mu, axis=0)

        lower_tri_covariance = None
        if covariance_matrix is not None:
            assert (
                x.ndim == covariance_matrix.ndim - 1
                and x.shape == covariance_matrix.shape[:-1]
            ), (
                f"The data point and the covariance need to be of the same shape. "
                f"Got x={x.shape} and y={covariance_matrix.shape}."
            )
            if covariance_matrix.ndim == 2:
                covariance_matrix = covariance_matrix[jnp.newaxis, ...]
            lower_tri_covariance = jax.vmap(jnp.linalg.cholesky)(covariance_matrix)

        lower_tri_precision = None
        if precision_matrix is not None:
            assert (
                x.ndim == precision_matrix.ndim - 1
                and x.shape == precision_matrix.shape[:-1]
            ), (
                f"The data point and the covariance need to be of the same shape. "
                f"Got x={x.shape} and y={precision_matrix.shape}."
            )
            if precision_matrix.ndim == 2:
                precision_matrix = precision_matrix[jnp.newaxis, ...]
            lower_tri_precision = jax.vmap(jnp.linalg.cholesky)(precision_matrix)

        if covariance_matrix is None and precision_matrix is None:
            n, d = mu.shape
            lower_tri_precision = jnp.expand_dims(jnp.eye(d), 0).repeat(n, axis=0)

        if lower_tri_covariance is not None:
            x_transformed = jax.vmap(jnp.linalg.solve)(
                lower_tri_covariance, x - mu
            )  # (n, d)
        elif lower_tri_precision is not None:
            x_transformed = jnp.einsum(
                "...mn, ...n->...m", lower_tri_precision, x - mu
            )  # (n, d)
        else:
            raise ValueError("Neither covariance nor precision matrices were given.")

        distance = jnp.linalg.norm(x_transformed, axis=-1)  # (n, )
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
    Similarity measure between a data point and a multivariate Normal using the squared Mahalanobis distance.
    If data point corresponds to a time series, the distance is calculated between the individual pairs.
    This implies that the series must be of equal length.

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
        x: jax.Array,
        mu: Optional[jax.Array] = None,
        covariance_matrix: Optional[jax.Array] = None,
        precision_matrix: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
         Estimate the Mahalanobis distance measure.

         Parameters
         ----------
        x: `jax.Array`
             The input data point of shape (d, ) if particle, or (n, d) if time series.
         mu: `jax.Array`, optional
             The mean data point of shape (d, ) if particle, or (n, d) if time series. If not provided, mu = 0.
         covariance_matrix: `jax.Array`, optional
             The covariance matrix of shape (d, d, ) if particle, or (n, d, d) if time series. If not provided,
             covariance_matrix = eye(d).
         precision_matrix: `jax.Array`, optional
             The precision matrix of shape (d, d, ) if particle, or (n, d, d) if time series. If not provided,
             precision_matrix = eye(d).

         Returns
         -------
         `jax.Array`:
             The estimated squared Mahalanobis distance of shape ().
        """
        if mu is None:
            mu = jnp.zeros_like(x)

        assert (
            x.ndim == mu.ndim and x.shape == mu.shape
        ), f"The data point and the mean need to be of the same shape. Got x={x.shape}."
        assert x.ndim <= 2 and mu.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {mu.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
            mu = jnp.expand_dims(mu, axis=0)

        lower_tri_covariance = None
        if covariance_matrix is not None:
            assert (
                x.ndim == covariance_matrix.ndim - 1
                and x.shape == covariance_matrix.shape[:-1]
            ), (
                f"The data point and the covariance need to be of the same shape. "
                f"Got x={x.shape} and y={covariance_matrix.shape}."
            )
            if covariance_matrix.ndim == 2:
                covariance_matrix = covariance_matrix[jnp.newaxis, ...]
            lower_tri_covariance = jax.vmap(jnp.linalg.cholesky)(covariance_matrix)

        lower_tri_precision = None
        if precision_matrix is not None:
            assert (
                x.ndim == precision_matrix.ndim - 1
                and x.shape == precision_matrix.shape[:-1]
            ), (
                f"The data point and the covariance need to be of the same shape. "
                f"Got x={x.shape} and y={precision_matrix.shape}."
            )
            if precision_matrix.ndim == 2:
                precision_matrix = precision_matrix[jnp.newaxis, ...]
            lower_tri_precision = jax.vmap(jnp.linalg.cholesky)(precision_matrix)

        if covariance_matrix is None and precision_matrix is None:
            n, d = mu.shape
            lower_tri_precision = jnp.expand_dims(jnp.eye(d), 0).repeat(n, axis=0)

        if lower_tri_covariance is not None:
            x_transformed = jax.vmap(jnp.linalg.solve)(
                lower_tri_covariance, x - mu
            )  # (n, d)
        elif lower_tri_precision is not None:
            x_transformed = jnp.einsum(
                "...mn, ...n->...m", lower_tri_precision, x - mu
            )  # (n, d)
        else:
            raise ValueError("Neither covariance nor precision matrices were given.")

        squared_distance = jnp.linalg.norm(x_transformed, axis=-1) ** 2  # (n, )

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
    not satisfy the triangle inequality.

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
    def construct(
        cls, distance: Optional[DistanceMeasures] = None
    ) -> "DynamicTimeWarping":
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

    def init_model_matrix(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Initialize the state for the Dynamic Time Warping distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data of shape = (n_x, d).
        y: `jax.Array`
            The second input data of shape = (n_y, d).

        Returns
        -------
        jax.Array:
            The model matrix for the dynamice time warping measure of shape (n_x + n_y - 1, n_y).
        """
        x = jnp.expand_dims(x, axis=1)
        y = jnp.expand_dims(y, axis=1)
        distance_matrix = jax.vmap(
            jax.vmap(self.distance, in_axes=(0, None)), in_axes=(None, 0)
        )(x, y)

        h, _ = distance_matrix.shape
        rows = []
        for row in range(h):
            rows.append(
                jnp.pad(
                    distance_matrix[row], (row, h - row - 1), constant_values=jnp.inf
                )
            )
        return jnp.stack(rows, axis=1)

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Estimate the Dynamic Time Warping distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_x, d) if time series.
        y: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_y, d) if time series.

        Returns
        -------
        `jax.Array`:
            The estimated Dynamic Time Warping distance of shape ().
        """
        assert (
            x.shape[-1] == y.shape[-1]
        ), f"The two inputs need to have the same dimensionality. Got x = {x.shape} and y = {y.shape}."
        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        if y.ndim == 1:
            y = jnp.expand_dims(y, axis=0)

        def _body_fn(carry: Sequence, anti_diagonal: jax.Array) -> Any:
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

            next_row = best + anti_diagonal
            next_row = jnp.pad(next_row, (1, 0), constant_values=jnp.inf)

            return (one_ago, next_row), next_row

        model_matrix = self.init_model_matrix(x, y)

        init = (
            jnp.pad(model_matrix[0], (1, 0), constant_values=jnp.inf),
            jnp.pad(
                model_matrix[1] + model_matrix[0, 0], (1, 0), constant_values=jnp.inf
            ),
        )
        carry, ys = jax.lax.scan(_body_fn, init, model_matrix[2:], unroll=2)
        return carry[1][-1]


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Discrete Frechet Distance --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class DiscreteFrechetDistance(DistanceMeasures):
    """
    Similarity measure between time series data using the discrete Frechet distance [1. 2]. It is the minimum length of
    a leash required for a dog and its owner to walk along their respective curves without backtracking. This implies
    that the series can be of different lengths. The discrete Frechet distance is a measure of similarity between two
    curves.

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
    def construct(
        cls, distance: Optional[DistanceMeasures] = None
    ) -> "DiscreteFrechetDistance":
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

    def init_model_matrix(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Initialize the state for the discrete Frechet distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data of shape = (n_x, d).
        y: `jax.Array`
            The second input data of shape = (n_y, d).

        Returns
        -------
        jax.Array:
            The model matrix for the discrete Frechet distance measure of shape (n_x + n_y - 1, n_y).
        """
        x = jnp.expand_dims(x, axis=1)
        y = jnp.expand_dims(y, axis=1)
        distance_matrix = jax.vmap(
            jax.vmap(self.distance, in_axes=(0, None)), in_axes=(None, 0)
        )(x, y)

        h, _ = distance_matrix.shape

        rows = []
        for row in range(h):
            rows.append(
                jnp.pad(
                    distance_matrix[row], (row, h - row - 1), constant_values=jnp.inf
                )
            )
        return jnp.stack(rows, axis=1)

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Estimate the discrete Frechet distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_x, d) if time series.
        y: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_y, d) if time series.

        Returns
        -------
        `jax.Array`:
            The estimated Dynamic Time Warping distance of shape ().
        """
        assert (
            x.shape[-1] == y.shape[-1]
        ), f"The two inputs need to have the same dimensionality. Got x = {x.shape} and y = {y.shape}."
        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        if y.ndim == 1:
            y = jnp.expand_dims(y, axis=0)

        def _body_fn(carry: Sequence, anti_diagonal: jax.Array) -> Any:
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

            next_row = jnp.maximum(best, anti_diagonal)
            next_row = jnp.pad(next_row, (1, 0), constant_values=jnp.inf)

            return (one_ago, next_row), next_row

        model_matrix = self.init_model_matrix(x, y)

        init = (
            jnp.pad(model_matrix[0], (1, 0), constant_values=jnp.inf),
            jnp.pad(
                jnp.maximum(model_matrix[1], model_matrix[0]),
                (1, 0),
                constant_values=jnp.inf,
            ),
        )

        carry, ys = jax.lax.scan(_body_fn, init, model_matrix[2:], unroll=2)
        return carry[1][-1]


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Sinkhorn Distance ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
@struct.dataclass
class OTTCostWrapper(CostFn):
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
    `OTTCostWrapper`
        An instance of the base cost function.
    """
    weights: Optional[Sequence[float]] = None
    distances: Optional[Sequence[DistanceMeasures]] = None

    @classmethod
    def construct(cls, *args: Any, **kwargs: Any) -> "OTTCostWrapper":
        """
        Construct the base cost function for the Sinkhorn distance measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the constructor.
        kwargs: `Any`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `OTTCostWrapper`
            An instance of the base cost function.
        """
        weights = [1.0, 1.0]
        distances = [
            SquaredEuclideanDistance.construct(),
            MinkowskiDistance.construct(p=1),
        ]
        return cls(weights=weights, distances=distances)

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "OTTCostWrapper":
        """
        Create an instance of the base cost function for the Sinkhorn distance measure.

        Parameters
        ----------
        args: `Any`
            The arguments to pass to the constructor.
        kwargs: `Any`
            The keyword arguments to pass to the constructor.

        Returns
        -------
        `OTTCostWrapper`
            An instance of the base cost function.
        """
        return cls.construct(*args, **kwargs)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
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

        # Spatial coordinate related cost
        total_cost += spatio_weights * spatio_dist(x[:-1], y[:-1]).squeeze()

        # Temporal coordinate related cost
        total_cost += temporal_weights * temporal_dist(x[-1:], y[-1:]).squeeze()
        return total_cost


@struct.dataclass
class SinkhornDistance(DistanceMeasures):
    """
    Similarity measure between data points using the Sinkhorn distance measure [1, 2]. The Sinkhorn distance is a
    regularized optimal transport distance that is a measure of similarity between two probability distributions. Here,
    we consider each point as a weighted particle.

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
    return_regularized_cost: bool = struct.field(default=False, pytree_node=False)

    @classmethod
    def construct(
        cls,
        epsilon: Optional[float] = None,
        cost_fn: Optional[CostFn] = None,
        return_regularized_cost: bool = False,
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
            cost_fn = OTTCostWrapper.construct()

        return cls(
            solver=sinkhorn.Sinkhorn(),
            epsilon=epsilon,
            cost_fn=cost_fn,
            return_regularized_cost=return_regularized_cost,
        )

    def init_geometry(self, x: jax.Array, y: jax.Array) -> Any:
        """
        Initialize the geometry for the Sinkhorn distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data of shape = (n_x, d).
        y: `jax.Array`
            The second input data of shape = (n_y, d).

        Returns
        -------
        `Any`
            The geometry of the Sinkhorn distance measure.
        """
        # Add time to given arrays based on a linear interpolation
        n_x, _ = x.shape
        x_extended = jnp.concatenate(
            (x, jnp.linspace(0, 1, n_x)[:, jnp.newaxis]), axis=-1
        )
        n_y, _ = y.shape
        y_extended = jnp.concatenate(
            (y, jnp.linspace(0, 1, n_y)[:, jnp.newaxis]), axis=-1
        )

        # Generate and return a geometry for a Linear OT problem
        geometry = pointcloud.PointCloud(
            x_extended, y_extended, cost_fn=self.cost_fn, epsilon=self.epsilon
        )
        return geometry

    def run(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Estimate the Sinkhorn distance measure.

        Parameters
        ----------
        x: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_x, d) if time series.
        y: `jax.Array`
            The input data point of shape (d, ) if particle, or (n_y, d) if time series.

        Returns
        -------
        `jax.Array`
            The estimated Sinkhorn distance of shape ().
        """
        assert (
            x.shape[-1] == y.shape[-1]
        ), f"The two inputs need to have the same dimensionality. Got x = { x.shape} and y = {y.shape}."
        assert x.ndim <= 2 and y.ndim <= 2, (
            f"The two inputs need to be of shape (d, ) if particle, or (n, d) if time series. "
            f"Got x = {x.shape} and y = {y.shape}."
        )

        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        if y.ndim == 1:
            y = jnp.expand_dims(y, axis=0)

        geometry = self.init_geometry(x, y)
        ot_problem = linear_problem.LinearProblem(geometry)
        solution = self.solver(ot_problem)
        if self.return_regularized_cost:
            return solution.reg_ot_cost
        return jnp.sum(solution.matrix * solution.geom.cost_matrix)
