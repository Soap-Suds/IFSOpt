# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

import functools
import itertools
from typing import Any, Callable, List, NoReturn, Optional, Sequence, Tuple, Union, Literal

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
import numpy as np

from . import costs as cost_module
from . import epsilon_scheduler as eps_scheduler
from . import math_utils as mu


@jtu.register_pytree_node_class
class Geometry:
  r"""Base class to define ground costs/kernels used in optimal transport.

  Args:
    cost_matrix: Cost matrix of shape ``[n, m]``.
    kernel_matrix: Kernel matrix of shape ``[n, m]``.
    epsilon: Regularization parameter or a scheduler.
    relative_epsilon: Whether ``epsilon`` refers to a fraction of the
      :attr:`mean_cost_matrix` or :attr:`std_cost_matrix`.
    scale_cost: option to rescale the cost matrix. Implemented scalings are
      'median', 'mean', 'std' and 'max_cost'. Alternatively, a float factor can
      be given to rescale the cost such that ``cost_matrix /= scale_cost``.
  """

  def __init__(
      self,
      cost_matrix: Optional[jnp.ndarray] = None,
      kernel_matrix: Optional[jnp.ndarray] = None,
      epsilon: Optional[Union[float, eps_scheduler.Epsilon]] = None,
      relative_epsilon: Optional[Literal["mean", "std"]] = None,
      scale_cost: Union[float, Literal["mean", "max_cost", "median",
                                       "std"]] = 1.0,
  ):
    self._cost_matrix = cost_matrix
    self._kernel_matrix = kernel_matrix
    self._epsilon_init = epsilon
    self._relative_epsilon = relative_epsilon
    self._scale_cost = scale_cost

  @property
  def cost_matrix(self) -> jnp.ndarray:
    """Cost matrix, recomputed from kernel if only kernel was specified."""
    if self._cost_matrix is None:
      eps = jnp.finfo(self._kernel_matrix.dtype).tiny
      cost = -jnp.log(self._kernel_matrix + eps)
      cost *= self.inv_scale_cost
      return cost if self._epsilon_init is None else self.epsilon * cost
    return self._cost_matrix * self.inv_scale_cost

  @property
  def mean_cost_matrix(self) -> float:
    """Mean of the :attr:`cost_matrix`."""
    n, m = self.shape
    tmp = self.apply_cost(jnp.full((n,), fill_value=1.0 / n))
    return jnp.sum((1.0 / m) * tmp)

  @property
  def std_cost_matrix(self) -> float:
    r"""Standard deviation of all values stored in :attr:`cost_matrix`.

    Uses the :meth:`apply_square_cost` to remain
    applicable to low-rank matrices, through the formula:

    .. math::
        \sigma^2=\frac{1}{nm}\left(\sum_{ij} C_{ij}^2 -
        (\sum_{ij}C_ij)^2\right).

    to output :math:`\sigma`.
    """
    n, m = self.shape
    tmp = self.apply_square_cost(jnp.full((n,), fill_value=1.0 / n))
    tmp = jnp.sum((1.0 / m) * tmp) - (self.mean_cost_matrix ** 2)
    return jnp.sqrt(jax.nn.relu(tmp))

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    """Kernel matrix.

    Either provided by user or recomputed from :attr:`cost_matrix`.
    """
    if self._kernel_matrix is None:
      return jnp.exp(-self._cost_matrix * self.inv_scale_cost / self.epsilon)
    return self._kernel_matrix ** self.inv_scale_cost

  @property
  def epsilon_scheduler(self) -> eps_scheduler.Epsilon:
    """Epsilon scheduler."""
    if isinstance(self._epsilon_init, eps_scheduler.Epsilon):
      return self._epsilon_init
    if self._relative_epsilon is None:
      if self._epsilon_init is not None:
        return eps_scheduler.Epsilon(self._epsilon_init)
      multiplier = eps_scheduler.DEFAULT_EPSILON_SCALE
      scale = jax.lax.stop_gradient(self.std_cost_matrix)
      return eps_scheduler.Epsilon(target=multiplier * scale)

    if self._relative_epsilon == "std":
      scale = jax.lax.stop_gradient(self.std_cost_matrix)
    elif self._relative_epsilon == "mean":
      scale = jax.lax.stop_gradient(self.mean_cost_matrix)
    else:
      raise ValueError(f"Invalid relative epsilon: {self._relative_epsilon}.")

    multiplier = (
        eps_scheduler.DEFAULT_EPSILON_SCALE
        if self._epsilon_init is None else self._epsilon_init
    )
    return eps_scheduler.Epsilon(target=multiplier * scale)

  @property
  def epsilon(self) -> float:
    """Epsilon regularization value."""
    return self.epsilon_scheduler.target

  @property
  def shape(self) -> Tuple[int, int]:
    """Shape of the geometry."""
    mat = (
        self._kernel_matrix if self._cost_matrix is None else self._cost_matrix
    )
    if mat is not None:
      return mat.shape
    return 0, 0

  @property
  def is_symmetric(self) -> bool:
    """Whether geometry cost/kernel is a symmetric matrix."""
    mat = self.kernel_matrix if self.cost_matrix is None else self.cost_matrix
    return self.is_square and jnp.all(mat == mat.T)

  @property
  def is_square(self) -> bool:
    """Whether geometry cost/kernel is a square matrix."""
    n, m = self.shape
    return (n == m)

  @property
  def inv_scale_cost(self) -> jnp.ndarray:
    """Compute and return inverse of scaling factor for cost matrix."""
    if self._scale_cost == "max_cost":
      return 1.0 / jnp.max(self._cost_matrix)
    if self._scale_cost == "mean":
      return 1.0 / jnp.mean(self._cost_matrix)
    if self._scale_cost == "median":
      return 1.0 / jnp.median(self._cost_matrix)
    if isinstance(self._scale_cost, (int, float)):
      return 1.0 / self._scale_cost
    raise ValueError(f"Scaling {self._scale_cost} not implemented.")

  @property
  def diag_cost(self) -> jnp.ndarray:
    """Diagonal of the cost matrix."""
    assert self.is_square, "Cost matrix must be square to compute diagonal."
    return jnp.diag(self.cost_matrix)

  def apply_lse_kernel(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      eps: float,
      vec: jnp.ndarray = None,
      axis: int = 0
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Apply :attr:`kernel_matrix` in log domain.

    This function applies the ground geometry's kernel in log domain, using
    a stabilized formulation.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      eps: float, regularization strength
      vec: jnp.ndarray [num_a or num_b,] , when not None, this has the effect of
        doing log-Kernel computations with an addition elementwise
        multiplication of exp(g / eps) by a vector. This is carried out by
        adding weights to the log-sum-exp function, and needs to handle signs
        separately.
      axis: summing over axis 0 when doing (2), or over axis 1 when doing (1)

    Returns:
      A jnp.ndarray corresponding to output above, depending on axis.
    """
    w_res, w_sgn = self._softmax(f, g, eps, vec, axis)
    remove = f if axis == 1 else g
    return w_res - jnp.where(jnp.isfinite(remove), remove, 0), w_sgn

  def apply_kernel(
      self,
      vec: jnp.ndarray,
      eps: Optional[float] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Apply :attr:`kernel_matrix` on positive scaling vector.

    Args:
      vec: jnp.ndarray [num_a or num_b] , scaling of size num_rows or
        num_cols of kernel_matrix
      eps: passed for consistency, not used yet.
      axis: standard kernel product if axis is 1, transpose if 0.

    Returns:
      a jnp.ndarray corresponding to output above, depending on axis.
    """
    if eps is None:
      kernel = self.kernel_matrix
    else:
      kernel = self.kernel_matrix ** (self.epsilon / eps)
    kernel = kernel if axis == 1 else kernel.T

    return jnp.dot(kernel, vec)

  def marginal_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Output marginal of transportation matrix from potentials."""
    h = (f if axis == 1 else g)
    z = self.apply_lse_kernel(f, g, self.epsilon, axis=axis)[0]
    return jnp.exp((z + h) / self.epsilon)

  def marginal_from_scalings(
      self,
      u: jnp.ndarray,
      v: jnp.ndarray,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Output marginal of transportation matrix from scalings."""
    u, v = (v, u) if axis == 0 else (u, v)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray
  ) -> jnp.ndarray:
    """Output transport matrix from potentials."""
    return jnp.exp(self._center(f, g) / self.epsilon)

  def transport_from_scalings(
      self, u: jnp.ndarray, v: jnp.ndarray
  ) -> jnp.ndarray:
    """Output transport matrix from pair of scalings."""
    return self.kernel_matrix * u[:, jnp.newaxis] * v[jnp.newaxis, :]

  def update_potential(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      log_marginal: jnp.ndarray,
      iteration: Optional[int] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Carry out one Sinkhorn update for potentials, i.e. in log space.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      log_marginal: targeted marginal
      iteration: used to compute epsilon from schedule, if provided.
      axis: axis along which the update should be carried out.

    Returns:
      new potential value, g if axis=0, f if axis is 1.
    """
    eps = self.epsilon_scheduler(iteration)
    app_lse = self.apply_lse_kernel(f, g, eps, axis=axis)[0]
    return eps * log_marginal - jnp.where(jnp.isfinite(app_lse), app_lse, 0)

  def update_scaling(
      self,
      scaling: jnp.ndarray,
      marginal: jnp.ndarray,
      iteration: Optional[int] = None,
      axis: int = 0,
  ) -> jnp.ndarray:
    """Carry out one Sinkhorn update for scalings, using kernel directly.

    Args:
      scaling: jnp.ndarray of num_a or num_b positive values.
      marginal: targeted marginal
      iteration: used to compute epsilon from schedule, if provided.
      axis: axis along which the update should be carried out.

    Returns:
      new scaling vector, of size num_b if axis=0, num_a if axis is 1.
    """
    eps = self.epsilon_scheduler(iteration)
    app_kernel = self.apply_kernel(scaling, eps, axis=axis)
    return marginal / jnp.where(app_kernel > 0, app_kernel, 1.0)

  def _center(self, f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    return f[:, jnp.newaxis] + g[jnp.newaxis, :] - self.cost_matrix

  def _softmax(
      self, f: jnp.ndarray, g: jnp.ndarray, eps: float,
      vec: Optional[jnp.ndarray], axis: int
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply softmax row or column wise, weighted by vec."""
    if vec is not None:
      if axis == 0:
        vec = vec.reshape((-1, 1))
      lse_output = mu.logsumexp(
          self._center(f, g) / eps, b=vec, axis=axis, return_sign=True
      )
      return eps * lse_output[0], lse_output[1]

    lse_output = mu.logsumexp(
        self._center(f, g) / eps, axis=axis, return_sign=False
    )
    return eps * lse_output, jnp.array([1.0])

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray, vec: jnp.ndarray, axis: int
  ) -> jnp.ndarray:
    """Apply lse_kernel to arbitrary vector while keeping track of signs."""
    lse_res, lse_sgn = self.apply_lse_kernel(
        f, g, self.epsilon, vec=vec, axis=axis
    )
    lse_res += f if axis == 1 else g
    return lse_sgn * jnp.exp(lse_res / self.epsilon)

  def apply_transport_from_potentials(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Apply transport matrix computed from potentials to a (batched) vec.

    This approach does not instantiate the transport matrix itself, but uses
    instead potentials to apply the transport using apply_lse_kernel, therefore
    guaranteeing stability and lower memory footprint.

    Args:
      f: jnp.ndarray [num_a,] , potential of size num_rows of cost_matrix
      g: jnp.ndarray [num_b,] , potential of size num_cols of cost_matrix
      vec: jnp.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to potentials f, g, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.

    Returns:
      ndarray of the size of vec.
    """
    if vec.ndim == 1:
      return self._apply_transport_from_potentials(
          f, g, vec[jnp.newaxis, :], axis
      )[0, :]
    return self._apply_transport_from_potentials(f, g, vec, axis)

  @functools.partial(jax.vmap, in_axes=[None, None, None, 0, None])
  def _apply_transport_from_scalings(
      self, u: jnp.ndarray, v: jnp.ndarray, vec: jnp.ndarray, axis: int
  ):
    u, v = (u, v * vec) if axis == 1 else (v, u * vec)
    return u * self.apply_kernel(v, eps=self.epsilon, axis=axis)

  def apply_transport_from_scalings(
      self,
      u: jnp.ndarray,
      v: jnp.ndarray,
      vec: jnp.ndarray,
      axis: int = 0
  ) -> jnp.ndarray:
    """Apply transport matrix computed from scalings to a (batched) vec.

    This approach does not instantiate the transport matrix itself, but
    relies instead on the apply_kernel function.

    Args:
      u: jnp.ndarray [num_a,] , scaling of size num_rows of cost_matrix
      v: jnp.ndarray [num_b,] , scaling of size num_cols of cost_matrix
      vec: jnp.ndarray [batch, num_a or num_b], vector that will be multiplied
        by transport matrix corresponding to scalings u, v, and geom.
      axis: axis to differentiate left (0) or right (1) multiply.

    Returns:
      ndarray of the size of vec.
    """
    if vec.ndim == 1:
      return self._apply_transport_from_scalings(
          u, v, vec[jnp.newaxis, :], axis
      )[0, :]
    return self._apply_transport_from_scalings(u, v, vec, axis)

  def potential_from_scaling(self, scaling: jnp.ndarray) -> jnp.ndarray:
    """Compute dual potential vector from scaling vector.

    Args:
      scaling: vector.

    Returns:
      a vector of the same size.
    """
    return self.epsilon * jnp.log(scaling)

  def scaling_from_potential(self, potential: jnp.ndarray) -> jnp.ndarray:
    """Compute scaling vector from dual potential.

    Args:
      potential: vector.

    Returns:
      a vector of the same size.
    """
    finite = jnp.isfinite(potential)
    return jnp.where(
        finite, jnp.exp(jnp.where(finite, potential / self.epsilon, 0.0)), 0.0
    )

  def apply_square_cost(self, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Apply elementwise-square of cost matrix to array (vector or matrix).

    This function applies the ground geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b], when the cost matrix itself is computed as a
    squared-Euclidean distance between vectors, and therefore admits an
    explicit low-rank factorization.

    Args:
      arr: array.
      axis: axis of the array on which the cost matrix should be applied.

    Returns:
      An array, [num_b, p] if axis=0 or [num_a, p] if axis=1.
    """
    return self.apply_cost(arr, axis=axis, fn=lambda x: x ** 2)

  def apply_cost(
      self,
      arr: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    """Apply :attr:`cost_matrix` to array (vector or matrix).

    This function applies the ground geometry's cost matrix, to perform either
    output = C arr (if axis=1)
    output = C' arr (if axis=0)
    where C is [num_a, num_b]

    Args:
      arr: jnp.ndarray [num_a or num_b, p], vector that will be multiplied by
        the cost matrix.
      axis: standard cost matrix if axis=1, transpose if 0
      fn: function to apply to cost matrix element-wise before the dot product
      is_linear: Whether ``fn`` is linear.

    Returns:
      An array, [num_b, p] if axis=0 or [num_a, p] if axis=1
    """
    if arr.ndim == 1:
      return self._apply_cost_to_vec(arr, axis=axis, fn=fn, is_linear=is_linear)
    app = functools.partial(
        self._apply_cost_to_vec, axis=axis, fn=fn, is_linear=is_linear
    )
    return jax.vmap(app, in_axes=1, out_axes=1)(arr)

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    """Apply ``[num_a, num_b]`` fn(cost) (or transpose) to vector.

    Args:
      vec: jnp.ndarray [num_a,] ([num_b,] if axis=1) vector
      axis: axis on which the reduction is done.
      fn: function optionally applied to cost matrix element-wise, before the
        doc product
      is_linear: Whether ``fn`` is linear.

    Returns:
      A jnp.ndarray corresponding to cost x vector
    """
    del is_linear
    matrix = self.cost_matrix.T if axis == 0 else self.cost_matrix
    if fn is not None:
      matrix = fn(matrix)
    return jnp.dot(matrix, vec)

  @property
  def dtype(self) -> jnp.dtype:
    """The data type."""
    if self._cost_matrix is not None:
      return self._cost_matrix.dtype
    return self._kernel_matrix.dtype

  def tree_flatten(self):
    return (
        self._cost_matrix,
        self._kernel_matrix,
        self._epsilon_init,
    ), {
        "scale_cost": self._scale_cost,
        "relative_epsilon": self._relative_epsilon,
    }

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    cost, kernel, epsilon = children
    return cls(cost, kernel_matrix=kernel, epsilon=epsilon, **aux_data)


@jax.tree_util.register_pytree_node_class
class Grid(Geometry):
  r"""Class describing the geometry of points taken in a Cartesian product.

  This class implements a geometry in which probability measures are supported
  on a :math:`d`-dimensional Cartesian grid, a Cartesian product of :math:`d`
  lists of values, each list being itself of size :math:`n_i`.

  The transportation cost between points in the grid is assumed to be separable,
  namely a sum of coordinate-wise cost functions, as in:

  .. math::

    cost(x,y) = \sum_{i=1}^d cost_i(x_i, y_i)

  where :math:`cost_i`: R x R → R.

  In such a regime, and despite the fact that the total number :math:`n_{total}`
  of points in the grid is exponential :math:`d` (namely :math:`\prod_i n_i`),
  applying a kernel in the context of regularized optimal transport can be
  carried out in time that is of the order of :math:`n_{total}^{(1+1/d)}` using
  convolutions, either in the original domain or log-space domain. This class
  precomputes :math:`d` :math:`n_i` x :math:`n_i` cost matrices (one per
  dimension) and implements these two operations by carrying out these
  convolutions one dimension at a time.

  Args:
    x: list of arrays of varying sizes, describing the locations of the grid.
      Locations are provided as a list of arrays, that is :math:`d`
      vectors of (possibly varying) size :math:`n_i`. The resulting grid
      is the Cartesian product of these vectors.
    grid_size: tuple of integers describing grid sizes, namely
      :math:`(n_1,...,n_d)`. This will only be used if x is None.
      In that case the grid will be assumed to lie in the hypercube
      :math:`[0,1]^d`, with the :math:`d` dimensions, described as points
      regularly sampled in :math:`[0,1]`.
    cost_fns: a sequence of :math:`d` cost functions, each being a cost taking
      two reals as inputs to output a real number.
    num_a: total size of grid. This parameters will be computed from other
      inputs.
    grid_dimension: dimension of grid. This parameters will be computed from
      other inputs.
    kwargs: keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
  """

  def __init__(
      self,
      x: Optional[Sequence[jnp.ndarray]] = None,
      grid_size: Optional[Sequence[int]] = None,
      cost_fns: Optional[Sequence[cost_module.CostFn]] = None,
      num_a: Optional[int] = None,
      grid_dimension: Optional[int] = None,
      **kwargs: Any,
  ):
    super().__init__(**kwargs)
    if (
        grid_size is not None and x is not None and num_a is not None and
        grid_dimension is not None
    ):
      self.grid_size = tuple(map(int, grid_size))
      self.x = x
      self.num_a = num_a
      self.grid_dimension = grid_dimension
    elif x is not None:
      self.x = x
      self.grid_size = tuple(xs.shape[0] for xs in x)
      self.num_a = np.prod(np.array(self.grid_size))
      self.grid_dimension = len(self.x)
    elif grid_size is not None:
      self.grid_size = tuple(map(int, grid_size))
      self.x = tuple(jnp.linspace(0, 1, n) for n in self.grid_size)
      self.num_a = np.prod(np.array(grid_size))
      self.grid_dimension = len(self.grid_size)
    else:
      raise ValueError("Input either grid_size tuple or grid locations x.")

    if cost_fns is None:
      cost_fns = [cost_module.SqEuclidean()]
    self.cost_fns = cost_fns
    self.kwargs = {
        "num_a": self.num_a,
        "grid_size": self.grid_size,
        "grid_dimension": self.grid_dimension,
        "relative_epsilon": self._relative_epsilon,
    }

  @property
  def geometries(self) -> List[Geometry]:
    """Cost matrices along each dimension of the grid."""
    geometries = []
    for dimension, cost_fn in itertools.zip_longest(
        range(self.grid_dimension), self.cost_fns, fillvalue=self.cost_fns[-1]
    ):
      x_values = self.x[dimension][:, jnp.newaxis]
      cost_matrix = cost_fn.all_pairs(x_values, x_values)
      geom = Geometry(
          cost_matrix=cost_matrix,
          epsilon=self._epsilon_init,
      )
      geometries.append(geom)
    return geometries

  @property
  def shape(self) -> Tuple[int, int]:
    return self.num_a, self.num_a

  @property
  def is_symmetric(self) -> bool:
    return True

  def apply_lse_kernel(
      self,
      f: jnp.ndarray,
      g: jnp.ndarray,
      eps: float,
      vec: Optional[jnp.ndarray] = None,
      axis: int = 0
  ) -> jnp.ndarray:
    """Apply grid kernel in log space. See notes in parent class for use case.

    Reshapes vector inputs below as grids, applies kernels onto each slice, and
    then expands the outputs as vectors.

    Args:
      f: jnp.ndarray, a vector of potentials
      g: jnp.ndarray, a vector of potentials
      eps: float, regularization strength
      vec: jnp.ndarray, if needed, a vector onto which apply the kernel weighted
        by f and g.
      axis: axis (0 or 1) along which summation should be carried out.

    Returns:
      a vector, the result of kernel applied in lse space onto vec.
    """
    f, g = jnp.reshape(f, self.grid_size), jnp.reshape(g, self.grid_size)

    if vec is not None:
      vec = jnp.reshape(vec, self.grid_size)

    if axis == 0:
      f, g = g, f

    for dimension in range(self.grid_dimension):
      g, vec = self._apply_lse_kernel_one_dimension(dimension, f, g, eps, vec)
      g -= jnp.where(jnp.isfinite(f), f, 0)

    if vec is None:
      vec = jnp.array(1.0)
    return g.ravel(), vec.ravel()

  def _apply_lse_kernel_one_dimension(self, dimension, f, g, eps, vec=None):
    """Permute axis & apply the kernel on a single slice."""
    indices = np.arange(self.grid_dimension)
    indices[dimension], indices[0] = 0, dimension
    f, g = jnp.transpose(f, indices), jnp.transpose(g, indices)

    if vec is not None:
      vec = jnp.transpose(vec, indices)

      def loop_body_vec(k, carry):
        softmax_res, softmax_sgn = carry
        f_k = f[:, k]
        g_k = g[:, k]
        vec_k = vec[:, k]

        slice_cost = (
            f_k[:, jnp.newaxis] + g_k[jnp.newaxis, :] -
            self.geometries[dimension].cost_matrix
        ) / eps

        res_k, sgn_k = mu.logsumexp(
            slice_cost, b=vec_k[jnp.newaxis, :], axis=1, return_sign=True
        )

        softmax_res = softmax_res.at[:, k].set(eps * res_k)
        softmax_sgn = softmax_sgn.at[:, k].set(sgn_k)

        return softmax_res, softmax_sgn

      init_res = jnp.zeros_like(f)
      init_sgn = jnp.zeros_like(f, dtype=jnp.int32)
      softmax_res, softmax_sgn = lax.fori_loop(
          0, f.shape[1], loop_body_vec, (init_res, init_sgn)
      )

      return eps * jnp.transpose(softmax_res, indices), jnp.transpose(softmax_sgn, indices)
    else:
      def loop_body_simple(k, carry_res):
        f_k = f[:, k]
        g_k = g[:, k]

        slice_cost = (
            f_k[:, jnp.newaxis] + g_k[jnp.newaxis, :] -
            self.geometries[dimension].cost_matrix
        ) / eps

        res_k = mu.logsumexp(slice_cost, axis=1)
        carry_res = carry_res.at[:, k].set(eps * res_k)

        return carry_res

      init_res = jnp.zeros_like(f)
      softmax_res = lax.fori_loop(0, f.shape[1], loop_body_simple, init_res)

      return jnp.transpose(softmax_res, indices), None

  def _apply_cost_to_vec(
      self,
      vec: jnp.ndarray,
      axis: int = 0,
      fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      is_linear: bool = False,
  ) -> jnp.ndarray:
    r"""Apply grid's cost matrix (without instantiating it) to a vector.

    The `apply_cost` operation on grids rests on the following identity.
    If it were to be cast as a [num_a, num_a] matrix, the corresponding cost
    matrix :math:`C` would be a sum of `grid_dimension` matrices, each of the
    form (here for the j-th slice)
    :math:`\tilde{C}_j : = 1_{n_1} \otimes \dots \otimes C_j \otimes 1_{n_d}`
    where each :math:`1_{n}` is the :math:`n\times n` square matrix full of 1's.

    Applying :math:`\tilde{C}_j` on a vector grid consists in carrying a tensor
    multiplication on the dimension of that vector reshaped as a grid, followed
    by a summation on all other axis while keeping dimensions.

    Args:
      vec: jnp.ndarray, flat vector of total size prod(grid_size).
      axis: axis 0 if applying transpose costs, 1 if using the original cost.
      fn: function optionally applied to cost matrix element-wise, before the
        dot product.
      is_linear: TODO.

    Returns:
      A jnp.ndarray corresponding to cost x matrix
    """
    del fn, is_linear
    vec = jnp.reshape(vec, self.grid_size)
    accum_vec = jnp.zeros_like(vec)
    indices = list(range(1, self.grid_dimension))
    for dimension, geom in enumerate(self.geometries):
      cost = geom.cost_matrix
      ind = indices.copy()
      ind.insert(dimension, 0)
      if axis == 0:
        cost = cost.T
      accum_vec += jnp.sum(
          jnp.tensordot(cost, vec, axes=([0], [dimension])),
          axis=indices,
          keepdims=True
      ).transpose(ind)
    return accum_vec.ravel()

  def apply_kernel(
      self,
      vec: jnp.ndarray,
      eps: Optional[float] = None,
      axis: Optional[int] = None
  ) -> jnp.ndarray:
    """Apply grid kernel on scaling vector.

    See notes in parent class for use.

    Reshapes scaling vector as a grid, applies kernels onto each slice, and
    then ravels backs the output as a vector.

    Args:
      vec: jnp.ndarray, a vector of scaling (>0) values.
      eps: float, regularization strength
      axis: axis (0 or 1) along which summation should be carried out.

    Returns:
      a vector, the result of kernel applied onto scaling.
    """
    vec = jnp.reshape(vec, self.grid_size)
    indices = list(range(1, self.grid_dimension))
    for dimension, geom in enumerate(self.geometries):
      kernel = geom.kernel_matrix
      kernel = kernel if eps is None else kernel ** (self.epsilon / eps)
      ind = indices.copy()
      ind.insert(dimension, 0)
      vec = jnp.tensordot(kernel, vec, axes=([0], [dimension])).transpose(ind)
    return vec.ravel()

  def transport_from_potentials(
      self, f: jnp.ndarray, g: jnp.ndarray, axis: int = 0
  ) -> NoReturn:
    """Not implemented, use :meth:`apply_transport_from_potentials` instead."""
    raise ValueError(
        "Grid geometry cannot instantiate a transport matrix, use",
        " apply_transport_from_potentials(...) if you wish to ",
        " apply the transport matrix to a vector, or use a point "
        " cloud geometry instead"
    )

  def transport_from_scalings(
      self, f: jnp.ndarray, g: jnp.ndarray, axis: int = 0
  ) -> NoReturn:
    """Not implemented, use :meth:`apply_transport_from_scalings` instead."""
    raise ValueError(
        "Grid geometry cannot instantiate a transport matrix, use ",
        "apply_transport_from_scalings(...) if you wish to ",
        "apply the transport matrix to a vector, or use a point "
        "cloud geometry instead."
    )

  @property
  def cost_matrix(self) -> jnp.ndarray:
    """Not implemented."""
    raise NotImplementedError(
        "Instantiating cost matrix is not implemented for grids."
    )

  @property
  def kernel_matrix(self) -> jnp.ndarray:
    """Not implemented."""
    raise NotImplementedError(
        "Instantiating kernel matrix is not implemented for grids."
    )

  @property
  def dtype(self) -> jnp.dtype:
    return self.x[0].dtype

  def coarsen(self, factor: int) -> "Grid":
    """Create a coarsened version of this grid by downsampling.

    Args:
      factor: Integer downsampling factor. Grid size will be divided by this.

    Returns:
      A new Grid instance with coarsened grid_size.
    """
    new_grid_size = tuple(max(1, s // factor) for s in self.grid_size)
    return Grid(
        grid_size=new_grid_size,
        cost_fns=self.cost_fns,
        epsilon=self._epsilon_init,
        relative_epsilon=self._relative_epsilon,
        scale_cost=self._scale_cost
    )

  def upsample(self, arr: jnp.ndarray, target_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Upsample a potential array to a target grid shape.

    Args:
      arr: Potential array on the current grid (flattened).
      target_shape: Target grid shape to upsample to.

    Returns:
      Upsampled array (flattened) matching the target grid.
    """
    current_shape = self.grid_size
    arr_reshaped = arr.reshape(current_shape)
    upsampled = jax.image.resize(
        arr_reshaped,
        shape=target_shape,
        method='bilinear'
    )
    return upsampled.ravel()

  def tree_flatten(self):
    return (self.x, self.cost_fns, self._epsilon_init), self.kwargs

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    x, cost_fns, epsilon = children
    return cls(x, cost_fns=cost_fns, epsilon=epsilon, **aux_data)