# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import geometry as geom_module


@jax.tree_util.register_pytree_node_class
class LinearProblem:
  r"""Linear OT problem.

  This class describes the main ingredients appearing in a linear OT problem.
  Namely, a ``geom`` object (including cost structure/points) describing point
  clouds or the support of measures, followed by probability masses ``a`` and
  ``b``. Unbalancedness of the problem is also kept track of, through two
  coefficients ``tau_a`` and ``tau_b``, which are both kept between 0 and 1
  (1 corresponding to a balanced OT problem).

  Args:
    geom: The ground geometry cost of the linear problem.
    a: The first marginal. If ``None``, it will be uniform.
    b: The second marginal. If ``None``, it will be uniform.
    tau_a: If :math:`<1`, defines how much unbalanced the problem is
      on the first marginal.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
      on the second marginal.
    init_f: Initial dual potential for the first marginal. If ``None``,
      the solver will initialize it.
    init_g: Initial dual potential for the second marginal. If ``None``,
      the solver will initialize it.
  """

  def __init__(
      self,
      geom: geom_module.Geometry,
      a: Optional[jnp.ndarray] = None,
      b: Optional[jnp.ndarray] = None,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      init_f: Optional[jnp.ndarray] = None,
      init_g: Optional[jnp.ndarray] = None
  ):
    self.geom = geom
    self._a = a
    self._b = b
    self.tau_a = tau_a
    self.tau_b = tau_b
    self.init_f = init_f
    self.init_g = init_g

  @property
  def a(self) -> jnp.ndarray:
    """First marginal."""
    if self._a is not None:
      return self._a
    n, _ = self.geom.shape
    return jnp.full((n,), fill_value=1.0 / n, dtype=self.dtype)

  @property
  def b(self) -> jnp.ndarray:
    """Second marginal."""
    if self._b is not None:
      return self._b
    _, m = self.geom.shape
    return jnp.full((m,), fill_value=1.0 / m, dtype=self.dtype)

  @property
  def is_balanced(self) -> bool:
    """Whether the problem is balanced."""
    return self.tau_a == 1.0 and self.tau_b == 1.0

  @property
  def is_uniform(self) -> bool:
    """True if no weights ``a,b`` were passed, and have defaulted to uniform."""
    return self._a is None and self._b is None

  @property
  def is_equal_size(self) -> bool:
    """True if square shape, i.e. ``n == m``."""
    return self.geom.shape[0] == self.geom.shape[1]

  @property
  def is_assignment(self) -> bool:
    """True if assignment problem."""
    return self.is_equal_size and self.is_uniform and self.is_balanced

  @property
  def epsilon(self) -> float:
    """Entropic regularization."""
    return self.geom.epsilon

  @property
  def dtype(self) -> jnp.dtype:
    """The data type of the geometry."""
    return self.geom.dtype

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return ([self.geom, self._a, self._b, self.init_f, self.init_g], {
        "tau_a": self.tau_a,
        "tau_b": self.tau_b
    })

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "LinearProblem":
    geom, a, b, init_f, init_g = children
    return cls(geom=geom, a=a, b=b, init_f=init_f, init_g=init_g, **aux_data)