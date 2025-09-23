# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

import abc

import jax.numpy as jnp
import jax.tree_util as jtu


@jtu.register_pytree_node_class
class CostFn(abc.ABC):
  """Base class for all costs."""

  @abc.abstractmethod
  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute cost between :math:`x` and :math:`y`.

    Args:
      x: Array.
      y: Array.

    Returns:
      The cost.
    """

  def all_pairs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of all pairwise costs.

    Args:
      x: Array of shape ``[n, ...]``.
      y: Array of shape ``[m, ...]``.

    Returns:
      Array of shape ``[n, m]`` of cost evaluations.
    """
    import jax
    return jax.vmap(lambda x_: jax.vmap(lambda y_: self(x_, y_))(y))(x)

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@jtu.register_pytree_node_class
class SqEuclidean(CostFn):
  r"""Squared Euclidean distance.

  Implemented as a translation invariant cost, :math:`h(z) = \|z\|^2`.
  """

  def norm(self, x: jnp.ndarray) -> jnp.ndarray:
    """Compute squared Euclidean norm for vector."""
    return jnp.sum(x ** 2, axis=-1)

  def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute minus twice the dot-product between vectors."""
    cross_term = -2.0 * jnp.vdot(x, y)
    return self.norm(x) + self.norm(y) + cross_term