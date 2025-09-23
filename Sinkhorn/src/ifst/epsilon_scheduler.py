# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Optional

import jax.numpy as jnp
import jax.tree_util as jtu

DEFAULT_EPSILON_SCALE = 0.05


@jtu.register_pytree_node_class
class Epsilon:
  r"""Scheduler class for the regularization parameter epsilon.

  An epsilon scheduler outputs a regularization strength, to be used by the
  Sinkhorn algorithm or variant, at any iteration count. That value is
  either the final, targeted regularization, or one that is larger, obtained by
  geometric decay of an initial multiplier.

  Args:
    target: The epsilon regularizer that is targeted.
    init: Initial value when using epsilon scheduling, understood as a multiple
      of the ``target``, following :math:`\text{init} \text{decay}^{\text{it}}`.
    decay: Geometric decay factor, :math:`\leq 1`.
  """

  def __init__(self, target: jnp.array, init: float = 1.0, decay: float = 1.0):
    assert decay <= 1.0, f"Decay must be <= 1, found {decay}."
    self.target = target
    self.init = init
    self.decay = decay

  def __call__(self, it: Optional[int]) -> jnp.array:
    """Intermediate regularizer value at a given iteration number.

    Args:
      it: Current iteration. If :obj:`None`, return :attr:`target`.

    Returns:
      The epsilon value at the iteration.
    """
    if it is None:
      return self.target
    multiple = jnp.maximum(self.init * (self.decay ** it), 1.0)
    return multiple * self.target

  def __repr__(self) -> str:
    return (
        f"{self.__class__.__name__}(target={self.target:.4f}, "
        f"init={self.init:.4f}, decay={self.decay:.4f})"
    )

  def tree_flatten(self):
    return (self.target,), {"init": self.init, "decay": self.decay}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, **aux_data)