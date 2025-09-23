# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import problem as prob_module


@jax.tree_util.register_pytree_node_class
class SinkhornInitializer:
  """Base class for Sinkhorn initializers."""

  def init_fu(
      self,
      ot_prob: prob_module.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling f_u.

    Args:
      ot_prob: Linear OT problem.
      lse_mode: Return potential if ``True``, scaling if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      potential/scaling, array of size ``[n,]``.
    """
    raise NotImplementedError

  def init_gv(
      self,
      ot_prob: prob_module.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    """Initialize Sinkhorn potential/scaling g_v.

    Args:
      ot_prob: Linear OT problem.
      lse_mode: Return potential if ``True``, scaling if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      potential/scaling, array of size ``[m,]``.
    """
    raise NotImplementedError

  def __call__(
      self,
      ot_prob: prob_module.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize Sinkhorn potentials/scalings f_u and g_v.

    Args:
      ot_prob: Linear OT problem.
      lse_mode: Return potentials if ``True``, scalings if ``False``.
      rng: Random number generator for stochastic initializers.

    Returns:
      The initial potentials/scalings.
    """
    if rng is None:
      rng = jax.random.PRNGKey(0)
    rng_f, rng_g = jax.random.split(rng, 2)
    fu = self.init_fu(ot_prob, lse_mode=lse_mode, rng=rng_f)
    gv = self.init_gv(ot_prob, lse_mode=lse_mode, rng=rng_g)

    mask_value = -jnp.inf if lse_mode else 0.0
    fu = jnp.where(ot_prob.a > 0.0, fu, mask_value)
    gv = jnp.where(ot_prob.b > 0.0, gv, mask_value)
    return fu, gv

  def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
    return [], {}

  @classmethod
  def tree_unflatten(
      cls, aux_data: Dict[str, Any], children: Sequence[Any]
  ) -> "SinkhornInitializer":
    return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class DefaultInitializer(SinkhornInitializer):
  """Default initialization of Sinkhorn dual potentials/primal scalings."""

  def init_fu(
      self,
      ot_prob: prob_module.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    del rng
    return jnp.zeros_like(ot_prob.a) if lse_mode else jnp.ones_like(ot_prob.a)

  def init_gv(
      self,
      ot_prob: prob_module.LinearProblem,
      lse_mode: bool,
      rng: Optional[jax.Array] = None,
  ) -> jnp.ndarray:
    del rng
    return jnp.zeros_like(ot_prob.b) if lse_mode else jnp.ones_like(ot_prob.b)