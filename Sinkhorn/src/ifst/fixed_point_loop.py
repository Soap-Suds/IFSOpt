# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Any, Callable

import jax
import jax.numpy as jnp


def fixpoint_iter(
    cond_fn: Callable[[int, Any, Any], bool],
    body_fn: Callable[[Any, Any, Any, Any], Any], min_iterations: int,
    max_iterations: int, inner_iterations: int, constants: Any, state: Any
):
  """Implementation of a fixed point loop.

  This fixed point loop iterator applies ``body_fn`` to a tuple
  ``(iteration, constants, state, compute_error)`` to output a new state, using
  context provided in iteration and constants.

  ``body_fn`` is iterated (inner_iterations -1) times, and one last time with
  the ``compute_error`` flag to ``True``, indicating that additional
  computational effort can be spent on recalculating the latest error
  (``errors`` are stored as the first element of the state tuple).

  upon termination of these ``inner_iterations``, the loop is continued if
  iteration is smaller than ``min_iterations``, stopped if equal/larger than
  ``max_iterations``, and interrupted if ``cond_fn`` returns False.

  Args:
    cond_fn : termination condition function
    body_fn : body loop instructions
    min_iterations : lower bound on the total amount of fixed point iterations
    max_iterations : upper bound on the total amount of fixed point iterations
    inner_iterations : number of iterations ``body_fn`` will be executed
      successively before calling ``cond_fn``.
    constants : constant (during loop) parameters passed on to body
    state : state variable

  Returns:
    outputs state returned by ``body_fn`` upon termination.
  """
  force_scan = (min_iterations == max_iterations)

  compute_error_flags = jnp.arange(inner_iterations) == inner_iterations - 1

  def max_cond_fn(iteration_state):
    iteration, state = iteration_state
    return jnp.logical_and(
        iteration < max_iterations,
        jnp.logical_or(
            iteration < min_iterations, cond_fn(iteration, constants, state)
        )
    )

  def unrolled_body_fn(iteration_state):

    def one_iteration(iteration_state, compute_error):
      iteration, state = iteration_state
      state = body_fn(iteration, constants, state, compute_error)
      iteration += 1
      return (iteration, state), None

    iteration_state, _ = jax.lax.scan(
        one_iteration, iteration_state, compute_error_flags
    )
    return (iteration_state, None) if force_scan else iteration_state

  if force_scan:
    (_, state), _ = jax.lax.scan(
        lambda carry, x: unrolled_body_fn(carry), (0, state),
        None,
        length=max_iterations // inner_iterations
    )
  else:
    _, state = jax.lax.while_loop(max_cond_fn, unrolled_body_fn, (0, state))
  return state