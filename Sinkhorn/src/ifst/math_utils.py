# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

import functools
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_jvp, nondiff_argnums=[1, 2, 3])
@functools.partial(jax.jit, static_argnames=("ord", "axis", "keepdims"))
def norm(
    x: jnp.ndarray,
    ord: Union[int, str, None] = None,
    axis: Union[None, Sequence[int], int] = None,
    keepdims: bool = False
) -> jnp.ndarray:
  """Computes order ord norm of vector, using `jnp.linalg` in forward pass.

  Evaluations of distances between a vector and itself using translation
  invariant costs, typically  norms, result in functions of the form
  ``lambda x : jnp.linalg.norm(x-x)``. Such functions output `NaN` gradients,
  because they involve computing the derivative of a negative exponent of 0
  (e.g. when differentiating the Euclidean norm, one gets a 0-denominator in the
  expression, see e.g. https://github.com/jax-ml/jax/issues/6484 for context).

  While this makes sense mathematically, in the context of optimal transport
  such distances between a point and itself can be safely ignored when they
  contribute to an OT cost (when, for instance, computing Sinkhorn divergences,
  involving computing the OT cost of a point cloud with itself).

  To avoid such `NaN` values, this custom norm implementation uses the
  double-where trick, to avoid having branches that output any `NaN`, and
  safely output a 0 instead.

  Args:
    x: Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
      is None. If both `axis` and `ord` are None, the 2-norm of ``x.ravel``
      will be returned.
    ord: `{non-zero int, jnp.inf, -jnp.inf, 'fro', 'nuc'}`, Order of the norm.
      The default is `None`, which is equivalent to `2` for vectors.
    axis: `{None, int, 2-tuple of ints}`, optional. If `axis` is an integer, it
      specifies the axis of `x` along which to compute the vector norms.
      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices, and
      the matrix norms of these matrices are computed.  If `axis` is None then
      either a vector norm (when `x` is 1-D) or a matrix norm (when `x` is 2-D)
      is returned. The default is None.
    keepdims: If set to True, the axes which are normed over are left in the
      result as dimensions with size one.  With this option the result will
      broadcast correctly against the original `x`.

  Returns:
    float or ndarray, Norm of the matrix or vector(s).
  """
  return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


@norm.defjvp
def norm_jvp(ord, axis, keepdims, primals, tangents):
  """Custom_jvp for norm, that returns 0.0 when evaluated at 0."""
  x, = primals
  x_is_zero = jnp.all(jnp.logical_not(x))
  clean_x = jnp.where(x_is_zero, jnp.ones_like(x), x)
  primals, tangents = jax.jvp(
      functools.partial(jnp.linalg.norm, ord=ord, axis=axis, keepdims=keepdims),
      (clean_x,), tangents
  )
  return primals, jnp.where(x_is_zero, 0.0, tangents)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 4))
def logsumexp(
    mat, axis=None, keepdims=False, b=None, return_sign=False
):
  return jax.scipy.special.logsumexp(
      mat, axis=axis, keepdims=keepdims, b=b, return_sign=return_sign
  )


@logsumexp.defjvp
def logsumexp_jvp(axis, keepdims, return_sign, primals, tangents):
  """Custom derivative rule for lse that does not blow up with -inf.

  This logsumexp implementation uses the standard jax one in forward mode but
  implements a custom rule to differentiate. Given the preference of jax for
  jvp over vjp, and the fact that this is a simple linear rule, jvp is used.
  This custom differentiation address issues when the output of lse is
  -inf (which corresponds to the case where all inputs in a slice are -inf,
  which happens typically when ``a`` or ``b`` weight vectors have zeros.)

  Although both exp(lse) and its derivative should be 0, automatic
  differentiation returns a NaN derivative because of a -inf - (-inf) operation
  appearing in the definition of centered_exp below. This is corrected in the
  implementation below.

  Args:
    axis: argument from original logsumexp
    keepdims: argument from original logsumexp
    return_sign: argument from original logsumexp
    primals: mat and b, the two arguments against which we differentiate.
    tangents: of same size as mat and b.

  Returns:
    original primal outputs + their tangent.
  """
  mat, b = primals
  tan_mat, tan_b = tangents
  lse = logsumexp(mat, axis, keepdims, b, return_sign)
  if return_sign:
    lse, sign = lse
  lse = jnp.where(jnp.isfinite(lse), lse, 0.0)

  if axis is not None:
    centered_exp = jnp.exp(mat - jnp.expand_dims(lse, axis=axis))
  else:
    centered_exp = jnp.exp(mat - lse)

  if b is None:
    res = jnp.sum(centered_exp * tan_mat, axis=axis, keepdims=keepdims)
  else:
    res = jnp.sum(b * centered_exp * tan_mat, axis=axis, keepdims=keepdims)
    res += jnp.sum(tan_b * centered_exp, axis=axis, keepdims=keepdims)
  if return_sign:
    return (lse, sign), (sign * res, jnp.zeros_like(sign))
  return lse, res