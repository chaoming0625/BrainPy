# -*- coding: utf-8 -*-

__all__ = [
  'vector_matmul_mask',
]

import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

_vmmm_prim = core.Primitive("vector_matmul_mask")


def vector_matmul_mask(V, L, R, *, seed, p):
  if V.dtype != jnp.float32:
    raise ValueError(f'"V" must be a vector of float32, while we got {V.dtype}')
  if L.dtype != jnp.float32:
    raise ValueError(f'"L" must be a vector of float32, while we got {L.dtype}')
  if R.dtype != jnp.float32:
    raise ValueError(f'"R" must be a vector of float32, while we got {R.dtype}')
  assert V.ndim == 1
  assert L.ndim == 2
  assert R.ndim == 2
  assert V.shape[0] == L.shape[0]
  assert L.shape[1] == R.shape[0]

  # bind operator
  return _vmmm_prim.bind(V,
                         L,
                         R,
                         m=V.shape[0],
                         k=L.shape[1],
                         n=R.shape[1],
                         seed=seed,
                         p=p)


def _vmmm_abstract(V, L, R, *, m, k, n, seed, p):
  return ShapedArray(shape=(n,), dtype=R.dtype)


_vmmm_prim.def_abstract_eval(_vmmm_abstract)
_vmmm_prim.def_impl(partial(xla.apply_primitive, _vmmm_prim))


def _vmmm_translation(c, V, L, R, *, m, k, n, seed, p, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')
    opaque = gpu_ops.build_matmul_descriptor(m, k, n, seed, p)
    return x_ops.CustomCallWithLayout(
      c,
      b'vector_matmul_mask',
      operands=(V, L, R),
      operand_shapes_with_layout=(c.get_shape(V), c.get_shape(L), c.get_shape(R)),
      shape_with_layout=x_shape(np.dtype(c.get_shape(R).element_type()), (n,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


xla.backend_specific_translations["gpu"][_vmmm_prim] = partial(_vmmm_translation, platform="gpu")
