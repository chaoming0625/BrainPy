# -*- coding: utf-8 -*-


import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import core
from jax.lax import scan
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching
from jax.lib import xla_client

from numba.cuda.random import create_xoroshiro128p_states
x_ops = xla_client.ops
x_shape = xla_client.Shape.array_shape

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'uniform',
  'uniform2',
]

uniform_p = core.Primitive("uniform")


def uniform(length, seed):
  # bind operator
  return uniform_p.bind(seed=seed, length=length)


def _mmm_abstract(*, length, seed):
  return ShapedArray(shape=(length,), dtype=np.float32)


def _mmm_translation(c, *, length, seed, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')

    opaque = gpu_ops.build_rand_sample_descriptor(length, seed)
    return x_ops.CustomCallWithLayout(
      c,
      b'uniform',
      operands=tuple(),
      operand_shapes_with_layout=tuple(),
      shape_with_layout=x_shape(np.dtype(np.float32), (length,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


uniform_p.def_abstract_eval(_mmm_abstract)
uniform_p.def_impl(partial(xla.apply_primitive, uniform_p))
xla.backend_specific_translations["gpu"][uniform_p] = partial(_mmm_translation, platform="gpu")


uniform2_p = core.Primitive("uniform2")


def uniform2(keys, length):
  return uniform2_p.bind(keys, length=length)


def _mmm2_abstract(keys, *, length):
  return ShapedArray(shape=(length * 3,), dtype=np.float32)


def _mmm2_translation(c, keys, *, length, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')

    opaque = gpu_ops.build_rand_sample_descriptor(length, length)
    return x_ops.CustomCallWithLayout(
      c,
      b'uniform2',
      operands=(keys, ),
      operand_shapes_with_layout=(c.get_shape(keys), ),
      shape_with_layout=x_shape(np.dtype(np.float32), (length * 3,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


uniform2_p.def_abstract_eval(_mmm2_abstract)
uniform2_p.def_impl(partial(xla.apply_primitive, uniform2_p))
xla.backend_specific_translations["gpu"][uniform2_p] = partial(_mmm2_translation, platform="gpu")







uniform3_p = core.Primitive("uniform3")


def uniform3(length, seed):
  # bind operator
  return uniform3_p.bind(seed=seed, length=length)


def _mmm3_abstract(*, length, seed):
  return ShapedArray(shape=(length,), dtype=np.float32)


def _mmm3_translation(c, *, length, seed, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')

    rngs = create_xoroshiro128p_states(length, seed)

    opaque = gpu_ops.build_rand_sample_descriptor(length, seed)
    return x_ops.CustomCallWithLayout(
      c,
      b'uniform2',
      operands=(rngs, ),
      operand_shapes_with_layout=(x_shape(np.dtype(np.float64), (length * 6,), (0,)), ),
      shape_with_layout=x_shape(np.dtype(np.float32), (length,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


uniform3_p.def_abstract_eval(_mmm3_abstract)
uniform3_p.def_impl(partial(xla.apply_primitive, uniform3_p))
xla.backend_specific_translations["gpu"][uniform3_p] = partial(_mmm3_translation, platform="gpu")
