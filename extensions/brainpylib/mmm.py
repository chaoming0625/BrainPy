# -*- coding: utf-8 -*-


import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import core
from jax.lax import scan
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching
from jax.lib import xla_client

x_ops = xla_client.ops
x_shape = xla_client.Shape.array_shape

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'masked_matmul',
]

_mmm_prim = core.Primitive("masked_matmul")


def masked_matmul(L, R, *, seed, p):
  if L.dtype != jnp.float32:
    raise ValueError(f'"L" must be a vector of float32, while we got {L.dtype}')
  if R.dtype != jnp.float32:
    raise ValueError(f'"R" must be a vector of float32, while we got {R.dtype}')
  assert L.ndim == 2
  assert R.ndim == 2
  assert L.shape[1] == R.shape[0]

  # bind operator
  return _mmm_prim.bind(L,
                        R,
                        m=L.shape[0],
                        k=L.shape[1],
                        n=R.shape[1],
                        seed=seed,
                        p=p)


def _mmm_abstract(L, R, *, m, k, n, seed, p):
  return ShapedArray(shape=(n,), dtype=R.dtype)


def _mmm_translation(c, L, R, *, m, k, n, seed, p, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')
    opaque = gpu_ops.build_matmul_descriptor(m, k, n, seed, p)
    return x_ops.CustomCallWithLayout(
      c,
      b'masked_matmul',
      operands=(L, R),
      operand_shapes_with_layout=(c.get_shape(L), c.get_shape(R)),
      shape_with_layout=x_shape(np.dtype(c.get_shape(R).element_type()), (n,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


def _mmm_batch(args, axes, *, m, k, n, seed, p):
  batch_axes, batch_args, non_batch_args = [], {}, {}
  for ax_i, ax in enumerate(axes):
    if ax is None:
      non_batch_args[f'ax{ax_i}'] = args[ax_i]
    else:
      batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jnp.moveaxis(args[ax_i], ax, 0)
      batch_axes.append(ax_i)

  def f(_, x):
    pars = tuple([(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
                  for i in range(len(axes))])
    return 0, _mmm_prim.bind(*pars, m=m, k=k, n=n, seed=seed, p=p)

  _, outs = scan(f, 0, batch_args)
  return outs, 0


_mmm_prim.def_abstract_eval(_mmm_abstract)
_mmm_prim.def_impl(partial(xla.apply_primitive, _mmm_prim))
batching.primitive_batchers[_mmm_prim] = _mmm_batch
xla.backend_specific_translations["gpu"][_mmm_prim] = partial(_mmm_translation, platform="gpu")
