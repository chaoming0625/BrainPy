# -*- coding: utf-8 -*-


import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import core
from jax.lax import scan
import jax.random as jr
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching
from jax.lib import xla_client
from .base import *

x_ops = xla_client.ops
x_shape = xla_client.Shape.array_shape

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'masked_matmul',
  'MATxMASK',
  'EventMATxMASK',
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

cuda_op_names = {
  '8K_8x256x512': b'mmm_8K_8x256x512',
  '8K_8x128x256': b'mmm_8K_8x128x256',
  '8K_16x128x512': b'mmm_8K_16x128x512',
  '8K_16x64x256': b'mmm_8K_16x64x256',
  '8K_32x64x512': b'mmm_8K_32x64x512',
  '8K_32x32x256': b'mmm_8K_32x32x256',
}


class MATxMASK(BrainPyOp):
  def __init__(self, seed, n, version='8K_8x128x256'):
    self.seed = seed
    self.n = n
    self.keys = jr.split(jr.PRNGKey(seed), n * 6)
    self.version = version
    assert version in cuda_op_names

  def __call__(self, mat, p):
    if mat.dtype != jnp.float32:
      raise ValueError(f'Must be a matrix of float32, while we got {mat.dtype}')
    assert mat.ndim == 2
    k, m = mat.shape
    return mat_mtp_mask_prim.bind(mat,
                                  self.keys,
                                  p=p,
                                  k=k,
                                  m=m,
                                  n=self.n,
                                  version=self.version)


def mat_mul_mask_abstract(mat, keys, *, k, m, n, p, version):
  return ShapedArray(shape=(k, n), dtype=mat.dtype)


def mat_mtp_mask_translation(c, mat, keys, *, k, m, n, p, version, platform="gpu"):
  if platform == "cpu":
    raise NotImplementedError
  elif platform == 'gpu':
    if gpu_ops is None: raise ValueError('Cannot find compiled gpu wheels.')

    opaque = gpu_ops.build_matmul_descriptor(m, k, n, 0, p)
    return x_ops.CustomCallWithLayout(
      c,
      cuda_op_names[version],
      operands=(mat, keys),
      operand_shapes_with_layout=(c.get_shape(mat), c.get_shape(keys)),
      shape_with_layout=x_shape(np.dtype(c.get_shape(mat).element_type()), (k, n), (1, 0)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


mat_mtp_mask_prim = core.Primitive("mat_mtp_mask_prim")
mat_mtp_mask_prim.def_abstract_eval(mat_mul_mask_abstract)
mat_mtp_mask_prim.def_impl(partial(xla.apply_primitive, mat_mtp_mask_prim))
xla.backend_specific_translations["gpu"][mat_mtp_mask_prim] = partial(mat_mtp_mask_translation, platform="gpu")

event_mmm_op_names = {
  8: b'',
  16: b'',
  32: b'',
  64: b'',
}


class EventMATxMASK(BrainPyOp):
  def __init__(self, seed, n, p, k, N_THREAD=1):
    self.seed = seed
    self.n = n
    self.p = p
    self.k = k
    self.N_THREAD = N_THREAD  # number of column for each thread
    self.keys = jr.split(jr.PRNGKey(seed), (n + N_THREAD - 1) // N_THREAD * 6)

    for k, v in event_mmm_op_names.items():
      if k > self.k:
        self.fn = v
        break
    else:
      all_fn = list(event_mmm_op_names.values())
      nums = np.asarray(list(event_mmm_op_names.keys()))
      for i in range(1, 4):
        diff = (i * nums - self.k) < self.k
        nonzeros = np.where(diff)[0]
        if len(nonzeros) > 1:
          self.fn = all_fn[nonzeros[0]]
          break
      else:
        self.fn = all_fn[0]

  def __call__(self, events, mat):
    if mat.dtype != jnp.float32:
      raise ValueError(f'Must be a matrix of float32, while we got {mat.dtype}')
    assert mat.ndim == 2
    assert mat.shape[0] == self.k
    return event_mmm_prim.bind(self.keys,
                               events,
                               mat,
                               p=self.p,
                               k=self.k,
                               m=mat.shape[1],
                               n=self.n,
                               fn=self.fn)


def event_mmm_abstract(keys, events, mat, **kwargs):
  return ShapedArray(shape=(kwargs['k'], kwargs['n']), dtype=mat.dtype)


def event_mmm_translation_gpu(c, keys, events, mat, **kwargs):
  if gpu_ops is None: raise ValueError('Cannot find compiled gpu wheels.')

  opaque = gpu_ops.build_matmul_descriptor(kwargs['m'], kwargs['k'], kwargs['n'], 0, kwargs['p'])
  return x_ops.CustomCallWithLayout(
    c,
    kwargs['fn'],
    operands=(keys, events, mat),
    operand_shapes_with_layout=(c.get_shape(keys), c.get_shape(events), c.get_shape(mat)),
    shape_with_layout=x_shape(np.dtype(c.get_shape(mat).element_type()),
                              (kwargs['k'], kwargs['n']),
                              (1, 0)),
    opaque=opaque,
  )


event_mmm_prim = core.Primitive("event_mat_mtp_mask")
event_mmm_prim.def_abstract_eval(event_mmm_abstract)
event_mmm_prim.def_impl(partial(xla.apply_primitive, event_mmm_prim))
xla.backend_specific_translations["gpu"][event_mmm_prim] = event_mmm_translation_gpu
