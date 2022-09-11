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

mmm_op_names = {
  '8K_sm': {'n=1': {'k=8': b'mmm_8K_1x8x128x256',
                    'k=16': b'mmm_8K_1x16x64x256',
                    'k=32': b'mmm_8K_1x32x32x256',
                    'k=64': b'mmm_8K_1x64x16x256', },
            'n=4': {'k=8': b'mmm_8K_4x8x128x256',
                    'k=16': b'mmm_8K_4x16x64x256',
                    'k=32': b'mmm_8K_4x32x32x256',
                    'k=64': b'mmm_8K_4x64x16x256', }
            },
  '4K_sm': {'n=1': {'k=8': b'mmm_4K_1x8x128x256',
                    'k=16': b'mmm_4K_1x16x64x256',
                    'k=32': b'mmm_4K_1x32x32x256',
                    'k=64': b'mmm_4K_1x64x16x256'},
            'n=4': {'k=8': b'mmm_4K_4x8x128x256',
                    'k=16': b'mmm_4K_4x16x64x256',
                    'k=32': b'mmm_4K_4x32x32x256',
                    'k=64': b'mmm_4K_4x64x16x256'}}
}

all_ks = np.asarray([8, 16, 32, 64])


def _get_k(k):
  ids = np.where((all_ks - k) >= 0)[0]
  if len(ids) > 0:
    return all_ks[ids[0]]
  divs = k // all_ks
  # mods = k % all_ks
  return all_ks[np.argmin(divs)]


class MATxMASK(BrainPyOp):
  def __init__(self, seed, n, k, N_THREAD=1, SM_size='4K'):
    self.seed = seed
    self.n = n
    self.k = k
    self.keys = jr.split(jr.PRNGKey(seed), (n + N_THREAD - 1) // N_THREAD * 6)

    self.n_size = f'n={N_THREAD}'
    self.SM_size = f"{SM_size}_sm"
    self.k_size = f'k={k}'
    self.fn = mmm_op_names[self.SM_size][self.n_size]

  def __call__(self, mat, p):
    if mat.dtype != jnp.float32:
      raise ValueError(f'Must be a matrix of float32, while we got {mat.dtype}')
    assert mat.ndim == 2
    k, m = mat.shape
    return mmm_prom.bind(mat,
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
      mmm_op_names[version],
      operands=(mat, keys),
      operand_shapes_with_layout=(c.get_shape(mat), c.get_shape(keys)),
      shape_with_layout=x_shape(np.dtype(c.get_shape(mat).element_type()), (k, n), (1, 0)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


mmm_prom = core.Primitive("mat_mtp_mask")
mmm_prom.def_abstract_eval(mat_mul_mask_abstract)
mmm_prom.def_impl(partial(xla.apply_primitive, mmm_prom))
xla.backend_specific_translations["gpu"][mmm_prom] = partial(mat_mtp_mask_translation, platform="gpu")
