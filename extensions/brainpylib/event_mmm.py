# -*- coding: utf-8 -*-


from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

from .base import *

x_ops = xla_client.ops
x_shape = xla_client.Shape.array_shape

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'EventMATxMASK',
]

event_mmm_op_names = {
  'v1': {'fp':
    {
      'k=4': b'event_mmm_fp_v1_4x64x256',
      'k=8': b'event_mmm_fp_v1_8x32x256',
      'k=16': b'event_mmm_fp_v1_16x16x256',
      'k=32': b'event_mmm_fp_v1_32x8x256',
    }
  },
  'v2': {'fp':
    {
      'k=4': b'event_mmm_fp_v2_4x64x256',
      'k=8': b'event_mmm_fp_v2_8x32x256',
      'k=16': b'event_mmm_fp_v2_16x16x256',
      'k=32': b'event_mmm_fp_v2_32x8x256',
    }
  },
}

all_ks = np.asarray([4, 8, 16, 32])


def _get_k(k):
  ids = np.where((all_ks - k) >= 0)[0]
  if len(ids) > 0: return all_ks[ids[0]]
  raise ValueError


class EventMATxMASK(BrainPyOp):
  def __init__(self, seed, n, p, k, version='v1'):
    self.seed = seed
    self.n = n
    self.k = k
    if version == 'v1':
      self.p = float(np.log((1 - p) if p < 1 else 1e-40).astype(np.float32))
    else:
      self.p = p
    self.fn = event_mmm_op_names[version]['fp'][f'k={_get_k(k)}']

  def __call__(self, events, mat):
    if mat.dtype != jnp.float32:
      raise ValueError(f'Must be a matrix of float32, while we got {mat.dtype}')
    assert mat.ndim == 2
    assert mat.shape[0] == self.k
    return event_mmm_prim.bind(events,
                               mat,
                               p=self.p,
                               k=self.k,
                               m=mat.shape[1],
                               n=self.n,
                               fn=self.fn,
                               seed=self.seed)


def event_mmm_abstract(events, mat, *, p, k, m, n, seed, fn):
  return ShapedArray(shape=(k, n), dtype=mat.dtype)


def event_mmm_translation_gpu(c, events, mat, *, p, k, m, n, seed, fn):
  if gpu_ops is None: raise ValueError('Cannot find compiled gpu wheels.')

  opaque = gpu_ops.build_matmul_descriptor(m, k, n, seed, p)
  return x_ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events, mat),
    operand_shapes_with_layout=(c.get_shape(events), c.get_shape(mat)),
    shape_with_layout=x_shape(np.dtype(c.get_shape(mat).element_type()), (k, n), (1, 0)),
    opaque=opaque,
  )


event_mmm_prim = core.Primitive("event_mat_mtp_mask")
event_mmm_prim.def_abstract_eval(event_mmm_abstract)
event_mmm_prim.def_impl(partial(xla.apply_primitive, event_mmm_prim))
xla.backend_specific_translations["gpu"][event_mmm_prim] = event_mmm_translation_gpu
