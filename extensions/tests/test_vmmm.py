# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm
from jax import lax, random
import jax
import jax.numpy as jnp
from brainpylib import vector_matmul_mask, event_sum

bm.disable_gpu_memory_preallocation()
from time import time


def try_jax():
  pivots_size = 4
  permutation_size = 2 * pivots_size
  pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=jnp.int32)
  r = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
  print(r)


def try_vmmm():
  m, k, n = int(1e4), 16, int(1e4)

  key = random.PRNGKey(1233)

  key, subkey = random.split(key)
  V = random.normal(subkey, (m,))
  key, subkey = random.split(key)
  L = random.normal(subkey, (m, k))
  key, subkey = random.split(key)
  R = random.normal(subkey, (k, n))

  # r = vector_matmul_mask(V, L, R, seed=1234, p=0.1)
  r = vector_matmul_mask(V, L, R, seed=1234, p=2.)

  r2 = V @ L @ R

  diff = (r - r2).flatten()
  plt.hist(bm.as_numpy(diff), bins=100)
  plt.show()
  # assert jnp.allclose(r, r2)

  print(r)


def compare_speed_of_vmmm_and_jax():
  p = 0.4
  k = 96
  # m, k, n = int(1e4), 16, int(1e4)

  f2 = jax.jit(lambda v, a, b: vector_matmul_mask(v, a, b, seed=1234, p=p))
  f = jax.jit(lambda v, a, b, m: (v @ ((a @ b) * mask)))

  key = random.PRNGKey(1233)
  for size in [1e3, 5e3, 1e4, 2e4]:
    m = n = int(size)
    key, subkey = random.split(key)
    V = random.normal(subkey, (m,))
    key, subkey = random.split(key)
    L = random.normal(subkey, (m, k))
    key, subkey = random.split(key)
    R = random.normal(subkey, (k, n))
    key, subkey = random.split(key)
    mask = random.uniform(subkey, (m, n)) < p

    print(f'm={m}, k={k}, n={n}, p={p}')

    _ = jax.block_until_ready(f2(V, L, R))
    t0 = time()
    _ = jax.block_until_ready(f2(V, L, R))
    print(f'VMMM time: {time() - t0} s')

    _ = jax.block_until_ready(f(V, L, R, mask))
    t0 = time()
    _ = jax.block_until_ready(f(V, L, R, mask))
    print(f'JAX  time: {time() - t0} s')
    print()


if __name__ == '__main__':
  # try_vmmm()
  compare_speed_of_vmmm_and_jax()
  # try_event_sum()
