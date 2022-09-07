# -*- coding: utf-8 -*-

import jax.random as jr
import jax.numpy as jnp

from brainpylib.random import uniform, uniform2, uniform3
from numba.cuda.random import create_xoroshiro128p_states


def test_uniform():
  r = uniform(length=32, seed=123)
  print(r)

  r = uniform(length=32, seed=123)
  print(r)

  r = uniform(length=130, seed=123)
  print(r)
  print(r.size)

  r = uniform(length=130, seed=123)
  print(r)
  print(r.size)


def test_uniform2():
  key = jr.PRNGKey(123)
  keys = jr.split(key, 33 * 6)
  # keys = jr.split(key, 1 * 6)

  r = uniform2(keys, length=33)
  print(r.reshape(3, -1))

  r = uniform2(keys, length=33)
  print(r.reshape(3, -1))


def test_uniform3():
  r = uniform3(seed=1, length=33)
  print(r)

  r = uniform3(seed=1, length=33)
  print(r)


if __name__ == '__main__':
    test_uniform2()
    # test_uniform3()
