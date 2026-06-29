from algorithms.ccc import ETC_helpers as etc
from algorithms.ccc import ETC_and_CCC as ccc
# import pdb
import jax.numpy as jnp
import jax

def test_ETC():
    key = jax.random.key(28)
    tol = 1e-6
    sym_seq = jax.random.normal(key, (14,), dtype=jnp.float32)
    sym_seq = jnp.array([1.5, 1.5, 3.0, 3.0, 8.9, 7.23, 4.56,5.55, 2.10, -1.2, -1.56, 0.33, 0.76])
    print(sym_seq)
    # sym_seq = jnp.ones((10,), dtype=jnp.float32)
    # sym_seq = sym_seq.at[2].set(3.0)
    max_length = int(sym_seq.shape[0])
    num_bins = 24 #int(jnp.max(sym_seq))
    print(num_bins)

    final_seq, iters, normal_N = ccc.ETC_jit(sym_seq, max_length, num_bins, tol)
    print(f"Final seq: {final_seq}\niterations: {iters}\nnormalised_N: {normal_N}\n")

def test_CCC():
    sym_seq_c = jax.random.normal(jax.random.key(10), (2, 50), dtype=jnp.float32)
    max_length = int(sym_seq_c.shape[1])
    print(max_length)
    INFO=(15, 15, 15)
    print(INFO)
    # num_bins = int(jnp.max(sym_seq_c))
    # sym_seq_e = sym_seq_c 
    mean, arr = ccc.CCC_calculation(sym_seq_c, INFO, (12, 12))
    print(mean, arr)

def test_binning_timeseries():
    # x = jnp.array([1.0, 2.33, 4.55, 0.9, 0.02, -0.3, 10.0])
    x = jnp.ones((10,), dtype=jnp.float32)
    binned = etc.bin_timeseries(x, 10, 12)
    print(binned)


if __name__=="__main__":
    # test_CCC()
    # test_binning_timeseries()
    test_ETC()
    
