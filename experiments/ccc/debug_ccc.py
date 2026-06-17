from algorithms.ccc import ETC_helpers as etc
from algorithms.ccc import ETC_and_CCC as ccc
# import pdb
import jax.numpy as jnp
import jax

def test_ETC():
    key = jax.random.key(28)
    sym_seq = jax.random.uniform(key, (121,), dtype=jnp.float32, minval=0.0, maxval=12.0)
    max_length = int(sym_seq.shape[0])
    num_bins = int(jnp.max(sym_seq))
    print(num_bins)

    final_seq, iters, normal_N = ccc.ETC_jit(sym_seq, max_length, num_bins)
    print(f"Final seq: {final_seq}\niterations: {iters}\nnormalised_N: {normal_N}\n")

def test_CCC():
    sym_seq_c = jnp.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1], dtype=jnp.int32).reshape(1, -1)
    max_length = int(sym_seq_c.shape[1])
    print(max_length)
    INFO=(max_length // 2, max_length - max_length // 2, max_length // 2)
    print(INFO)
    num_bins = int(jnp.max(sym_seq_c))
    sym_seq_e = sym_seq_c 
    mean, arr, wins = ccc.CCC_calculation_2vec(sym_seq_c, sym_seq_e, num_bins, num_bins, INFO)
    print(mean, arr, wins)

def test_binning_timeseries():
    x = jnp.array([1.0, 2.33, 4.55, 0.9, 0.02, -0.3, 10.0])
    binned = etc.bin_timeseries(x, 7, 300)
    print(binned)


if __name__=="__main__":
    # test_CCC()
    # test_binning_timeseries()
    test_ETC()
    
