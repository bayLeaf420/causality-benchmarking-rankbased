import jax
import jax.numpy as jnp
from functools import partial 

@partial(jax.jit, static_argnums=(1, 2))
def find_pairs_fixed_length(
    input_array: jnp.ndarray,
    max_length: int,
    num_bins: int,
) -> tuple[jnp.ndarray, int, jnp.ndarray, float]:
    """
    Inputs:
    1. input_array: Jax array consisting of [valid_symbols -1 ... -1]
    2. max_length: Length of input_array. It is the length at which ETC algorithm began. 
    3. num_bins: INITIAL number of bins at which our symbolic sequence began. 

    Outputs:
    1. most_freq_pair: Most frequent pair in the form of a jax array. If a tie, lower hash chosen.
    2. count: Number of times the most_freq_pair occurs
    3. hashed_safe: Hash values of pairs after validity mask has been applied.
    4. 
    """

    M = max_length + num_bins
    first = input_array[:-1]
    second = input_array[1:]
    valid_pair = (first != -1) & (second != -1)

    # Overlap rule (equal‑symbol pairs)
    valid_symbols = input_array != -1
    change_points = jnp.concatenate([
        jnp.array([True]),
        (input_array[1:] != input_array[:-1]) & valid_symbols[1:]
    ])
    indices = jnp.arange(max_length)
    run_starts = jnp.where(change_points, indices, 0)
    run_starts = jax.lax.cummax(run_starts)
    offset = indices - run_starts
    pair_offset = offset[:-1]

    equal_pair = (first == second) & valid_pair
    keep_equal = (pair_offset % 2) == 0
    keep_different = (first != second) & valid_pair
    final_valid = keep_different | (equal_pair & keep_equal)

    # Hash valid pairs, set invalid ones to -1
    hashed = first * M + second
    hashed_safe = jnp.where(final_valid, hashed, -1)

    # Sort and count frequencies with jnp.unique
    unique, counts = jnp.unique(
        hashed_safe, return_counts=True, size=max_length - 1
    )
    # Ignore the bin for -1 (which will be first if present)
    # unique[0] may be -1; if so, set its count to 0
    counts = jnp.where(unique != -1, counts, 0)

    N = jnp.sum(counts) 
    shannon_entropy_1 = jnp.where(
        N > 0,
        -jnp.sum(jnp.where(counts != 0, (counts / N) * jnp.log2(counts / N), 0.0)),
        0.0,
        )

    # Handle no valid pairs
    L = jnp.sum(valid_symbols.astype(jnp.int32))
    no_pairs = (L < 2) | (jnp.sum(final_valid) == 0)

    max_idx = jnp.argmax(counts)
    max_hash = unique[max_idx]
    a = max_hash // M
    b = max_hash % M
    most_freq_pair = jnp.where(no_pairs,
                               jnp.array([-1, -1], dtype=input_array.dtype),
                               jnp.array([a, b], dtype=input_array.dtype))
    count = jnp.where(no_pairs, 0, counts[max_idx])

    return most_freq_pair, count, hashed_safe, shannon_entropy_1

@partial(jax.jit, static_argnums=(1, 2))
def substitute(
    sym_seq: jnp.ndarray,
    max_length: int,
    num_bins: int,
    most_freq_pair: jnp.ndarray,
    hashed_array: jnp.ndarray,
) -> jnp.ndarray:
    # new symbol = current maximum + 1  (correctly increments each iteration)
    new_sym = jnp.max(sym_seq) + 1
    a, b = most_freq_pair[0], most_freq_pair[1]
    M = max_length + num_bins 
    pair_hash = a * M + b

    # Second element of each replaced pair
    swap_mask = jnp.concatenate([
        jnp.array([False], dtype=bool),
        hashed_array == pair_hash
    ])

    # First element of replaced pair → new_sym
    first_of_pair = jnp.roll(swap_mask, shift=-1).at[-1].set(False)
    modified_seq = jnp.where(first_of_pair, new_sym, sym_seq)

    keep_mask = ~swap_mask

    # Compute destination indices for kept elements
    # cumsum gives 1‑based index where each kept element goes
    dest = jnp.cumsum(keep_mask) - 1                  # 0‑based
    # Write non‑kept elements into a trash bin at index max_length
    indices_to_write = jnp.where(keep_mask, dest, max_length)
    values_to_write = jnp.where(keep_mask, modified_seq, 0)

    # Scatter all: trash bin will be overwritten but we ignore it
    out = jnp.full(max_length + 1, -1, dtype=sym_seq.dtype)
    out = out.at[indices_to_write].set(values_to_write)

    return out[:max_length]

def bin_timeseries(x: jnp.ndarray, max_length: int, num_bins: int) -> jnp.ndarray:
    """
    Inputs:
    1. x: shape(L,) use vmap if required. 

    Outputs:
    1. 
    Takes in floating point jax array, outputs symbolic sequence in [1, num_bins]
    dtype int32.

    VMapping: axis 0 is perpendicular to the time direction. axis 1 is time. 
    So, num_bins will look like: np.array([bins_cause, bins_effect], dtype=np.int32)
    """
    # breakpoint()
    if x.shape[-1] != max_length:
        raise ValueError(f"x shape is different than max value.\nx.shape = {x.shape}\nmax_length = {max_length}")

    x_min = jnp.min(x)
    x_max = jnp.max(x)
    x_range = x_max - x_min
    jax.debug.callback(
        lambda r: (_ for _ in ()).throw(ValueError("x_range is <= 0")) if r <= 0 else None, x_range
    )
    
    x_norm = jnp.where(x != -1, (x - x_min) / x_range, -1)
    return (jnp.floor(x_norm * (num_bins - 1)) + 1).astype(jnp.int32)

@partial(jax.jit, static_argnums=(1,))
def dimensionsToOne(symSeqMatrix: jnp.ndarray, stride: int) -> jnp.ndarray:
    """
    Vectorised replacement for the pure-Python dimensionsToOne.

    Maps each column of a (2, L) integer matrix to a unique 1-based symbol
    using a base encoding:
        symbol = (row0 - 1) * stride + row1
    where stride = effect_bins (the alphabet size of row 1).

    Output range: [1, cause_bins * effect_bins]  →  nb_joint = cause_bins * effect_bins  ✓
    Works under jit / vmap because it contains no Python control flow.
    """
    return ((symSeqMatrix[1] - 1) * stride + symSeqMatrix[0]).astype(jnp.int32)
