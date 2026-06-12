import jax
import jax.numpy as jnp
from functools import partial 

@partial(jax.jit, static_argnums=(1, 2))
def find_pairs_fixed_length(
    input_array: jnp.ndarray,
    max_length: int,
    num_bins: int,
) -> tuple[jnp.ndarray, int, jnp.ndarray]:

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

    return most_freq_pair, count, hashed_safe

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
    return (symSeqMatrix[0] - 1) * stride + symSeqMatrix[1]