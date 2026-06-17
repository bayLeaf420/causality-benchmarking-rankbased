import jax
import jax.numpy as jnp
from functools import partial 
from typing import Tuple

from algorithms.ccc.ETC_helpers import find_pairs_fixed_length, substitute, dimensionsToOne, bin_timeseries
# @partial(jax.vmap, in_axes=(0, 0))

@partial(jax.jit, static_argnums=(1, 2))
def ETC_jit(input_data, max_length, num_bins):
    """
    ETC with a static-bound fori_loop instead of while_loop.

    A sequence of valid length L needs at most L-1 substitutions,
    so max_length - 1 iterations is a safe upper bound.
    Inactive iterations are masked out: they still execute but write nothing.
    """
    # breakpoint()
    sym_seq = bin_timeseries(input_data, max_length, num_bins) # BIN timeseries, important
    initial_valid_len = jnp.sum(sym_seq != -1)

    def body(_, state):
        seq, iters = state
        # Check whether this iteration should actually do work
        active = jnp.sum(seq != -1) > 1

        pair, _cnt, hashed = find_pairs_fixed_length(seq, max_length, num_bins)
        new_seq = substitute(seq, max_length, num_bins, pair, hashed)

        # Only commit the update if the sequence hadn't already converged
        seq   = jnp.where(active, new_seq, seq)
        iters = iters + active.astype(jnp.int32)

        # jax.debug.print("iter={}, valid_count={}, active={}, pair={}", iters, jnp.sum(seq != -1), active, pair)
        jax.debug.print("iters: {}", iters)
        return seq, iters

    init = (sym_seq, jnp.array(0, dtype=jnp.int32))
    final_seq, iters = jax.lax.fori_loop(0, max_length - 1, body, init)

    normalN = jnp.where(
        initial_valid_len > 1,
        iters.astype(jnp.float32) / (initial_valid_len - 1),
        0.0,
    )
    return final_seq, iters, normalN


# --------------- NEW CCC ------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 2))
def CCC_calculation(
        input_data: jax.Array, # (2, L) shape. 
        time_ranges: Tuple[int],
        num_bins:Tuple[int, int],
) -> Tuple[float, jax.Array, int]:
    """
    Inputs:
    1. input_data is a jax.Array. Assuming dims are (n_timeseries, n_timepoints). CAUSES MUST COME FIRST
    2. cause_vars and effect_vars: ints stating the number of cause and effect timeseries. 
    3. time_ranges is a tuple stating the needed windows. 
    4. num_bins is a jnp.array stating the number or bins of each time series.  
    """
    breakpoint()
    (past, present, jump) = time_ranges
    max_length = input_data.shape[1] # inout_data shaped (2,  L)

    if past + present > max_length:
        raise ValueError("past + present are greater than time-series initial size")

    num_wins = (max_length - (past + present)) // jump # Last smol window will be ignored
    
    # Split input data into cause and effect matrices
    # cause_vector = jax.lax.slice_in_dim(input_data, 0, 1, axis=0) # Shape is (1, L)
    effect_vector = jax.lax.slice_in_dim(input_data, 0, 1, axis=0).reshape(-1)
    # Extract bins
    cause_bins = num_bins[0]
    effect_bins = num_bins[1]
    joint_bins = max(cause_bins, effect_bins)

    # MAKE JOINT AT THE BEGINNING ITSELF DON'T MAKE IT EVERY ITERATION
    joint_vector = dimensionsToOne(input_data, int(effect_bins)).reshape(-1) # Shape is again (1, L)

    def single_window_calc(win_ind: int) -> None:

        # Define required slices
        effect_present = jax.lax.dynamic_slice(effect_vector, (win_ind*jump + past,), (present,),allow_negative_indices=False)
        joint_past = jax.lax.dynamic_slice(joint_vector, (win_ind*jump,), (past,), allow_negative_indices=False)
        effect_past = jax.lax.dynamic_slice(effect_vector, (win_ind*jump,), (past,), allow_negative_indices=False)

        _, _, ETC_joint_full = ETC_jit(jnp.concatenate([joint_past, effect_present], dtype=jnp.int32), past + present, joint_bins)
        _, _, ETC_joint_past = ETC_jit(joint_past, past, joint_bins)
        _, _, ETC_effect_full = ETC_jit(jnp.concatenate([effect_past, effect_present]), past + present, effect_bins)
        _, _, ETC_effect_past = ETC_jit(effect_past, past, effect_bins)

        causal_dynamic_complexity = ETC_joint_full - ETC_joint_past 
        control_dynamic_complexity = ETC_effect_full - ETC_effect_past

        CCC_val = causal_dynamic_complexity - control_dynamic_complexity

        return CCC_val

    CCC_array = jax.vmap(single_window_calc)(jnp.arange(num_wins))
    CCC_mean = jnp.mean(CCC_array, axis=0, dtype=jnp.float32)

    return CCC_mean, CCC_array 
