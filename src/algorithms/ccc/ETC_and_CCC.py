import jax
import jax.numpy as jnp
# import numpy as np
from functools import partial 
from typing import Tuple

from algorithms.ccc.ETC_helpers import find_pairs_fixed_length, substitute, dimensionsToOne, bin_timeseries
# @partial(jax.vmap, in_axes=(0, 0))

# @partial(jax.jit, static_argnums=(1, 2, 3, 4))
def ETC_jit(input_data, max_length, num_bins, H_1_tolerance, is_joint=False):
    """
    Inputs:
        input_data: shape (L,) for scalar, shape (2, L) for joint
        max_length: static i
        num_bins:   static int (or tuple of 2 ints if is_joint=True)
        is_joint:   static bool — if True, input_data is (2, L) raw float,
                    bin each row separately then combine via dimensionsToOne

    ETC with a static-bound fori_loop instead of while_loop.

    A sequence of valid length L needs at most L-1 substitutions,
    so max_length - 1 iterations is a safe upper bound.
    Inactive iterations are masked out: they still execute but write nothing.
    """
    # breakpoint()
    if is_joint:
        cause_bins, effect_bins = num_bins
        cause_binned  = bin_timeseries(input_data[0], max_length, cause_bins)
        effect_binned = bin_timeseries(input_data[1], max_length, effect_bins)
        joint_input   = jnp.stack([cause_binned, effect_binned], axis=0)
        sym_seq       = dimensionsToOne(joint_input, effect_bins).reshape(-1)
        actual_bins   = cause_bins * effect_bins

    else:
        sym_seq     = bin_timeseries(input_data, max_length, num_bins)
        actual_bins = num_bins

    # sym_seq = bin_timeseries(input_data, max_length, num_bins) # BIN timeseries, important
    initial_valid_len = jnp.sum(sym_seq != -1)

    jax.debug.callback(
        lambda v, m: (_ for _ in ()).throw(ValueError(f"initial_valid_len={v} != max_length={m}")) if v != m else None,
        initial_valid_len, max_length
    )

    def body(_, state):
        seq, iters = state
        # Check if 'Shannon Entropy' is above tolerance
        # Check whether this iteration should actually do work
        
        pair, _cnt, hashed, H_1 = find_pairs_fixed_length(seq, max_length, actual_bins)
        active = jnp.logical_and(jnp.sum(jnp.where(seq != -1, 1.0, 0.0)) > 1, H_1 > H_1_tolerance) 

        # print(f"\n{hashed}\n")
        # jax.debug.print("seq ={}", seq)
        # jax.debug.print("iter={}, valid_count={}, active={}, pair={}, Ent: {}", iters, jnp.sum(seq != -1), active, pair, H_1)

        new_seq = substitute(seq, max_length, actual_bins, pair, hashed)

        # Only commit the update if the sequence hadn't already converged
        # jax.debug.print("Sequence before={}", seq)
        seq   = jnp.where(active, new_seq, seq)
        iters = iters + active.astype(jnp.int32)

        # jax.debug.print("iters: {}", iters)
        
        return seq, iters

    init = (sym_seq, jnp.array(0, dtype=jnp.int32))
    final_seq, iters = jax.lax.fori_loop(0, max_length - 1, body, init)
    # seq, iters = init
    # for i in range(max_length-1):
        # print(seq)
        # seq, iters = body(i, (seq, iters))
    
    # final_seq = seq
    # print(final_seq)
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
        H_1_tolerance: float,
) -> Tuple[float, jax.Array, int]:
    """
    Inputs:
    1. input_data is a jax.Array. Assuming dims are (n_timeseries, n_timepoints). CAUSES MUST COME FIRST
    2. cause_vars and effect_vars: ints stating the number of cause and effect timeseries. 
    3. time_ranges is a tuple stating the needed windows. 
    4. num_bins is a jnp.array stating the number or bins of each time series.  
    """
    # breakpoint()
    (past, present, jump) = time_ranges
    max_length = input_data.shape[1] # inout_data shaped (2,  L)

    if past + present > max_length:
        raise ValueError("past + present are greater than time-series initial size")

    num_wins = (max_length - (past + present)) // jump + 1# Last smol window will be ignored
    
    # Split input data into cause and effect matrices
    cause_vector = jax.lax.slice(input_data, (0, 0), (1, max_length)).reshape(max_length) # Shape is (1, L)
    effect_vector = jax.lax.slice(input_data, (1, 0), (2, max_length)).reshape(max_length)
    jax.debug.print("Cause vec: {}\neffect vec: {}", cause_vector[0:10], effect_vector[0:10])
    # Extract bins
    cause_bins = num_bins[0]
    effect_bins = num_bins[1]
    # joint_bins = cause_bins * effect_bins
   
    # MAKE JOINT AT THE BEGINNING ITSELF DON'T MAKE IT EVERY ITERATION
    # joint_vector = dimensionsToOne(input_data, int(effect_bins)).reshape(-1) # Shape is again (1, L)

    def single_window_calc(win_ind: int) -> None:

        # Define required slices
        effect_present = jax.lax.dynamic_slice(effect_vector, (win_ind*jump + past,), (present,),allow_negative_indices=False)
        # joint_past = jax.lax.dynamic_slice(joint_vector, (win_ind*jump,), (past,), allow_negative_indices=False)
        effect_past = jax.lax.dynamic_slice(effect_vector, (win_ind*jump,), (past,), allow_negative_indices=False)
        cause_past = jax.lax.dynamic_slice(cause_vector, (win_ind*jump,), (past,), allow_negative_indices=False)

        # jax.debug.print("joint_full: {}\n", "="*40)
        _, _, ETC_joint_full = ETC_jit(
            jnp.stack([
                jnp.concatenate([cause_past, effect_present], dtype=jnp.float32), 
                jnp.concatenate([effect_past, effect_present], dtype=jnp.float32),
            ], 
            dtype=jnp.float32,
            ),
            past + present, 
            (cause_bins, effect_bins),
            H_1_tolerance,
            is_joint=True,
        )
        # jax.debug.print("joint_past: {}\n", "="*40)
        _, _, ETC_joint_past = ETC_jit(
            jnp.stack([cause_past, effect_past], dtype=jnp.float32), 
            past, 
            (cause_bins, effect_bins),
            H_1_tolerance,
            is_joint=True,
        )
        # jax.debug.print("effect_full: {}\n", "="*40)
        _, _, ETC_effect_full = ETC_jit(jnp.concatenate([effect_past, effect_present], dtype=jnp.float32), past + present, effect_bins, H_1_tolerance)
        # jax.debug.print("effect_past: {}\n", "="*40)
        _, _, ETC_effect_past = ETC_jit(effect_past, past, effect_bins, H_1_tolerance)

        causal_dynamic_complexity = ETC_joint_full - ETC_joint_past 
        control_dynamic_complexity = ETC_effect_full - ETC_effect_past
        
        # jax.debug.print(
        #     "ETC_joint_full: {}\nETC_joint_past: {}\nETC_effect_full: {}\nETC_effect_past: {}",
        #     ETC_joint_full,
        #     ETC_joint_past,
        #     ETC_effect_full,
        #     ETC_effect_past,                        
        #    )
        
        CCC_val = - causal_dynamic_complexity + control_dynamic_complexity

        return CCC_val

    CCC_array = jax.vmap(single_window_calc)(jnp.arange(num_wins))
    CCC_mean = jnp.mean(CCC_array, axis=0, dtype=jnp.float32)

    return CCC_mean, CCC_array 
