import jax
import jax.numpy as jnp
from functools import partial 

from algorithms.ETC_helpers import find_pairs_fixed_length, substitute, dimensionsToOne

@partial(jax.jit, static_argnums=(1, 2))
def ETC_jit(sym_seq, max_length, num_bins):
    """
    ETC with a static-bound fori_loop instead of while_loop.

    A sequence of valid length L needs at most L-1 substitutions,
    so max_length - 1 iterations is a safe upper bound.
    Inactive iterations are masked out: they still execute but write nothing.
    """
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
        return seq, iters

    init = (sym_seq, jnp.array(0, dtype=jnp.int32))
    final_seq, iters = jax.lax.fori_loop(0, max_length - 1, body, init)

    normalN = jnp.where(
        initial_valid_len > 1,
        iters.astype(jnp.float32) / (initial_valid_len - 1),
        0.0,
    )
    return final_seq, iters, normalN

@partial(jax.jit, static_argnums=(2, 3, 4))
@partial(jax.vmap, in_axes = (0, 0, None, None, None, None))
def CCC_calculation_2vec(cause_seq, effect_seq, cause_bins, effect_bins, INFO):
    """
    Fully JAX-traceable version of CCC_calculation_2vec.

    cause_seq, effect_seq : 1-D jnp arrays of shape (T,)

    Outer callers can vmap over any leading batch dimensions, e.g.
        batched = jax.vmap(jax.vmap(
            partial(CCC_calculation_2vec, cause_bins=cb, effect_bins=eb, INFO=INFO),
            in_axes=(0, 0)),   # n_rand
            in_axes=(0, 0)     # n_param_var
        )(f_out, x_out)        # shapes (n_rand, n_param_var, T)
    """
    lPast  = INFO[0]
    lPres  = INFO[1]
    lJump  = INFO[2]

    T = cause_seq.shape[0]          # static — it's a shape
    lCheck = lPast + lPres

    # ── static sizes (all derived from INFO / shape, never from traced values) ─
    max_len_effect = lCheck
    max_len_joint  = lCheck + lPres          # lPast + 2*lPres

    nb_effect = effect_bins
    # Upper bound: dimensionsToOne maps (cause_sym, eff_sym) → one int.
    # At most cause_bins * effect_bins distinct symbols.
    nb_joint  = cause_bins * effect_bins

    # numWindows is fully static because T and INFO values are static.
    numWindows = (T - lCheck) // lJump + 1

    # ── per-window computation, vectorised with vmap ────────────────────────
    def single_window(ii):
        start   = ii * lJump
        p_start = start + lPast           # start of the "present" portion

        # ── effect-only terms ──────────────────────────────────────────────
        eff_full = jax.lax.dynamic_slice(effect_seq, (start,),   (lCheck,))
        eff_past = jax.lax.dynamic_slice(effect_seq, (start,),   (lPast,))

        eff_full_pad = jnp.pad(eff_full, (0, max_len_effect - lCheck), constant_values=-1)
        eff_past_pad = jnp.pad(eff_past, (0, max_len_effect - lPast),  constant_values=-1)

        _, _, norm_eff_full = ETC_jit(eff_full_pad, max_len_effect, nb_effect)
        _, _, norm_eff_past = ETC_jit(eff_past_pad, max_len_effect, nb_effect)
        CC_noCause = norm_eff_full - norm_eff_past

        # ── joint cause–effect terms ───────────────────────────────────────
        cause_full  = jax.lax.dynamic_slice(cause_seq,  (start,),   (lCheck,))
        eff_present = jax.lax.dynamic_slice(effect_seq, (p_start,), (lPres,))
        cause_past  = jax.lax.dynamic_slice(cause_seq,  (start,),   (lPast,))
        eff_past2   = jax.lax.dynamic_slice(effect_seq, (start,),   (lPast,))

        # Full window: [cause & effect for lCheck cols] ++ [eff_present repeated, lPres cols]
        mat_full = jnp.concatenate([
            jnp.stack([cause_full,  eff_full],    axis=0),   # (2, lCheck)
            jnp.stack([eff_present, eff_present], axis=0),   # (2, lPres)
        ], axis=1)                                            # (2, lCheck+lPres)

        joint_full_seq = dimensionsToOne(mat_full, effect_bins)   # (lCheck+lPres,)
        joint_full_pad = jnp.pad(
            joint_full_seq, (0, max_len_joint - (lCheck + lPres)), constant_values=-1
        )

        # Past window
        mat_past = jnp.stack([cause_past, eff_past2], axis=0)   # (2, lPast)
        joint_past_seq = dimensionsToOne(mat_past, effect_bins)  # (lPast,)
        joint_past_pad = jnp.pad(
            joint_past_seq, (0, max_len_joint - lPast), constant_values=-1
        )

        _, _, norm_joint_full = ETC_jit(joint_full_pad, max_len_joint, nb_joint)
        _, _, norm_joint_past = ETC_jit(joint_past_pad, max_len_joint, nb_joint)
        CC_full = norm_joint_full - norm_joint_past

        return CC_noCause - CC_full

    # vmap single_window over the window indices (replaces the Python for-loop)
    CCC_array = jax.vmap(single_window)(jnp.arange(numWindows))

    return jnp.mean(CCC_array), CCC_array, numWindows


# ── example: batch over (n_rand, n_param_var) outputs of x_tensor ────────────

@partial(jax.jit, static_argnums=(2, 3, 4))
def batched_CCC(x_out, f_out, cause_bins, effect_bins, INFO):
    """
    x_out, f_out : shape (n_rand, n_param_var, T)
    Returns CCC_values of shape (n_rand, n_param_var).
    """
    fn = partial(CCC_calculation_2vec,
                 cause_bins=cause_bins, effect_bins=effect_bins, INFO=INFO)
    return jax.vmap(jax.vmap(fn, in_axes=(0, 0)), in_axes=(0, 0))(f_out, x_out)

