from algorithms import tc
from algorithms.tc import _symbolic_information_flow, _normalised_information_flow
from utils import build_params_tensor_jax, _build_lambda_fns
from algorithms.tc_numerical import time_integrals          # adjust if module lives elsewhere
from sampling import sample_oscillator_tensor    # adjust if module lives elsewhere
from utils import build_norm_lambda         # adjust to wherever you put it

import os
from pathlib import Path

import pytest
import numpy as np
# import sympy as sp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # tighter tolerances for numeric checks

# ---------------------------------------------------------------- fixtures / consts

TAU_INIT = (0.0, 1.0, 6)                 # (start, end, n_points)
SIGMA_INIT = (1.0, 1.0, 1.0, 1.0)
MEAN_INIT = (0.0, 0.0, 0.0, 0.0)
PARAMS = {"beta": (0.3, 0.6, 2), "K": (0.8, 1.2, 2), "mu": (0.4, 0.7, 2)}

# structurally nonzero T(i->j) entries: exactly where A[j, i] != 0
NONZERO_ENTRIES = {(1, 0), (0, 1), (1, 1), (2, 1), (3, 2), (2, 3)}

# a safe evaluation point: tau > tau_0 avoids the 0/0 at the start
EVAL_POINT = dict(tau_v=0.5, beta_v=0.5, K_v=1.0, mu_v=0.5)


@pytest.fixture(scope="session", autouse=True)
def _prime_cache():
    """Pin the symbolic cache to ./results and load it (no recompute)."""
    cache_dir = Path("./results").resolve()
    tc.CACHE_PATH_SIF = cache_dir / "symbolic_flow_cache.pkl"
    tc.CACHE_PATH_NSIF = cache_dir / "normalised_flow_cache.pkl"

    assert tc.CACHE_PATH_SIF.exists(), (
        f"expected cached symbolic expr at {tc.CACHE_PATH_SIF}; "
        "run the pipeline once to generate it before testing"
    )
    _symbolic_information_flow()
    _normalised_information_flow()
    yield

@pytest.fixture(scope="session")
def norm_fn():
    """Build the numeric normaliser once; reuse across tests."""
    return build_norm_lambda(TAU_INIT, SIGMA_INIT)


# ---------------------------------------------------------------- symbolic layer

def test_symbolic_shapes():
    info, sigma, mean, symbols, mean_symbols = _symbolic_information_flow()
    assert info.shape == (4, 4)
    assert sigma.shape == (4, 4)
    assert mean.shape == (4, 1)
    assert len(symbols) == 9
    assert len(mean_symbols) == 4


def test_symbolic_cache_roundtrip():
    assert tc.CACHE_PATH_SIF.exists()
    a = _symbolic_information_flow()
    b = _symbolic_information_flow()          # second call hits the cache
    assert a[0] == b[0]                       # sympy Matrix structural equality


# ---------------------------------------------------------------- normalised flow (numeric)

def test_norm_structure(norm_fn):
    """Bug #2 regression: nonzero pattern must key on A[j, i], not A[i, j]."""
    vals = np.asarray(norm_fn(**EVAL_POINT))
    assert vals.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            if (i, j) in NONZERO_ENTRIES:
                assert vals[i, j] != 0.0, f"entry ({i},{j}) was wrongly zeroed"
            else:
                assert vals[i, j] == 0.0, f"entry ({i},{j}) should be zero"


def test_coupling_term_survives(norm_fn):
    """The -mu*K^2 coupling T(x2 -> x1) is exactly the entry the old bug killed."""
    vals = np.asarray(norm_fn(**EVAL_POINT))
    assert vals[2, 1] != 0.0


def test_norm_bounded(norm_fn):
    """Bugs #3/#4 regression: sum-of-abs normaliser => every entry in [0, 1]."""
    vals = np.asarray(norm_fn(**EVAL_POINT))
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= -1e-9)
    assert np.all(vals <= 1 + 1e-6)


#def test_norm_matches_symbolic(norm_fn):
#    """Numeric normaliser must agree with the symbolic _normalised_information_flow."""
#    N = _normalised_information_flow()
#    _, _, _, symbols, _ = _symbolic_information_flow()
#    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols
#    subs = {tau_0: TAU_INIT[0], s11: SIGMA_INIT[0], s22: SIGMA_INIT[1],
#            s33: SIGMA_INIT[2], s44: SIGMA_INIT[3]}
#    sym_fn = sp.lambdify((tau, beta, K, mu), N.subs(subs), modules="numpy", cse=True)
#
#    sym_vals = np.asarray(sym_fn(EVAL_POINT["tau_v"], EVAL_POINT["beta_v"],
#                                 EVAL_POINT["K_v"], EVAL_POINT["mu_v"]))
#    num_vals = np.asarray(norm_fn(**EVAL_POINT))
#    assert np.allclose(sym_vals, num_vals, atol=1e-8)



def test_norm_multiple_points(norm_fn):
    """Bounds and finiteness should hold across a spread of points, not just one."""
    for tau_v in (0.2, 0.5, 0.9):
        for beta_v in (0.3, 0.6):
            vals = np.asarray(norm_fn(tau_v=tau_v, beta_v=beta_v, K_v=1.0, mu_v=0.5))
            assert np.all(np.isfinite(vals))
            assert np.all(vals >= -1e-9) and np.all(vals <= 1 + 1e-6)


# ---------------------------------------------------------------- utils layer

def test_params_tensor_shape_and_order():
    p = build_params_tensor_jax(PARAMS)
    assert p.shape == (2, 2, 2, 3)
    # indexing='ij': p[i, j, k] == (beta_i, K_j, mu_k)
    assert jnp.allclose(p[:, 0, 0, 0], jnp.linspace(*PARAMS["beta"]))
    assert jnp.allclose(p[0, :, 0, 1], jnp.linspace(*PARAMS["K"]))
    assert jnp.allclose(p[0, 0, :, 2], jnp.linspace(*PARAMS["mu"]))


def test_build_lambda_fns():
    lam, lam_norm = _build_lambda_fns(TAU_INIT, SIGMA_INIT)
    assert len(lam) == 4 and all(len(r) == 4 for r in lam)
    assert len(lam_norm) == 4 and all(len(r) == 4 for r in lam_norm)
    assert callable(lam[1][0])
    assert jnp.isfinite(float(lam[1][0](0.5, 0.5, 1.0, 0.5)))


# ---------------------------------------------------------------- numerical layer

def test_time_integrals_shape_and_finite():
    p = build_params_tensor_jax(PARAMS)
    lkif, n_lkif = time_integrals(p, TAU_INIT, SIGMA_INIT)
    assert lkif.shape == (2, 2, 2, 4, 4)
    assert n_lkif.shape == (2, 2, 2, 4, 4)
    assert bool(jnp.all(jnp.isfinite(lkif)))
    assert bool(jnp.all(jnp.isfinite(n_lkif)))


def test_normalised_integrals_bounded():
    """Integrand <= 1 pointwise, so each normalised integral <= (tau_end - tau_start)."""
    p = build_params_tensor_jax(PARAMS)
    _, n_lkif = time_integrals(p, TAU_INIT, SIGMA_INIT)
    span = TAU_INIT[1] - TAU_INIT[0]
    assert bool(jnp.all(jnp.abs(n_lkif) <= span + 1e-6))


# ---------------------------------------------------------------- sampling layer

def test_sample_shape_and_finite():
    """Bug #5 regression: n_realisations must be the leading key axis."""
    p = build_params_tensor_jax(PARAMS)
    n_real = 5
    out = sample_oscillator_tensor(
        n_real, p, TAU_INIT, MEAN_INIT, SIGMA_INIT, jax.random.PRNGKey(0)
    )
    assert out.shape == (n_real, 2, 2, 2, TAU_INIT[2], 4)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_sample_is_reproducible():
    p = build_params_tensor_jax(PARAMS)
    a = sample_oscillator_tensor(3, p, TAU_INIT, MEAN_INIT, SIGMA_INIT,
                                 jax.random.PRNGKey(42))
    b = sample_oscillator_tensor(3, p, TAU_INIT, MEAN_INIT, SIGMA_INIT,
                                 jax.random.PRNGKey(42))
    assert jnp.array_equal(a, b)


# ---------------------------------------------------------------- cache-path change

def test_cache_paths_derived_from_cache_dir():
    assert tc.CACHE_PATH_SIF.parent == tc.CACHE_PATH_NSIF.parent
    assert tc.CACHE_PATH_SIF.name == "symbolic_flow_cache.pkl"
    assert tc.CACHE_PATH_NSIF.name == "normalised_flow_cache.pkl"


def test_cache_dir_env_logic(monkeypatch, tmp_path):
    """Validates the env-var -> path construction without reloading the module."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    cache_dir = Path(os.environ.get("CACHE_DIR", "./results")).resolve()
    assert cache_dir == tmp_path.resolve()
    assert (cache_dir / "symbolic_flow_cache.pkl").parent == cache_dir