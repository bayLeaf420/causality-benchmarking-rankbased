from lkif_ode import (        # adjust import path to your layout # type: ignore
    make_system_matrix,
    raw_flow,
    normalise_flow,
    _integrate_single,
    normalised_lkif_grid,
)
from utils import build_params_tensor_jax   # adjust if it lives elsewhere

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)  # solver tolerances need float64

# ---------------------------------------------------------------- consts

TAU_INIT = (0.0, 1.0, 6)
SIGMA_INIT = (1.0, 1.0, 1.0, 1.0)
PARAMS = {"beta": (0.3, 0.6, 2), "K": (0.8, 1.2, 2), "mu": (0.4, 0.7, 2)}

# structurally nonzero T(i->j) entries: exactly where A[j, i] != 0
NONZERO_ENTRIES = {(1, 0), (0, 1), (1, 1), (2, 1), (3, 2), (2, 3)}


# ---------------------------------------------------------------- system matrix

def test_system_matrix_structure():
    A = make_system_matrix(0.5, 1.0, 0.4)
    assert A.shape == (4, 4)
    # the coupling term -mu*K^2 sits at [1, 2]
    assert jnp.isclose(A[1, 2], -0.4 * 1.0**2)
    assert jnp.isclose(A[1, 1], -0.5)      # damping
    assert jnp.isclose(A[3, 2], -1.0**2)   # second oscillator restoring force
    # rows that must be pure kinematic identities
    assert jnp.allclose(A[0], jnp.array([0.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(A[2], jnp.array([0.0, 0.0, 0.0, 1.0]))


# ---------------------------------------------------------------- raw flow

def test_raw_flow_diagonal_equals_A_diag():
    """flow[j,j] = A[j,j]*Sigma[j,j]/Sigma[j,j] = A[j,j] for any nonzero variance."""
    A = make_system_matrix(0.5, 1.0, 0.4)
    Sigma = jnp.eye(4) * 2.0 + 0.3 * (jnp.ones((4, 4)) - jnp.eye(4))  # SPD-ish, nonzero diag
    Sigma = 0.5 * (Sigma + Sigma.T)
    flow = raw_flow(A, Sigma)
    assert jnp.allclose(jnp.diagonal(flow), jnp.diagonal(A))


def test_raw_flow_structural_pattern():
    """flow[i,j] nonzero iff A[j,i] != 0 (given a full covariance)."""
    A = make_system_matrix(0.5, 1.0, 0.4)
    Sigma = jnp.full((4, 4), 0.5) + jnp.eye(4)     # all entries nonzero, symmetric
    flow = np.asarray(raw_flow(A, Sigma))
    for i in range(4):
        for j in range(4):
            if (i, j) in NONZERO_ENTRIES:
                assert flow[i, j] != 0.0, f"({i},{j}) should be nonzero"
            else:
                assert flow[i, j] == 0.0, f"({i},{j}) should be zero"


def test_raw_flow_zero_variance_guard():
    """A zero variance on the diagonal must not produce nan/inf."""
    A = make_system_matrix(0.5, 1.0, 0.4)
    Sigma = jnp.diag(jnp.array([1.0, 1.0, 0.0, 1.0]))  # column-2 variance is 0
    flow = np.asarray(raw_flow(A, Sigma))
    assert np.all(np.isfinite(flow))


# ---------------------------------------------------------------- normalise

def test_normalise_bounded_and_finite():
    A = make_system_matrix(0.5, 1.0, 0.4)
    Sigma = jnp.full((4, 4), 0.4) + jnp.eye(4)
    n = np.asarray(normalise_flow(raw_flow(A, Sigma)))
    assert np.all(np.isfinite(n))
    assert np.all(n >= -1e-12)
    assert np.all(n <= 1 + 1e-9)


def test_normalise_column_sums_to_one():
    """Every nonzero column of |flow|/colsum(|flow|) must sum to exactly 1."""
    A = make_system_matrix(0.5, 1.0, 0.4)
    Sigma = jnp.full((4, 4), 0.4) + jnp.eye(4)
    n = np.asarray(normalise_flow(raw_flow(A, Sigma)))
    col_sums = n.sum(axis=0)
    for j, s in enumerate(col_sums):
        if s != 0.0:
            assert np.isclose(s, 1.0), f"column {j} sums to {s}, expected 1"


def test_normalise_zero_column_guard():
    """An all-zero flow column must yield zeros, not nan."""
    flow = jnp.zeros((4, 4)).at[0, 0].set(3.0)   # only column 0 has mass
    n = np.asarray(normalise_flow(flow))
    assert np.all(np.isfinite(n))
    assert np.allclose(n[:, 1:], 0.0)            # empty columns stay 0
    assert np.isclose(n[0, 0], 1.0)


# ---------------------------------------------------------------- covariance ODE

def test_covariance_matches_matrix_exponential():
    """dSigma/dt = A Sigma + Sigma A^T has closed form Sigma(t) = e^{At} Sigma0 e^{At}^T.

    Integrate the covariance block alone and compare to the analytic solution.
    """
    import diffrax
    beta, K, mu = 0.5, 1.0, 0.4
    A = make_system_matrix(beta, K, mu)
    Sigma0 = jnp.diag(jnp.asarray(SIGMA_INIT) ** 2)
    t1 = 1.0

    def vf(t, y, args):
        return args @ y + y @ args.T

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Tsit5(),
        t0=0.0, t1=t1, dt0=0.01, y0=Sigma0, args=A,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
    )
    numeric = np.asarray(sol.ys[-1]) # type: ignore

    An = np.asarray(A)
    E = expm(An * t1)
    analytic = E @ np.asarray(Sigma0) @ E.T
    assert np.allclose(numeric, analytic, atol=1e-6)


# ---------------------------------------------------------------- integrate single

def test_integrate_single_shape_and_finite():
    params = jnp.array([0.5, 1.0, 0.4])
    out = _integrate_single(params, TAU_INIT, SIGMA_INIT, 1e-6, 1e-9, 100_000)
    assert out.shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(out)))


def test_integrate_single_zero_interval_is_zero():
    """Integrating over [t0, t0] must give the zero matrix (empty interval)."""
    params = jnp.array([0.5, 1.0, 0.4])
    out = _integrate_single(params, (0.0, 0.0, 2), SIGMA_INIT, 1e-6, 1e-9, 100_000)
    assert np.allclose(np.asarray(out), 0.0)


def test_integrate_single_structural_pattern():
    """Integrated flow keeps the T(i->j) nonzero pattern (masked entries stay 0)."""
    params = jnp.array([0.5, 1.0, 0.4])
    out = np.asarray(_integrate_single(params, TAU_INIT, SIGMA_INIT, 1e-6, 1e-9, 100_000))
    for i in range(4):
        for j in range(4):
            if (i, j) not in NONZERO_ENTRIES:
                assert np.isclose(out[i, j], 0.0), f"({i},{j}) should integrate to 0"


# ---------------------------------------------------------------- grid driver

def test_grid_shape_and_finite():
    p = build_params_tensor_jax(PARAMS)
    nlkif = normalised_lkif_grid(p, TAU_INIT, SIGMA_INIT)
    assert nlkif.shape == (2, 2, 2, 4, 4)
    assert np.all(np.isfinite(np.asarray(nlkif)))


def test_grid_matches_single():
    """Each grid cell must equal _integrate_single on that cell's params."""
    p = build_params_tensor_jax(PARAMS)
    nlkif = np.asarray(normalised_lkif_grid(p, TAU_INIT, SIGMA_INIT))
    # spot-check one corner
    corner = np.asarray(
        _integrate_single(p[0, 0, 0], TAU_INIT, SIGMA_INIT, 1e-6, 1e-9, 100_000)
    )
    assert np.allclose(nlkif[0, 0, 0], corner, atol=1e-6)


def test_grid_bounded_by_interval():
    """Normalised integrand <= 1 pointwise => each integral <= (tau_end - tau_start)."""
    p = build_params_tensor_jax(PARAMS)
    nlkif = np.asarray(normalised_lkif_grid(p, TAU_INIT, SIGMA_INIT))
    span = TAU_INIT[1] - TAU_INIT[0]
    assert np.all(np.abs(nlkif) <= span + 1e-6)