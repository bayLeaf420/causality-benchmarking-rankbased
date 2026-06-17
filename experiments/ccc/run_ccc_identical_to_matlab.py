# import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat

from algorithms.ccc.ETC_and_CCC import CCC_calculation #, bin_timeseries
from oscillators import damped_driven_oscillator, driving_force

t = jnp.linspace(0, 10, 500)
n_curves = 40
omega_vals = jnp.linspace(0.2, 4.2, n_curves)

x = jnp.zeros((n_curves, 500))
F = jnp.zeros((n_curves, 500))
theoretical_causality = np.zeros(n_curves)

params = [25, 8, 32, 0.4, 0, 1.7, 0]

for i, w in enumerate(omega_vals):
    params[-1] = w # Last one of params is omega
    x_i, A, B, C, alpha = damped_driven_oscillator(t, params)
    C = jnp.abs(jnp.real(C))
    A = jnp.abs(jnp.real(A))
    B = jnp.abs(jnp.real(B))

    x = x.at[i].set(x_i)
    theoretical_causality[i] = C / (C + (jnp.sum(jnp.exp(-alpha * t) * (A + B)) / 500))
    F = F.at[i].set(driving_force(t, params))

input_data = jnp.stack([F, x], dtype=jnp.float32)
print(input_data.shape)


# --- CCC loop ---
CCC_mean = np.zeros((n_curves,))


# cause_seq = F[jj], effect_seq = x[jj]; batch over jj via vmap (built into CCC_calculation_2vec)
for i in range(n_curves):
    curr_data = input_data[:, i, :]
    curr_data = curr_data.reshape(2, 500)
    # print(f"\n{'='*50}\nCurrent data shape is {curr_data.shape}\n{'='*50}\n")
    # if curr_data != (2, 500):
        # raise ValueError("Current data has wrong dims")
    CCC_mean[i], _ = CCC_calculation(curr_data, (90, 90, 90), (12, 12))
    
plt.plot(np.array(omega_vals), CCC_mean, label=f"numBins = {12}")
plt.xlabel("Driving Frequency values")
plt.ylabel("CCC values")
plt.legend()
plt.show()

# --- Theoretical causality plot ---
plt.figure()
plt.plot(np.array(omega_vals), theoretical_causality, color="black")
plt.show()

# --- Save ---
np.savez("./experiments/ccc/CCC_Comparison_Py.npz", CCC_mean)
# np.savez("./experiments/ccc/CCC_All_Windows_Py.npz", ccc_all_windows)
