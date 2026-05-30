import os
import sys
import numpy as np
import sympy as sp
# import jax 
# import jax.numpy as jnp
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath('theoretical_causality'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Going to project parent folder
from theoretical_causality.sym_information import symbolic_information_flow
from oscillators.oscillators import damped_driven_oscillator_numpy
# from runs.theoretical_runs import create_params_grid

def evaluate_theoretical_causality(var, parameters, time_dict):
    """
    Evaluates information flow T_(3->2) over a specified time interval.

    Inputs:
    ------------
    mean: np.array([mu1, mu2, mu3, mu4])
        Initial mean vector for gaussian over initial conditions. 
    var: np.array([sigma11**2, sigma22**2, sigma33**2, sigma44**2])
        Represents the diagonal of covariance matrix. 
    parameters: list([k, b, m, omega])
        Oscillator parameters. 
    time_list: dict('start', 'stop', 'num')
        Time config. 

    Outputs:
    ------------
    np.array:
        Evaluated T(3->2) at each time point

    1. symbolic_information_flow() imports the symbolic expression for T(i->j)
       for all i, j for which the flow is not 0 as a dict. 
    2. We choose 3->2 and 2->1 from the dict as that is x2 -> v1 and v1 -> x1.  
    3. We evaluate the expression at each timepoint as a lambda function. 
    4. We return the resulting arrays. 
    """

    # 1. Unpack inputs
    k_val, b_val, m_val, omega_val = parameters
    start, stop, num_timepoints = time_dict['start'], time_dict['stop'], time_dict['num']
    
    # 2. Generate time array
    t_array = np.linspace(start, stop, num_timepoints)

    # 3. Get symbolic expression
    flow_dict, covar_matrix = symbolic_information_flow()
    expr32 = flow_dict['3->2']
    expr21 = flow_dict['2->1']

    # 4. Re-create matching symbols to map inputs correctly
    t, t0 = sp.symbols('t t0', real=True)
    k, b, m, omega = sp.symbols('k b m omega', real=True)
    s11, s22, s33, s44 = sp.symbols('sigma11 sigma22 sigma33 sigma44', real=True)

    # 5. Build substitutions dictionary to input into expression
    substitution_map = {
        t0: start,
        m: m_val,
        b: b_val,
        k: k_val,
        # F0: F0_val,
        omega: omega_val,
        s11: var[0],
        s22: var[1],
        s33: var[2],
        s44: var[3],
    }

    # 6. Substitute constants and leave 't', make function
    num32 = expr32.subs(substitution_map)
    num21 = expr21.subs(substitution_map)
    num_s11 = covar_matrix[0, 0].subs(substitution_map)
    t32_numpy = sp.lambdify(t, num32, modules='numpy')
    t21_numpy = sp.lambdify(t, num21, modules='numpy')
    sigma11_numpy = sp.lambdify(t, num_s11, modules='numpy')
    

    # 7. Evaluate over array of time-points
    result32 = t32_numpy(t_array)
    result21 = t21_numpy(t_array)
    result_sigma11 = sigma11_numpy(t_array)
    

    # 8. If result evaluates to a single scalar, broadcast it 
    # to full array, so it still plots. 
    if isinstance(result32, (int, float, np.number)):
        result32 = np.full_like(t_array, result32)
    if isinstance(result21, (int, float, np.number)):
        result21 = np.full_like(t_array, result21)
    if isinstance(result_sigma11, (int, float, np.number)):
        result_sigma11 = np.full_like(t_array, result_sigma11)

    return result32, result21, result_sigma11

def plot_theoretical_causality(mu, var, parameters, time_dict):

    # 1. Calculate the information flow values, check shape
    t32_values, t21_values, s11_values = evaluate_theoretical_causality(var, parameters, time_dict)

    # 2. Check for NaNs
    for name, dataset in [("T(3->2)", t32_values), ("T(2->1)", t21_values), ("S(1,1)", s11_values)]:
        if np.any(np.isnan(dataset)):
            nan_count = np.sum(np.isnan(dataset))
            nan_indices = np.where(np.isnan(dataset))[0]
            print(f"[WARNING] {name} failed check! Found {nan_count} NaN(s) at indices: {nan_indices}")
        else:
            print(f"[SUCCESS] {name} check passed. No NaNs detected.")

    # 3. Make t_array again for plotting X-axis
    start, stop, num_timepoints = time_dict['start'], time_dict['stop'], time_dict['num']
    t_array = np.linspace(start, stop, num_timepoints)

    # 4. Find oscillator mean value for plot:

    # note that damped driven osc func takes params as [k, b, m, x0, v0, F0, omega]
    # x1_0, v1_0 obtained from mu[0:2]
    # x2_0, v2_0 are (-F0/(m*omega**2))*cos(-delta) and d/dt(x2_0) as x2 = (-F0/(m*omega**2))*cos(omega*t). Therefore F0 = (mu_3)*(-m*omega**2)
    # parameters list is [k, b, m, omega]
    x1_0, v1_0, x2_0, _ = mu
    k_val, b_val, m_val, omega_val = parameters
    F0 = x2_0 * (-m_val * (omega_val**2))

    osc_params = [k_val, b_val, m_val, x1_0, v1_0, F0, omega_val]
    osc1_mean, *rest = damped_driven_oscillator_numpy(t_array,osc_params)

    # Find std-dev, calc bounds for cloud
    osc1_std = np.sqrt(s11_values)
    lower_bound = osc1_mean - osc1_std
    upper_bound = osc1_mean + osc1_std

    # =========================================================================
    # VISUALIZATION DECK
    # =========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # --- Top Subplot: Oscillator 3 to Oscillator 2 ---
    ax1.plot(t_array, t32_values, color='#1f77b4', linewidth=2.5, label=r'$T_{3\rightarrow 2}$ (Causality Flow)')
    ax1.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax1.set_title('Theoretical Information Flow Pathways', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Flow Rate (Nats/unit time)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    # --- Bottom Subplot: Oscillator 2 to Oscillator 1 ---
    ax2.plot(t_array, t21_values, color='#ff7f0e', linewidth=2.5, label=r'$T_{2\rightarrow 1}$ (Causality Flow)')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax2.set_xlabel('Time ($t$)', fontsize=11)
    ax2.set_ylabel('Flow Rate (Nats/unit time)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    # --- Another subplot for the oscillator 1 itself ---
    # 1. Plot centre mean trajectory
    ax3.plot(t_array, osc1_mean, color='#2ca02c', linewidth=2, label='Damped Driven Oscillator x1')
    ax3.fill_between(
        t_array,
        lower_bound,
        upper_bound,
        color='#2ca02c',
        alpha=0.25,
        label=r'Evolution of Variance cloud ($\pm 1\sigma$)'
    )
    ax3.set_xlabel('Time ($t$)', fontsize=11)
    ax3.set_ylabel('Time ($x_1$)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        
    # Optimize layout and show
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    plot_theoretical_causality([1, 0.6, -0.2216, 0], [5, 5, 5, 5], [100, 0.8, 1, 14], {'start': 0, 'stop': 1, 'num': 2000})