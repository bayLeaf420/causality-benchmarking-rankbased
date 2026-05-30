import sympy as sp

# This library tells theoretical causality between coupled oscillators

def symbolic_information_flow():

    # Define time symbols (real so that sympy can simplify the math)
    t, t0 = sp.symbols('t t0', real=True)
    # Define oscillator parameters
    k, b, m, omega = sp.symbols('k b m omega', real = True)
    # Covariances = To be used for calculating flow
    s11, s22, s33, s44 = sp.symbols('sigma11 sigma22 sigma33 sigma44', real=True)

    #Define system matrix A
    A = sp.Matrix([
        [0, 1, 0, 0],
        [-k, -b/m, -omega**2, 0],
        [0, 0, 0, 1],
        [0, 0, -omega**2, 0],
    ])

    # Define initial covariance matrix
    Sigma_0 = sp.Matrix([
        [s11**2, 0, 0, 0],
        [0, s22**2, 0, 0],
        [0, 0, s33**2, 0],
        [0, 0, 0, s44**2],
    ])

    # tau represents (t - t0)
    tau = t - t0
    exp_At = (A * tau).exp()

    # Sigma_t is covariance matrix as a function of time
    Sigma_t = exp_At * Sigma_0 * exp_At.T

    # Extract terms needed for T_(3 -> 2)
    # sigma_22_t = Sigma_t[1, 1]
    # sigma_23_t = Sigma_t[1, 2]
    # a_23 = A[1, 2] # This is -omega**2

    # T_3_to_2 = a_23 * (sigma_23_t / sigma_22_t)

    information_flow = {'1->2': A[1, 0] * (Sigma_t[1, 0]/Sigma_t[1, 1]),
                        '2->1': A[0, 1] * (Sigma_t[0, 1]/Sigma_t[0, 0]),
                        '3->2': A[1, 2] * (Sigma_t[1, 2]/Sigma_t[1, 1]),
                        '4->3': A[2, 3] * (Sigma_t[2, 3]/Sigma_t[2, 2]),
                        '3->4': A[3, 2] * (Sigma_t[3, 2]/Sigma_t[3, 3]),
                        }

    return information_flow, Sigma_t

"""
# Get the dictionary of information flows
flow_dict = symbolic_information_flow()

# Open the file and write each flow on its own line
with open('output.txt', 'w') as f:
    for direction, expression in flow_dict.items():
        latex_str = sp.latex(expression)
        # Option A: Writes with the label (e.g., "1->2: \frac{...}{...}")
        f.write(f"{direction}: {latex_str}\n")
        
        # Option B: If you ONLY want the raw math values without labels, use this instead:
        # f.write(f"{latex_str}\n")
"""
