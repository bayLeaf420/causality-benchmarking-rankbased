import matplotlib.pyplot as plt
import numpy as np
from sampling import _lambdify_oscillator

def test_lambdify_1():
    tau_init = (0.0, 10.0, 200)
    mean_init = (0.8, 0.0, 0.0, 1.5)
    sigma_init = (1.0, 1.0, 1.0, 1.0)
    mean_lambda, sigma_lambda = _lambdify_oscillator(tau_init, mean_init, sigma_init)
    beta, K, mu = 1.0, 2.3, 0.4

    vectorised_mean = np.vectorize(mean_lambda)
    # vectorised_std_dev = 
    tau_vec = np.linspace(tau_init[0], tau_init[1], tau_init[2])
    mean_vec = vectorised_mean(tau_vec, beta, K, mu)[1]

    fig, axes = plt.subplot(2, 1, figsize=(8, 6))
    axes[0].plot(tau_vec, mean_vec)
    # axes[1].plot

    return None 

if __name__=="__main__":
    test_lambdify_1()

