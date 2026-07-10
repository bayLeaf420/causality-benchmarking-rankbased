from algorithms.tc_numerical import time_integrals 
from utils import build_params_tensor_jax

SAVING_DIR = 

BETA_CONF = (0.25, 2.25, 5)
K_CONF = (0.25, 2.25, 6)
MU_CONF = (0.25, 2.25, 5)
N_SAMPLES = 10

TAU_CONF = (0.0, 10.0, 400)
MEAN_CONF = (1.0, 0.0, -1.3, 0.0)
SIGMA_CONF = (0.25, 0.25, 0.25, 0.25)

def run_and_save_tc() -> None:
    params_dict = {
        'beta': BETA_CONF,
        'K': K_CONF,
        'mu': MU_CONF,
    }
    params_tensor = build_params_tensor_jax(params_dict)
    time_integrals_tc = time_integrals(
        params_tensor,
        TAU_CONF,
        SIGMA_CONF,
    )


