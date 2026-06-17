import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import jax.numpy as jnp
# import jax
from algorithms.ccc.ETC_helpers import dimensionsToOne



def test_difference():
    data_matlab = loadmat('./experiments/ccc/CCC_Comparison_Array.mat')
    data_py = np.load('./experiments/ccc/CCC_Comparison_Py.npz')

    print(data_py.files)
    var_names = [k for k in data_matlab.keys() if not k.startswith('__')]
    print("Your MATLAB variables:", var_names)

    data_py = data_py['arr_0']
    data_matlab = data_matlab['CCC_array'].reshape(*data_py.shape)
    print(f"MATLAB shape: {data_matlab.shape}, Python shape: {data_py.shape}")
    
    diff_arr = ((data_py - data_matlab)/data_matlab) * 100

    non_zero_values = diff_arr[diff_arr != 1]
    flat_data = diff_arr.ravel()
    print(non_zero_values)

    counts, bins, patches = plt.hist(flat_data, bins='auto', histtype='stepfilled', alpha=0.6)
    plt.xticks(bins, labels=[f"{x:.2f}" for x in bins], rotation=90, fontsize=8)

    # 3. Add vertical gridlines that line up perfectly with the bin edges
    plt.grid(axis='x', color='red', linestyle='--', alpha=0.5) 
    plt.grid(axis='y', alpha=0.3)

    plt.xlabel('Error in JAX CCC (% age)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
def test_allwindows_difference():
    data_matlab = loadmat('./experiments/ccc/CCC_All_Windows_MATLAB.mat')
    data_py = np.load('./experiments/ccc/CCC_All_Windows_Py.npz')

    print(data_py.files)
    var_names = [k for k in data_matlab.keys() if not k.startswith('__')]
    print("Your MATLAB variables:", var_names)

    data_py = data_py['arr_0']
    data_matlab = data_matlab['CCC_all_windows']
    data_matlab = data_matlab.T
    print(data_matlab.shape, data_py.shape)
    
    # diff_arr = data_matlab - data_py
    diff_arr = data_matlab - data_py

    # non_zero_values = diff_arr[diff_arr != 1]
    # flat_data = diff_arr.ravel()
    # print(non_zero_values)
    """
    print("MATLAB row 0:", data_matlab[0, :])
    print("Python row 0:", data_py[0, :])
    print("MATLAB row 5:", data_matlab[5, :])
    print("Python row 5:", data_py[5, :])
    """

    
    for i in range(4):
        counts, bins, patches = plt.hist(diff_arr[:, i], bins='auto', histtype='stepfilled', alpha=0.6)
        plt.xticks(bins, labels=[f"{x:.2f}" for x in bins], rotation=90, fontsize=8)

        # 3. Add vertical gridlines that line up perfectly with the bin edges
        plt.grid(axis='x', color='red', linestyle='--', alpha=0.5) 
        plt.grid(axis='y', alpha=0.3)

        plt.xlabel(f"Difference between MATLAB CCC and JAX CCC in window {i}")
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    
def test_jax_lax_slicing():
    a = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # x1 = jax.lax.dynamic_slice_in_dim(a, 0, 1, axis = 0)
    x1 = dimensionsToOne(a, 8)
    print(x1.shape)
    print(x1)

if __name__=="__main__":
    test_difference()
    # test_allwindows_difference()
    # test_jax_lax_slicing()