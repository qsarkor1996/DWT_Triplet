import pywt
import numpy as np

def apply_dwt_multiresolution(data, wavelet='db1', level=3):
    """
    Apply Discrete Wavelet Transform (DWT) multiresolution analysis to the data.

    Args:
        data (numpy.ndarray): Input signal data (2D array: samples x time).
        wavelet (str): Wavelet type (default is 'db1').
        level (int): Level of decomposition (default is 3).

    Returns:
        list of numpy.ndarray: List containing DWT coefficients for each signal.
    """
    dwt_coefficients = []
    for signal in data:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        dwt_coefficients.append(coeffs)
    return dwt_coefficients


def apply_dwt_to_noisy_training_data(noisy_train_data, wavelet='db1', level=3):
    """
    Apply DWT multiresolution analysis to all classes in noisy training data.

    Args:
        noisy_train_data (dict): Dictionary of noisy training data split by class.
        wavelet (str): Wavelet type (default is 'db1').
        level (int): Level of decomposition (default is 3).

    Returns:
        dict: Dictionary containing DWT coefficients for each class.
    """
    dwt_data = {}
    for class_name, class_array in noisy_train_data.items():
        dwt_data[class_name] = apply_dwt_multiresolution(class_array, wavelet=wavelet, level=level)
        print(f"Applied DWT to class '{class_name}' with wavelet '{wavelet}' and level {level}.")
    return dwt_data