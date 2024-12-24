import numpy as np

def add_awgn(data, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to the data.

    Args:
        data (numpy.ndarray): Input data (e.g., signals).
        snr_db (float): Signal-to-Noise Ratio (SNR) in decibels.

    Returns:
        numpy.ndarray: Data with AWGN added.
    """
    # Calculate signal power
    signal_power = np.mean(data ** 2, axis=1, keepdims=True)

    # Convert SNR from decibels to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(*data.shape)

    # Add noise to the signal
    noisy_data = data + noise
    return noisy_data


def add_awgn_to_training_data(train_data, snr_db):
    """
    Add AWGN to all classes of training data.

    Args:
        train_data (dict): Dictionary where keys are class names and values are numpy arrays of data.
        snr_db (float): Signal-to-Noise Ratio (SNR) in decibels.

    Returns:
        dict: Dictionary of noisy training data for each class.
    """
    noisy_train_data = {}
    for class_name, class_array in train_data.items():
        noisy_train_data[class_name] = add_awgn(class_array, snr_db)
        print(f"Added AWGN to class '{class_name}' with SNR = {snr_db} dB.")
    return noisy_train_data