import os
import numpy as np
import scipy.io as sio

def load_mat_data(data_path):
    """
    Load data from a .mat file.

    Args:
        data_path (str): Path to the .mat file.

    Returns:
        dict: Data dictionary from the .mat file.
    """
    try:
        mat_data = sio.loadmat(data_path)
        print(f"Data loaded successfully from {data_path}")
        return mat_data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {data_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading .mat file: {e}")


def save_data(data, output_path):
    """
    Save data to a .npy file.

    Args:
        data (any): Data to be saved (e.g., numpy array, dictionary).
        output_path (str): Path to save the .npy file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, data)
    print(f"Data saved to {output_path}")


def min_max_scaling(data, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to the data.

    Args:
        data (numpy.ndarray): Input data array to be scaled.
        feature_range (tuple): Desired range for scaling (default is (0, 1)).

    Returns:
        numpy.ndarray: Scaled data.
    """
    min_val, max_val = feature_range
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    range_diff = data_max - data_min
    range_diff[range_diff == 0] = 1e-8  # Avoid division by zero
    scaled_data = (data - data_min) / range_diff
    return scaled_data * (max_val - min_val) + min_val


def split_data(data, slices):
    """
    Split data into different classes based on provided slice indices.

    Args:
        data (numpy.ndarray): Input data to be split.
        slices (dict): Dictionary with class names as keys and tuple (start, end) as values.

    Returns:
        dict: Dictionary of split data for each class.
    """
    class_data = {}
    try:
        for name, (start, end) in slices.items():
            class_data[name] = data[start:end, :]
        print("Data split into classes successfully.")
        return class_data
    except IndexError as e:
        raise IndexError(f"IndexError: {e} - Check data shape and slice indices.")


def save_class_data(class_data, output_folder):
    """
    Save class data as .npy files for each class.

    Args:
        class_data (dict): Dictionary with class names as keys and data as values.
        output_folder (str): Path to save class data files.
    """
    os.makedirs(output_folder, exist_ok=True)
    for class_name, class_array in class_data.items():
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        output_file = os.path.join(class_folder, f"{class_name}.npy")
        np.save(output_file, class_array)
        print(f"Saved data for class '{class_name}' to {output_file}.")