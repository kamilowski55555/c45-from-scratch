"""
Data loading utilities for C4.5 Decision Tree.

Handles reading datasets from text files with automatic separator detection.
"""

import numpy as np


def load_data(filename, separator='auto'):
    """
    Reads data from a text file and returns a NumPy array.

    Supports automatic detection of common separators (comma, semicolon,
    tab, pipe, space).

    Args:
        filename: Path to the data file
        separator: Character to split by. Use 'auto' to detect automatically,
                   or specify ',', ';', '\\t', etc.

    Returns:
        numpy.ndarray: 2D array of string values

    Example:
        >>> data = load_data('weather.csv')
        >>> data = load_data('data.txt', separator='\\t')
    """
    data_list = []
    detected_separator = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if separator == 'auto' and detected_separator is None:
                for sep in [',', ';', '\t', '|', ' ']:
                    if sep in line:
                        detected_separator = sep
                        break
                if detected_separator is None:
                    detected_separator = ','

            active_sep = detected_separator if separator == 'auto' else separator
            row = line.split(active_sep)
            data_list.append(row)

    return np.array(data_list)


def train_test_split(data, test_ratio=0.2, shuffle=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(data)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return data[train_indices], data[test_indices]

