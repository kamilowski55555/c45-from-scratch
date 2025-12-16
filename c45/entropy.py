"""
Entropy and Information Theory calculations for C4.5 algorithm.

This module contains the core mathematical functions:
- Entropy (information measure)
- Information Gain (ID3 criterion)
- Split Info and Gain Ratio (C4.5 improvement over ID3)
"""

import numpy as np


def calculate_entropy(counts):
    """
    Calculates entropy I(P) based on counts.

    Entropy measures the impurity/disorder in a dataset.
    Formula: H(S) = -Σ p_i * log2(p_i)

    Args:
        counts: Array or list of class counts

    Returns:
        float: Entropy value (0 = pure, higher = more mixed)

    Example:
        >>> calculate_entropy([5, 5])  # 50-50 split
        1.0
        >>> calculate_entropy([10, 0])  # Pure
        0.0
    """
    counts = np.array(list(counts))
    total = np.sum(counts)

    if total == 0:
        return 0.0

    probs = counts / total
    # Filter out zero probabilities to avoid log2(0) error
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calculate_attribute_info(data, attribute_idx, decision_idx=-1):
    """
    Calculates the Information Function (Info_x(T)) for a specific attribute.

    This measures the expected entropy after splitting on the attribute.
    Formula: Info_x(T) = Σ (|Ti| / |T|) * Info(Ti)

    Args:
        data: NumPy array of the dataset
        attribute_idx: Index of the attribute column
        decision_idx: Index of the decision/target column (default: last)

    Returns:
        float: Weighted average entropy after split
    """
    total_rows = len(data)
    attribute_column = data[:, attribute_idx]
    unique_values = np.unique(attribute_column)

    info_attribute = 0.0

    for value in unique_values:
        # Select rows where attribute equals the current value
        mask = (attribute_column == value)
        subset_decisions = data[mask, decision_idx]

        # Count decision classes in this subset
        _, subset_counts = np.unique(subset_decisions, return_counts=True)

        # Calculate entropy for this subset
        subset_entropy = calculate_entropy(subset_counts)

        # Weight by subset size
        weight = len(subset_decisions) / total_rows
        info_attribute += weight * subset_entropy

    return info_attribute


def calculate_information_gain(system_entropy, attribute_info):
    """
    Calculates Information Gain (Gain(X)).

    Measures how much information an attribute provides.
    Formula: Gain(X) = Info(T) - Info_x(T)

    Args:
        system_entropy: Entropy of the whole dataset
        attribute_info: Weighted entropy after splitting on attribute

    Returns:
        float: Information gain (higher = better attribute)
    """
    return system_entropy - attribute_info


def calculate_split_info(attribute_counts):
    """
    Calculates SplitInfo(X, T) - the intrinsic information of a split.

    This penalizes attributes with many unique values.
    Formula: SplitInfo(X, T) = -Σ (|Ti| / |T|) * log2(|Ti| / |T|)

    Args:
        attribute_counts: Counts of each unique value in the attribute

    Returns:
        float: Split information value
    """
    return calculate_entropy(attribute_counts)


def calculate_gain_ratio(gain, split_info):
    """
    Calculates GainRatio(X, T) - the C4.5 improvement over ID3.

    Normalizes information gain by split info to avoid bias
    toward attributes with many values.
    Formula: GainRatio(X, T) = Gain(X, T) / SplitInfo(X, T)

    Args:
        gain: Information gain value
        split_info: Split information value

    Returns:
        float: Gain ratio (higher = better attribute)
    """
    if split_info == 0:
        return 0.0
    return gain / split_info

