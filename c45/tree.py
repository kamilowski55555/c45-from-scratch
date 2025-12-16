"""
Decision Tree classes for C4.5 algorithm.

Contains TreeNode and DecisionTree classes for building and using
decision trees with the C4.5 (Gain Ratio) criterion.
"""

import numpy as np
from .entropy import (
    calculate_entropy,
    calculate_attribute_info,
    calculate_information_gain,
    calculate_split_info,
    calculate_gain_ratio,
)


class TreeNode:
    """
    A node in the decision tree.

    Can be either:
    - Internal node: splits on an attribute
    - Leaf node: makes a decision

    Attributes:
        attribute_index: Index of attribute to split on (internal nodes)
        attribute_name: Name of attribute (if provided)
        decision: Class label (leaf nodes)
        children: Dict mapping attribute values to child nodes
        is_leaf: Whether this is a leaf node
    """

    def __init__(self, attribute_index=None, attribute_name=None, decision=None):
        self.attribute_index = attribute_index
        self.attribute_name = attribute_name
        self.decision = decision
        self.children = {}
        self.is_leaf = False

    def set_as_leaf(self, decision):
        """Convert this node to a leaf with the given decision."""
        self.is_leaf = True
        self.decision = decision
        self.attribute_index = None
        self.attribute_name = None
        self.children = {}


class DecisionTree:
    """
    C4.5 Decision Tree Classifier.

    Uses Gain Ratio criterion to select the best attribute for splitting,
    which improves upon ID3's Information Gain by normalizing for
    attributes with many values.

    Attributes:
        root: The root TreeNode of the built tree
        feature_names: Optional list of attribute names

    Example:
        >>> tree = DecisionTree()
        >>> tree.fit(data, decision_idx=-1)
        >>> prediction = tree.predict(sample)
        >>> tree.print_tree()
    """

    def __init__(self, feature_names=None):
        """
        Initialize the decision tree.

        Args:
            feature_names: Optional list of names for each attribute
        """
        self.root = None
        self.feature_names = feature_names
        self.decision_idx = None

    def fit(self, data, decision_idx=-1):
        """
        Build the decision tree from training data.

        Args:
            data: NumPy array where each row is a sample
            decision_idx: Index of the target/decision column (default: last)

        Returns:
            self: The fitted tree
        """
        self.decision_idx = decision_idx if decision_idx >= 0 else data.shape[1] + decision_idx
        self.root = self._build_tree(data, self.decision_idx)
        return self

    def predict(self, sample):
        """
        Predict the class for a single sample.

        Args:
            sample: List or array of attribute values

        Returns:
            The predicted class label
        """
        return self._predict_single(self.root, sample)

    def predict_batch(self, samples):
        """
        Predict classes for multiple samples.

        Args:
            samples: 2D array where each row is a sample

        Returns:
            List of predicted class labels
        """
        return [self.predict(sample) for sample in samples]

    def score(self, data, decision_idx=-1):
        """
        Calculate accuracy on a dataset.

        Args:
            data: NumPy array of samples with true labels
            decision_idx: Index of the true label column

        Returns:
            float: Accuracy (0.0 to 1.0)
        """
        if decision_idx < 0:
            decision_idx = data.shape[1] + decision_idx

        correct = 0
        for row in data:
            # Create sample without decision column
            sample = list(row)
            true_label = sample.pop(decision_idx)
            # Reconstruct sample for prediction
            predicted = self.predict(row)
            if predicted == true_label:
                correct += 1
        return correct / len(data)

    def print_tree(self, node=None, indent="", edge_name=None):
        """
        Print a text representation of the tree.

        Args:
            node: Starting node (default: root)
            indent: Current indentation string
            edge_name: Label of the edge leading to this node
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            if edge_name:
                print(f"{indent}[{edge_name}] → Decision: {node.decision}")
            else:
                print(f"{indent}Decision: {node.decision}")
            return

        # Get attribute name
        attr_name = node.attribute_name
        if attr_name is None and self.feature_names and node.attribute_index < len(self.feature_names):
            attr_name = self.feature_names[node.attribute_index]
        if attr_name is None:
            attr_name = f"Attribute[{node.attribute_index}]"

        if edge_name:
            print(f"{indent}[{edge_name}] → {attr_name}?")
        else:
            print(f"{indent}{attr_name}?")

        new_indent = indent + "    "
        for value, child in node.children.items():
            self.print_tree(child, new_indent, edge_name=value)

    def _build_tree(self, data, decision_idx):
        """Recursively build the decision tree."""
        node = TreeNode()

        # Check if all decisions are the same (pure node)
        decisions = data[:, decision_idx]
        if len(np.unique(decisions)) == 1:
            node.set_as_leaf(decisions[0])
            return node

        # Find best attribute using Gain Ratio
        num_attributes = data.shape[1]
        best_gain_ratio = -1
        best_attr_idx = -1

        # Calculate system entropy
        _, decision_counts = np.unique(decisions, return_counts=True)
        system_entropy = calculate_entropy(decision_counts)

        # Evaluate each attribute
        for i in range(num_attributes):
            if i == decision_idx:
                continue

            # Calculate metrics
            info_attr = calculate_attribute_info(data, i, decision_idx)
            gain = calculate_information_gain(system_entropy, info_attr)

            attr_col = data[:, i]
            _, attr_counts = np.unique(attr_col, return_counts=True)
            split_info = calculate_split_info(attr_counts)
            gain_ratio = calculate_gain_ratio(gain, split_info)

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attr_idx = i

        # If no good split found, make leaf with majority class
        if best_gain_ratio <= 0.0:
            majority = self._get_majority_class(data, decision_idx)
            node.set_as_leaf(majority)
            return node

        # Create internal node
        node.attribute_index = best_attr_idx
        if self.feature_names and best_attr_idx < len(self.feature_names):
            node.attribute_name = self.feature_names[best_attr_idx]

        # Split data and recurse
        best_col_data = data[:, best_attr_idx]
        unique_values = np.unique(best_col_data)

        for val in unique_values:
            mask = (data[:, best_attr_idx] == val)
            subset_data = data[mask]
            child_node = self._build_tree(subset_data, decision_idx)
            node.children[val] = child_node

        return node

    def _predict_single(self, node, sample):
        """Traverse tree to predict class for a sample."""
        if node.is_leaf:
            return node.decision

        # Get attribute value from sample
        attr_value = sample[node.attribute_index]

        # Follow the appropriate branch
        if attr_value in node.children:
            return self._predict_single(node.children[attr_value], sample)
        else:
            # Unknown value - return most common child decision
            # (Simple fallback strategy)
            for child in node.children.values():
                if child.is_leaf:
                    return child.decision
            return list(node.children.values())[0].decision

    def _get_majority_class(self, data, decision_idx):
        """Returns the most common decision in the dataset."""
        decisions = data[:, decision_idx]
        values, counts = np.unique(decisions, return_counts=True)
        return values[np.argmax(counts)]

