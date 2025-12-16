"""
C4.5 Decision Tree - Learning from Scratch

A minimal implementation of the C4.5 decision tree algorithm.
The goal is to understand how decision trees work at a fundamental level
by building them from scratch.
"""

from .entropy import calculate_entropy, calculate_information_gain, calculate_split_info, calculate_gain_ratio
from .tree import DecisionTree, TreeNode
from .data import load_data

__all__ = [
    'DecisionTree',
    'TreeNode',
    'load_data',
    'calculate_entropy',
    'calculate_information_gain',
    'calculate_split_info',
    'calculate_gain_ratio',
]

