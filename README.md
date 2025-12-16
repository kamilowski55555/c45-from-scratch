# C4.5 Decision Tree - Learning from Scratch

My implementation of the **C4.5 decision tree algorithm** from scratch using only NumPy.

This is a minimal decision tree classifier that uses Gain Ratio for attribute selection. The goal is to understand how decision trees work at a fundamental level by building them from scratch.

## What's Inside

```
c45/
├── entropy.py         # Entropy, Information Gain, Gain Ratio calculations
├── tree.py            # DecisionTree and TreeNode classes
├── data.py            # Data loading utilities
└── __init__.py

demo.py                # Demo with multiple datasets
gielda.txt             # Stock market dataset (Polish)
car.data               # UCI Car Evaluation dataset
```

## The Algorithm

C4.5 is an improvement over ID3, created by Ross Quinlan. The key difference:

- **ID3** uses Information Gain → biased toward attributes with many values
- **C4.5** uses Gain Ratio → normalizes by Split Info to avoid this bias

```
Gain Ratio = Information Gain / Split Info
```

## Quick Example

```python
from c45 import DecisionTree, load_data

# Load data
data = load_data('stock.txt')

# Build the tree
tree = DecisionTree(feature_names=['Age', 'Dividend', 'P/E Ratio'])
tree.fit(data, decision_idx=-1)

# Visualize
tree.print_tree()

# Predict
sample = ['new', 'yes', 'hwr', '?']
prediction = tree.predict(sample)
print(f"Prediction: {prediction}")
```

## Understanding Entropy

```python
from c45 import calculate_entropy

# Pure dataset (all same class) - entropy = 0
calculate_entropy([10, 0])  # 0.0

# Perfectly mixed (50-50) - maximum entropy
calculate_entropy([5, 5])   # 1.0

# Multi-class entropy
calculate_entropy([4, 4, 4])  # 1.585
```

## Training on Real Data

```python
from c45 import DecisionTree, load_data
from c45.data import train_test_split

# Load UCI Car Evaluation dataset
data = load_data('car.data')

# Split into train/test
train_data, test_data = train_test_split(data, test_ratio=0.2, random_state=42)

# Build tree
feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
tree = DecisionTree(feature_names=feature_names)
tree.fit(train_data, decision_idx=-1)

# Evaluate
accuracy = tree.score(test_data, decision_idx=-1)
print(f"Test Accuracy: {accuracy:.2%}")
```

## Run the Demo

```bash
python demo.py
```

The demo includes:
- Basic entropy calculations
- Building a tree on stock market data
- Classic weather/play tennis example
- Training on UCI Car Evaluation dataset

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Entropy** | Measures impurity/disorder in a dataset (0 = pure, 1 = mixed) |
| **Information Gain** | How much entropy is reduced by splitting on an attribute |
| **Split Info** | Intrinsic information of the split itself |
| **Gain Ratio** | Information Gain normalized by Split Info (C4.5's criterion) |

## Requirements

- Python 3.7+
- NumPy

## How It Works

1. **Calculate entropy** of the current dataset
2. For each attribute:
   - Calculate **Information Gain**
   - Calculate **Split Info**  
   - Compute **Gain Ratio** = Info Gain / Split Info
3. Select attribute with **highest Gain Ratio**
4. Split data and **recursively** build child nodes
5. Stop when all instances have the same class

