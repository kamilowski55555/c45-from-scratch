"""
C4.5 Decision Tree Demo

This demo shows how to use the C4.5 decision tree implementation:
1. Basic entropy calculations
2. Building a tree from a simple dataset
3. Training on the UCI Car Evaluation dataset
"""

from c45 import DecisionTree, load_data, calculate_entropy
from c45.data import train_test_split
import numpy as np


def demo_entropy():
    """Demonstrate entropy calculations."""
    print("=" * 60)
    print("PART 1: Understanding Entropy")
    print("=" * 60)

    # Pure dataset (all same class) - entropy = 0
    pure_counts = [10, 0]
    print(f"\nPure dataset [10, 0]: Entropy = {calculate_entropy(pure_counts):.4f}")

    # Perfectly mixed (50-50) - entropy = 1
    mixed_counts = [5, 5]
    print(f"Mixed dataset [5, 5]: Entropy = {calculate_entropy(mixed_counts):.4f}")

    # Slightly imbalanced
    imbalanced_counts = [7, 3]
    print(f"Imbalanced [7, 3]:    Entropy = {calculate_entropy(imbalanced_counts):.4f}")

    # Multi-class
    multiclass_counts = [4, 4, 4]
    print(f"3 classes [4, 4, 4]:  Entropy = {calculate_entropy(multiclass_counts):.4f}")

    print("\n-> Higher entropy = more disorder/uncertainty")
    print("-> Lower entropy = more purity/certainty")


def demo_stock_market():
    """Build a tree for the stock market prediction dataset."""
    print("\n" + "=" * 60)
    print("PART 2: Stock Market Prediction (stock.txt)")
    print("=" * 60)

    # Load the stock market dataset
    data = load_data('stock.txt')
    print(f"\nDataset shape: {data.shape}")
    print(f"Sample rows:")
    for i in range(min(3, len(data))):
        print(f"  {data[i]}")

    # Define feature names
    feature_names = ['Age', 'Dividend', 'P/E Ratio', 'Decision']

    # Build the tree
    tree = DecisionTree(feature_names=feature_names[:3])
    tree.fit(data, decision_idx=-1)

    print("\n--- Decision Tree Structure ---")
    tree.print_tree()

    # Make predictions
    print("\n--- Predictions ---")
    test_samples = [
        ['old', 'yes', 'swr', 'down'],
        ['new', 'no', 'hwr', 'up'],
        ['mid', 'yes', 'hwr', 'down'],
    ]

    for sample in test_samples:
        prediction = tree.predict(sample)
        actual = sample[-1]
        status = "OK" if prediction == actual else "MISS"
        print(f"  {sample[:3]} -> Predicted: {prediction}, Actual: {actual} [{status}]")


def demo_manual_tree():
    """Show how to manually create and use a tree."""
    print("\n" + "=" * 60)
    print("PART 3: Quick Example - Weather Dataset")
    print("=" * 60)

    # Classic weather/play tennis dataset
    weather_data = [
        ['sunny', 'hot', 'high', 'weak', 'no'],
        ['sunny', 'hot', 'high', 'strong', 'no'],
        ['overcast', 'hot', 'high', 'weak', 'yes'],
        ['rain', 'mild', 'high', 'weak', 'yes'],
        ['rain', 'cool', 'normal', 'weak', 'yes'],
        ['rain', 'cool', 'normal', 'strong', 'no'],
        ['overcast', 'cool', 'normal', 'strong', 'yes'],
        ['sunny', 'mild', 'high', 'weak', 'no'],
        ['sunny', 'cool', 'normal', 'weak', 'yes'],
        ['rain', 'mild', 'normal', 'weak', 'yes'],
        ['sunny', 'mild', 'normal', 'strong', 'yes'],
        ['overcast', 'mild', 'high', 'strong', 'yes'],
        ['overcast', 'hot', 'normal', 'weak', 'yes'],
        ['rain', 'mild', 'high', 'strong', 'no'],
    ]

    data = np.array(weather_data)

    feature_names = ['outlook', 'temperature', 'humidity', 'wind']

    tree = DecisionTree(feature_names=feature_names)
    tree.fit(data, decision_idx=-1)

    print("\nWeather -> Play Tennis?")
    print("-" * 40)
    tree.print_tree()

    # Test prediction
    print("\n--- Test Prediction ---")
    test = ['sunny', 'cool', 'high', 'weak', '?']
    prediction = tree.predict(test)
    print(f"Weather: outlook={test[0]}, temp={test[1]}, humidity={test[2]}, wind={test[3]}")
    print(f"Play tennis? -> {prediction}")


def demo_car_evaluation():
    """Build a tree for the UCI Car Evaluation dataset."""
    print("\n" + "=" * 60)
    print("PART 4: Car Evaluation Dataset (car.data)")
    print("=" * 60)

    # Load the car evaluation dataset
    data = load_data('car.data')
    print(f"\nDataset shape: {data.shape}")
    print(f"Classes: {set(data[:, -1])}")

    # Feature names for the car dataset
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    # Split into train/test
    train_data, test_data = train_test_split(data, test_ratio=0.2, random_state=42)
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    # Build the tree
    print("\nBuilding decision tree...")
    tree = DecisionTree(feature_names=feature_names)
    tree.fit(train_data, decision_idx=-1)

    # Evaluate
    train_accuracy = tree.score(train_data, decision_idx=-1)
    test_accuracy = tree.score(test_data, decision_idx=-1)

    print(f"\n--- Results ---")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Testing Accuracy:  {test_accuracy:.2%}")

    # Show first few levels of the tree
    print("\n--- Decision Tree Structure ---")
    tree.print_tree()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   C4.5 DECISION TREE - DEMO")
    print("=" * 60)

    demo_entropy()
    demo_stock_market()
    demo_manual_tree()

    # Only run car demo if the file exists
    try:
        demo_car_evaluation()
    except FileNotFoundError:
        print("\n[Skipping car.data demo - file not found]")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
