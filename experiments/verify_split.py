import numpy as np
from experiments.utils import get_dataset

def main():
    print("Checking Covertype dataset loading and class distribution...")
    X_train, X_test, y_train, y_test = get_dataset("covertype", random_state=42)
    
    print("\n--- Train set split ---")
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    for c, count in zip(train_classes, train_counts):
        print(f"Class {c}: {count} samples ({count / len(y_train) * 100:.2f}%)")
        
    print("\n--- Test set split ---")
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    for c, count in zip(test_classes, test_counts):
        print(f"Class {c}: {count} samples ({count / len(y_test) * 100:.2f}%)")
        
    # Check if the minority class (which is class 4 / 3 0-indexed) survived.
    # Class 4 is 3. Let's make sure it is in both train and test.
    assert 3 in train_classes, "Minority class (class 4 / target index 3) missing from train set!"
    assert 3 in test_classes, "Minority class (class 4 / target index 3) missing from test set!"
    print("\n=> Verification successful: Minority class exists in both train and test partitions.")

if __name__ == "__main__":
    main()
