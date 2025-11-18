"""
Data loading and preprocessing for CIFAR-10 dataset
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10 # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
from typing import Tuple

class CIFAR10DataLoader:
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    # Returns: Tuple of (X_train, y_train, X_test, y_test)
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("Loading CIFAR-10 dataset...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Image shape: {X_train.shape[1:]}")

        # Preprocess
        X_train, X_test = self._preprocess_images(X_train, X_test)
        y_train, y_test = self._preprocess_labels(y_train, y_test)

        # Store for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        print(f"\nAfter preprocessing:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        return X_train, y_train, X_test, y_test

    # Normalize pixel values to (0, 1) range
    def _preprocess_images(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalize:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

        return X_train, X_test

    # Convert labels to one-hot encoding
    def _preprocess_labels(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        return y_train, y_test

    # Get human-readable class name from index
    def get_class_name(self, class_index: int) -> str:
        return self.CLASS_NAMES[class_index]