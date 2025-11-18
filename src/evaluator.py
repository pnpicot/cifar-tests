"""
Model evaluation and metrics generation
"""

import numpy as np
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class ModelEvaluator:
    def __init__(self, model: Sequential, model_name: str):
        self.model = model
        self.model_name = model_name
        self.test_loss = None
        self.test_accuracy = None
        self.predictions = None
        self.y_true = None

    def evaluate(self, X_test, y_test, verbose: int = 0) -> tuple:
        """
        Evaluate model on test set

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity level
        """

        print(f"\n{'=' * 60}")
        print(f"Evaluating {self.model_name}")
        print('=' * 60)

        # Evaluate
        self.test_loss, self.test_accuracy = self.model.evaluate(
            X_test, y_test, verbose=verbose
        )

        print(f"Test Loss: {self.test_loss:.4f}")
        print(f"Test Accuracy: {self.test_accuracy * 100:.2f}")

        # Generate predictions
        self.predictions = self.model.predict(X_test, verbose=0)
        self.y_true = y_test

        return self.test_loss, self.test_accuracy

    def get_confusion_matrix(self, class_names: list = None) -> np.ndarray:
        """
        Generate confusion matrix

        Args:
            class_names: List of class names
        """

        if self.predictions is None:
            raise ValueError("Run evaluate() first")

        # Convert one-hot to class indices
        y_pred_classes = np.argmax(self.predictions, axis=1)
        y_true_classes = np.argmax(self.y_true, axis=1)

        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        return cm

    def get_classification_report(self, class_names: list) -> str:
        """
        Generate detailed classification report

        Args:
            class_names: List of class names
        """

        if self.predictions is None:
            raise ValueError("Run evaluate() first")

        y_pred_classes = np.argmax(self.predictions, axis=1)
        y_true_classes = np.argmax(self.y_true, axis=1)

        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=class_names,
            digits=4
        )

        return report

    def get_per_class_accuracy(self, class_names: list) -> dict:
        """
        Calculate accuracy for each class

        Args:
            class_names: List of class names
        """

        if self.predictions is None:
            raise ValueError("Run evaluate() first")

        y_pred_classes = np.argmax(self.predictions, axis=1)
        y_true_classes = np.argmax(self.y_true, axis=1)
        per_class_acc = {}

        for i, class_name in enumerate(class_names):
            # Find samples of this class
            mask = y_true_classes == i

            if mask.sum() > 0:
                class_correct = (y_pred_classes[mask] == i).sum()
                class_total = mask.sum()
                accuracy = class_correct / class_total
                per_class_acc[class_name] = accuracy

        return per_class_acc

    def get_top_k_accuracy(self, k: int = 2) -> float:
        """
        Calculate top-k accuracy

        Args:
            k: Number of top predictions to consider
        """

        if self.predictions is None:
            raise ValueError("Run evaluate() first")

        y_true_classes = np.argmax(self.y_true, axis=1)

        # Get top-k predictions
        top_k_preds = np.argsort(self.predictions, axis=1)[:, -k:]

        # Check if true class is in top-k
        correct = np.array([
            y_true in top_k_preds[i] for i, y_true in enumerate(y_true_classes)
        ])

        return correct.mean()

    def get_summary(self) -> dict:
        """
        Get evaluation summary
        """

        return {
            'model_name': self.model_name,
            'test_loss': self.test_loss,
            'test_accuracy': self.test_accuracy,
            'total_parameters': self.model.count_params()
        }