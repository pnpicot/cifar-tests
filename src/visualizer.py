"""
Visualization functions for training results and model comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

class ResultsVisualizer:
    def __init__(self, save_dir: Path = Path('results/plots')):
        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style('whitegrid')

        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def plot_training_history(
        self,
        history,
        model_name: str,
        save: bool = True
    ):
        """
        Plot training and validation loss/accuracy curves

        Args:
            history: Keras History object from model.fit()
            model_name: Name of the model
            save: Whether to save the plot
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)

        # Plot loss
        ax1.plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.save_dir / f'{model_name}_training_history.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved training history plot: {filepath}")

        plt.show()
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list,
        model_name: str,
        save: bool = True
    ):
        """
        Plot confusion matrix as a heatmap

        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
            save: Whether to save the plot
        """

        plt.figure(figsize=(12, 10))

        # Normalize by row (actual class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion'},
            square=True
        )

        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save:
            filepath = self.save_dir / f'{model_name}_confusion_matrix.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix: {filepath}")

        plt.show()
        plt.close()

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        save: bool = True
    ):
        """
        Plot comparison of all models

        Args:
            results_df: DataFrame with columns: model_name, test_accuracy, test_loss, etc.
            save: Whether to save the plot
        """

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        models = results_df['model_name']

        # Plot 1: Test Accuracy
        ax1 = axes[0, 0]
        bars = ax1.bar(models, results_df['test_accuracy'] * 100, color='steelblue', edgecolor='black')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()

            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Test Loss
        ax2 = axes[0, 1]
        bars = ax2.bar(models, results_df['test_loss'], color='coral', edgecolor='black')
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()

            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Parameters
        ax3 = axes[1, 0]
        bars = ax3.bar(models, results_df['parameters'] / 1e6, color='lightgreen', edgecolor='black')
        ax3.set_ylabel('Parameters (Milliosn)', fontsize=12)
        ax3.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()

            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.2f}M',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 4: Training Time
        ax4 = axes[1, 1]
        bars = ax4.bar(models, results_df['training_time_seconds'] / 60, color='plum', edgecolor='black')
        ax4.set_ylabel('Training Time (Minutes)', fontsize=12)
        ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()

            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1f}m',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save:
            filepath = self.save_dir / 'model_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison: {filepath}")

        plt.show()
        plt.close()

    def plot_per_class_accuracy(
        self,
        per_class_acc: dict,
        model_name: str,
        save: bool = True
    ):
        """
        Plot per-class accuracy as a bar chart

        Args:
            per_class_acc: Dictionary mapping class name to accuracy
            model_name: Name of the model
            save: Whether to save the plot
        """

        plt.figure(figsize=(12, 6))
        classes = list(per_class_acc.keys())
        accuracies = [per_class_acc[c] * 100 for c in classes]

        bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='black')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.title(f'{model_name} - Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            filepath = self.save_dir / f'{model_name}_per_class_accuracy.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved per-class accuracy plot: {filepath}")

        plt.show()
        plt.close()