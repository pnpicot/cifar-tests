"""
Train and evaluate all models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CIFAR10DataLoader
from src.models import create_model
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import ResultsVisualizer
import pandas as pd
import json

def train_and_evaluate_model(
    model_type: str,
    X_train, y_train, X_test, y_test,
    class_names: list,
    visualizer: ResultsVisualizer,
    epochs: int = 50,
    batch_size: int = 128
):
    """
    Train and evaluate a single model

    Args:
        model_type: Type of model to create
        X_train, y_train: Training data
        X_test, y_test: Test data
        class_names: List of class names
        visualizer: Visualizer instance
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Dictionary with all results
    """

    print(f"\n{'=' * 60}")
    print(f"Training Model: {model_type.upper()}")
    print('=' * 60)

    # Create model
    model = create_model(model_type)

    # Train
    trainer = ModelTrainer(model, model_name=model_type)

    trainer.compile(learning_rate=0.001)

    history = trainer.train(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        use_early_stopping=True,
        patience=10
    )

    # Evaluate
    evaluator = ModelEvaluator(model, model_name=model_type)
    test_loss, test_accuracy = evaluator.evaluate(X_test, y_test)

    # Get detailed metrics
    confusion_matrix = evaluator.get_confusion_matrix(class_names)
    per_class_acc = evaluator.get_per_class_accuracy(class_names)
    top2_acc = evaluator.get_top_k_accuracy(k=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_history(history, model_type, save=True)
    visualizer.plot_confusion_matrix(confusion_matrix, class_names, model_type, save=True)
    visualizer.plot_per_class_accuracy(per_class_acc, model_type, save=True)

    # Get training summary
    training_summary = trainer.get_training_summary()

    # Combine all results
    results = {
        'model_name': model_type,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'top2_accuracy': top2_acc,
        'parameters': model.count_params(),
        'training_time_seconds': training_summary['training_time_seconds'],
        'epochs_trained': training_summary['epochs_trained'],
        'best_val_accuracy': training_summary['best_val_accuracy'],
        'per_class_accuracy': per_class_acc
    }

    print(f"Completed {model_type}")
    print(f"- Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"- Top-2 Accuracy: {top2_acc * 100:.2f}%")
    print(f"- Training Time: {training_summary['training_time_seconds'] / 60:.1f} minutes")

    return results

def main():
    """
    Main training pipeline
    """

    print('=' * 60)
    print("CIFAR-10 CNN Comparison - Training All Models")
    print('=' * 60)

    # Configuration
    MODELS_TO_TRAIN = ['dense', 'simple_cnn', 'deep_cnn', 'regularized_cnn']
    EPOCHS = 50
    BATCH_SIZE = 128

    # Load data
    print(f"\n{'=' * 60}")
    print("Loading data...")
    print('=' * 60)

    loader = CIFAR10DataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()

    # Create visualizer
    visualizer = ResultsVisualizer()

    # Train all models
    all_results = []

    for model_type in MODELS_TO_TRAIN:
        try:
            results = train_and_evaluate_model(
                model_type,
                X_train, y_train, X_test, y_test,
                loader.CLASS_NAMES,
                visualizer,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            all_results.append(results)
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue

    # Create comparison DataFrame
    print(f"\n{'=' * 60}")
    print("Generating comparison plots...")
    print('=' * 60)

    results_df = pd.DataFrame([{
        'model_name': r['model_name'],
        'test_accuracy': r['test_accuracy'],
        'test_loss': r['test_loss'],
        'parameters': r['parameters'],
        'training_time_seconds': r['training_time_seconds']
    } for r in all_results])

    # Plot comparison
    visualizer.plot_model_comparison(results_df, save=True)

    # Save results to CSV
    results_path = Path('results/metrics/comparison_results.csv')

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved comparison results: {results_path}")

    # Save detailed results to JSON
    json_path = Path('results/metrics/detailed_results.json')

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)

    print(f"Saved detailed results: {json_path}")

    # Print final summary
    print(f"\n{'=' * 60}")
    print("Final Results Summary")
    print('=' * 60)
    print(results_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print('All Training Complete')
    print('=' * 60)
    print("\nResults saved to:")
    print("- Models: results/models/")
    print("- Plots: results/plots/")
    print("- Metrics: results/metrics/")

if __name__ == "__main__":
    main()