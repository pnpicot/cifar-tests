"""
Model compilation and training
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # pyright: ignore[reportMissingImports]
from pathlib import Path
import time

class ModelTrainer:
    def __init__(
        self,
        model: Sequential,
        model_name: str,
        save_dir: Path = Path('results/models')
    ):
        self.model = model
        self.model_name = model_name
        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = None
        self.training_time = 0

    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ):
        """
        Compile the model

        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """

        if metrics is None:
            metrics = ['accuracy']

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

        print(f"Model compiled with {optimizer} (lr={learning_rate})")

    def train(
        self,
        X_train,
        y_train,
        batch_size: int = 128,
        epochs: int = 100,
        validation_split: float = 0.1,
        use_early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 1
    ):
        """
        Train the model

        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            validation_split: Fraction of data used for validation
            use_early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Verbosity level (0, 1, or 2)
        """

        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name}")
        print('=' * 60)

        # Prepare callbacks
        callbacks = []

        # Early stopping to prevent overfitting
        if use_early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )

            callbacks.append(early_stop)

        # Model checkpoint to save the best model
        checkpoint_path = self.save_dir / f"{self.model_name}_best.keras"

        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )

        callbacks.append(checkpoint)

        # Train model
        start_time = time.time()

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        self.training_time = time.time() - start_time

        print(f"\nTraining complete in {self.training_time:.1f}s")

        # Save final model
        final_path = self.save_dir / f"{self.model_name}_final.keras"

        self.model.save(str(final_path))
        print(f"Model saved to {final_path}")

        return self.history

    def get_training_summary(self) -> dict:
        """
        Get summary of training results
        """

        if self.history is None:
            return {}

        history = self.history.history
        epochs_trained = len(history['loss'])

        return {
            'model_name': self.model_name,
            'epochs_trained': epochs_trained,
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'training_time_seconds': self.training_time,
            'parameters': self.model.count_params()
        }