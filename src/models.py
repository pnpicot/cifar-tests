"""
CNN model architecture for CIFAR-10 classification
"""

from tensorflow import keras
from tensorflow.keras import layers, regularizers # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, input_shape: tuple = (32, 32, 3), num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    @abstractmethod
    def build(self) -> Sequential:
        """Build and return the model architecture"""
        pass

    def get_model(self) -> Sequential:
        """Get the built model"""
        if self.model is None:
            self.model = self.build()

        return self.model

    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.model = self.build()

        self.model.summary()

class DenseBaselineModel(BaseModel):
    """Model 1: Fully connected baseline (no convolutions)"""

    def build(self) -> Sequential:
        model = Sequential([
            layers.Flatten(input_shape=self.input_shape),
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='DenseBaseline')

        return model

class SimpleCNNModel(BaseModel):
    """Model 2: Simple CNN with 2 convolutional layers"""

    def build(self) -> Sequential:
        model = Sequential([
            # First convolutional block
            layers.Conv2D(
                32,
                (3, 3),
                activation='relu',
                input_shape=self.input_shape,
                name='conv_1'
            ),
            layers.MaxPooling2D((2, 2), name='pool_1'),

            # Second convolutional block
            layers.Conv2D(
                64,
                (3, 3),
                activation='relu',
                name='conv_2'
            ),
            layers.MaxPooling2D((2, 2), name='pool_2'),

            # Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='SimpleCNN')

        return model

class DeepCNNModel(BaseModel):
    """Model 3: Deeper CNN with VGG-style architecture"""

    def build(self) -> Sequential:
        model = Sequential([
            # First convolutional block
            layers.Conv2D(
                64,
                (3, 3),
                activation='relu',
                padding='same',
                input_shape=self.input_shape,
                name='conv_1_1'
            ),
            layers.Conv2D(
                64,
                (3, 3),
                activation='relu',
                padding='same',
                name='conv_1_2'
            ),
            layers.MaxPooling2D((2, 2), name='pool_1'),

            # Second convolutional block
            layers.Conv2D(
                128,
                (3, 3),
                activation='relu',
                padding='same',
                name='conv_2_1'
            ),
            layers.Conv2D(
                128,
                (3, 3),
                activation='relu',
                padding='same',
                name='conv_2_2'
            ),
            layers.MaxPooling2D((2, 2), name='pool_2'),

            # Third convolutional block
            layers.Conv2D(
                256,
                (3, 3),
                activation='relu',
                padding='same',
                name='conv_3_1'
            ),
            layers.Conv2D(
                256,
                (3, 3),
                activation='relu',
                padding='same',
                name='conv_3_2'
            ),
            layers.MaxPooling2D((2, 2), name='pool_3'),

            # Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='DeepCNN')

        return model

class RegularizedCNNModel(BaseModel):
    """Model 4: Deep CNN with dropout and batch normalization"""

    def build(self) -> Sequential:
        model = Sequential([
            # First convolutional block
            layers.Conv2D(
                64,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                input_shape=self.input_shape,
                name='conv_1_1'
            ),
            layers.BatchNormalization(name='bn_1_1'),
            layers.Conv2D(
                64,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name='conv_1_2'
            ),
            layers.BatchNormalization(name='bn_1_2'),
            layers.MaxPooling2D((2, 2), name='pool_1'),
            layers.Dropout(0.25, name='dropout_1'),

            # Second convolutional block
            layers.Conv2D(
                128,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name='conv_2_1'
            ),
            layers.BatchNormalization(name='bn_2_1'),
            layers.Conv2D(
                128,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name='conv_2_2'
            ),
            layers.BatchNormalization(name='bn_2_2'),
            layers.MaxPooling2D((2, 2), name='pool_2'),
            layers.Dropout(0.25, name='dropout_2'),

            # Third convolutional block
            layers.Conv2D(
                256,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name='conv_3_1'
            ),
            layers.BatchNormalization(name='bn_3_1'),
            layers.Conv2D(
                256,
                (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=regularizers.l2(0.001),
                name='conv_3_2'
            ),
            layers.BatchNormalization(name='bn_3_2'),
            layers.MaxPooling2D((2, 2), name='pool_3'),
            layers.Dropout(0.25, name='dropout_3'),

            # Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_1'),
            layers.Dropout(0.5, name='dropout_4'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='RegularizedCNN')

        return model

def create_model(
    model_type: str,
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10
) -> Sequential:
    """
    Factory to create models by name.

    Args:
        model_type: Type of model ('dense', 'simple_cnn', 'deep_cnn', 'regularized_cnn')
        input_shape: Input image shape
        num_classes: Number of output classes
    """

    models = {
        'dense': DenseBaselineModel,
        'simple_cnn': SimpleCNNModel,
        'deep_cnn': DeepCNNModel,
        'regularized_cnn': RegularizedCNNModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    model_class = models[model_type]

    return model_class(input_shape, num_classes).get_model()