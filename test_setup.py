"""Quick test to verify all libraries are installed correctly."""
import sys

print("Testing imports...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print(f"✓ Seaborn {sns.__version__}")
except ImportError as e:
    print(f"✗ Seaborn import failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")
    sys.exit(1)

print("\n✓ All libraries installed successfully!")
print("\nChecking GPU availability...")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")