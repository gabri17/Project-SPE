import os
from pathlib import Path

# Get current working directory
CWD = Path(os.getcwd())

# Dataset configuration
BASE_PATH = CWD / "datasets" / "CSV Files"
PART_FILES = [
    'UNSW-NB15_1.csv',
    'UNSW-NB15_2.csv',
    'UNSW-NB15_3.csv',
    'UNSW-NB15_4.csv'
]

# Analysis output paths
ANALYSIS_DIR = CWD / "analysis_outputs"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Model configuration
MODELS = {
    "binary": {
        "Logistic Regression": {"max_iter": 2000, "random_state": 42},
        "Random Forest": {"n_estimators": 200, "max_depth": 20, "random_state": 42},
        "Neural Network": {"hidden_layer_sizes": (100, 50), "max_iter": 500, "random_state": 42},
        "Naive Bayes": {},
        "Decision Tree": {"max_depth": 15, "random_state": 42}
    },
    "multiclass": {
        "Logistic Regression": {"max_iter": 3000, "multi_class": "ovr", "random_state": 42},
        "Random Forest": {"n_estimators": 300, "max_depth": 25, "random_state": 42},
        "Neural Network": {"hidden_layer_sizes": (150, 100), "max_iter": 700, "random_state": 42},
        "Naive Bayes": {},
        "Decision Tree": {"max_depth": 20, "random_state": 42}
    }
}

# Evaluation parameters
CV_SPLITS = 5
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.9
TEST_SIZE = 0.2  # For train-test split