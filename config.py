"""Configuration file with hyperparameters and paths."""
from pathlib import Path


# paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "assets/data"
DATA_PREPROCESSED_DIR = ROOT / "assets/data_preprocessed"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# training
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 1 #TODO: erhöhen
LR = 1e-4
WEIGHT_DECAY = 1e-5


# model
IN_CHANNELS = 3
BASE_FEATURES = 32
NUM_CLASSES = 3000 # adjust to Kuzushiji label count
DETECTOR_HEATMAP_SIGMA = 2


# training modes (choose one or both for comparison)
USE_DETECTOR_HEAD = False  # Option 1: Traditional detection (heatmap + bbox regression)
USE_ROI_ATTENTION = True   # Option 2: Predict boxes from attention patterns
DETECTION_LOSS_WEIGHT = 1.0  # Weight for detection losses
ROI_BOX_LOSS_WEIGHT = 0.5    # Weight for ROI box loss


# validation (optional during training)
RUN_VALIDATION = True       # Whether to run validation during training
VALIDATION_FREQ = 1         # Validate every N epochs
VALIDATION_BATCHES = 50     # Max batches per validation (set to None for full validation)


# teacher forcing
TEACHER_FORCING_PROB = 0.5


# misc
SEED = 42
