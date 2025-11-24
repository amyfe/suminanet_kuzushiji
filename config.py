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
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 2 #TODO: erhöhen
LR = 1e-4
WEIGHT_DECAY = 1e-5


# model
IN_CHANNELS = 3
BASE_FEATURES = 32
NUM_CLASSES = 3000 # adjust to Kuzushiji label count
DETECTOR_HEATMAP_SIGMA = 2


# teacher forcing
TEACHER_FORCING_PROB = 0.5


# misc
SEED = 42
