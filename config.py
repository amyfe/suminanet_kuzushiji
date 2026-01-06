"""Configuration file with hyperparameters and paths."""
from pathlib import Path


# paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "assets/data"
DATA_PREPROCESSED_DIR = ROOT / "assets/data_preprocessed"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# For validation and testing, exclude some books
EXCLUDE_BOOKS = {
    "200021925",
    "200022050",
    "200025191",
    "umgy00000",
}

# training
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WORKERS = 2  
NUM_EPOCHS = 20 
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0  # Gradient clipping threshold

# images
IMAGE_SIZE = (512, 512)  # (H, W) centralized resize for training/validation


# model
IN_CHANNELS = 3
BASE_FEATURES = 32
NUM_CLASSES = 3000 # adjust to Kuzushiji label count
DETECTOR_HEATMAP_SIGMA = 2


# training modes (choose one or both for comparison)
# Option 1
USE_DETECTOR_HEAD = False
DETECTION_LOSS_WEIGHT = 1.0
# Option 2: Predict boxes from attention patterns
USE_ROI_ATTENTION = True  
USE_MIXED_PRECISION = True
ROI_BOX_LOSS_WEIGHT = 0.1    


# validation (optional during training)
RUN_VALIDATION = True       # Whether to run validation during training
VALIDATION_FREQ = 1         # Validate every N epochs
VALIDATION_BATCHES = 50     # Max batches per validation (set to None for full validation)


# teacher forcing
TEACHER_FORCING_PROB = 0.5


# misc
SEED = 42
