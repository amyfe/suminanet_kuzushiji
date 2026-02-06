"""Configuration file with hyperparameters and paths."""
from pathlib import Path


# paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "assets/data"
DATA_PREPROCESSED_DIR = ROOT / "assets/data_preprocessed"
DATA_ZIP_DIR = ROOT / "assets/data_cached"
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
BATCH_SIZE = 2  # Reduced from 4 to fit in 16GB GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Increased to maintain effective batch size of 8
NUM_WORKERS = 2  
NUM_EPOCHS = 25 
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0  # Gradient clipping threshold
CTC_WARMUP_EPOCHS = 0  
# images
IMAGE_SIZE = (256, 256)


# model
IN_CHANNELS = 3
BASE_FEATURES = 32
# NUM_CLASSES will be set dynamically from vocab (vocab_size includes special tokens)
# Actual data has ~2906 characters + 4 special tokens (PAD, UNK, SOS, EOS) = ~2910 total
NUM_CLASSES = 3000  # Keep as upper bound, actual value set in training from vocab.vocab_size
DETECTOR_HEATMAP_SIGMA = 2


# training modes
# Stage 1: Detection (DetectorHead predicts WHERE characters are)
USE_DETECTOR_HEAD = True
DETECTION_LOSS_WEIGHT = 1.0

# Stage 2: Recognition (Classifier/Decoder predicts WHAT characters are)
# ROI Attention is DISABLED - it was architecturally broken (1D encoder used as fake 2D)
USE_ROI_ATTENTION = False  
USE_MIXED_PRECISION = True
ROI_BOX_LOSS_WEIGHT = 0.1  # Not used when USE_ROI_ATTENTION=False    


# validation (optional during training)
RUN_VALIDATION = True       # Whether to run validation during training
VALIDATION_FREQ = 1         # Validate every N epochs
VALIDATION_BATCHES = 50     # Max batches per validation (set to None for full validation)


# teacher forcing
TEACHER_FORCING_PROB = 0.5


# misc
SEED = 42
