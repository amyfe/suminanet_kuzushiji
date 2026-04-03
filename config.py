"""Configuration file with hyperparameters and paths for Stage 1 + Option-C Stage 2."""

from pathlib import Path
import torch


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "assets/data"
DATA_PREPROCESSED_DIR = ROOT / "assets/data_preprocessed"
DATA_ZIP_DIR = ROOT / "assets/data_cached"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Optional: books excluded from validation/test experiments
EXCLUDE_BOOKS = {
    "200021925",
    "200022050",
    "200025191",
    "umgy00000",
}


# ============================================================
# Global / Runtime
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMAGE_SIZE = (256, 256)

USE_MIXED_PRECISION = True
NUM_WORKERS = 2

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
GRAD_CLIP = 1.0

NUM_EPOCHS = 5


# ============================================================
# Optimizer
# ============================================================

LR = 1.0841999197654953e-4
WEIGHT_DECAY = 9.431774935513243e-5


# ============================================================
# Backbone / Shared Model Defaults
# ============================================================

IN_CHANNELS = 3
BASE_FEATURES = 32

# Upper bound / placeholder only.
# Real vocabulary size is created dynamically from annotations.
NUM_CLASSES = 3000


# ============================================================
# Stage 1: Detection
# ============================================================

STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "stage1_detection"

STAGE1_DROPOUT_RATE = 0.33763617734186435

STAGE1_FOCAL_ALPHA = 0.22380365509800024
STAGE1_FOCAL_GAMMA = 1.6449122167800316
STAGE1_POS_WEIGHT = 2.2202569170802624

STAGE1_BBOX_WEIGHT = 0.10087921713096845
STAGE1_HEATMAP_SIGMA = 1.3143263749771124

STAGE1_EARLY_STOPPING_PATIENCE = 0   # 0 = disabled


# ============================================================
# Stage 2: Hybrid Recognition (Option C)
# ============================================================

STAGE2_CHECKPOINT_DIR = CHECKPOINT_DIR / "stage2_hybrid"

# Freeze policy for first stable Option-C training
STAGE2_FREEZE_BACKBONE = True
STAGE2_FREEZE_DETECTOR = True
STAGE2_DROPOUT_RATE = 0.33763617734186435

# -------------------------
# Detector -> Proposal stage
# -------------------------

STAGE2_DET_SCORE_THRESH = 0.40
STAGE2_DET_TOP_K = 256
STAGE2_DET_NMS_IOU = 0.50
STAGE2_DET_MIN_BOX_SIZE = 1.0

# -------------------------
# Shared feature projection
# -------------------------

STAGE2_PROJ_DIM = 256
STAGE2_PROJ_DROPOUT = 0.10

# -------------------------
# ROI pooling
# -------------------------

STAGE2_ROI_SIZE = (8, 6)
STAGE2_ROI_FEAT_DIM = 256
STAGE2_ROI_POOL_DROPOUT = 0.10

# -------------------------
# ROI refinement
# -------------------------

STAGE2_REFINE_HIDDEN_DIM = 256
STAGE2_REFINE_DROPOUT = 0.10

# Whether to add an auxiliary ROI classification head
STAGE2_USE_AUX_HEAD = True

# -------------------------
# Reading order
# -------------------------

STAGE2_READING_ORDER_LINE_THRESH_RATIO = 0.60

# -------------------------
# ROI token projection
# -------------------------

STAGE2_TOKEN_DIM = 256
STAGE2_TOKEN_HIDDEN_DIM = 256
STAGE2_TOKEN_DROPOUT = 0.10
STAGE2_TOKEN_USE_SCORE_BRANCH = True

# -------------------------
# Context encoder
# -------------------------

STAGE2_CONTEXT_HIDDEN_DIM = 256

# -------------------------
# Decoder
# -------------------------

STAGE2_DECODER_EMBED_DIM = 128
STAGE2_DECODER_HIDDEN_DIM = 256
STAGE2_DECODER_NUM_LAYERS = 1
STAGE2_DECODER_DROPOUT = 0.10
STAGE2_DECODER_LABEL_SMOOTHING = 0.0
STAGE2_DECODER_EOS_WEIGHT = 3.0

# Teacher forcing schedule
STAGE2_TF_START = 1.0
STAGE2_TF_END = 1.0
STAGE2_TF_SCHEDULE = "linear"

# -------------------------
# Refinement target matching
# -------------------------

STAGE2_REFINE_POS_IOU = 0.50
STAGE2_REFINE_NEG_IOU = 0.20
STAGE2_REFINE_POS_WEIGHT = 1.0

# -------------------------
# Stage-2 loss weights
# -------------------------

STAGE2_LAMBDA_BOX = 0.50
STAGE2_LAMBDA_DELTA = 1.00
STAGE2_LAMBDA_SCORE = 0.30
STAGE2_LAMBDA_AUX = 1.0
STAGE2_LAMBDA_DECODER = 0


# ============================================================
# Validation / Evaluation
# ============================================================

RUN_VALIDATION = True
VALIDATION_FREQ = 1
VALIDATION_BATCHES = 50

# For later free-decoding evaluation
STAGE2_VAL_MAX_DECODE_LEN = 384

FREEZE_BACKBONE = True
FREEZE_DETECTOR = True
STAGE2_DEBUG_BATCH_STATS = True
