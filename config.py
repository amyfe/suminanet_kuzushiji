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
NUM_WORKERS = 8

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
STAGE2_DROPOUT_RATE = 0.15

# -------------------------
# Detector -> Proposal stage
# -------------------------

DET_SCORE_THRESH = 0.051290347220558016
DET_TOP_K = 1200
DET_NMS_IOU = 0.5920639234211412
DET_MIN_BOX_SIZE = 1.8012387816266866

# -------------------------
# Shared feature projection
# -------------------------

STAGE2_PROJ_DIM = 256
STAGE2_PROJ_DROPOUT = 0.10

# -------------------------
# ROI pooling
# -------------------------

STAGE2_ROI_SIZE = (8, 8)
STAGE2_ROI_FEAT_DIM = 384
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
STAGE2_TOKEN_HIDDEN_DIM = 512
STAGE2_TOKEN_DROPOUT = 0.10
STAGE2_TOKEN_USE_SCORE_BRANCH = True

# -------------------------
# Context encoder
# -------------------------

STAGE2_CONTEXT_HIDDEN_DIM = 384
STAGE2_CONTEXT_NUM_LAYERS = 2

# -------------------------
# Decoder
# -------------------------

STAGE2_DECODER_EMBED_DIM = 128
STAGE2_DECODER_HIDDEN_DIM = 256
STAGE2_DECODER_NUM_LAYERS = 1
STAGE2_DECODER_DROPOUT = 0.10
STAGE2_DECODER_LABEL_SMOOTHING = 0.03
STAGE2_DECODER_EOS_WEIGHT = 2.5
STAGE2_ACTION_AUX_WEIGHT = 0.35

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
VALIDATION_BATCHES = 10

# For later free-decoding evaluation
STAGE2_VAL_MAX_DECODE_LEN = 352

FREEZE_BACKBONE = True
FREEZE_DETECTOR = True
STAGE2_DEBUG_BATCH_STATS = False
STAGE2_DEBUG_AUX_ALIGNMENT = False
STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT = 20


# ============================================================
# Phase-based training (e.g. for Option-C Stage 2)
# ============================================================


STAGE2_PHASE = "B"   # "A" oder "B"

# Phase A (context encoder enabled, decoder still frozen)
STAGE2_PHASE_A_EPOCHS = 15
STAGE2_PHASE_A_LAMBDA_BOX = 0.5
STAGE2_PHASE_A_LAMBDA_DELTA = 1.0
STAGE2_PHASE_A_LAMBDA_SCORE = 0.3
STAGE2_PHASE_A_LAMBDA_AUX = 1.0
STAGE2_PHASE_A_LAMBDA_DECODER = 0.0
STAGE2_PHASE_A_TF_START = 1.0
STAGE2_PHASE_A_TF_END = 0.974432737452076

# Phase B
STAGE2_PHASE_B_EPOCHS = 10
STAGE2_PHASE_B_LAMBDA_BOX = 0.35
STAGE2_PHASE_B_LAMBDA_DELTA = 0.75
STAGE2_PHASE_B_LAMBDA_SCORE = 0.2
STAGE2_PHASE_B_LAMBDA_AUX = 0.6
STAGE2_PHASE_B_LAMBDA_DECODER = 1.0
STAGE2_LAMBDA_FREE_PREFIX = 0.35
STAGE2_LAMBDA_ACTION = 0.15
STAGE2_PHASE_B_TF_START = 0.9
STAGE2_PHASE_B_TF_END = 0.4
STAGE2_ACTION_ALIGNMENT_START_EPOCH = 1
STAGE2_ACTION_ALIGNMENT_FULL_EPOCH = 4

# Free-decoding stabilization (Phase B)
STAGE2_PHASE_B_FREE_REPETITION_PENALTY = 0.1
STAGE2_PHASE_B_FREE_BLOCK_IMMEDIATE_REPEAT = True
STAGE2_PHASE_B_FREE_MIN_STEPS = 36
STAGE2_PHASE_B_FREE_MAX_STALL_STEPS = 8
STAGE2_PHASE_B_FREE_FORCE_STOP_ON_STALL = False
STAGE2_PHASE_B_FREE_STOP_THRESHOLD = 0.0
STAGE2_PHASE_B_FREE_MIN_PROGRESS_FOR_EOS = 0.0
STAGE2_PHASE_B_FREE_MIN_COVERAGE_FOR_EOS = 0.0
STAGE2_PHASE_B_TRAIN_FREE_REPETITION_PENALTY = 0.0
STAGE2_PHASE_B_TRAIN_FREE_BLOCK_IMMEDIATE_REPEAT = False
STAGE2_PHASE_B_TRAIN_FREE_MIN_STEPS = 10
STAGE2_PHASE_B_TRAIN_FREE_MAX_STALL_STEPS = 4
STAGE2_PHASE_B_TRAIN_FREE_FORCE_STOP_ON_STALL = False
STAGE2_PHASE_B_TRAIN_FREE_STOP_THRESHOLD = 0.0
STAGE2_PHASE_B_TRAIN_FREE_MIN_PROGRESS_FOR_EOS = 0.0
STAGE2_PHASE_B_TRAIN_FREE_MIN_COVERAGE_FOR_EOS = 0.0
STAGE2_FREE_PREFIX_EVERY_N_STEPS = 256
STAGE2_TRAIN_FREE_DECODE_BATCHES = 2
STAGE2_VAL_FREE_DECODE_BATCHES = 4
STAGE2_TRAIN_DECODER_STATS_EVERY_N_STEPS = 64
STAGE2_TRAIN_DECODER_STATS_MAX_SAMPLES = 4
STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS = 8
STAGE2_TRAIN_FREE_DIAG_MAX_LEN = 128
STAGE2_FREE_RECOVERY_WINDOW = 5
STAGE2_FREE_RECOVERY_WEIGHT = 0.35
STAGE2_FREE_PREFIX_MIN_TOKENS = 4
STAGE2_USE_CHUNKED_FREE_DECODE = True
STAGE2_USE_CHUNKED_FREE_TRAINING = True
STAGE2_FREE_TEACHER_PREFIX_STEPS = 6
STAGE2_LOG_PREDICTIONS = True
STAGE2_PREDICTION_SAMPLES = 2
STAGE2_LAMBDA_STOP = 0.2
STAGE2_LAMBDA_STOP_FREE_SCALE = 0.3
STAGE2_LAMBDA_ACTION_FREE_SCALE = 1.5
STAGE2_DECODER_STOP_POS_WEIGHT = 3.0
STAGE2_PHASE_B_STOP_THRESHOLD = 0.1

# Runtime logging controls (recommended for cluster runs)
STAGE2_ENABLE_TQDM = True
STAGE2_PROGRESS_POSTFIX_EVERY_N_STEPS = 70


# ============================================================
# KuroNet Recognizer (simplified per-ROI classifier)
# ============================================================
# Replaces the seq2seq decoder with a direct per-ROI character
# classifier: detect → refine → sort by reading order → classify.
# All ROI pipeline components (pool, refine, ordering, tokens)
# are reused unchanged from Stage 2 hybrid.
# ============================================================

KURONET_CHECKPOINT_DIR       = CHECKPOINT_DIR / "kuronet_recognizer"
KURONET_CHECKPOINT_DIR.mkdir(exist_ok=True)

KURONET_USE_CONTEXT          = True        # Keep BiGRU context encoder
KURONET_CLASSIFIER_HIDDEN    = 512         # MLP hidden dim for char classifier

# ROI pooling — larger crop to preserve dakuten (voiced-mark) strokes
KURONET_ROI_SIZE             = (12, 12)    # was (8, 8); (12,12) gives 2.25× more pixels per ROI

# Detection params (DET_*) are shared — defined once above, used by Stage 2 and KuroNet.
# Once Optuna (jobs 9182-9184) finishes, update DET_* in the Stage 2 section above.

# Assembled CER: filter out false-positive proposals before transcription
# (ordered_mask includes all 159 props/img; ~18 are FPs that add insertion errors)
KURONET_CER_SCORE_THRESH     = 0.5

# Loss weights (single-phase training, no scheduling)
KURONET_LAMBDA_CHAR          = 1.0         # Primary: per-ROI character classification
KURONET_LAMBDA_BOX           = 0.35        # Box regression on positive ROIs
KURONET_LAMBDA_DELTA         = 0.75        # Delta regression on positive ROIs
KURONET_LAMBDA_SCORE         = 0.20        # ROI quality BCE

# Optimizer
KURONET_LR                   = 1.0841999197654953e-4
KURONET_LR_ETA_MIN           = 1e-6       # Cosine annealing floor
KURONET_WEIGHT_DECAY         = 5e-4       # was 9.4e-5; raised to fight overfitting

# Training
KURONET_EPOCHS               = 50
KURONET_GRAD_ACCUM_STEPS     = 4
KURONET_ENABLE_TQDM          = True
KURONET_PROGRESS_POSTFIX_N   = 70
KURONET_LOG_PREDICTIONS      = True
KURONET_PREDICTION_SAMPLES   = 2
KURONET_VALIDATION_BATCHES   = 10
