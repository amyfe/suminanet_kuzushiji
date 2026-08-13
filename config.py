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

EXCLUDE_BOOKS = set()

EXCLUDE_PAGES = {
    "200021925": ["200021925_00003_2.jpg", "200021925_00007_2.jpg", "200021925_00012_1.jpg", "200021925_00012_2.jpg", "200021925_00013_2.jpg"],
    "200022050": ["200022050_00004_2.jpg", "200022050_00006_2.jpg", "200022050_00007_2.jpg", "200022050_00010_1.jpg", "200022050_00014_2.jpg"],
    "200025191": ["200025191_00021_2.jpg", "200025191_00039_2.jpg", "200025191_00057_1.jpg", "200025191_00061_1.jpg", "200025191_00061_2.jpg"],
    "umgy00000": ["umgy001_004.jpg", "umgy002_032.jpg", "umgy006_034.jpg", "umgy010_023.jpg", "umgy012_025.jpg"],
}


# ============================================================
# Global / Runtime
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMAGE_SIZE = (512, 512)

USE_MIXED_PRECISION = True
NUM_WORKERS = 8

STAGE1_BATCH_SIZE = 32
STAGE2_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
GRAD_CLIP = 1.0

NUM_EPOCHS = 30
DENSITY_GRID          = 8
DENSITY_FACTOR        = 5.0
AVG_GT_PER_IMAGE      = 236


# ============================================================
# Optimizer
# ============================================================

LR =  0.00055
WEIGHT_DECAY = 3.0827509590757714e-06


# ============================================================
# Backbone / Shared Model Defaults
# ============================================================

IN_CHANNELS = 3

# Backbone selection.
# 'unet'            -- UNet from scratch (no pretrained weights)
# 'efficientnet_b2' -- EfficientNet-B2 + FPN decoder, ImageNet pretrained
BACKBONE_TYPE = "efficientnet_b2"
BACKBONE_BASE_FEATURES = 64


# ============================================================
# Stage 1: Detection
# ============================================================

STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "stage1_detection"

STAGE1_DROPOUT_RATE = 0.3510765163667262

STAGE1_FOCAL_ALPHA = 0.25658617128945466
STAGE1_FOCAL_GAMMA = 1.5881990201080256
STAGE1_POS_WEIGHT = 1.536665269500476

STAGE1_BBOX_WEIGHT = 0.13048169580424154
# DIoU loss added alongside SmoothL1: directly optimises IoU (which validation measures at ≥0.5)
STAGE1_DIOU_WEIGHT = STAGE1_BBOX_WEIGHT
STAGE1_HEATMAP_SIGMA = 1.7545869700715266         
STAGE1_FOCAL_POS_THRESHOLD = 0.479126803723303     
STAGE1_EARLY_STOPPING_PATIENCE = 5   
STAGE1_SIGMA_FLOOR = 1.5
STAGE1_SIGMA_CEIL  = 3.0
STAGE1_SIGMA_SCALE = 0.20

# ============================================================
# Stage 2: Hybrid Recognition
# ============================================================

STAGE2_DROPOUT_RATE = 0.15
SUMINANET_EPOCHS               = 50
SUMINANET_EARLY_STOPPING_PATIENCE = 10  

# -------------------------
# Detector -> Proposal stage
# -------------------------

DET_SCORE_THRESH = 0.65
DET_TOP_K = 700
DET_NMS_IOU = 0.6454054005824295
DET_MIN_BOX_SIZE = 1.0539268301125129

# -------------------------
# Shared feature projection
# -------------------------

STAGE2_PROJ_DIM = 256
STAGE2_PROJ_DROPOUT = 0.10

# -------------------------
# ROI pooling
# -------------------------

STAGE2_ROI_SIZE = (16, 16) 
STAGE2_ROI_FEAT_DIM = 384
STAGE2_ROI_POOL_DROPOUT = 0.10

# -------------------------
# ROI refinement
# -------------------------

STAGE2_REFINE_HIDDEN_DIM = 512
STAGE2_REFINE_DROPOUT = 0.10
STAGE2_USE_AUX_HEAD = True

# -------------------------
# Reading order
# -------------------------

STAGE2_READING_ORDER_LINE_THRESH_RATIO = 0.60
SUMINANET_READING_ORDER_CONFIDENCE = 0.5

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
# Unidirectional GRU: the sequence is already in reading order (right→left columns, top→bottom).
# The backward GRU pass of a BiGRU only learns reversed reading order — trivial and wastes 50%
# of context capacity. "gru" = forward only; "bigru" = bidirectional.
STAGE2_CONTEXT_MODE = "gru"

# -------------------------
# Loss Functions
# -------------------------

STAGE2_CONTINUATION_WEIGHT = 0.3
STAGE2_CONTINUATION_ALPHA = 0.5

# -------------------------
# Refinement target matching
# -------------------------

STAGE2_REFINE_POS_IOU = 0.45
STAGE2_REFINE_NEG_IOU = 0.20
STAGE2_REFINE_POS_WEIGHT = 1.0
STAGE2_USE_HUNGARIAN = True


# ============================================================
# Validation / Evaluation
# ============================================================

RUN_VALIDATION = True
VALIDATION_FREQ = 1
VALIDATION_BATCHES = None  # None = full validation set each epoch

FREEZE_BACKBONE = True
FREEZE_DETECTOR = True
STAGE2_DEBUG_BATCH_STATS = False
STAGE2_DEBUG_AUX_ALIGNMENT = False
STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT = 20

# Runtime logging controls
STAGE2_ENABLE_TQDM = True
STAGE2_PROGRESS_POSTFIX_EVERY_N_STEPS = 70
STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS = 8


# ============================================================
# Warmup training (ROI pipeline pre-training for SuminaNet)
# ============================================================

WARMUP_EPOCHS = 40
WARMUP_LAMBDA_BOX = 0.5
WARMUP_LAMBDA_DELTA = 1.0
WARMUP_LAMBDA_SCORE = 0.3
WARMUP_LAMBDA_AUX = 1.0

# ============================================================
# More Stage 2 / SuminaNet hyperparameters
# ============================================================

SUMINANET_CHECKPOINT_DIR       = CHECKPOINT_DIR /"suminanet_recognizer"
SUMINANET_CHECKPOINT_DIR.mkdir(exist_ok=True)

SUMINANET_USE_CONTEXT          = True
SUMINANET_CONTEXT_BLOCK_GAP_FACTOR = 2.5
SUMINANET_COPY_PASTE_PROB      = 0.4

SUMINANET_CLASSIFIER_HIDDEN    = 512
SUMINANET_USE_CROP_ENCODER     = True
SUMINANET_CROP_ENCODER_SIZE    = (112, 112)
# Two-phase freeze schedule:
# Phase 1 (epochs 1–FREEZE_AFTER): unfrozen — EfficientNet-B0 adapts ImageNet → Kuzushiji.
# Phase 2 (epoch FREEZE_AFTER+1+): auto-frozen — only out_proj + crop_fusion keep training.
#   Drops ~5.3M params from backprop → no longer bottleneck at 112px.
SUMINANET_FREEZE_CROP_ENCODER       = False  # start unfrozen
SUMINANET_FREEZE_CROP_ENCODER_AFTER = 5      # freeze EfficientNet-B0 weights after this epoch
SUMINANET_CROP_ENCODER_CHUNK_SIZE   = 512
SUMINANET_ROI_POOL_OUTPUT_SIZE = (4, 4)
SUMINANET_DET_SCORE_THRESH     = 0.33821750481132784
SUMINANET_DELTA_SCALE_XY       = 0.25
SUMINANET_DELTA_SCALE_WH       = 0.20

SUMINANET_CER_SCORE_THRESH     = 0.35

SUMINANET_NUM_ALTERNATES        = 3

# Furigana / noise-box filter: proposals whose area is below this fraction of the
# median character area are discarded.  Only applied when at least SUMINANET_FURIGANA_MIN_SAMPLES
SUMINANET_FURIGANA_AREA_RATIO  = 0.25
SUMINANET_FURIGANA_MIN_SAMPLES = 5

# ============================================================
# SuminaNet Loss / Optimizer / Training Hyperparameters
# ============================================================

SUMINANET_LAMBDA_CHAR          = 1.0         
SUMINANET_LAMBDA_BOX           = 0.0         
SUMINANET_LAMBDA_DELTA         = 0.3391212189425109       
SUMINANET_LAMBDA_SCORE         = 0.25001127318731137        
SUMINANET_BG_WEIGHT            = 0.06213050022661147         
SUMINANET_STRONG_BG_WEIGHT     = 0.185146        
SUMINANET_BG_SCORE_GATE        = 0.55
SUMINANET_RESIDUAL_SCALE_INIT  = 0.5

SUMINANET_FOCAL_GAMMA          = 1.203618254887657
SUMINANET_RARE_CHAR_THRESH     = 50
SUMINANET_HARD_NEG_WEIGHT      = 1.231901422995889   
SUMINANET_HARD_NEG_TOP_K       = 20    
SUMINANET_LAMBDA_SCRIPT        = 0.2627635863899684

SUMINANET_LR                   = 0.0004789618199206086
SUMINANET_LR_ETA_MIN           = 1e-6       
SUMINANET_WEIGHT_DECAY         = 0.0003042318888471685


# Effective batch = STAGE2_BATCH_SIZE * SUMINANET_GRAD_ACCUM_STEPS * GPU count
SUMINANET_GRAD_ACCUM_STEPS     = 1
SUMINANET_ENABLE_TQDM          = True
SUMINANET_PROGRESS_POSTFIX_N   = 70
SUMINANET_LOG_PREDICTIONS      = True
SUMINANET_PREDICTION_SAMPLES   = 2
SUMINANET_VALIDATION_BATCHES   = None   # None = full validation set each epoch


# ============================================================
# SAM2 Illustration Hard-Negative Weighting
# ============================================================
SAM2_PREPROCESSING = True
SAM2_MASKS_DIR = ROOT / "assets/data_sam2_masks"
SAM2_CHECKPOINT = ROOT / "checkpoints/sam2/sam2.1_hiera_large.pt"
SAM2_HARD_NEG_WEIGHT = 4.0
# Masks whose area is between these two fractions are treated as illustrations.
# Lower bound: individual characters are small — a region above 10 % is not a character.
# Upper bound: anything above 60 % is the page background (parchment), not an illustration.
SAM2_ILLUS_AREA_THRESH     = 0.10 
SAM2_ILLUS_AREA_MAX_THRESH = 0.60  
SAM2_ILLUS_AREA_SOFT_MARGIN = 0.02
SAM2_ILLUS_MIN_AREA = 1_000     # px²
SAM2_ILLUS_PRED_IOU_THRESH  = 0.85
SAM2_ILLUS_STABILITY_THRESH = 0.90
SAM2_PROPOSALS_PREPROCESSING = True
SAM2_PROPOSALS_DIR = ROOT / "assets/sam2_proposals"

# ============================================================
# Translation Pipeline (Claude API resilience + OCR-confidence handling)
# ============================================================
WEBSITE_CHECKPOINT_DIR = CHECKPOINT_DIR / "C_gru_efficientnet_sam2"  / "suminanet_recognizer" / "suminanet_best.pt"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL ="claude-sonnet-4-6"
OPENROUTER_MODEL = "anthropic/" + MODEL
TARGET_LANGUAGES = {"en": "English", "de": "German"}

TRANSLATION_API_MAX_RETRIES = 3
TRANSLATION_API_TIMEOUT_SEC = 30.0

# Chars reaching translate_text() already survived SUMINANET_CER_SCORE_THRESH
# (0.35) upstream in run_inference(); this is a second, softer "still shaky"
# bar used to bracket-annotate individual characters for Claude.
TRANSLATION_UNCERTAIN_SCORE_THRESH = 0.6

# Minimum run length of kana between kanji before it's flagged as possible
FURIGANA_MIN_RUN_LENGTH = 2

TRANSLATION_MAX_TOKENS_FLOOR = 2000
TRANSLATION_MAX_TOKENS_CEILING = 8000
TRANSLATION_MAX_TOKENS_CHARS_MULTIPLIER = 4
TRANSLATION_MAX_INPUT_CHARS = 1500

# ============================================================
# Backend API — Rate Limiting
# ============================================================

TRANSLATE_RATE_LIMIT_MAX_REQUESTS = 5
TRANSLATE_RATE_LIMIT_WINDOW_SECONDS = 600
TRANSCRIBE_RATE_LIMIT_MAX_REQUESTS = 10
TRANSCRIBE_RATE_LIMIT_WINDOW_SECONDS = 60
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
TRANSCRIBE_INFERENCE_TIMEOUT_SEC = 60.0

