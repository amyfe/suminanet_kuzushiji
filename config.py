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

IMAGE_SIZE = (512, 512)

USE_MIXED_PRECISION = True
NUM_WORKERS = 4

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
GRAD_CLIP = 1.0

NUM_EPOCHS = 20


# ============================================================
# Optimizer
# ============================================================

LR =  0.0019040586163242641
WEIGHT_DECAY = 9.431774935513243e-5


# ============================================================
# Backbone / Shared Model Defaults
# ============================================================

IN_CHANNELS = 3
BASE_FEATURES = 32

# Backbone selection.
# 'unet'            -- UNet from scratch (no pretrained weights)
# 'efficientnet_b2' -- EfficientNet-B2 + FPN decoder, ImageNet pretrained,
#                      outputs at stride=2 (H//2, W//2). Requires timm.
BACKBONE_TYPE = "unet"

# Output channel count for both backbone types.
# Changing this requires Stage 1 retraining.
BACKBONE_BASE_FEATURES = 64

# Upper bound / placeholder only.
# Real vocabulary size is created dynamically from annotations.
NUM_CLASSES = 3000


# ============================================================
# Stage 1: Detection
# ============================================================

STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "stage1_detection"

STAGE1_DROPOUT_RATE = 0.4904013263270167

STAGE1_FOCAL_ALPHA = 0.3519203997395026
STAGE1_FOCAL_GAMMA = 1.881061777549089
STAGE1_POS_WEIGHT = 4.896873402520948

STAGE1_BBOX_WEIGHT = 0.5587297156424271             # raised from 0.10; bbox regression now gets meaningful gradient
STAGE1_HEATMAP_SIGMA = 0.7249217915919892         # re-tune via train_f1 Optuna; formula now uses sqrt(area)*0.20
STAGE1_FOCAL_POS_THRESHOLD = 0.3      # pixels with Gaussian target >= this treated as positives (was 0.5)

STAGE1_EARLY_STOPPING_PATIENCE = 0   # 0 = disabled

# Gaussian sigma bounds for heatmap targets.
# Floor raised from 0.5 → 1.5 so small chars (area < ~100px²) get a learnable
# positive region rather than a near-single-pixel spike.
# Ceil raised from 2.0 → 3.0 to give large chars proportionally wider coverage.
STAGE1_SIGMA_FLOOR = 1.5
STAGE1_SIGMA_CEIL  = 3.0

# Spatial density filter — suppresses illustration FPs in the proposal set.
# The image is divided into GRID×GRID cells; any cell with more than
# DENSITY_FACTOR × (AVG_GT / GRID²) predictions is trimmed to that cap.
STAGE1_DENSITY_GRID          = 8     # cells per side (32×32 px per cell at 256×256)
STAGE1_DENSITY_FACTOR        = 3.0   # allowed = factor × expected chars/cell
STAGE1_AVG_GT_PER_IMAGE      = 236   # from training stats (used to set expected density)

#  {'lr': 0.0019040586163242641, 
#  'weight_decay': 7.610321317306418e-06, 
#  'batch_size': 2, 
#  'dropout_rate': 0.4904013263270167, 
#  'focal_alpha': 0.3519203997395026, 
#  'focal_gamma': 1.881061777549089, 
#  'pos_weight': 4.896873402520948, 
#  'bbox_weight': 0.5587297156424271, 
#  'heatmap_sigma': 0.7249217915919892, 
#  'bbox_radius': 0, 
#  'pos_threshold': 0.29142092186467783}

# ============================================================
# Stage 2: Hybrid Recognition (Option C)
# ============================================================

STAGE2_CHECKPOINT_DIR = CHECKPOINT_DIR / "stage2_hybrid"

# Freeze policy for first stable Option-C training
STAGE2_DROPOUT_RATE = 0.15

# -------------------------
# Detector -> Proposal stage
# -------------------------

DET_SCORE_THRESH = 0.65
DET_TOP_K = 500
DET_NMS_IOU = 0.6
DET_MIN_BOX_SIZE = 2.0

# -------------------------
# Shared feature projection
# -------------------------

STAGE2_PROJ_DIM = 256
STAGE2_PROJ_DROPOUT = 0.10

# -------------------------
# ROI pooling
# -------------------------

STAGE2_ROI_SIZE = (12, 12)   # matched to KURONET_ROI_SIZE for warmup weight transfer
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

# -------------------------
# Refinement target matching
# -------------------------

STAGE2_REFINE_POS_IOU = 0.45
STAGE2_REFINE_NEG_IOU = 0.20
STAGE2_REFINE_POS_WEIGHT = 1.0


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
# Warmup training (ROI pipeline pre-training for KuroNet)
# ============================================================

WARMUP_EPOCHS = 15
WARMUP_LAMBDA_BOX = 0.5
WARMUP_LAMBDA_DELTA = 1.0
WARMUP_LAMBDA_SCORE = 0.3
WARMUP_LAMBDA_AUX = 1.0


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
KURONET_CLASSIFIER_HIDDEN    = 256         # matches STAGE2_REFINE_HIDDEN_DIM → enables warm-start from aux_head_context

# Pretrained EfficientNet-B0 applied to raw image crops (Clanuwat VGG-16 equivalent).
# Adds ImageNet pretrained features on top of UNet ROI features before refinement.
# Enable after warmup retraining; freeze keeps GPU cost low (only projection trains).
KURONET_USE_CROP_ENCODER     = True
KURONET_CROP_ENCODER_SIZE    = (96, 96)
KURONET_FREEZE_CROP_ENCODER  = False

# ROI pooling — larger crop to preserve dakuten (voiced-mark) strokes
KURONET_ROI_SIZE             = STAGE2_ROI_SIZE  # must stay equal to STAGE2_ROI_SIZE for warmup weight transfer

# Detection params (DET_*) are shared — defined once above, used by Stage 2 and KuroNet.
# Once Optuna (jobs 9182-9184) finishes, update DET_* in the Stage 2 section above.

# Assembled CER: filter out false-positive proposals before transcription
# (ordered_mask includes all 159 props/img; ~18 are FPs that add insertion errors)
KURONET_CER_SCORE_THRESH     = 0.35

# Loss weights (single-phase training, no scheduling)
KURONET_LAMBDA_CHAR          = 1.0         # Primary: per-ROI character classification
KURONET_LAMBDA_BOX           = 0.0         # Box regression disabled — delta is the correct parameterization
KURONET_LAMBDA_DELTA         = 0.75        # Delta regression on positive ROIs
KURONET_LAMBDA_SCORE         = 0.20        # ROI quality BCE
KURONET_BG_WEIGHT            = 0.1         # Weight for in-column negative ROIs (normal text-region FPs)
KURONET_STRONG_BG_WEIGHT     = 0.45        # Weight for isolated/oversized negatives (likely illustrations)
KURONET_BG_SCORE_GATE        = 0.55        # Min sigmoid(refine_score) to suppress a BG prediction at inference

# Focal loss for character classification (gamma=0 → plain cross-entropy)
# Reduced from 2.0 to 1.0: focal loss downweights easy examples, which hurts learning
# when classifier is pre-trained (not random). Gentler focal loss with warmup initialization.
KURONET_FOCAL_GAMMA          = 1.0

# Character augmentation: threshold below which a character is considered rare.
# Rare-char images get stronger augmentation (Option A) and are oversampled (Option B).
# Training stats: freq < 20 → 2508 classes (59% of vocab, 1.2% of tokens)
#                 freq < 50 → 3103 classes (73% of vocab, 2.9% of tokens)
KURONET_RARE_CHAR_THRESH     = 50

# Hard negative mining: extra loss weight applied when the model confuses a tracked pair.
# Pairs are derived from the previous epoch's top-K validation confusion pairs.
KURONET_HARD_NEG_WEIGHT      = 1.5   # multiplier on CE loss for confused pairs
KURONET_HARD_NEG_TOP_K       = 50    # how many top confusion pairs to track per epoch

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
KURONET_VALIDATION_BATCHES   = None   # None = full validation set each epoch
