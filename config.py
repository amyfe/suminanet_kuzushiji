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
NUM_WORKERS = 8

BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
GRAD_CLIP = 1.0

NUM_EPOCHS = 30


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
# 'efficientnet_b2' -- EfficientNet-B2 + FPN decoder, ImageNet pretrained,
#                      outputs at stride=2 (H//2, W//2). Requires timm.
BACKBONE_TYPE = "efficientnet_b2"
BACKBONE_BASE_FEATURES = 64
NUM_CLASSES = 3000


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
# plus a centre-distance penalty that keeps predictions anchored near GT centres.
# Weight kept equal to BBOX_WEIGHT so both objectives carry similar gradient magnitude.
STAGE1_DIOU_WEIGHT = 0.1345315263632802
STAGE1_HEATMAP_SIGMA = 1.7545869700715266         
STAGE1_FOCAL_POS_THRESHOLD = 0.479126803723303     

STAGE1_EARLY_STOPPING_PATIENCE = 5   
STAGE1_SIGMA_FLOOR = 1.5
STAGE1_SIGMA_CEIL  = 3.0
# Multiplier in adaptive sigma: sigma_eff = sigma * sqrt(bw * bh) * SIGMA_SCALE
# With stride=2 (512→256 output), effective ranges at STAGE1_HEATMAP_SIGMA=0.7249:
#   char  15px → bw=7.5  → sigma_eff = 1.09 → clamped to floor 1.5
#   char  28px → bw=14.0 → sigma_eff = 2.03 (active range)
#   char  42px → bw=21.0 → sigma_eff = 3.04 → clamped to ceil  3.0
# Characters outside [~20px, ~42px] are always at floor or ceil; only mid-size
# characters are sensitive to STAGE1_HEATMAP_SIGMA.
STAGE1_SIGMA_SCALE = 0.20
STAGE1_DENSITY_GRID          = 8     
STAGE1_DENSITY_FACTOR        = 5.0   
STAGE1_AVG_GT_PER_IMAGE      = 236   


# ============================================================
# Stage 2: Hybrid Recognition (Option C)
# ============================================================

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
STAGE2_CONTEXT_MODE = "bigru"

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
# Warmup training (ROI pipeline pre-training for KuroNet)
# ============================================================

WARMUP_EPOCHS = 15
WARMUP_LAMBDA_BOX = 0.5
WARMUP_LAMBDA_DELTA = 1.0
WARMUP_LAMBDA_SCORE = 0.3
WARMUP_LAMBDA_AUX = 1.0

KURONET_CHECKPOINT_DIR       = CHECKPOINT_DIR / "kuronet_recognizer"
KURONET_CHECKPOINT_DIR.mkdir(exist_ok=True)

KURONET_USE_CONTEXT          = True
KURONET_CONTEXT_BLOCK_GAP_FACTOR = 2.5
KURONET_COPY_PASTE_PROB      = 0.4

KURONET_CLASSIFIER_HIDDEN    = 512
KURONET_USE_CROP_ENCODER     = True
# 112px → 3×3 feature map before global-avg-pool (vs 2×2 at 64/80px).
# Meaningful spatial detail improvement for stroke discrimination (Clanuwat insight).
KURONET_CROP_ENCODER_SIZE    = (112, 112)
# Two-phase freeze schedule:
# Phase 1 (epochs 1–FREEZE_AFTER): unfrozen — EfficientNet-B0 adapts ImageNet → Kuzushiji.
# Phase 2 (epoch FREEZE_AFTER+1+): auto-frozen — only out_proj + crop_fusion keep training.
#   Drops ~5.3M params from backprop → no longer bottleneck at 112px.
KURONET_FREEZE_CROP_ENCODER       = False  # start unfrozen
KURONET_FREEZE_CROP_ENCODER_AFTER = 5      # freeze EfficientNet-B0 weights after this epoch
# A dense page of Kuzushiji can yield hundreds of ROIs per image; with
# BATCH_SIZE=4 and DET_TOP_K up to 500, a single training batch can produce
# ~1500+ crops. Running all of them through EfficientNet-B0 in one unchunked
# forward call (with gradients, while unfrozen) can exceed a 16GB GPU. Process
# crops through the encoder in chunks of this size instead — bounds peak
# memory independent of how many ROIs a batch happens to contain. Note this
# changes BatchNorm statistics from "computed over all N crops" to "computed
# per chunk" during training (eval-mode BN uses running stats, unaffected).
KURONET_CROP_ENCODER_CHUNK_SIZE   = 256
# Pool output size: the AdaptiveAvgPool2d target after the conv layers, before the MLP.
# This is DISTINCT from STAGE2_ROI_SIZE:
#   ROI_SIZE    = how finely tv_roi_align samples from the feature map (conv input resolution)
#   POOL_OUTPUT = how many spatial cells the downstream MLP receives
# 4×4=16 cells captures coarse stroke layout (enough for character discrimination) while
# keeping the Linear layer small (conv_ch*16 inputs vs conv_ch*256 for a 16×16 pool).
# Increase only if the MLP is a bottleneck — requires retraining from scratch.
KURONET_ROI_POOL_OUTPUT_SIZE = (4, 4)

# Per-ROI detector threshold for KuroNet proposal extraction.
# Intentionally lower than DET_SCORE_THRESH (Stage 1 validation threshold):
# KuroNet's classifier filters the expanded candidate set, so casting a wider
# net here trades a few extra FP proposals for better GT recall.
KURONET_DET_SCORE_THRESH     = 0.26

# Box delta smoothing: tanh-scaled dx/dy and dw/dh in ROIRefinementHead.
# Larger values allow bigger shifts per step but destabilize early training.
KURONET_DELTA_SCALE_XY       = 0.25
KURONET_DELTA_SCALE_WH       = 0.20

KURONET_CER_SCORE_THRESH     = 0.35

# Furigana / noise-box filter: proposals whose area is below this fraction of the
# median character area are discarded.  Only applied when at least KURONET_FURIGANA_MIN_SAMPLES
# characters are detected (fewer samples → unreliable median → skip filter).
KURONET_FURIGANA_AREA_RATIO  = 0.25
KURONET_FURIGANA_MIN_SAMPLES = 5

KURONET_LAMBDA_CHAR          = 1.0         
KURONET_LAMBDA_BOX           = 0.0         
KURONET_LAMBDA_DELTA         = 0.75        
KURONET_LAMBDA_SCORE         = 0.20        
KURONET_BG_WEIGHT            = 0.1         
KURONET_STRONG_BG_WEIGHT     = 0.45        
KURONET_BG_SCORE_GATE        = 0.55

# Minimum sigmoid(refine_score) for a box to "anchor" column/row geometry in
# ROIReadingOrder's reading-order clustering. Low-confidence boxes (weak/
# duplicate detections, gap-filled synthetic proposals) are excluded from
# establishing column positions, then snapped to the nearest anchor-derived
# column afterward -- otherwise their x-centers smear the gaps between real
# columns on dense pages, collapsing column detection.
KURONET_READING_ORDER_CONFIDENCE = 0.5

# Initial value for the learnable residual scale in ROIRefinementHead.
# Stored as a sigmoid-constrained nn.Parameter (logit-parameterised).
# sigmoid(logit(0.5)) = 0.5 — identical to the old hardcoded constant at init time.
KURONET_RESIDUAL_SCALE_INIT  = 0.5

KURONET_FOCAL_GAMMA          = 1.0

KURONET_RARE_CHAR_THRESH     = 50

KURONET_HARD_NEG_WEIGHT      = 1.5   
KURONET_HARD_NEG_TOP_K       = 20    
KURONET_LAMBDA_SCRIPT        = 0.15

# Optimizer
KURONET_LR                   = 1.5e-4
KURONET_LR_ETA_MIN           = 1e-6       
KURONET_WEIGHT_DECAY         = 5e-4       # was 9.4e-5; raised to fight overfitting

# Training
KURONET_EPOCHS               = 50
KURONET_EARLY_STOPPING_PATIENCE = 7   # epochs without composite-score improvement before stopping
KURONET_GRAD_ACCUM_STEPS     = 4
KURONET_ENABLE_TQDM          = True
KURONET_PROGRESS_POSTFIX_N   = 70
KURONET_LOG_PREDICTIONS      = True
KURONET_PREDICTION_SAMPLES   = 2
KURONET_VALIDATION_BATCHES   = None   # None = full validation set each epoch


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
