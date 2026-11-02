"""Phase 1 — session init stages."""
from .stage1_validation import validate_source, ValidationResult
from .stage2_gn_source import (fit_source, extract_eye_params, compute_mouth_score,
                               THETA_OPEN, MOUTH_EXP_COLS)
from .stage3_rasterise import (subpass_A_neutral, subpass_B_mouth, synthesis_fallback,
                               merge_texture, detect_open_mouth_frames)
from .stage4_target_neutral import fit_target_neutral, TargetFrameData, TargetNeutralResult
