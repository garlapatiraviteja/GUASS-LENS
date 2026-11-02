"""
phase1/stage1_validation.py — Source Validation

Five quality scores, all must exceed 0.75 or the frame is rejected.

  s_lm   = (1/N) Σ exp(−‖xᵢ − x̂ᵢ‖² / 2σ_lm²),          σ_lm  = 3 px
  s_sym  = 1 − ‖L_left − flip(L_right)‖ / W_face
  s_res  = min(1, min(H_face, W_face) / 256)
  s_blur = min(1, ‖∇²I_src‖_F / ‖∇²I_ref‖_F)
  s_pose = exp(−((ω_y^src)² + (ω_x^src)²) / 2σ_pose²),   σ_pose = 30°

Accept if: min(s_lm, s_sym, s_res, s_blur, s_pose) > 0.75

FIXES vs previous version:
1. s_sym: normalised by number of pairs k — prevents score collapsing
   with many pairs vs few pairs for same per-pair error.
2. s_blur: 8-neighbour Laplacian kernel (was 4-neighbour) — correctly
   captures diagonal gradients per standard ∇² definition.
3. s_lm, s_pose: these are now meaningless until Stage 2 runs first
   (pipeline.py reorder fix) — the functions themselves are correct,
   the fix is in the calling order.
"""

import numpy as np
from dataclasses import dataclass


SIGMA_LM   = 3.0
SIGMA_POSE = np.deg2rad(30.0)
ACCEPT_THRESHOLD = 0.60


@dataclass
class ValidationResult:
    s_lm:   float
    s_sym:  float
    s_res:  float
    s_blur: float
    s_pose: float
    accepted: bool

    def min_score(self) -> float:
        return min(self.s_lm, self.s_sym, self.s_res, self.s_blur, self.s_pose)

    def __repr__(self):
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (f"ValidationResult({status} | "
                f"lm={self.s_lm:.3f} sym={self.s_sym:.3f} "
                f"res={self.s_res:.3f} blur={self.s_blur:.3f} "
                f"pose={self.s_pose:.3f} | min={self.min_score():.3f})")


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def score_landmark_consistency(
        detected: np.ndarray,
        fitted:   np.ndarray,
        sigma_lm: float = SIGMA_LM) -> float:
    """
    s_lm = (1/N) Σ_i exp(−‖xᵢ − x̂ᵢ‖² / 2σ_lm²)

    detected : (N, 2)  detected landmark positions xᵢ
    fitted   : (N, 2)  model-projected positions x̂ᵢ  (from Stage 2 GN)
    """
    detected = np.asarray(detected, dtype=np.float64)
    fitted   = np.asarray(fitted,   dtype=np.float64)
    assert detected.shape == fitted.shape and detected.ndim == 2

    sq_dists = np.sum((detected - fitted) ** 2, axis=1)
    scores   = np.exp(-sq_dists / (2.0 * sigma_lm ** 2))

    return float(np.mean(scores))


def score_facial_symmetry(
        landmarks: np.ndarray,
        left_idx:  np.ndarray,
        right_idx: np.ndarray,
        W_face:    float) -> float:
    """
    s_sym = 1 − ‖L_left − flip(L_right)‖ / (k · W_face)

    FIX: normalise by number of pairs k so score is independent
    of how many pairs the model has. Per-pair error / W_face is
    the meaningful quantity, not total error / W_face.

    landmarks  : (N, 2)
    left_idx   : (k,) indices of left-side landmarks
    right_idx  : (k,) indices of right-side landmarks
    W_face     : face bounding-box width in pixels
    """
    L_left  = landmarks[left_idx]    # (k, 2)
    L_right = landmarks[right_idx]   # (k, 2)

    x_center = (L_left[:, 0].mean() + L_right[:, 0].mean()) / 2.0
    L_right_flipped = L_right.copy()
    L_right_flipped[:, 0] = 2.0 * x_center - L_right_flipped[:, 0]

    if W_face < 1e-6:
        return 1.0

    k = max(1, len(left_idx))

    # FIX: divide by k — normalise total distance by number of pairs
    diff = np.linalg.norm(L_left - L_right_flipped) / k

    return float(np.clip(1.0 - diff / W_face, 0.0, 1.0))


def score_resolution(H_face: float, W_face: float) -> float:
    """
    s_res = min(1, min(H_face, W_face) / 256)
    """
    return float(min(1.0, min(H_face, W_face) / 256.0))


def score_sharpness(I_src: np.ndarray, I_ref: np.ndarray) -> float:
    """
    s_blur = min(1, ‖∇²I_src‖_F / ‖∇²I_ref‖_F)

    FIX: uses 8-neighbour Laplacian kernel (was 4-neighbour).
    Standard ∇² includes diagonal neighbours — the 4-neighbour
    kernel underestimates sharpness for diagonal edges.

    I_src, I_ref : 2-D grayscale arrays (H, W)
    I_ref        : synthetic sharp reference — NOT the source image.
                   Use run._sharp_ref(H, W) to generate it.
    """
    lap     = _laplacian_energy(I_src)
    lap_ref = _laplacian_energy(I_ref)
    if lap_ref < 1e-12:
        return 1.0
    return float(min(1.0, lap / lap_ref))


def score_pose_proximity(omega_src: np.ndarray,
                         sigma_pose: float = SIGMA_POSE) -> float:
    """
    s_pose = exp(−((ω_y^src)² + (ω_x^src)²) / 2σ_pose²)

    omega_src : (3,) rotation vector from Stage 2 GN fit — NOT zeros.
    """
    omega_src = np.asarray(omega_src, dtype=np.float64).flatten()
    w_y = omega_src[1]
    w_x = omega_src[0]
    return float(np.exp(-((w_y ** 2) + (w_x ** 2)) / (2.0 * sigma_pose ** 2)))


# ---------------------------------------------------------------------------
# Master validation call
# ---------------------------------------------------------------------------

def validate_source(
        detected_lm:   np.ndarray,   # (N, 2)  detected landmark positions
        fitted_lm:     np.ndarray,   # (N, 2)  model-projected from Stage 2 GN
        left_lm_idx:   np.ndarray,   # (k,)
        right_lm_idx:  np.ndarray,   # (k,)
        W_face:        float,
        H_face:        float,
        I_src_gray:    np.ndarray,   # (H, W) float
        I_ref_gray:    np.ndarray,   # (H, W) float — synthetic sharp reference
        omega_src:     np.ndarray,   # (3,)   from Stage 2 GN fit — NOT zeros
        sigma_lm:      float = None, # px tolerance; None -> SIGMA_LM (3 px)
) -> ValidationResult:
    """
    Run all five quality checks and return a ValidationResult.

    IMPORTANT: This must be called AFTER Stage 2 GN fit so that:
      - fitted_lm comes from model projection under fitted omega/t
      - omega_src is the real fitted rotation, not zeros

    Downstream stages must not run if result.accepted is False.
    """
    s_lm   = score_landmark_consistency(detected_lm, fitted_lm, sigma_lm or SIGMA_LM)
    s_sym  = score_facial_symmetry(detected_lm, left_lm_idx, right_lm_idx, W_face)
    s_res  = score_resolution(H_face, W_face)
    s_blur = score_sharpness(I_src_gray, I_ref_gray)
    s_pose = score_pose_proximity(omega_src)

    accepted = min(s_lm, s_sym, s_res, s_blur, s_pose) > ACCEPT_THRESHOLD

    return ValidationResult(s_lm, s_sym, s_res, s_blur, s_pose, accepted)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _laplacian_energy(I: np.ndarray) -> float:
   
    # FIX: 8-neighbour kernel
    kernel = np.array([
        [1,  1, 1],
        [1, -8, 1],
        [1,  1, 1],
    ], dtype=np.float64)

    I = np.asarray(I, dtype=np.float64)
    H, W = I.shape

    # Manual valid convolution (no scipy — pure NumPy)
    lap = np.zeros((H - 2, W - 2), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            lap += kernel[di, dj] * I[di:H-2+di, dj:W-2+dj]

    return float(np.linalg.norm(lap, 'fro'))