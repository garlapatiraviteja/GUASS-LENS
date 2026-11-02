"""
face_recogniser/landmark_correspondence.py
==========================================
Maps the 68 standard facial landmark points (dlib / iBUG convention)
to 3DMM vertex indices.

The 68-point scheme:
  0-16   jawline           (17 pts)
  17-21  right eyebrow     ( 5 pts)
  22-26  left  eyebrow     ( 5 pts)
  27-30  nose bridge       ( 4 pts)
  31-35  nose tip/base     ( 5 pts)
  36-41  right eye         ( 6 pts)
  42-47  left  eye         ( 6 pts)
  48-59  outer lip         (12 pts)
  60-67  inner lip         ( 8 pts)

Each 68-point landmark is mapped to:
  - A 3DMM vertex index (lm_vidx)
  - Its facial region  (for sub-model routing)

For a real 3DMM (Basel Face Model, FLAME, etc.) use the official
correspondence table from that model's release.  The table below
is a structurally correct placeholder mapping that mirrors the
BFM2009 → iBUG-68 correspondence published by Booth et al. 2016.
Replace `VERTEX_INDICES` with your model's actual values.

Reference groups (used by Stage 2 regional Jacobian blocks):
  EYE_LM_IDX    → ψ_eye  block
  BROW_LM_IDX   → γ_brow block
  NOSE_LM_IDX   → δ_nose block
  MOUTH_LM_IDX  → β_mouth (subset of β block)
  JAW_LM_IDX    → boundary residual (Phase 3 Stage 1)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 68-point → 3DMM vertex index table
# Replace with your model's actual correspondence values.
# ─────────────────────────────────────────────────────────────────────────────

# UNUSED IN THE CURRENT PIPELINE — run.py always passes
# lm_vidx=np.arange(model.V), so this table's fallback path in
# __init__.py never executes. Only valid for the synthetic 7x7 grid —
# replace with the real BFM/FLAME correspondence table before reviving.

VERTEX_INDICES = np.array([
    # ── Synthetic 49-vertex (7×7 grid) model mapping ──────────────────────
    # Grid: vi(row, col) = row*7 + col
    # row 0=forehead(y=80) … row 6=chin(y=-80)
    # col 0=face-left(x<0) … col 3=centre … col 6=face-right(x>0)
    # iBUG "right" = viewer's right = face's left = cols 0-2 (x<0)
    #
    # NOTE: a real 3DMM (BFM/FLAME) would use the BFM→iBUG-68 table from
    # Booth et al. 2016.  Replace this block with that table when you switch
    # to a real 3DMM.  The indices below are geometrically correct for the
    # synthetic 7×7 grid and ensure all sub-region Jacobian blocks fire.

    # Jawline (0-16) — bottom-row outer ring
    35, 42, 43, 44, 44, 45, 45,   # 0-6:  left side jaw
    45,                             # 7:    chin centre
    45, 45, 46, 46, 47, 47, 48, 48, 41,  # 8-16: right side jaw

    # Right eyebrow (17-21) — row 0, cols 1-2  [eye_idx contains 1,2]
     1,  1,  2,  2,  2,

    # Left eyebrow (22-26) — row 0, cols 4-5   [eye_idx contains 4,5]
     4,  4,  5,  5,  5,

    # Nose bridge (27-30) — centre column rows 1-2
    10, 17, 17, 24,

    # Nose base (31-35) — row 3, cols 2-4      [nose_idx: 23,24,25]
    23, 23, 24, 25, 25,

    # Right eye (36-41) — rows 1-2, cols 1-2   [eye_idx: 8,9,15,16]
     8,  8,  9, 16, 15, 15,

    # Left eye (42-47) — rows 1-2, cols 4-5    [eye_idx: 11,12,18,19]
    11, 11, 12, 19, 18, 18,

    # Outer lip (48-59) — rows 4-5, cols 2-4
    30, 30, 31, 32, 32, 33,
    39, 39, 38, 37, 37, 36,

    # Inner lip (60-67)
    30, 31, 31, 32,
    39, 38, 38, 37,
], dtype=np.intp)

assert len(VERTEX_INDICES) == 68, "Must have exactly 68 landmark vertices"


# ─────────────────────────────────────────────────────────────────────────────
# Region index groups  (indices INTO the 68-point array, not vertex IDs)
# ─────────────────────────────────────────────────────────────────────────────

JAW_LM_IDX   = np.arange(0,  17, dtype=np.intp)   # jawline — boundary mask
RBROW_LM_IDX = np.arange(17, 22, dtype=np.intp)   # right brow
LBROW_LM_IDX = np.arange(22, 27, dtype=np.intp)   # left  brow
BROW_LM_IDX  = np.arange(17, 27, dtype=np.intp)   # all brow
NOSE_LM_IDX  = np.arange(27, 36, dtype=np.intp)   # nose bridge + base
REYE_LM_IDX  = np.arange(36, 42, dtype=np.intp)   # right eye
LEYE_LM_IDX  = np.arange(42, 48, dtype=np.intp)   # left  eye
EYE_LM_IDX   = np.arange(36, 48, dtype=np.intp)   # both eyes
MOUTH_LM_IDX = np.arange(48, 68, dtype=np.intp)   # outer + inner lip

# Symmetry pairs for s_sym validation score
# Each row = (left_lm_index, right_lm_index) — mirrored pairs
SYMMETRY_PAIRS = np.array([
    # jaw
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    # brow
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    # eye
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    # lip corners
    (48, 54), (60, 64),
], dtype=np.intp)

LEFT_SYM_IDX  = SYMMETRY_PAIRS[:, 0]
RIGHT_SYM_IDX = SYMMETRY_PAIRS[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lm_vidx(subset: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lm_positions_subset_indices, vertex_indices_subset).

    subset: array of landmark indices (0-67) to select.
            If None, returns all 68.

    Usage:
        lm_idx, lm_vidx = get_lm_vidx()
        lm_idx, lm_vidx = get_lm_vidx(EYE_LM_IDX)
    """
    if subset is None:
        subset = np.arange(68, dtype=np.intp)
    return subset, VERTEX_INDICES[subset]


def region_mask_for_vertex(vertex_id: int) -> str:
    """Return region name for a given 3DMM vertex id."""
    eye_verts  = set(VERTEX_INDICES[EYE_LM_IDX].tolist())
    brow_verts = set(VERTEX_INDICES[BROW_LM_IDX].tolist())
    nose_verts = set(VERTEX_INDICES[NOSE_LM_IDX].tolist())
    mouth_verts = set(VERTEX_INDICES[MOUTH_LM_IDX].tolist())
    jaw_verts  = set(VERTEX_INDICES[JAW_LM_IDX].tolist())

    if vertex_id in eye_verts:   return "eye"
    if vertex_id in brow_verts:  return "brow"
    if vertex_id in nose_verts:  return "nose"
    if vertex_id in mouth_verts: return "mouth"
    if vertex_id in jaw_verts:   return "jaw"
    return "skin"


# def mouth_expression_selector(A_exp: np.ndarray,
#                                beta: np.ndarray,
#                                mouth_cols: np.ndarray = None) -> float:
#     """
#     β_mouth = ‖A_exp,mouth^T β‖  > θ_open = 0.15

#     A_exp     : (V, 3, 64) full expression basis
#     beta      : (64,)
#     mouth_cols: indices into dim-64 that govern jaw/lip  (default: [0,1,2,3])

#     Returns scalar mouth-open score.
#     """
#     if mouth_cols is None:
#         from . . phase1.stage2_gn_source import MOUTH_EXP_COLS
#         mouth_cols = MOUTH_EXP_COLS
#     beta_mouth = beta[mouth_cols]
#     return float(np.linalg.norm(beta_mouth))

def mouth_expression_selector(A_exp, beta, mouth_cols=None):
    if mouth_cols is None:
        from ..phase1.stage2_gn_source import MOUTH_EXP_COLS
        mouth_cols = MOUTH_EXP_COLS
    beta_mouth = beta[mouth_cols]
    return float(np.linalg.norm(beta_mouth))