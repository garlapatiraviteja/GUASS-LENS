"""
face_recogniser/landmark_detector.py
======================================
Landmark initialisation — spec-faithful.

Your spec does NOT use a separate landmark detector (no CLM, no ASM,
no patch response maps).  The landmarks are initialised from the 3DMM
mean shape and then refined entirely by the Gauss-Newton fit in
Phase 1 Stage 2.

This module provides:
  1. Project 3DMM mean shape → initial (N, 2) landmark positions.
  2. Per-organ refinement warm-starts — offset each organ's initial
     position using the known mean shape geometry:
       eyes   → S_eye_ref projected
       brows  → S_brow_3dmm projected
       nose   → S_nose_3dmm projected
       mouth  → outer lip vertices projected
  3. A thin wrapper (LandmarkInitialiser) used by FaceRecogniser.

The GN in Phase 1 Stage 2 is responsible for the actual fitting.
"""

import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Per-organ initial projections
# ─────────────────────────────────────────────────────────────────────────────


def project_vertices(S, K, t):
    P = S + t[None, :]
    Z = P[:, 2]
    uvh = (K @ P.T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        pi = uvh[:, :2] / Z[:, None]
    pi[:, 1] = 2*K[1,2] - pi[:, 1]   # must match geometry.py exactly
    pi[Z <= 0] = np.nan
    return pi

def init_eye_landmarks(S_eye_ref: np.ndarray, eye_idx: np.ndarray,
                       K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project eye reference shape → (V_eye, 2) initial positions."""
    return project_vertices(S_eye_ref, K, t)


def init_brow_landmarks(S_bar: np.ndarray, brow_idx: np.ndarray,
                        K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project brow vertices of mean shape → (V_brow, 2)."""
    return project_vertices(S_bar[brow_idx], K, t)


def init_nose_landmarks(S_bar: np.ndarray, nose_idx: np.ndarray,
                        K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project nose vertices of mean shape → (V_nose, 2)."""
    return project_vertices(S_bar[nose_idx], K, t)


# ─────────────────────────────────────────────────────────────────────────────
# Full initialiser
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkInitialiser:
    """
    Produces initial (N, 2) landmark positions from the 3DMM mean shape.

    This replaces any external landmark detector:
      - No patch responses
      - No CLM / ASM
      - No Haar / CNN
    The GN fit in Phase 1 Stage 2 IS the detector.

    Usage
    -----
        init = LandmarkInitialiser(model, K)
        lm_init, t_init = init.run(img_h, img_w, lm_vidx)
        # Feed lm_init → Stage 1 validation
        # Feed lm_init, t_init → Stage 2 GN as warm-start
    """

    def __init__(self, model, K: np.ndarray, face_scale: float = 0.65):
        """
        model      : MorphableModel
        K          : (3,3) camera intrinsics
        face_scale : target face width as fraction of image width
        """
        self.model       = model
        self.K           = K
        self.face_scale  = face_scale

    def run(self, img_h: int, img_w: int,
            lm_vidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute initial landmark positions and translation.

        Returns
        -------
        lm_init  : (N, 2)  pixel positions for lm_vidx vertices
        t_init   : (3,)    initial camera translation for Stage 2 GN
        """
        from .face_detector import init_from_mean_shape, select_landmark_subset
        lm_all, t_init, _ = init_from_mean_shape(
            self.model.S_bar, self.K, img_h, img_w, self.face_scale
        )
        lm_init = select_landmark_subset(lm_all, lm_vidx)
        return lm_init, t_init
