# """
# face_recogniser/face_detector.py
# ==================================
# Face localisation — spec-faithful implementation.

# Your spec has NO separate face detector (no Haar, no CNN, no skin mask).
# The spec's "face recognition" IS the GN fitting in Phase 1 Stage 2.

# What this module provides is the INITIALISATION that feeds the GN:

#   1. Project the 3DMM mean shape S̄ under a frontal camera pose
#      → gives pixel positions for all 68 landmark vertices as initial guess.

#   2. Optionally refine the initial translation (t) so the projected mean
#      shape is roughly centred on the image.  This is a simple closed-form
#      step — no cascade, no colour mask:

#        t_x = (img_cx - π(S̄; ω=0, t=0)_cx) / f_x
#        t_y = (img_cy - π(S̄; ω=0, t=0)_cy) / f_y
#        t_z = f_x * face_scale / mean_shape_width_3D

#   3. Return the projected 2-D positions as the warm-start for Stage 1
#      validation and Stage 2 GN.

# The GN in Stage 2 takes over from there — the fitting IS the detection.
# Quality gate (s_lm, s_sym, s_res, s_blur, s_pose) decides accept/reject.

# FaceBox is kept purely as a data container for the image crop region —
# it has no detection logic.
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # FaceBox — crop region only, no detection
# # ─────────────────────────────────────────────────────────────────────────────

# @dataclass
# class FaceBox:
#     """
#     Axis-aligned face region in pixel coordinates.
#     Derived from the projected mean-shape bounding box, not from a detector.
#     """
#     x: int      # left
#     y: int      # top
#     w: int      # width
#     h: int      # height
#     score: float = 1.0   # always 1.0 — region is from the model, not a detector


# # ─────────────────────────────────────────────────────────────────────────────
# # Mean-shape initialisation
# # ─────────────────────────────────────────────────────────────────────────────

# def init_from_mean_shape(
#         S_bar:     np.ndarray,   # (V, 3) 3DMM mean shape
#         K:         np.ndarray,   # (3, 3) camera intrinsics
#         img_h:     int,
#         img_w:     int,
#         face_scale: float = 0.5, # target face width as fraction of image width
# ) -> tuple[np.ndarray, np.ndarray, FaceBox]:
#     """
#     Initialise landmark positions and translation by projecting
#     the 3DMM mean shape centred on the image.

#     Steps (pure geometry — no detector):
#       1. Set ω = 0 (frontal), compute initial t so S̄ projects to image centre.
#       2. Scale t_z so the mean face spans face_scale × img_w pixels.
#       3. Project all vertices → 2-D landmark initialisations.
#       4. Derive FaceBox from projected bounding box.

#     Returns
#     -------
#     lm_init  : (V, 2)  projected 2-D positions for all vertices
#     t_init   : (3,)    initial translation vector
#     box      : FaceBox centred on image
#     """
#     # Step 1: find mean face width in 3-D
#     x3d     = S_bar[:, 0]
#     w3d     = x3d.max() - x3d.min() + 1e-9   # face width in model space


#     centroid_3d = S_bar.mean(axis=0)   # (3,)


#     return lm_init, t_init, box


# # ─────────────────────────────────────────────────────────────────────────────
# # Landmark subset selector
# # ─────────────────────────────────────────────────────────────────────────────


# # ─────────────────────────────────────────────────────────────────────────────
# # Convenience: single entry point matching old FaceRecogniser API
# # ─────────────────────────────────────────────────────────────────────────────

# def initialise_landmarks(
#         S_bar:      np.ndarray,   # (V, 3) mean shape
#         K:          np.ndarray,   # (3, 3) camera matrix
#         lm_vidx:    np.ndarray,   # (N,)   landmark vertex indices
#         img_h:      int,
#         img_w:      int,
#         face_scale: float = 0.5,
# ) -> tuple[np.ndarray, np.ndarray, FaceBox]:
#     """
#     Full initialisation pipeline matching the spec:
#       - Project mean shape → t_init, lm_init_all
#       - Select N landmark positions
#       - Return FaceBox

#     Returns
#     -------
#     lm_init   : (N, 2)  initial landmark positions
#     t_init    : (3,)    initial translation for the GN
#     box       : FaceBox
#     """
#     lm_all, t_init, box = init_from_mean_shape(S_bar, K, img_h, img_w, face_scale)
#     lm_init = select_landmark_subset(lm_all, lm_vidx)
#     return lm_init, t_init, box


# # ─────────────────────────────────────────────────────────────────────────────
# # Resize helper (kept for image preprocessing — not detection)
# # ─────────────────────────────────────────────────────────────────────────────


"""
face_recogniser/face_detector.py
==================================
Face localisation — spec-faithful implementation.

Your spec has NO separate face detector (no Haar, no CNN, no skin mask).
The spec's "face recognition" IS the GN fitting in Phase 1 Stage 2.

What this module provides is the INITIALISATION that feeds the GN:

  1. Project the 3DMM mean shape S̄ under a frontal camera pose
     → gives pixel positions for all 68 landmark vertices as initial guess.

  2. Optionally refine the initial translation (t) so the projected mean
     shape is roughly centred on the image.  This is a simple closed-form
     step — no cascade, no colour mask:

       t_x = (img_cx - π(S̄; ω=0, t=0)_cx) / f_x
       t_y = (img_cy - π(S̄; ω=0, t=0)_cy) / f_y
       t_z = f_x * face_scale / mean_shape_width_3D

  3. Return the projected 2-D positions as the warm-start for Stage 1
     validation and Stage 2 GN.

The GN in Stage 2 takes over from there — the fitting IS the detection.
Quality gate (s_lm, s_sym, s_res, s_blur, s_pose) decides accept/reject.

FaceBox is kept purely as a data container for the image crop region —
it has no detection logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FaceBox — crop region only, no detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    """
    Axis-aligned face region in pixel coordinates.
    Derived from the projected mean-shape bounding box, not from a detector.
    """
    x: int      # left
    y: int      # top
    w: int      # width
    h: int      # height
    score: float = 1.0   # always 1.0 — region is from the model, not a detector

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2

    def crop(self, img: np.ndarray, pad: float = 0.10) -> np.ndarray:
        """Return the face region from img with a small padding."""
        H, W = img.shape[:2]
        p    = int(max(self.w, self.h) * pad)
        x1   = max(0, self.x - p);   y1 = max(0, self.y - p)
        x2   = min(W, self.x+self.w+p); y2 = min(H, self.y+self.h+p)
        return img[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# Mean-shape initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_from_mean_shape(
        S_bar:     np.ndarray,   # (V, 3) 3DMM mean shape
        K:         np.ndarray,   # (3, 3) camera intrinsics
        img_h:     int,
        img_w:     int,
        face_scale: float = 0.65, # target face width as fraction of image width
) -> tuple[np.ndarray, np.ndarray, FaceBox]:
    """
    Initialise landmark positions and translation by projecting
    the 3DMM mean shape centred on the image.

    Steps (pure geometry — no detector):
      1. Set ω = 0 (frontal), compute initial t so S̄ projects to image centre.
      2. Scale t_z so the mean face spans face_scale × img_w pixels.
      3. Project all vertices → 2-D landmark initialisations.
      4. Derive FaceBox from projected bounding box.

    Returns
    -------
    lm_init  : (V, 2)  projected 2-D positions for all vertices
    t_init   : (3,)    initial translation vector
    box      : FaceBox centred on image
    """
    # Step 1: find mean face width in 3-D
    x3d     = S_bar[:, 0]
    w3d     = x3d.max() - x3d.min() + 1e-9   # face width in model space

    fx = K[0, 0]; fy = K[1, 1]
    cx_k = K[0, 2]; cy_k = K[1, 2]

    centroid_3d = S_bar.mean(axis=0)   # (3,)

    # FIX: compute the depth that makes the face span face_scale × img_w pixels.
    # Formula from spec: z = fx * w3d / (face_scale * img_w)
    # Old code used max(existing_depth, target_depth) which always kept the
    # synthetic model at its existing z (~498), making target_depth (~240) 
    # unreachable and face_scale completely ineffective.
    # Now we ALWAYS use the target depth — existing model z is irrelevant.
    target_depth = fx * w3d / (face_scale * img_w)
    target_depth = max(target_depth, 1.0)   # must stay in front of camera

    # Translation to:
    #   1. Centre the face on the image (t_x, t_y cancel the centroid offset)
    #   2. Place it at target_depth (t_z shifts from existing z to target)
    t_x = -centroid_3d[0]
    t_y = -centroid_3d[1]
    t_z = target_depth - centroid_3d[2]   # shift model to target depth

    # t_proj: landmarks warm-start — face centred at target_depth
    t_proj = np.array([t_x, t_y, t_z], dtype=np.float64)

    # t_init for GN: start slightly deeper so GN has a non-zero residual
   
    t_init = np.array([t_x, t_y, t_z + target_depth * 0.3], dtype=np.float64)

    # Step 4: project all vertices under (ω=0, t_proj) for lm_init
    P_cam = S_bar + t_proj[None, :]          # (V, 3)
    Z     = P_cam[:, 2]
    uv_h  = (K @ P_cam.T).T                  # (V, 3)
    with np.errstate(invalid='ignore', divide='ignore'):
        lm_init = uv_h[:, :2] / Z[:, None]
    # FIX (v-flip): geometry.project()/project_mesh() flip v about cy
    # ("v-flip fix": model +Y is up, image +v is down). This warm-start
    # projection did NOT, so lm_init was the mean face mirrored top-to-bottom
    # relative to what the Stage-2/Phase-2 GN projects. Stage-2 GN then spent
    # all its iterations inflating alpha/beta to fold the mesh onto an
    # upside-down template (geo_res stuck ~4000, Z range 286 -> 1000).
    # Must stay identical to geometry.project_mesh.  Do NOT apply the flip
    # anywhere else that already goes through project_mesh.
    lm_init[:, 1] = 2.0 * cy_k - lm_init[:, 1]
    lm_init[Z <= 0] = np.nan

    # Step 5: FaceBox from projected bounding box (finite pts only)
    valid = np.isfinite(lm_init[:, 0])
    if valid.any():
        xs = lm_init[valid, 0]; ys = lm_init[valid, 1]
        x0 = int(np.floor(xs.min())); y0 = int(np.floor(ys.min()))
        x1 = int(np.ceil(xs.max()));  y1 = int(np.ceil(ys.max()))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(img_w, x1); y1 = min(img_h, y1)
        box = FaceBox(x0, y0, x1-x0, y1-y0, score=1.0)
    else:
        # Fallback: full image
        box = FaceBox(0, 0, img_w, img_h, score=1.0)

    return lm_init, t_init, box


# ─────────────────────────────────────────────────────────────────────────────
# Landmark subset selector
# ─────────────────────────────────────────────────────────────────────────────

def select_landmark_subset(
        lm_init:  np.ndarray,    # (V, 2)
        lm_vidx:  np.ndarray,    # (N,)  vertex indices for the N landmarks
) -> np.ndarray:
    """
    Return (N, 2) initial 2-D positions for the selected landmark vertices.
    This is the warm-start fed to Phase 1 Stage 1 validation and Stage 2 GN.
    """
    return lm_init[lm_vidx]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: single entry point matching old FaceRecogniser API
# ─────────────────────────────────────────────────────────────────────────────

def initialise_landmarks(
        S_bar:      np.ndarray,   # (V, 3) mean shape
        K:          np.ndarray,   # (3, 3) camera matrix
        lm_vidx:    np.ndarray,   # (N,)   landmark vertex indices
        img_h:      int,
        img_w:      int,
        face_scale: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, FaceBox]:
    """
    Full initialisation pipeline matching the spec:
      - Project mean shape → t_init, lm_init_all
      - Select N landmark positions
      - Return FaceBox

    Returns
    -------
    lm_init   : (N, 2)  initial landmark positions
    t_init    : (3,)    initial translation for the GN
    box       : FaceBox
    """
    lm_all, t_init, box = init_from_mean_shape(S_bar, K, img_h, img_w, face_scale)
    lm_init = select_landmark_subset(lm_all, lm_vidx)
    return lm_init, t_init, box


# ─────────────────────────────────────────────────────────────────────────────
# Resize helper (kept for image preprocessing — not detection)
# ─────────────────────────────────────────────────────────────────────────────

def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-NumPy bilinear resize. Used only for image normalisation."""
    H, W = img.shape[:2]
    y_src = np.linspace(0, H-1, out_h)
    x_src = np.linspace(0, W-1, out_w)
    xx, yy = np.meshgrid(x_src, y_src)
    x0 = np.floor(xx).astype(int); x1 = np.clip(x0+1, 0, W-1)
    y0 = np.floor(yy).astype(int); y1 = np.clip(y0+1, 0, H-1)
    fx = xx-x0; fy = yy-y0
    if img.ndim == 2:
        return ((1-fx)*(1-fy)*img[y0,x0]+fx*(1-fy)*img[y0,x1]
               +(1-fx)*fy*img[y1,x0]+fx*fy*img[y1,x1]).astype(img.dtype)
    fx=fx[:,:,None]; fy=fy[:,:,None]
    return ((1-fx)*(1-fy)*img[y0,x0]+fx*(1-fy)*img[y0,x1]
           +(1-fx)*fy*img[y1,x0]+fx*fy*img[y1,x1]).astype(img.dtype)