"""
face_recogniser/__init__.py
============================
Spec-faithful face recognition front-end.

Your spec uses NO external face detector.
Recognition = GN fitting (Phase 1 Stage 2).
This module provides the INITIALISATION that seeds the GN.

Pipeline:
  1. Project 3DMM mean shape S̄ under centred frontal pose
     → (N, 2) initial landmark positions + t_init
  2. FaceResult wraps these for the pipeline stages

The five quality scores (s_lm, s_sym, s_res, s_blur, s_pose) in
Phase 1 Stage 1 ARE the face detection/validation gate.

FIX (this version): FaceResult.t_init restored.
It had been commented out and replaced with t_fitted, which broke the
existing (unmodified) t_init wiring in pipeline.py's process_frame:
    _t_init = _r.t_init if hasattr(_r, 't_init') else None
Since t_init never existed on FaceResult, hasattr() was always False,
and every target frame silently fell back to fit_target_frame's own
default_t=[0,0,500] instead of the per-image warm-start depth computed
by init_from_mean_shape() / initialise_landmarks(). t_init was already
being computed locally inside process() (see the `lm_init, t_init, box =
initialise_landmarks(...)` call below) — it just wasn't being saved onto
the returned result. No other logic changes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .face_detector import (
    FaceBox, init_from_mean_shape,
    select_landmark_subset, initialise_landmarks
)
from .landmark_detector import LandmarkInitialiser
from .landmark_correspondence import VERTEX_INDICES

@dataclass
class FaceResult:
    """
    Initial landmark positions derived from 3DMM mean-shape projection.
    These are warm-starts for Phase 1 Stage 2 GN — not detector outputs.
    """
    landmarks:      np.ndarray   # (N, 2) initial pixel positions
    lm_vidx:        np.ndarray   # (N,)   3DMM vertex indices
    t_init:         np.ndarray   # (3,)   initial (pre-fit) translation warm-start for GN — FIX: restored
    t_fitted:       np.ndarray   # (3,)   translation AFTER GN fitting (fit_source result)
    box:            FaceBox      # projected bounding box
    scores:         np.ndarray   # (N,) all 1.0 at init — updated after GN

    @property
    def W_face(self): return float(self.box.w)
    @property
    def H_face(self): return float(self.box.h)


class FaceRecogniser:
    """
    Spec-faithful face recogniser.

    Initialises landmarks from 3DMM mean shape.
    The GN fit in Phase 1 Stage 2 IS the recognition/detection.

    Parameters
    ----------
    model        : MorphableModel  (required — used for mean-shape projection)
    K            : (3,3) camera intrinsics
    face_scale   : target face width as fraction of image width (default 0.5)
    """

    def __init__(self, model=None, K=None, face_scale: float = 0.65,
                 force_K_rebuild: bool = True, landmark_provider=None):
        # landmark_provider: optional callable(img_rgb) -> LandmarkSet (perception/).
        # When set, the fit targets REAL image landmarks (SAM masks) instead of the
        # model's own mean-shape projection, and the face box / validation tolerance
        # come from the image too.
        self.landmark_provider = landmark_provider
        self.model       = model
        self.K           = K
        self.face_scale  = face_scale
        self._init       = None  # LandmarkInitialiser, built lazily
        # FIX [13]: the previous "_K_is_default = (K is None)" check only
        # rebuilt K per-image when NO K was supplied at construction time.
        # But run.py / app.py always call FaceRecogniser(model=model, K=K)
        # with a real (non-None) K built from load_model(model_path, Hs, Ws)
        # — i.e. built from the SOURCE image's dimensions. That made
        # _K_is_default False for every call, so the per-image rebuild
        # branch in process() never fired, and the SAME source-sized K
        # (e.g. cx=91.5, cy=137.5 for a 183x275 source) was reused for the
        # 400x400 target. The target's face then projects far off-centre
        # (~23%/34% instead of 50%/50%), which is the exact bug seen in
        # the [DEBUG-INIT] log lines staying identical across both calls.
        #
        # force_K_rebuild=True (new default) makes process() ALWAYS rebuild
        # K from each call's actual image dims, regardless of what K was
        # passed at construction. Set force_K_rebuild=False only if you
        # have a genuine externally-calibrated K that must not change
        # (e.g. real camera intrinsics that are physically fixed
        # regardless of how the image is resized/cropped).
        self.force_K_rebuild = force_K_rebuild
        self._K_is_default   = (K is None) or force_K_rebuild

    def _ensure_init(self, img_h: int, img_w: int):
        """Build LandmarkInitialiser lazily (needs model + K)."""
        if self._init is None:
            assert self.model is not None, "FaceRecogniser needs a model"
            assert self.K is not None,     "FaceRecogniser needs camera K"
            self._init = LandmarkInitialiser(self.model, self.K, self.face_scale)

    def process(self, img_rgb: np.ndarray,
            lm_vidx: np.ndarray = None) -> Optional[FaceResult]:
        H, W = img_rgb.shape[:2]
        self._ensure_init(H, W)

        # FIX [3]: rebuild K from this image's actual dims every call.
        # The recogniser is shared between source (e.g. 275x183) and target
        # (e.g. 400x400) calls.  Using a stale K built from source dims puts
        # the principal point ~63px left and ~109px above centre on a 400x400
        # image → warm-start landmarks land in the wrong region → GN diverges.
        # If K was explicitly supplied at init (e.g. from a real calibration),
        # honour it.  Otherwise rebuild from image dims using the standard
        # heuristic fx = fy = max(H,W)*1.2, cx = W/2, cy = H/2.
        from ..geometry import make_K as _make_K
        if self._K_is_default:
            K_img = _make_K(max(H, W) * 1.2, max(H, W) * 1.2, W / 2.0, H / 2.0)
        else:
            K_img = self.K   # caller supplied a real calibrated K — use it

        if lm_vidx is None:
            lm_vidx = np.clip(VERTEX_INDICES, 0, self.model.V - 1)

        ext = self.landmark_provider(img_rgb) if self.landmark_provider is not None else None
        if ext is not None:
            lm_vidx = ext.vidx

        # Step 1: mean-shape warm-start — use per-image K
        lm_init, t_init, box = initialise_landmarks(
            self.model.S_bar, K_img, lm_vidx, H, W, self.face_scale
        )
        if ext is not None:                      # real landmarks replace the template ones
            lm_init, t_init, box = ext.xy.copy(), ext.t_init.copy(), ext.box
        # Step 2: GN fit to get actual face landmarks
        from ..phase1.stage2_gn_source import fit_source
        from ..shape_model import SourceState
        from ..geometry import project_mesh

        V_nose = len(self.model.nose_idx)
        init_state = SourceState(
            alpha      = np.zeros(80),
            beta       = np.zeros(64),
            omega      = np.zeros(3),
            t          = t_init.copy(),
            psi_eye    = np.zeros(6),
            gamma_brow = np.zeros(8),
            delta_nose = np.zeros(3 * V_nose),
        )
        fitted = fit_source(
            model      = self.model,
            K          = K_img,
            landmarks  = lm_init,
            lm_vidx    = lm_vidx,
            I_src      = img_rgb,
            init_state = init_state,
        )

        # Step 3: re-project fitted shape to get refined landmark positions
        S_fitted = self.model.full_shape(
            fitted.alpha, fitted.beta,
            fitted.psi_eye, fitted.gamma_brow, fitted.delta_nose,
            psi_mouth=fitted.psi_mouth, gamma_ear=fitted.gamma_ear
            )
        lm_all, _ = project_mesh(S_fitted, fitted.omega, fitted.t, K_img)
        lm_refined = lm_all[lm_vidx]

        result = FaceResult(
            landmarks     = (ext.xy if ext is not None else lm_refined),
            lm_vidx       = lm_vidx,
            t_init        = t_init,      # FIX: save the pre-fit warm-start depth
            t_fitted      = fitted.t,
            box           = box,
            scores        = np.ones(len(lm_vidx), dtype=np.float64),
        )
        result.fitted_state = fitted
        result.lm_sigma_px = ext.sigma_lm_px if ext is not None else None
        result.K = K_img   # expose per-image K for downstream (stage4, phase2)
        return result