"""
face_params/extractor.py — high-level API

    from face_params import FaceParamExtractor
    ex = FaceParamExtractor()                    # synthetic model, or FaceParamExtractor(model_path="bfm.npz")
    p  = ex.fit_image(img)                       # img: HxWx3 uint8 or float
    p.eye, p.mouth_openness, p.head_pose_deg     # readable parameters
    p.region_points_2d["nose"]                   # (n,2) pixel positions of the nose vertices

    params = ex.track_video(frames)              # list[FaceParams], one per frame

IMPORTANT — where the 2D landmarks come from
--------------------------------------------
By default (landmarks=None) this uses the original project's strategy: project the
model's mean shape, then let the Gauss-Newton fit pull it onto the image using the
appearance term. That works on clean, roughly frontal, centred faces. For real-world
use, pass landmarks from a real detector (e.g. MediaPipe / dlib / FAN) together with
`lm_vidx`, the model-vertex index each landmark corresponds to.
"""
from __future__ import annotations

import contextlib, io, os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .geometry import make_K, project_mesh, rodrigues
from .shape_model import MorphableModel, SourceState
from .face_recogniser import FaceRecogniser
from .face_recogniser.face_detector import initialise_landmarks
from .phase1.stage2_gn_source import (fit_source, extract_eye_params,
                                   compute_mouth_score, THETA_OPEN)
from .phase1.stage4_target_neutral import fit_target_neutral, TargetFrameData
from .phase1.stage3_rasterise import subpass_A_neutral, synthesis_fallback, merge_texture
from .mesh_utils import compute_vertex_normals
from .phase2.per_frame_tracking import (NewtonState, newtonian_warmstart, fit_target_frame,
                            quality_gate)


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------

@dataclass
class FaceParams:
    """All parameters extracted for one image / frame."""
    alpha:      np.ndarray               # (80,)  identity
    beta:       np.ndarray               # (64,)  expression
    omega:      np.ndarray               # (3,)   head rotation (Rodrigues vector)
    t:          np.ndarray               # (3,)   head translation (camera space)
    psi_eye:    np.ndarray               # (6,)   [aperture, gaze_h, gaze_v] x (L, R)
    gamma_brow: np.ndarray               # (8,)   brow
    delta_nose: np.ndarray               # (3*V_nose,) nose offsets
    psi_mouth:  np.ndarray               # (4,)   mouth
    gamma_ear:  np.ndarray               # (4,)   ear
    K:          np.ndarray               # (3,3)  camera intrinsics used
    model:      MorphableModel = field(repr=False, default=None)
    quality:    Optional[float] = None   # tracker confidence in [0,1] (video only)
    decision:   Optional[str]   = None   # PROCEED / CONSERVATIVE / FALLBACK (video only)
    texture:    Optional[np.ndarray] = field(default=None, repr=False)  # (H,W,3) baked UV colour texture (Stage 3)
    uv:         Optional[np.ndarray] = field(default=None, repr=False)  # (V,2) UV coords of every vertex

    # ---- readable views ---------------------------------------------------
    @property
    def eye(self) -> dict:
        """aperture_left/right, gaze_h_left/right, gaze_v_left/right."""
        return extract_eye_params(self.psi_eye)

    @property
    def mouth_openness(self) -> float:
        """‖β_mouth‖ — compare against THETA_OPEN (0.15) for open/closed."""
        return compute_mouth_score(self.beta)

    @property
    def mouth_open(self) -> bool:
        return self.mouth_openness > THETA_OPEN

    @property
    def head_pose_deg(self) -> dict:
        """Approximate yaw / pitch / roll in degrees (from R = Rodrigues(omega))."""
        R = rodrigues(self.omega)
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
        roll  = np.arctan2(R[1, 0], R[0, 0])
        return {"yaw": float(np.degrees(yaw)), "pitch": float(np.degrees(pitch)),
                "roll": float(np.degrees(roll))}

    # ---- geometry ---------------------------------------------------------
    def shape(self) -> np.ndarray:
        """(V,3) fitted 3D mesh in model space (before pose)."""
        m = self.model
        return m.full_shape(self.alpha, self.beta, self.psi_eye, self.gamma_brow,
                            self.delta_nose, psi_mouth=self.psi_mouth,
                            gamma_ear=self.gamma_ear)

    def project(self) -> np.ndarray:
        """(V,2) pixel positions of every model vertex."""
        px, _ = project_mesh(self.shape(), self.omega, self.t, self.K)
        return px

    def normals(self, smooth_iters: int = 2) -> np.ndarray:
        """(V,3) area-weighted, smoothed vertex normals of the fitted mesh (model space)."""
        return compute_vertex_normals(self.shape(), self.model.faces, smooth_iters=smooth_iters)

    @property
    def region_points_2d(self) -> dict:
        """Pixel positions of each facial region's vertices."""
        px = self.project()
        m = self.model
        out = {"eye": px[m.eye_idx], "brow": px[m.brow_idx], "nose": px[m.nose_idx]}
        if len(m.mouth_idx): out["mouth"] = px[m.mouth_idx]
        if len(m.ear_idx):   out["ear"]   = px[m.ear_idx]
        return out

    @property
    def region_points_3d(self) -> dict:
        """Camera-space 3D positions of each region's vertices."""
        S = self.shape()
        P = (rodrigues(self.omega) @ S.T).T + self.t
        m = self.model
        out = {"eye": P[m.eye_idx], "brow": P[m.brow_idx], "nose": P[m.nose_idx]}
        if len(m.mouth_idx): out["mouth"] = P[m.mouth_idx]
        if len(m.ear_idx):   out["ear"]   = P[m.ear_idx]
        return out

    # ---- export -----------------------------------------------------------
    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("alpha", "beta", "omega", "t", "psi_eye", "gamma_brow",
              "delta_nose", "psi_mouth", "gamma_ear", "K")}
        if self.quality is not None:
            d["quality"] = self.quality
        if self.texture is not None:
            d["texture"] = self.texture
        if self.uv is not None:
            d["uv"] = self.uv
        return d

    def save(self, path: str) -> None:
        np.savez(path, **self.to_dict())


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def load_model_npz(path: str, return_extras: bool = False):
    """Load a 3DMM from .npz (keys: S_bar,A_id,A_exp,B_eye,B_brow,S_eye_ref,
    eye_idx,brow_idx,nose_idx,faces  [+ ear_idx,mouth_idx,B_ear,B_mouth])."""
    d = np.load(path, allow_pickle=False)
    opt = lambda k, default: d[k] if k in d else default
    model = MorphableModel(
        S_bar=d["S_bar"], A_id=d["A_id"], A_exp=d["A_exp"],
        B_eye=d["B_eye"], B_brow=d["B_brow"], S_eye_ref=d["S_eye_ref"],
        eye_idx=d["eye_idx"].astype(np.intp), brow_idx=d["brow_idx"].astype(np.intp),
        nose_idx=d["nose_idx"].astype(np.intp), faces=d["faces"].astype(np.intp),
        ear_idx=opt("ear_idx", np.array([], dtype=np.intp)).astype(np.intp),
        mouth_idx=opt("mouth_idx", np.array([], dtype=np.intp)).astype(np.intp),
        B_ear=opt("B_ear", np.zeros((0, 3, 4))),
        B_mouth=opt("B_mouth", np.zeros((0, 3, 4))),
    )
    if return_extras:       # uv / mouth_mask are needed for the Stage-3 texture bake
        return model, (d["uv"] if "uv" in d else None), \
               (d["mouth_mask"].astype(bool) if "mouth_mask" in d else None)
    return model


def _to_float01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.float64)
    return img / 255.0 if img.max() > 1.5 else img


# ----------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------

class FaceParamExtractor:
    """
    Parameters
    ----------
    model       : MorphableModel, or None → synthetic model (demo only)
    model_path  : .npz 3DMM file (used when `model` is None)
    K           : fixed camera intrinsics. None → built per image as
                  fx=fy=1.2·max(H,W), cx=W/2, cy=H/2
    face_scale  : initial face width as fraction of image width
    verbose     : False (default) silences the fitters' per-iteration debug prints
    (uv / mouth mask for the texture bake come from the .npz, or the synthetic model)
    """

    def __init__(self, model: Optional[MorphableModel] = None,
                 model_path: Optional[str] = None, K: Optional[np.ndarray] = None,
                 face_scale: float = 0.65, verbose: bool = False):
        self.verbose = verbose
        uv, mouth_mask = None, None
        if model is None:
            if model_path and os.path.isfile(model_path):
                model, uv, mouth_mask = load_model_npz(model_path, return_extras=True)
            else:
                from .synthetic_model import make_synthetic_model, make_synthetic_uv
                with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())):
                    model = make_synthetic_model()
                    uv = make_synthetic_uv(model.V, model_S_bar=model.S_bar)
        self.model      = model
        self.uv         = uv                 # None → no texture bake possible
        self.mouth_uv_mask = mouth_mask      # None → all-False (same as synthetic model)
        self._fixed_K   = K
        self.face_scale = face_scale
        self.lm_vidx    = np.arange(model.V, dtype=np.intp)   # use every vertex
        self._rec = FaceRecogniser(model=model,
                                   K=K if K is not None else make_K(1, 1, 0, 0),
                                   face_scale=face_scale,
                                   force_K_rebuild=(K is None))

    # ---- helpers ----------------------------------------------------------
    def _K_for(self, img) -> np.ndarray:
        if self._fixed_K is not None:
            return self._fixed_K
        H, W = img.shape[:2]
        s = max(H, W) * 1.2
        return make_K(s, s, W / 2.0, H / 2.0)

    def _params_from_state(self, st, K, quality=None, decision=None, alpha=None) -> FaceParams:
        z4 = np.zeros(4)
        return FaceParams(
            alpha=np.asarray(st.alpha if alpha is None else alpha), beta=np.asarray(st.beta),
            omega=np.asarray(st.omega), t=np.asarray(st.t),
            psi_eye=np.asarray(st.psi_eye), gamma_brow=np.asarray(st.gamma_brow),
            delta_nose=np.asarray(getattr(st, "delta_nose", np.zeros(3 * len(self.model.nose_idx)))),
            psi_mouth=np.asarray(getattr(st, "psi_mouth", z4)),
            gamma_ear=np.asarray(getattr(st, "gamma_ear", z4)),
            K=K, model=self.model, quality=quality, decision=decision, uv=self.uv)

    # ---- single image -----------------------------------------------------
    def _quiet(self):
        return contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())

    def bake_texture(self, img: np.ndarray, params: FaceParams) -> Optional[np.ndarray]:
        """
        Stage 3: bake the image's colours into the model's UV layout using the
        fitted mesh in `params` (same procedure as FaceParamPipeline.init_session,
        neutral pass + mouth synthesis fallback). Returns (H,W,3) float, or None
        when the extractor has no UV layout.
        """
        if self.uv is None:
            return None
        I = _to_float01(img)
        H, W = I.shape[:2]
        mask = self.mouth_uv_mask
        if mask is None or mask.shape != (H, W):
            mask = np.zeros((H, W), dtype=bool)
        st = SourceState(alpha=params.alpha, beta=params.beta, omega=params.omega,
                         t=params.t, psi_eye=params.psi_eye, gamma_brow=params.gamma_brow,
                         delta_nose=params.delta_nose)
        st.psi_mouth, st.gamma_ear = params.psi_mouth, params.gamma_ear
        with self._quiet():
            T = subpass_A_neutral(model=self.model, state=st, uv=self.uv, K=params.K,
                                  I_src=I, img_h=H, img_w=W, mouth_uv_mask=mask)
            T_synth = None
            if mask.any():
                ys, xs = np.where(mask)
                T_synth = synthesis_fallback(T, int(xs.mean()) if len(xs) else W // 2, mask)
            return merge_texture(T, None, mask, T_synth)

    def fit_image(self, img: np.ndarray, landmarks: Optional[np.ndarray] = None,
                  lm_vidx: Optional[np.ndarray] = None,
                  texture: bool = True) -> FaceParams:
        """Fit all face parameters to one image. texture=True also runs Stage 3
        (UV colour-texture bake) and returns it as `params.texture`."""
        with self._quiet():
            p = self._fit_image(img, landmarks, lm_vidx)
        if texture:
            p.texture = self.bake_texture(img, p)
        return p

    def _fit_image(self, img, landmarks, lm_vidx):
        I = _to_float01(img)
        K = self._K_for(I)
        if landmarks is None:
            r = self._rec.process(I, self.lm_vidx if lm_vidx is None else lm_vidx)
            return self._params_from_state(r.fitted_state, getattr(r, "K", K))

        lm_vidx = self.lm_vidx if lm_vidx is None else np.asarray(lm_vidx, dtype=np.intp)
        landmarks = np.asarray(landmarks, dtype=np.float64)
        H, W = I.shape[:2]
        _, t_init, _ = initialise_landmarks(self.model.S_bar, K, lm_vidx, H, W,
                                            self.face_scale)
        init = SourceState(alpha=np.zeros(80), beta=np.zeros(64), omega=np.zeros(3),
                           t=t_init.copy(), psi_eye=np.zeros(6), gamma_brow=np.zeros(8),
                           delta_nose=np.zeros(3 * len(self.model.nose_idx)))
        st = fit_source(self.model, K, landmarks, lm_vidx, I, init_state=init)
        return self._params_from_state(st, K)

    # ---- video ------------------------------------------------------------
    def track_video(self, frames: Sequence[np.ndarray],
                    landmarks: Optional[Sequence[np.ndarray]] = None,
                    lm_vidx: Optional[np.ndarray] = None,
                    n_identity_frames: int = 10) -> list:
        """
        Track parameters through a video.

        1. Identity (α) and neutral expression are fitted jointly over the first
           `n_identity_frames` frames (≥10 needed; padded with jittered copies if fewer).
        2. Every frame: Newtonian pose prediction → per-frame joint GN
           (β, ω, t, ψ_eye, γ_brow, ψ_mouth) with α frozen → quality gate.
        """
        with self._quiet():
            return self._track_video(frames, landmarks, lm_vidx, n_identity_frames)

    def _track_video(self, frames, landmarks, lm_vidx, n_identity_frames):
        lm_vidx = self.lm_vidx if lm_vidx is None else np.asarray(lm_vidx, dtype=np.intp)
        imgs = [_to_float01(f) for f in frames]
        if not imgs:
            return []
        K = self._K_for(imgs[0])

        def lm_for(i):
            if landmarks is not None:
                return np.asarray(landmarks[i], dtype=np.float64), None
            r = self._rec.process(imgs[i], lm_vidx)
            return r.landmarks, r

        # ---- identity over first frames
        id_frames, recs = [], []
        for i in range(min(n_identity_frames, len(imgs))):
            lm, r = lm_for(i)
            recs.append(r)
            t0 = r.t_fitted if r is not None else np.array([0.0, 0.0, 500.0])
            id_frames.append(TargetFrameData(landmarks=lm, lm_vidx=lm_vidx,
                                             omega=np.zeros(3), t=t0))
        rng = np.random.default_rng(0)
        base = list(id_frames)
        while len(id_frames) < 10:               # fit_target_neutral needs >= 10
            b = base[len(id_frames) % len(base)]
            id_frames.append(TargetFrameData(
                landmarks=b.landmarks + rng.uniform(-0.5, 0.5, b.landmarks.shape),
                lm_vidx=b.lm_vidx, omega=b.omega.copy(), t=b.t.copy()))
        ident = fit_target_neutral(model=self.model, K=K, frames=id_frames)

        # ---- per frame
        results, prev_tgt = [], None
        t_start = id_frames[0].t.copy()
        newton = NewtonState(q=np.zeros(3), q_dot=np.zeros(3), t=t_start, t_dot=np.zeros(3))
        quality, prev_lm = 1.0, None
        for i, I in enumerate(imgs):
            lm, r = lm_for(i)
            if prev_lm is None:
                prev_lm = lm.copy()
            pred, _ = newtonian_warmstart(prev=newton, tau=np.zeros(3),
                                          f_flow=np.zeros_like(lm), lm_prev=prev_lm,
                                          quality=quality)
            t_init = r.t_init if (r is not None and hasattr(r, "t_init")) else None
            st, _ = fit_target_frame(
                model=self.model, alpha_tgt=ident.alpha_tgt, beta_neutral=ident.beta_neutral,
                K=K, I_tgt=I, landmarks=lm, lm_vidx=lm_vidx, q_pred=pred.q,
                sigma_q=0.1 if i > 0 else 10.0, t_pred=pred.t, sigma_t=10.0,
                prev_state=prev_tgt, frame_index=i, t_init=t_init)

            S = self.model.full_shape(
                ident.alpha_tgt, st.beta, st.psi_eye, st.gamma_brow,
                np.zeros(3 * len(self.model.nose_idx)),
                psi_mouth=st.psi_mouth, gamma_ear=st.gamma_ear)
            proj, _ = project_mesh(S, st.omega, st.t, K)
            proj_lm = proj[lm_vidx]
            H, W = I.shape[:2]
            ok = (np.isfinite(proj_lm).all(1) & (proj_lm[:, 0] >= 0) & (proj_lm[:, 0] < W)
                  & (proj_lm[:, 1] >= 0) & (proj_lm[:, 1] < H))
            if ok.sum() < 3:
                quality, decision = 0.5, "CONSERVATIVE"
            else:
                z = np.zeros((int(ok.sum()), 2))
                quality, decision = quality_gate(lm[ok], proj_lm[ok], st.omega, z, z)

            if decision != "FALLBACK":           # same update rule as the original
                newton = NewtonState(q=st.omega.copy(), q_dot=st.omega - newton.q,
                                     t=st.t.copy(),   t_dot=st.t - newton.t)
            prev_tgt, prev_lm = st, lm.copy()

            results.append(self._params_from_state(st, K, quality, decision,
                                                   alpha=ident.alpha_tgt))
        return results
