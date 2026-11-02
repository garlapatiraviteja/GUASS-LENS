"""
pipeline.py — Face-Parameter Extraction Pipeline Orchestrator

Wires Phase 1 (session init: S2 → S1 → S3 → S4) → Phase 2 (per-frame tracking)
→ Phase 3 (temporal state). The swap-mesh build, SH relighting, mask and
Laplacian-pyramid blend of the original FaceSwapPipeline are removed; what is
left is everything that EXTRACTS information from faces:

  Phase 1  S2  GN fit of the source image       → identity α, expression β, pose, eye/brow/nose/mouth/ear
           S1  validation of that fit           → 5 quality scores (s_lm, s_sym, s_res, s_blur, s_pose)
           S3  UV texture bake (source colour)   → T_uv_src  (optional / reusable UV skin-colour map)
           S4  multi-frame identity fit         → α_tgt, β_neutral for the video/driving face
  Phase 2  per frame: Newtonian pose prior → joint GN (β, ω, t, ψ_eye, γ_brow, ψ_mouth, γ_ear)
           → quality gate (PROCEED / CONSERVATIVE / FALLBACK)
  Phase 3  per frame: velocity update (+ optional GN optical flow on the raw frames)

Log prefixes are numbered sequentially per phase — Phase 1: S1–S4 · Phase 2: S1 prior, S2 fit,
S3 quality gate · Phase 3: S1 flow, S2 motion state — and app.py's log panel groups on them.
"""

import numpy as np
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from .shape_model import MorphableModel, SourceState
from .geometry import rodrigues, project_mesh, make_K

from .phase1.stage1_validation import validate_source, ValidationResult


class SourceRejected(ValueError):
    """Validation rejected the source; .validation holds the scores (used for the verdict)."""
    def __init__(self, msg, validation):
        super().__init__(msg)
        self.validation = validation

from .phase1.stage2_gn_source import fit_source, THETA_OPEN, compute_mouth_score
from .phase1.stage3_rasterise import (
    subpass_A_neutral, subpass_B_mouth,
    synthesis_fallback, merge_texture,
    detect_open_mouth_frames,
)
from .phase1.stage4_target_neutral import (
    fit_target_neutral, TargetFrameData, TargetNeutralResult
)
from .phase2.per_frame_tracking import (
    NewtonState, newtonian_warmstart,
    fit_target_frame, TargetFrameState,
    quality_gate,
)
from .phase3.tracking import gn_optical_flow, velocity_update
from .diagnose_landmark_offset import draw_landmark_overlay
from .extractor import FaceParams


# ---------------------------------------------------------------------------
# Session data (fixed after Phase 1)
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """Everything computed once during Phase 1."""
    source_state:     SourceState
    T_uv_src:         np.ndarray      # (H, W, 3) source albedo texture (UV space)
    alpha_tgt:        np.ndarray      # (80,)  identity of the tracked/driving face
    beta_tgt_neutral: np.ndarray      # (64,)  its neutral expression
    mouth_uv_mask:    np.ndarray      # (H, W) bool
    uv:               np.ndarray      # (V, 2)
    w_mouth_max:      float = 1.0
    validation:       Optional[ValidationResult] = None


# ---------------------------------------------------------------------------
# Per-frame inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    I_tgt:     np.ndarray   # (H, W, 3) float
    landmarks: np.ndarray   # (N_lm, 2)
    lm_vidx:   np.ndarray   # (N_lm,)


@dataclass
class FrameResult:
    params:         FaceParams           # everything extracted for this frame
    tgt_state:      TargetFrameState
    delta_beta:     np.ndarray           # β − β_neutral
    quality:        float
    decision:       str
    proj_landmarks: np.ndarray           # (N_lm, 2) model landmarks projected into the frame
    region_error:   dict                 # mean px error per region (eye/brow/nose/mouth/ear/all)


@dataclass
class FrameState:
    newton:        NewtonState
    prev_tgt:      Optional[TargetFrameState]
    prev_I:        Optional[np.ndarray]     # previous raw frame (only used for optical flow)
    prev_lm:       np.ndarray
    flow:          np.ndarray
    quality:       float
    frame_idx:     int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FaceParamPipeline:

    def __init__(
            self,
            model:         MorphableModel,
            K:             np.ndarray,
            uv:            np.ndarray,
            mouth_uv_mask: np.ndarray,
    ):
        self.model         = model
        self.K             = K
        self.uv            = uv
        self.mouth_uv_mask = mouth_uv_mask

    # ------------------------------------------------------------------
    # Phase 1 — Session Init  (unchanged from the original project)
    # ------------------------------------------------------------------

    def init_session(
            self,
            I_src:            np.ndarray,
            src_landmarks:    np.ndarray,
            src_lm_vidx:      np.ndarray,
            src_lm_left_idx:  np.ndarray,
            src_lm_right_idx: np.ndarray,
            tgt_frames:       list,
            open_mouth_frame_list: list = None,
            src_init_state    = None,
            K_tgt:            np.ndarray = None,
            src_face_box      = None,
            skip_refit:       bool = False,
            strict_validation: bool = True,
            lm_sigma_px:      float = None,
    ) -> SessionData:
        """
        Run all 4 Phase-1 stages and return frozen SessionData.

        FIX: Stage order is now S2 → S1 → S3 → S4 per PDF maths.
        Stage 1 needs fitted_lm and omega from Stage 2.
        """
        H, W = I_src.shape[:2]
        K_tgt_eff = K_tgt if K_tgt is not None else self.K

        # Face box for resolution score
        if src_face_box is not None:
            W_face_eff = float(src_face_box.w)
            H_face_eff = float(src_face_box.h)
        else:
            W_face_eff = float(W)
            H_face_eff = float(H)

        # ── STAGE 2 FIRST — GN fit on source ──────────────────────────────
        # Must run before Stage 1 to get real omega and fitted landmarks
        if skip_refit and src_init_state is not None:
            src_state: SourceState = src_init_state
            print("[Phase1/S2] Skipping re-fit — using converged state from FaceRecogniser")
        else:
            src_state: SourceState = fit_source(
                model      = self.model,
                K          = self.K,
                landmarks  = src_landmarks,
                lm_vidx    = src_lm_vidx,
                I_src      = I_src,
                init_state = src_init_state,
            )

        print(f"[Phase1/S2] Source fit complete. omega={src_state.omega}, t={src_state.t}")
        print(f"[Phase1/S2] psi_eye={src_state.psi_eye}")
        print(f"[Phase1/S2] gamma_brow={src_state.gamma_brow}")
        print(f"[Phase1/S2] delta_nose norm={np.linalg.norm(src_state.delta_nose):.4f}")
        print(f"[Phase1/S2] gamma_ear={src_state.gamma_ear}")
        print(f"[Phase1/S2] alpha norm={np.linalg.norm(src_state.alpha):.4f}  "
              f"beta norm={np.linalg.norm(src_state.beta):.4f}")
        #print(f"[Phase1/S2] mouth_score={np.linalg.norm(src_state.beta[:4]):.4f}")
        mouth_score = compute_mouth_score(src_state.beta)
        print(f"[Phase1/S2] mouth_score={mouth_score:.4f}")
        
        # if mouth_score > THETA_OPEN:
        #     raise ValueError(
        #         f"Source photo not neutral enough (mouth_score={mouth_score:.3f} "
        #         f"> {THETA_OPEN}). Please provide a more neutral-expression source photo."
        #     )

        # ── STAGE 1 SECOND — validate with real Stage 2 outputs ───────────
        # FIX: project fitted landmarks from Stage 2 state
        S_fitted = self.model.full_shape(
                    src_state.alpha, src_state.beta,
                    src_state.psi_eye, src_state.gamma_brow,
                    src_state.delta_nose,   # FIX: was np.zeros(3 * len(self.model.nose_idx))
                    psi_mouth=src_state.psi_mouth, gamma_ear=src_state.gamma_ear
                )
        fitted_lm_all, _ = project_mesh(
            S_fitted, src_state.omega, src_state.t, self.K
        )
        fitted_lm = fitted_lm_all[src_lm_vidx]

        # FIX: synthetic sharp reference — NOT source image
        I_ref_gray = _sharp_ref(H, W)

        val_result: ValidationResult = validate_source(
            detected_lm  = src_landmarks,
            fitted_lm    = fitted_lm,           # FIX: real projected
            left_lm_idx  = src_lm_left_idx,
            right_lm_idx = src_lm_right_idx,
            W_face       = W_face_eff,
            H_face       = H_face_eff,
            I_src_gray   = _rgb_to_gray(I_src),
            I_ref_gray   = I_ref_gray,          # FIX: synthetic sharp
            omega_src    = src_state.omega,     # FIX: real omega from S2
            sigma_lm     = lm_sigma_px,
        )
        print(f"[Phase1/S1] {val_result}")

        # All 5 scores must exceed 0.75 per PDF
        if not val_result.accepted:
            if strict_validation:
                raise SourceRejected(f"Source rejected by validation: {val_result}", val_result)
            print(f"[Phase1/S1] WARNING — rejected but continuing (strict_validation=False)")

        # ── STAGE 3 — Rasterisation ────────────────────────────────────────
        T_neutral = subpass_A_neutral(
            model          = self.model,
            state          = src_state,
            uv             = self.uv,
            K              = self.K,
            I_src          = I_src,
            img_h          = H,
            img_w          = W,
            mouth_uv_mask  = self.mouth_uv_mask,
        )
        print(f"[Phase1/S3A] T_neutral min={T_neutral.min():.3f} max={T_neutral.max():.3f} "
              f"nonzero_frac={np.count_nonzero(T_neutral.sum(-1))/T_neutral[:,:,0].size:.3f}")

        T_mouth, w_mouth_max = _run_subpass_b(
            self.model, src_state, self.K, self.uv,
            self.mouth_uv_mask, open_mouth_frame_list or []
        )
        print(f"[Phase1/S3B] T_mouth={'None (using fallback)' if T_mouth is None else 'extracted'}"
              f"  w_mouth_max={w_mouth_max}")

        T_synth = None
        if T_mouth is None and self.mouth_uv_mask.any():
            ys, xs = np.where(self.mouth_uv_mask)
            u_lip  = int(xs.mean()) if len(xs) else W // 2
            T_synth = synthesis_fallback(T_neutral, u_lip, self.mouth_uv_mask)

        T_uv_src = merge_texture(T_neutral, T_mouth, self.mouth_uv_mask, T_synth)

        # Debug eye UV region
        eye_uv       = self.uv[self.model.eye_idx]
        u_min, v_min = eye_uv.min(axis=0)
        u_max, v_max = eye_uv.max(axis=0)
        c1 = int(np.clip(u_min * (T_uv_src.shape[1]-1), 0, T_uv_src.shape[1]-2))
        c2 = int(np.clip(u_max * (T_uv_src.shape[1]-1), 1, T_uv_src.shape[1]-1))
        r1 = int(np.clip(v_min * (T_uv_src.shape[0]-1), 0, T_uv_src.shape[0]-2))
        r2 = int(np.clip(v_max * (T_uv_src.shape[0]-1), 1, T_uv_src.shape[0]-1))
        patch = T_uv_src[r1:r2+1, c1:c2+1]
        print(f"[Phase1/S3] Texture baked. w_mouth_max={w_mouth_max}")
        print(f"[Phase1/S3] T_uv_src shape={T_uv_src.shape} "
              f"min={T_uv_src.min():.3f} max={T_uv_src.max():.3f}")

        # Debug UV band coverage
        for v_band in [0.15, 0.33, 0.50, 0.67]:
            row = int(v_band * T_uv_src.shape[0])
            band = T_uv_src[max(0, row-2):row+3]
            zf   = float((band.sum(-1) == 0).mean())

        # ── STAGE 4 — Identity / Neutral Fit ───────────────────────────────────
        tgt_result: TargetNeutralResult = fit_target_neutral(
            model  = self.model,
            K      = K_tgt_eff,
            frames = tgt_frames,
        )
        print(f"[Phase1/S4] Identity fit complete.")
        print(f"[Phase1/S4] alpha norm={np.linalg.norm(tgt_result.alpha_tgt):.4f}  "
              f"beta_neutral norm={np.linalg.norm(tgt_result.beta_neutral):.4f}")

        return SessionData(
            validation        = val_result,
            source_state      = src_state,
            T_uv_src          = T_uv_src,
            alpha_tgt         = tgt_result.alpha_tgt,
            beta_tgt_neutral  = tgt_result.beta_neutral,
            mouth_uv_mask     = self.mouth_uv_mask,
            uv                = self.uv,
            w_mouth_max       = w_mouth_max,
        )

    # ------------------------------------------------------------------
    # Helper — source parameters as a FaceParams object
    # ------------------------------------------------------------------

    def source_params(self, session: SessionData) -> FaceParams:
        s = session.source_state
        return FaceParams(
            alpha=s.alpha, beta=s.beta, omega=s.omega, t=s.t, psi_eye=s.psi_eye,
            gamma_brow=s.gamma_brow, delta_nose=s.delta_nose,
            psi_mouth=getattr(s, "psi_mouth", np.zeros(4)),
            gamma_ear=getattr(s, "gamma_ear", np.zeros(4)),
            K=self.K, model=self.model)

    # ------------------------------------------------------------------
    # Phase 2+3 — Per-frame tracking
    # ------------------------------------------------------------------

    def init_frame_state(
            self,
            first_frame: FrameData,
            session:     SessionData,
    ) -> FrameState:
        return FrameState(
            newton = NewtonState(
                q     = np.zeros(3),
                q_dot = np.zeros(3),
                t     = session.source_state.t.copy(),
                t_dot = np.zeros(3),
            ),
            prev_tgt = None,
            prev_I   = None,
            prev_lm  = first_frame.landmarks.copy(),
            flow     = np.zeros_like(first_frame.landmarks),
            quality  = 1.0,
            frame_idx = 0,
        )

    def process_frame(
            self,
            session: SessionData,
            frame:   FrameData,
            state:   FrameState,
            use_flow: bool = False,
            ear_from_source: bool = True,
            debug_overlay_dir: Optional[str] = None,
    ) -> tuple:
        """
        One frame of tracking.

        use_flow         : run GN optical flow between consecutive RAW frames
                           (slow, pure-Python) and feed it to the Newtonian prior.
        ear_from_source  : freeze γ_ear to the source fit's value (ears are rigid;
                           original design decision). False → fit freely per frame.
        debug_overlay_dir: if set, writes detected-vs-projected landmark overlays there.
        """
        H, W  = frame.I_tgt.shape[:2]
        model = self.model
        K = make_K(max(H, W)*1.2, max(H, W)*1.2, W/2.0, H/2.0)
        print(f"\n[P2/S1] ── Frame {state.frame_idx} ──────────────────────────")

        newton_pred, lm_init = newtonian_warmstart(
            prev    = state.newton,
            tau     = np.zeros(3),
            f_flow  = state.flow,
            lm_prev = state.prev_lm,
            quality = state.quality,
        )
        print(f"[P2/S1] newton_pred.q={newton_pred.q}  "
              f"lm_init range x:{lm_init[:,0].min():.1f}~{lm_init[:,0].max():.1f}  "
              f"y:{lm_init[:,1].min():.1f}~{lm_init[:,1].max():.1f}")

        _rec_tgts = getattr(session, '_recognised_tgts', None)
        _t_init = None
        if _rec_tgts:
            _r = _rec_tgts[min(state.frame_idx, len(_rec_tgts)-1)]
            _t_init = _r.t_init if hasattr(_r, 't_init') else None

        tgt_state, delta_beta = fit_target_frame(
            model         = model,
            alpha_tgt     = session.alpha_tgt,
            beta_neutral  = session.beta_tgt_neutral,
            K             = K,
            I_tgt         = frame.I_tgt,
            landmarks     = frame.landmarks,
            lm_vidx       = frame.lm_vidx,
            q_pred        = newton_pred.q,
            sigma_q       = 0.1 if state.frame_idx > 0 else 10.0,
            t_pred        = newton_pred.t,
            sigma_t       = 10.0,
            prev_state    = state.prev_tgt,
            frame_index   = state.frame_idx,
            t_init        = _t_init,
            gamma_ear_fixed = session.source_state.gamma_ear if ear_from_source else None,
        )
        if debug_overlay_dir:
            os.makedirs(debug_overlay_dir, exist_ok=True)
            draw_landmark_overlay(
                I_tgt=frame.I_tgt, landmarks=frame.landmarks, model=model,
                tgt_state=tgt_state, alpha_tgt=session.alpha_tgt, K=K,
                out_path=os.path.join(debug_overlay_dir,
                                      f"debug_landmarks_frame{state.frame_idx}.png"),
            )
        print(f"[P2/S2] omega={tgt_state.omega}  t={tgt_state.t}")
        print(f"[P2/S2] delta_beta norm={np.linalg.norm(delta_beta):.4f}  "
              f"delta_beta[:4]={delta_beta[:4]}")
        print(f"[P2/S2] psi_eye={tgt_state.psi_eye}")

        # ---- tracked face mesh (target identity + tracked expression) ----
        S_tgt = model.full_shape(
            session.alpha_tgt, tgt_state.beta, tgt_state.psi_eye, tgt_state.gamma_brow,
            np.zeros(3 * len(model.nose_idx)),
            psi_mouth=tgt_state.psi_mouth, gamma_ear=tgt_state.gamma_ear)
        p_all, _ = project_mesh(S_tgt, tgt_state.omega, tgt_state.t, K)

        # ---- alignment diagnostics (per region) ----
        proj_lm  = p_all[frame.lm_vidx]
        valid_lm = ((proj_lm[:, 0] >= 0) & (proj_lm[:, 0] < W) &
                    (proj_lm[:, 1] >= 0) & (proj_lm[:, 1] < H) &
                    np.isfinite(proj_lm).all(axis=1))
        err_all = np.linalg.norm(proj_lm - frame.landmarks, axis=1)
        region_error = {"all": float(err_all.mean())}
        lm_list = frame.lm_vidx.tolist()
        for name, ridx in [("eye", model.eye_idx), ("brow", model.brow_idx),
                           ("nose", model.nose_idx),
                           ("mouth", getattr(model, "mouth_idx", np.array([], dtype=np.intp))),
                           ("ear",   getattr(model, "ear_idx",   np.array([], dtype=np.intp)))]:
            rows = [lm_list.index(vi) for vi in ridx if vi in lm_list]
            if not rows:
                continue
            rows = np.array(rows, dtype=np.intp)
            region_error[name] = float(err_all[rows].mean())

        # ---- quality gate ----
        if valid_lm.sum() < 3:
            c_t, decision = 0.50, "CONSERVATIVE"
        else:
            f_prev = state.flow[valid_lm] if state.flow is not None \
                     else np.zeros((valid_lm.sum(), 2))
            c_t, decision = quality_gate(
                landmarks = frame.landmarks[valid_lm],
                projected = proj_lm[valid_lm],
                omega     = tgt_state.omega,
                f_t       = f_prev,
                f_pred    = f_prev,
            )
        print(f"[P2/S3] quality={c_t:.3f}  decision={decision}")

        # ---- Phase 3: optical flow (optional) + Newton state update ----
        if use_flow and state.prev_I is not None:
            flow_new = gn_optical_flow(I_curr=frame.I_tgt, I_prev=state.prev_I,
                                       points=frame.landmarks)
            print(f"[P3/S1] flow mean magnitude={np.linalg.norm(flow_new, axis=1).mean():.3f}px")
        else:
            flow_new = np.zeros_like(frame.landmarks, dtype=np.float64)
            print("[P3/S1] flow disabled / first frame — flow initialised to zero")

        if decision == "FALLBACK":
            new_newton = state.newton
            print("[P3/S2] FALLBACK — motion state kept from previous frame")
        else:
            q_dot_new, t_dot_new = velocity_update(
                tgt_state.omega, state.newton.q, tgt_state.t, state.newton.t)
            new_newton = NewtonState(q=tgt_state.omega.copy(), q_dot=q_dot_new,
                                     t=tgt_state.t.copy(),     t_dot=t_dot_new)
            print(f"[P3/S2] motion state updated  q_dot={np.round(q_dot_new, 4)}  "
                  f"t_dot={np.round(t_dot_new, 3)}")

        new_state = FrameState(
            newton=new_newton, prev_tgt=tgt_state, prev_I=frame.I_tgt if use_flow else None,
            prev_lm=frame.landmarks.copy(), flow=flow_new, quality=c_t,
            frame_idx=state.frame_idx + 1)

        params = FaceParams(
            alpha=np.asarray(session.alpha_tgt), beta=tgt_state.beta, omega=tgt_state.omega,
            t=tgt_state.t, psi_eye=tgt_state.psi_eye, gamma_brow=tgt_state.gamma_brow,
            delta_nose=np.zeros(3 * len(model.nose_idx)), psi_mouth=tgt_state.psi_mouth,
            gamma_ear=tgt_state.gamma_ear, K=K, model=model, quality=c_t, decision=decision)

        return FrameResult(params=params, tgt_state=tgt_state, delta_beta=delta_beta,
                           quality=c_t, decision=decision, proj_landmarks=proj_lm,
                           region_error=region_error), new_state


# ---------------------------------------------------------------------------
# Module-level helpers (used by init_session)
# ---------------------------------------------------------------------------

def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]


def _sharp_ref(H: int, W: int) -> np.ndarray:
    """
    Synthetic sharp reference for s_blur denominator.
    FIX: was returning grayscale of source image → s_blur=1.0 always.
    Now returns synthetic high-frequency pattern so s_blur measures
    source sharpness relative to a genuinely sharp reference.
    """
    x  = np.linspace(0, 4*np.pi, W)
    y  = np.linspace(0, 4*np.pi, H)
    XX, YY = np.meshgrid(x, y)
    return (np.sin(10*XX) * np.cos(10*YY) * 0.5 + 0.5)


def _run_subpass_b(model, src_state, K, uv, mouth_uv_mask, open_mouth_frames):
    """Run subpass B — returns (T_mouth, w_mouth_max)."""
    if not open_mouth_frames:
        return None, 0.3
    T_mouth = subpass_B_mouth(
        model             = model,
        alpha_src         = src_state.alpha,
        uv                = uv,
        K                 = K,
        open_mouth_frames = open_mouth_frames,
        mouth_uv_mask     = mouth_uv_mask,
        img_h             = mouth_uv_mask.shape[0],
        img_w             = mouth_uv_mask.shape[1],
    )
    w_mouth_max = 1.0 if T_mouth is not None else 0.3
    return T_mouth, w_mouth_max
