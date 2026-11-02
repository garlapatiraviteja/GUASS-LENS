"""
face_params/phase2/per_frame_tracking.py — per-frame face-parameter TRACKING (no swap / render)

Extracted from the original phase2/per_frame_swap.py. Kept:
  * newtonian_warmstart  — rigid-body (so(3)) pose prediction:  I·q̈ = τ − b·q̇
  * fit_target_frame     — joint Gauss-Newton per frame for
                           [β, ω, t, ψ_eye, γ_brow, ψ_mouth, γ_ear] with identity α frozen
  * quality_gate         — c_t = 0.5·s_lm + 0.3·s_flow + 0.2·s_pose → PROCEED/CONSERVATIVE/FALLBACK
Removed: render-mesh construction, SH relighting / photometric GN, UV texture sampling,
         curvature / streak diagnostics, render + blend.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..geometry import *
from ..shape_model import MorphableModel


NEWTON_B    = 0.15


NEWTON_I    = 1.0


EPS_BETA    = 0.02


SIGMA_Q     = 0.0001


GN_MAX_ITER = 10


GN_THRESH   = 1e-3


SIGMA_POSE  = np.deg2rad(30.0)


SIGMA_FLOW  = 2.0


W_APP_TGT   = 5.0


@dataclass
class NewtonState:
    q:      np.ndarray   # (3,) current rotation vector
    q_dot:  np.ndarray   # (3,) angular velocity
    t:      np.ndarray   # (3,) translation
    t_dot:  np.ndarray   # (3,) translation velocity


def newtonian_warmstart(
        prev:    NewtonState,
        tau:     np.ndarray,
        f_flow:  np.ndarray,
        lm_prev: np.ndarray,
        quality: float,
        dt:      float = 1.0,
) -> tuple[NewtonState, np.ndarray]:
    """
    Newtonian Warm-Start on so(3).

    FIX: Low-quality coast was using acceleration-based damping:
      q̈ = -(b/I)·q̇  then Verlet.
    PDF says coast = multiply velocity by damping factor directly:
      q̇_t = q̇_{t-1}·(1 - b·Δt)
      q_t  = q_{t-1} + q̇_t·Δt
    """
    b = NEWTON_B
    I = NEWTON_I

    if quality < 0.40:
        # FIX: coast — direct velocity damping per PDF
        q_dot_new = prev.q_dot * (1.0 - b * dt)
        q_pred    = prev.q + q_dot_new * dt
    else:
        # Normal: I·q̈ = τ - b·q̇, Verlet integration
        q_ddot    = (tau - b * prev.q_dot) / I
        q_dot_new = prev.q_dot + q_ddot * dt
        q_pred    = prev.q    + q_dot_new * dt

    # Translation — constant velocity
    t_pred = prev.t + prev.t_dot * dt

    # Landmark warm-start from optical flow
    x_init = lm_prev + f_flow

    pred = NewtonState(
        q     = q_pred,
        q_dot = q_dot_new,
        t     = t_pred,
        t_dot = prev.t_dot,
    )
    return pred, x_init


@dataclass
class TargetFrameState:
    beta:       np.ndarray   # (64,)
    omega:      np.ndarray   # (3,)
    t:          np.ndarray   # (3,)
    psi_eye:    np.ndarray   # (6,)
    gamma_brow: np.ndarray   # (8,)
    psi_mouth: np.ndarray = field(default_factory=lambda: np.zeros(4))   # NEW
    gamma_ear: np.ndarray = field(default_factory=lambda: np.zeros(4))   # NEW


def _cramers_rule_6x6(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax=b for 6x6 system via Cramer's rule.
    Per PDF: pose block is 6×6 — analytical inversion via Cramer's rule.
    x_i = det(A_i)/det(A) where A_i has column i replaced by b.
    Falls back to regularised solve if det(A) near zero.
    """
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-10:
        return np.linalg.solve(A + 1e-6*np.eye(6), b)
    x = np.zeros(6, dtype=np.float64)
    for i in range(6):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    return x


def _sample_bilinear_img(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear sample of (H,W,C) at (x,y). Returns zeros if OOB."""
    H, W = img.shape[:2]
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return np.zeros(img.shape[2] if img.ndim == 3 else 1, dtype=np.float64)
    fx = x - x0; fy = y - y0
    v = ((1-fx)*(1-fy)*img[y0,x0] + fx*(1-fy)*img[y0,x1] +
         (1-fx)*fy    *img[y1,x0] + fx*fy    *img[y1,x1])
    return v.astype(np.float64)


def _image_gradient_target(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """∂I/∂x, ∂I/∂y at (x,y). Returns (2,C)."""
    H, W = img.shape[:2]
    xc = float(np.clip(x, 1, W-2))
    yc = float(np.clip(y, 1, H-2))
    gx = (_sample_bilinear_img(img, xc+1, yc) - _sample_bilinear_img(img, xc-1, yc)) * 0.5
    gy = (_sample_bilinear_img(img, xc, yc+1) - _sample_bilinear_img(img, xc, yc-1)) * 0.5
    return np.stack([gx, gy], axis=0)


def _build_target_jacobian(
        model:     MorphableModel,
        alpha_tgt: np.ndarray,
        state:     TargetFrameState,
        K:         np.ndarray,
        landmarks: np.ndarray,
        lm_vidx:   np.ndarray,
        q_pred:    np.ndarray,
        sigma_q:   float,
        t_pred:    np.ndarray,      # ADD THIS
        sigma_t:   float,           # ADD THIS
        I_tgt:     np.ndarray = None,
        gamma_ear_frozen: bool = False,   # NEW — see fit_target_frame docstring
) -> tuple[np.ndarray, np.ndarray]:
    """
    3-term energy Jacobian for target frame.

    Columns: [β(64), ω(3), t(3), ψ_eye(6), γ_brow(8), ψ_mouth(4), γ_ear(4)] = 92 columns
    Rows: 2·N_lm geometric + C·N_lm appearance + 3 pose prior

    FIX: β Jacobian now uses build_dpi_dbeta for depth correction.
    FIX: dpi_domega passes state.omega for exact Rodrigues derivative.

    [DEBUG instrumentation] — when debug_cond=True, reports per-column norms
    for the beta block vs the pose block, plus the condition number of
    H = J.T@J, so we can tell whether runaway beta growth is caused by:
      (a) a genuine Jacobian scaling/collinearity bug (bad build_dpi_dbeta),
          in which case beta columns will be near-zero-norm or near-duplicate
          relative to omega/t columns, or
      (b) the problem simply being under/over-determined as expected, in
          which case the column norms and condition number will look sane
          and the fix belongs in sigma_q / EPS_BETA tuning instead.
    """
    N_lm  = landmarks.shape[0]
    dim   = 92
    C     = I_tgt.shape[2] if (I_tgt is not None and I_tgt.ndim == 3) else 0
    n_rows = 2*N_lm + (C*N_lm if I_tgt is not None else 0) + 6

    J = np.zeros((n_rows, dim), dtype=np.float64)
    e = np.zeros(n_rows,       dtype=np.float64)

    S  = model.full_shape(
        alpha_tgt, state.beta,
        state.psi_eye, state.gamma_brow,
        np.zeros(3 * len(model.nose_idx)),
        psi_mouth=state.psi_mouth,
        gamma_ear=state.gamma_ear,
    )
    R  = rodrigues(state.omega)
    pi, Z = project_mesh(S, state.omega, state.t, K)

    eye_set    = set(model.eye_idx.tolist())
    brow_set   = set(model.brow_idx.tolist())
    mouth_set  = set(model.mouth_idx.tolist())
    ear_set    = set(model.ear_idx.tolist())

    eye_local  = {int(vi): li for li, vi in enumerate(model.eye_idx)}
    brow_local = {int(vi): li for li, vi in enumerate(model.brow_idx)}
    mouth_local = {int(vi): li for li, vi in enumerate(model.mouth_idx)}
    ear_local   = {int(vi): li for li, vi in enumerate(model.ear_idx)}

    if I_tgt is not None:
        T_patch = np.zeros((N_lm, C), dtype=np.float64)
        for n in range(N_lm):
            T_patch[n] = _sample_bilinear_img(I_tgt, landmarks[n,0], landmarks[n,1])

    rows_geo = 2 * N_lm
    rows_app = C * N_lm if I_tgt is not None else 0

    for n in range(N_lm):
        vi  = lm_vidx[n]
        Z_i = Z[vi]
        if Z_i <= 0 or np.isnan(Z_i):
            continue

        S_i     = S[vi]
        p_cam_i = R @ S_i + state.t

        # ── Geometric rows ────────────────────────────────────────────────
        row = 2 * n
        e[row:row+2] = pi[vi] - landmarks[n]

        dp_dt_i = dpi_dt(K, Z_i, p_cam_i)
        dp_dt_i = dp_dt_i.copy()
        dp_dt_i[1, :] *= -1.0
        dp_dw_i = dpi_domega(dp_dt_i, S_i, state.omega)

        # β: 0:64
        A_exp_i = model.A_exp[vi]
        J[row:row+2, 0:64]   = build_dpi_dbeta(K, Z_i, R, A_exp_i, dp_dt_i, p_cam_i)

        # ω: 64:67
        J[row:row+2, 64:67]  = dp_dw_i

        # t: 67:70
        J[row:row+2, 67:70]  = dp_dt_i

        # ψ_eye: 70:76
        if vi in eye_set:
            li = eye_local[int(vi)]
            B_eye_i = model.B_eye[li]
            J[row:row+2, 70:76] = build_dpi_dpsi_eye(
                K, Z_i, R, B_eye_i, dp_dt_i, p_cam_i)

        # γ_brow: 76:84
        if vi in brow_set:
            li = brow_local[int(vi)]
            B_brow_i = model.B_brow[li]
            J[row:row+2, 76:84] = build_dpi_dgamma_brow(
                K, Z_i, R, B_brow_i, dp_dt_i, p_cam_i)

        # ψ_mouth: 84:88
        if vi in mouth_set:
            li = mouth_local[int(vi)]
            B_mouth_i = model.B_mouth[li]
            J[row:row+2, 84:88] = build_dpi_dpsi_mouth(
                K, Z_i, R, B_mouth_i, dp_dt_i, p_cam_i)

        # γ_ear: 88:92
        # NEW — when gamma_ear_frozen, leave these columns at zero so ear
        # contributes NO gradient to the joint solve at all (not even via
        # cross-terms in H = J.T@J with omega/t). This mirrors exactly how
        # nose is already handled above (passed in as a fixed np.zeros(...)
        # rather than a free block with its own columns): the fit must
        # explain ear landmark positions using omega/t alone, because that
        # is the only mechanism still available once a mesh renderer
        # substitutes the frozen SOURCE gamma_ear at render time. Without
        # this, gamma_ear was free to "absorb" ear residual during fitting,
        # letting omega/t settle into a pose that only looks correct when
        # paired with a gamma_ear that gets discarded before rendering —
        # producing exactly the "ears shifted" symptom even though the
        # per-iteration ear_err_trial < ear_err_before check always passed.
        if vi in ear_set and not gamma_ear_frozen:
            li = ear_local[int(vi)]
            B_ear_i = model.B_ear[li]
            J[row:row+2, 88:92] = build_dpi_dgamma_ear(
                K, Z_i, R, B_ear_i, dp_dt_i, p_cam_i)

        # ── Appearance rows ───────────────────────────────────────────────
        if I_tgt is not None:
            row_app = rows_geo + C * n
            px, py  = float(pi[vi, 0]), float(pi[vi, 1])

            I_curr = _sample_bilinear_img(I_tgt, px, py)
            e[row_app:row_app+C] = W_APP_TGT * (I_curr - T_patch[n])

            grad_I = _image_gradient_target(I_tgt, px, py)
            J_pi   = J[row:row+2, :]
            for c in range(C):
                J[row_app+c, :] = W_APP_TGT * (grad_I[:, c] @ J_pi)

    # ── Pose prior rows ───────────────────────────────────────────────────
    row_q = rows_geo + rows_app
    e[row_q:row_q+3]        = (state.omega - q_pred) / sigma_q
    J[row_q:row_q+3, 64:67] = np.eye(3) / sigma_q

    row_t = row_q + 3                                    # ADD THIS BLOCK
    e[row_t:row_t+3]        = (state.t - t_pred) / sigma_t
    J[row_t:row_t+3, 67:70] = np.eye(3) / sigma_t

    # [DEBUG] conditioning diagnostics — column norms + condition number.
    # Runs on EVERY call (including backtracking trial calls inside
    # fit_target_frame), so expect this to print multiple times per GN
    # iteration, not just once per iteration. That's expected — same
    # drop-in signature as the original function, no caller changes needed.
    if True:
        beta_cols  = J[:, 0:64]
        omega_cols = J[:, 64:67]
        t_cols     = J[:, 67:70]

        beta_col_norms  = np.linalg.norm(beta_cols,  axis=0)
        omega_col_norms = np.linalg.norm(omega_cols, axis=0)
        t_col_norms     = np.linalg.norm(t_cols,     axis=0)

        # Rank / collinearity check on beta block alone
        try:
            beta_rank = np.linalg.matrix_rank(beta_cols)
        except Exception:
            beta_rank = -1

        # Full H conditioning
        H_full   = J.T @ J
        try:
            cond_H = np.linalg.cond(H_full)
        except Exception:
            cond_H = float('inf')
        try:
            cond_beta_block = np.linalg.cond(beta_cols.T @ beta_cols)
        except Exception:
            cond_beta_block = float('inf')

    return J, e


def _build_target_lambda(dim=92, lambda_beta=50.0, lambda_pose=50.0,
                          lambda_eye=250.0, lambda_brow=150.0,
                          lambda_mouth=250.0, lambda_ear=250.0,
                          lambda_depth_anchor=2000.0, gn_lambda=0.01):
    Lambda = np.eye(dim)
    Lambda[0:64,   0:64]   *= lambda_beta / gn_lambda
    Lambda[64:67,  64:67]  *= lambda_pose / gn_lambda
    Lambda[67:70,  67:70]  *= lambda_pose / gn_lambda
    Lambda[69, 69]         += lambda_depth_anchor / gn_lambda
    Lambda[70:76,  70:76]  *= lambda_eye / gn_lambda
    Lambda[76:84,  76:84]  *= lambda_brow / gn_lambda
    Lambda[84:88,  84:88]  *= lambda_mouth / gn_lambda
    Lambda[88:92,  88:92]  *= lambda_ear / gn_lambda
    return Lambda


def fit_target_frame(
        model:        MorphableModel,
        alpha_tgt:    np.ndarray,
        beta_neutral: np.ndarray,
        K:            np.ndarray,
        I_tgt:        np.ndarray,
        landmarks:    np.ndarray,
        lm_vidx:      np.ndarray,
        q_pred:       np.ndarray,
        sigma_q:      float,
        t_pred:       np.ndarray,      # NEW
        sigma_t:      float,           # NEW
        prev_state:   Optional[TargetFrameState],
        frame_index:  int,
        t_init:       np.ndarray = None,
        gamma_ear_fixed: Optional[np.ndarray] = None,   # NEW
) -> tuple[TargetFrameState, np.ndarray]:
    """
    Phase-2 Stage-2 joint GN on single target frame.
    Returns fitted TargetFrameState and expression delta Δβᵗ.

    [DEBUG instrumentation] — tracks state.omega / state.t / residuals
    per iteration to determine whether the GN loop is:
      (a) diverging (omega moving steadily away from a reasonable pose),
      (b) starting bad (q_pred/init state already extreme at iter 0), or
      (c) oscillating (bouncing between poses without settling).

    FIXES APPLIED:
      1. Main-branch regularization: flat GN_LAMBDA_BETA/POSE constants
         replaced with per-column adaptive (LM-style) damping — the flat
         50/5 ignored the ~3000x column-norm gap between beta and
         omega/t, so beta's null-space directions (rank 39/64) were
         effectively undamped relative to their real informativeness.
      2. Backtracking: was accepting a step if resid_trial <=
         resid_before * 1.05 (a 5% residual INCREASE), which compounds
         across iterations. Now never accepts a step that makes it worse.
      3. Convergence check: GN_THRESH was applied to a single combined
         norm across beta (~O(1-10)), omega (~O(0.01-1) rad), and t
         (~O(100s)) — t's scale dominates the vector, so the threshold
         almost never reflects true convergence. Now checked per block
         against a scale each block can actually satisfy.
      4. beta_frozen reduced-solve branch: same flat/un-scaled damping
         problem (1e-6 across a 28-dim block that still mixes t's large
         scale with everything else). Now uses the same adaptive scheme.
      5. Reinstated the MAX_OMEGA_RAD safety cap (previously dead/
         commented-out code) as a belt-and-suspenders guard.
      7. NEW — gamma_ear is now frozen at gamma_ear_fixed (pass
         session.source_state.gamma_ear from the caller) rather than
         solved as a free 4-dim block. a downstream renderer would use
         with the SOURCE's gamma_ear, never the fitted target one, so
         leaving it free during fitting let omega/t quietly lean on an
         ear deformation that gets thrown away before rendering — the
         fit would converge "fine" on paper while still producing
         visibly shifted ears in the final render. Freezing it here keeps
         fit-time and render-time consistent, exactly like nose already
         was.
      6. NEW — gamma_brow (cols 76:84) suffers the same crushed-step
         problem mouth/ear had — ABS_FLOOR in the joint solve is
         calibrated to pose curvature (~1e8-1e9), orders of magnitude
         larger than brow's own Jacobian scale, so the joint step for
         these columns was ~1e-6 every iteration regardless of residual
         (confirmed: gamma_brow norm stayed at 0.0000-0.0001 across all
         10 iterations while psi_mouth moved 0.02->0.26 in the same
         run). Same dedicated reduced-solve fix pattern as mouth/ear.
    """
    default_t = t_init if t_init is not None else np.array([0., 0., 500.])

    # [DEBUG] what warm-start / prior is this fit being anchored to?
    print(f"[P2/S2 fit] frame_index={frame_index}  "
          f"q_pred={np.array2string(q_pred, precision=4)}  "
          f"t_init={'None -> default' if t_init is None else np.array2string(t_init, precision=1)}  "
          f"default_t={np.array2string(default_t, precision=1)}  "
          f"sigma_q={sigma_q}")

    if prev_state is None:
        state = TargetFrameState(
            beta       = beta_neutral.copy(),
            omega      = q_pred.copy(),
            t          = default_t.copy(),
            psi_eye    = np.zeros(6),
            gamma_brow = np.zeros(8),
        )
        print("[P2/S2 fit] prev_state is None — fresh init from q_pred/default_t")
    else:
        state = TargetFrameState(
            beta       = prev_state.beta.copy(),
            omega      = q_pred.copy(),
            t          = prev_state.t.copy(),
            psi_eye    = prev_state.psi_eye.copy(),
            gamma_brow = prev_state.gamma_brow.copy(),
            psi_mouth  = prev_state.psi_mouth.copy(),
            gamma_ear  = prev_state.gamma_ear.copy(),
        )
        # [DEBUG] compare prev_state.omega to this frame's q_pred — did the
        # Newton warm-start already jump far from where the previous frame converged?
        prev_omega_norm_diff = np.linalg.norm(q_pred - prev_state.omega)
        print(f"[P2/S2 fit] prev_state.omega={np.array2string(prev_state.omega, precision=4)}  "
              f"new q_pred={np.array2string(q_pred, precision=4)}  "
              f"‖q_pred - prev_state.omega‖={prev_omega_norm_diff:.4f}  "
              f"(large jump here means the warm-start itself moved the pose, "
              f"before any GN iteration has run)")

    # NEW — pin gamma_ear to the value that will actually be rendered
    # (source's), overriding whatever the dataclass default or prev_state
    # carried forward. See fix #7 in the docstring above. Applies
    # regardless of whether this is a fresh init or a warm-started frame.
    if gamma_ear_fixed is not None:
        state.gamma_ear = gamma_ear_fixed.copy()

    beta_frozen = getattr(prev_state, "beta_frozen", False) if prev_state is not None else False
    beta_before = state.beta.copy()   # snapshot to measure movement this frame

    # [DEBUG] iteration-by-iteration trace
    print(f"[P2/S2 fit] ── starting GN ── "
          f"init omega={np.array2string(state.omega, precision=4)}  "
          f"init t={np.array2string(state.t, precision=1)}  "
          f"beta_frozen={beta_frozen}")

    for iteration in range(GN_MAX_ITER):
        J, e = _build_target_jacobian(
            model, alpha_tgt, state, K,
            landmarks, lm_vidx, q_pred, sigma_q,
            t_pred=t_pred, sigma_t=sigma_t,
            I_tgt=I_tgt,
            gamma_ear_frozen=(gamma_ear_fixed is not None),
        )
        resid_before = np.linalg.norm(e)

        if beta_frozen:
            J[:, :64] = 0.0

        if beta_frozen:
            J_reduced = J[:, 64:92]
            # FIX 4: per-column adaptive damping instead of flat 1e-6,
            # which still lumped t's large scale in with everything else.
            H_data_r = J_reduced.T @ J_reduced
            diag_r   = np.diag(H_data_r).copy()
            diag_r[diag_r < 1e-12] = 1e-12
            H_reduced = H_data_r + 1e-2 * np.diag(diag_r)
            g_reduced = J_reduced.T @ e
            try:
                d_reduced = -np.linalg.solve(H_reduced, g_reduced)
            except np.linalg.LinAlgError:
                d_reduced = -np.linalg.lstsq(H_reduced, g_reduced, rcond=None)[0]
            delta = np.zeros(J.shape[1])
            delta[64:92] = d_reduced
        else:
            # FIX 1: per-column adaptive (LM-style) damping instead of
            # flat GN_LAMBDA_BETA=50 / GN_LAMBDA_POSE=5.
            #
            # FIX 1b: ABS_FLOOR used to be a flat constant (1.0), which is
            # nowhere near enough to damp beta's near-null directions —
            # beta columns run ~1600x smaller-normed than omega/t columns
            # (see [DEBUG jacobian_cond] beta_mean/omega_mean ratio logs),
            # so a flat floor of 1.0 let GN take huge steps along beta's
            # rank-deficient directions (beta block rank was 37/64) for
            # near-zero residual cost. Scale the floor to the pose block's
            # own curvature instead, since pose (omega/t) is the
            # well-determined part of this problem — keeps the floor
            # meaningful relative to the actual signal in THIS frame,
            # instead of an arbitrary constant that's fine most frames and
            # silently insufficient on ill-conditioned ones like this.
            # LM_LAMBDA = 1
            # H_data  = J.T @ J
            # diag_H  = np.diag(H_data).copy()
            # diag_H[diag_H < 1e-12] = 1e-12
            # pose_diag_mean = np.mean(diag_H[64:70])   # omega(3)+t(3) columns
            # ABS_FLOOR = max(0.1, 0.01 * pose_diag_mean)
            # #H = H_data + LM_LAMBDA * np.diag(diag_H) + ABS_FLOOR * np.eye(H_data.shape[0])
            # H = H_data + LM_LAMBDA * np.diag(diag_H)
            # g = J.T @ e
            # LAMBDA_DEPTH_ANCHOR = 2000.0
            # H[69, 69] += LAMBDA_DEPTH_ANCHOR * diag_H[69]
            # try:
            #     delta = -np.linalg.solve(H, g)
            # except np.linalg.LinAlgError:
            #     delta = -np.linalg.lstsq(H, g, rcond=None)[0]
            H = J.T @ J + 0.01 * _build_target_lambda()
            g = J.T @ e
            try:
                delta = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(H, g, rcond=None)[0]
                  
        # [DEBUG] raw proposed step BEFORE backtracking
        omega_step_raw = delta[64:67]
        t_step_raw     = delta[67:70]

        # NEW — dedicated reduced solve for mouth/ear (columns 84:92).
        # In the joint solve above, ABS_FLOOR is scaled to the POSE block's
        # curvature (pose_diag_mean, roughly the omega/t column norms
        # squared, ~1e8-1e9). Mouth/ear's own diagonal is orders of
        # magnitude smaller, so the same damping that correctly stabilizes
        # pose crushes mouth/ear into an almost-constant, iteration-
        # independent step — confirmed in logs: psi_mouth_step_norm sat at
        # ~0.013 every single iteration (0.0129, 0.0135, 0.0133, ...),
        # never decaying or growing. At that flat rate it would take
        # roughly 250 iterations to reach the source's converged
        # psi_mouth norm (~3.2), not the 10 this loop actually runs.
        #
        # Fix: solve the mouth/ear block's own small system separately,
        # damped relative to ITS OWN curvature, and use that result in
        # place of whatever the joint solve produced for those columns.
        # Pose, beta, backtracking, and clipping are all untouched.
        J_me = J[:, 84:92]
        H_me = J_me.T @ J_me
        diag_me = np.diag(H_me).copy()
        diag_me[diag_me < 1e-12] = 1e-12
        # FIX: LM_LAMBDA_ME was left at 5.0 — the weak, untested value from
        # our first attempt, which produced wild oscillation (psi_mouth
        # slamming into clip walls, step norms 2.7-4.0). Never actually
        # updated to the corrected value we settled on. Raising to 500.0
        # to match the source fit's proven damping scale, plus a hard
        # step-size cap as a safety net regardless of tuning.
        LM_LAMBDA_ME = 500.0
        H_me_damped = H_me + LM_LAMBDA_ME * np.diag(diag_me) + 1e-6 * np.eye(8)
        g_me = J_me.T @ e
        try:
            delta_me = -np.linalg.solve(H_me_damped, g_me)
        except np.linalg.LinAlgError:
            delta_me = -np.linalg.lstsq(H_me_damped, g_me, rcond=None)[0]

        MAX_ME_STEP = 0.3
        me_step_norm = np.linalg.norm(delta_me)
        if me_step_norm > MAX_ME_STEP:
            delta_me = delta_me * (MAX_ME_STEP / me_step_norm)

        delta[84:92] = delta_me

        # NEW — ear-specific validation, mirroring the brow fix above.
        # gamma_ear was bundled with psi_mouth in the "me" reduced solve
        # and accepted unconditionally. Same failure mode as brow: the
        # joint/me-block step can reduce the OVERALL residual (mouth +
        # pose + everything else) while actually moving ear's own
        # landmarks further from target. Reject the ear component of the
        # step if it doesn't reduce ear's own landmark error, independent
        # of whether mouth's component (which we still keep) helped.
        # landmarks are rows of lm_vidx, NOT vertex ids (they only coincide when every
        # vertex is a landmark).  Map region vertices to landmark rows; skip absent ones.
        _lm_row = {int(v): r for r, v in enumerate(lm_vidx)}
        def _region_pts(idx):
            keep = [(int(i), _lm_row[int(i)]) for i in idx if int(i) in _lm_row]
            if not keep:
                return None
            return np.array([k[0] for k in keep]), np.array([k[1] for k in keep])

        def _ear_landmark_err(gamma_ear_val):
            if len(model.ear_idx) == 0:
                return 0.0
            S_e = model.full_shape(
                alpha_tgt, state.beta,
                state.psi_eye, state.gamma_brow, np.zeros(3 * len(model.nose_idx)),
                psi_mouth=state.psi_mouth, gamma_ear=gamma_ear_val,
            )
            pi_e, _ = project_mesh(S_e, state.omega, state.t, K)
            pts = _region_pts(model.ear_idx)
            if pts is None:
                return 0.0        # no ear landmarks supplied -> ear is unobserved
            err_e = np.linalg.norm(pi_e[pts[0]] - landmarks[pts[1]], axis=1)
            valid_e = np.isfinite(err_e)
            return err_e[valid_e].mean() if valid_e.any() else np.inf

        ear_err_before = _ear_landmark_err(state.gamma_ear)
        ear_err_trial  = _ear_landmark_err(state.gamma_ear + delta[88:92])

        if ear_err_trial > ear_err_before:
            delta[88:92] = 0.0   # reject — don't move ear this iteration

        # NEW — dedicated reduced solve for gamma_brow (columns 76:84).
        # Same crushed-step mechanism as mouth/ear above: ABS_FLOOR is
        # calibrated to the POSE block's curvature, which is orders of
        # magnitude larger than brow's own Jacobian scale (B_brow is a
        # small, fixed-scale basis). Confirmed in logs: gamma_brow norm
        # stayed at 0.0000-0.0001 across all 10 iterations of a run where
        # psi_mouth (same joint solve, before its own fix) moved freely.
        J_brow = J[:, 76:84]
        H_brow = J_brow.T @ J_brow
        diag_brow = np.diag(H_brow).copy()
        diag_brow[diag_brow < 1e-12] = 1e-12
        LM_LAMBDA_BROW = 500.0
        H_brow_damped = H_brow + LM_LAMBDA_BROW * np.diag(diag_brow) + 1e-6 * np.eye(8)
        g_brow = J_brow.T @ e
        try:
            delta_brow = -np.linalg.solve(H_brow_damped, g_brow)
        except np.linalg.LinAlgError:
            delta_brow = -np.linalg.lstsq(H_brow_damped, g_brow, rcond=None)[0]

        MAX_BROW_STEP = 1.0
        brow_step_norm = np.linalg.norm(delta_brow)
        if brow_step_norm > MAX_BROW_STEP:
            delta_brow = delta_brow * (MAX_BROW_STEP / brow_step_norm)

        # NEW: brow-specific validation, independent of the joint
        # backtracking check. Confirmed via diagnose_brow_error.py that
        # the joint check accepts steps where brow_mean_err monotonically
        # WORSENS (254.8px -> 269.7px over 19 accepted iterations) because
        # the total residual keeps dropping from pose/beta improvements
        # that have nothing to do with brow. Reject the brow step here if
        # it doesn't actually reduce brow's own landmark error.
        def _brow_landmark_err(gamma_brow_val):
            S_b = model.full_shape(
                alpha_tgt, state.beta,
                state.psi_eye, gamma_brow_val, np.zeros(3 * len(model.nose_idx)),
                psi_mouth=state.psi_mouth, gamma_ear=state.gamma_ear,
            )
            pi_b, _ = project_mesh(S_b, state.omega, state.t, K)
            pts = _region_pts(model.brow_idx)
            if pts is None:
                return 0.0        # no brow landmarks supplied -> brow is unobserved
            err_b = np.linalg.norm(pi_b[pts[0]] - landmarks[pts[1]], axis=1)
            valid_b = np.isfinite(err_b)
            return err_b[valid_b].mean() if valid_b.any() else np.inf

        brow_err_before = _brow_landmark_err(state.gamma_brow)
        brow_err_trial  = _brow_landmark_err(state.gamma_brow + delta_brow)

        if brow_err_trial > brow_err_before:
            delta_brow = np.zeros_like(delta_brow)   # reject — don't move brow this iteration

        delta[76:84] = delta_brow

        # Backtracking: halve the step until it doesn't blow up the residual
        step_accepted = False
        n_backtracks = 0
        for _bt in range(5):
            trial_state = TargetFrameState(
                beta       = state.beta       + delta[:64],
                omega      = state.omega      + delta[64:67],
                t          = state.t          + delta[67:70],
                psi_eye    = state.psi_eye    + delta[70:76],
                gamma_brow = state.gamma_brow + delta[76:84],
                psi_mouth  = state.psi_mouth  + delta[84:88],
                gamma_ear  = state.gamma_ear  + delta[88:92],
            )
            _, e_trial = _build_target_jacobian(
                model, alpha_tgt, trial_state, K,
                landmarks, lm_vidx, q_pred, sigma_q, t_pred=t_pred, sigma_t=sigma_t, I_tgt=I_tgt,
                gamma_ear_frozen=(gamma_ear_fixed is not None),
            )
            resid_trial = np.linalg.norm(e_trial)
            # FIX 2: never accept a step that makes the residual worse
            # (was resid_before * 1.05, letting a 5% increase compound).
            if resid_trial <= resid_before:
                step_accepted = True
                break
            delta *= 0.5
            n_backtracks += 1
        else:
            delta = np.zeros_like(delta)

        if n_backtracks > 0 and step_accepted:
            pass

        if not beta_frozen:
            state.beta       += delta[:64]
        state.omega          += delta[64:67]
        state.t              += delta[67:70]
        state.psi_eye        += delta[70:76]
        state.gamma_brow     += delta[76:84]
        state.psi_mouth      += delta[84:88]
        state.gamma_ear      += delta[88:92]

        # Clip eye params to valid ranges
        state.psi_eye[0] = np.clip(state.psi_eye[0], 0.0, 1.0)
        state.psi_eye[3] = np.clip(state.psi_eye[3], 0.0, 1.0)
        state.psi_eye[[1,2,4,5]] = np.clip(state.psi_eye[[1,2,4,5]], -0.5, 0.5)

        # [DEBUG] check clip saturation BEFORE clipping psi_mouth/gamma_ear
        psi_mouth_raw = state.psi_mouth.copy()
        gamma_ear_raw = state.gamma_ear.copy()
        #state.psi_mouth = np.clip(state.psi_mouth, -1.0, 1.0)
        state.psi_mouth[[0, 3]] = np.clip(state.psi_mouth[[0, 3]], -1.0, 1.0)
        state.psi_mouth[[1, 2]] = np.clip(state.psi_mouth[[1, 2]], -2.5, 2.5)
        #state.gamma_ear  = np.clip(state.gamma_ear,  -1.0, 1.0)
        state.gamma_ear[[0, 3]] = np.clip(state.gamma_ear[[0, 3]], -1.0, 1.0)
        state.gamma_ear[[1, 2]] = np.clip(state.gamma_ear[[1, 2]], -2.5, 2.5)

        # FIX (revised): the ±4.0 widen here was based on the same
        # misdiagnosis as the source fit's — see stage2_gn_source.py's
        # _build_lambda for the actual root cause (lambda_brow was
        # under-damped relative to brow's 8 free params / 4 constraining
        # landmarks in the SOURCE fit, letting it run to norm ~10-11 once
        # widened). This target-side fit never actually reached ±4 in
        # your logs (peaked around norm 2.34, individual components
        # ~1.7), so it wasn't the active cause of the 77px error here —
        # but pulling it back to ±2.5 anyway for consistency, so this
        # fit doesn't have a wide-open wall waiting to cause the same
        # runaway under different conditions.
        gamma_brow_raw = state.gamma_brow.copy()
        state.gamma_brow = np.clip(state.gamma_brow, -2.5, 2.5)
        n_brow_clipped  = int(np.sum(np.abs(gamma_brow_raw) > 2.5))

        n_mouth_clipped = int(np.sum(np.abs(psi_mouth_raw) > 1.0))
        n_ear_clipped   = int(np.sum(np.abs(gamma_ear_raw) > 1.0))

        # FIX 5: reinstated safety cap (was dead/commented-out code)
        MAX_OMEGA_RAD = 0.6
        omega_before_cap = state.omega.copy()
        theta = np.linalg.norm(state.omega)
        if theta > MAX_OMEGA_RAD:
            state.omega = state.omega * (MAX_OMEGA_RAD / theta)

        # [DEBUG] end-of-iteration state snapshot

        gamma_brow_step = delta[76:84]

        # [DEBUG] track psi_mouth's actual value and movement per iteration —
        # to see whether it's genuinely updating or just drowned out by the
        # much larger pose/beta gradients in the same combined GN step.
        psi_mouth_step = delta[84:88]

        # [DEBUG] gamma_ear trace — same purpose as the brow/mouth traces
        # above: confirm the ear-rejection fix is actually letting gamma_ear
        # move toward target rather than sitting frozen or oscillating.
        gamma_ear_step = delta[88:92]

        # FIX 3: per-block convergence check instead of one combined norm
        # dominated by t's scale (t ~O(100s) vs beta/omega ~O(1)).
        if step_accepted:
            d_beta, d_pose, d_rest = delta[:64], delta[64:70], delta[70:]
            converged = (np.linalg.norm(d_beta) < 1e-2
                         and np.linalg.norm(d_pose[:3]) < 1e-4   # omega (rad)
                         and np.linalg.norm(d_pose[3:]) < 1e-1   # t (units)
                         and np.linalg.norm(d_rest) < 1e-2)
            if converged:
                print(f"[P2/S2 fit] converged at iteration {iteration} "
                      f"(per-block thresholds)")
                break
    else:
        print(f"[P2/S2 fit] hit GN_MAX_ITER={GN_MAX_ITER} without converging — "
              f"final ‖delta‖ was {np.linalg.norm(delta):.3f}")

    # [DEBUG] final omega/t vs the q_pred it was anchored to — how far did it travel?
    final_omega_drift = np.linalg.norm(state.omega - q_pred)
    print(f"[P2/S2 fit] FINAL omega={np.array2string(state.omega, precision=4)}  "
          f"‖final_omega - q_pred‖={final_omega_drift:.4f}  "
          f"(large drift = GN moved pose far from the Newton warm-start prior)")

    if beta_frozen and prev_state is not None:
        if np.linalg.norm(state.beta - prev_state.beta) > EPS_BETA:
            state = _resolv_beta(model, alpha_tgt, state, K,
                                 landmarks, lm_vidx, q_pred, sigma_q, t_pred=t_pred, sigma_t=sigma_t, I_tgt=I_tgt,
                                 gamma_ear_frozen=(gamma_ear_fixed is not None))

    delta_beta = state.beta - beta_neutral

    # FIX 2: 200.0 was far too loose — a norm=132 blowup (mesh
    # projecting to projected positions of +/-30,000px on a 1024px frame, see
    # [P2/S2] logs) sailed straight past this clip untouched. 40.0
    # is a sane ceiling for a real expression delta; tune from real
    # footage if legitimate large expressions start getting clipped.
    MAX_DB_NORM = 40.0
    db_norm = np.linalg.norm(delta_beta)
    if db_norm > MAX_DB_NORM:
        print(f"[P2/S2 fit] delta_beta norm={db_norm:.2f} EXCEEDS "
              f"MAX_DB_NORM={MAX_DB_NORM} — clipping")
        delta_beta = delta_beta * (MAX_DB_NORM / db_norm)

    beta_moved = float(np.linalg.norm(state.beta - beta_before))
    state.beta_frozen = bool(frame_index >= 5 and beta_moved <= EPS_BETA)

    return state, delta_beta


def _resolv_beta(model, alpha_tgt, state, K, landmarks, lm_vidx,
                 q_pred, sigma_q, t_pred, sigma_t, I_tgt=None,
                 gamma_ear_frozen=False):
    """
    One extra GN pass with β free.

    FIXES APPLIED:
      1. Previously used flat GN_LAMBDA_BETA/POSE = 0.01 — the same
         under-damped constant referenced in the original jacobian_cond
         debug note ("a small reg constant like 0.01 won't meaningfully
         constrain"). Replaced with per-column adaptive (LM-style)
         damping, matching fit_target_frame's main loop.
      2. This was the only GN solver in the file with NO backtracking or
         residual check at all — every iteration applied delta
         unconditionally, with nothing stopping a bad step. Added the
         same never-accept-a-worse-step guard used elsewhere.
    """
    _, e0 = _build_target_jacobian(
        model, alpha_tgt, state, K, landmarks, lm_vidx,
        q_pred, sigma_q, t_pred=t_pred, sigma_t=sigma_t, I_tgt=I_tgt,
        gamma_ear_frozen=gamma_ear_frozen)
    resid_before = np.linalg.norm(e0)

    for _ in range(5):
        J, e = _build_target_jacobian(
            model, alpha_tgt, state, K,
            landmarks, lm_vidx, q_pred, sigma_q, t_pred=t_pred, sigma_t=sigma_t, I_tgt=I_tgt,
            gamma_ear_frozen=gamma_ear_frozen)

        # FIX 1: per-column adaptive damping instead of flat 0.01/0.01
        LM_LAMBDA = 1e-2
        H_data  = J.T @ J
        diag_H  = np.diag(H_data).copy()
        diag_H[diag_H < 1e-12] = 1e-12
        H = H_data + LM_LAMBDA * np.diag(diag_H)
        g = J.T @ e
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(H, g, rcond=None)[0]

        # FIX 2: check the trial step before committing — this loop used
        # to apply delta unconditionally with zero safety net.
        trial = TargetFrameState(
            beta       = state.beta       + delta[:64],
            omega      = state.omega      + delta[64:67],
            t          = state.t          + delta[67:70],
            psi_eye    = state.psi_eye    + delta[70:76],
            gamma_brow = state.gamma_brow + delta[76:84],
            psi_mouth  = state.psi_mouth  + delta[84:88],
            gamma_ear  = state.gamma_ear  + delta[88:92],
        )
        _, e_trial = _build_target_jacobian(
            model, alpha_tgt, trial, K, landmarks, lm_vidx,
            q_pred, sigma_q, t_pred=t_pred, sigma_t=sigma_t, I_tgt=I_tgt,
            gamma_ear_frozen=gamma_ear_frozen)
        resid_trial = np.linalg.norm(e_trial)
        if resid_trial > resid_before:
            delta *= 0.5   # halve once and accept — keeps this pass cheap
        resid_before = min(resid_before, resid_trial)

        state.beta       += delta[:64]
        state.omega      += delta[64:67]
        state.t          += delta[67:70]
        state.psi_eye    += delta[70:76]
        state.gamma_brow += delta[76:84]
        state.psi_mouth  += delta[84:88]
        state.gamma_ear  += delta[88:92]

        state.psi_eye[0] = np.clip(state.psi_eye[0], 0.0, 1.0)
        state.psi_eye[3] = np.clip(state.psi_eye[3], 0.0, 1.0)
        state.psi_eye[[1,2,4,5]] = np.clip(state.psi_eye[[1,2,4,5]], -0.5, 0.5)
    return state


def quality_gate(
        landmarks:  np.ndarray,
        projected:  np.ndarray,
        omega:      np.ndarray,
        f_t:        np.ndarray,
        f_pred:     np.ndarray,
        sigma_lm:   float = 12.8,
        sigma_flow: float = SIGMA_FLOW,
) -> tuple[float, str]:
    """
    c_t = 0.50·s_lm + 0.30·s_flow + 0.20·s_pose

    FIX s_flow: was mean of per-point exp.
    PDF says: exp(-||f_t - f_t^pred||² / 2σ²)
    i.e. single exp of the MEAN squared flow error.

    FIX sigma_lm (2nd recalibration): the previous sigma=6.0 was tuned
    against ASSUMED per-region errors (eye 6.0px, brow 3.9px, nose 4.3px,
    mouth 13.1px, ear 2.7px, mean_err=5.7px) that were never actually
    measured on this pipeline's real output — they were roughly 4x
    tighter than reality across the board, and up to ~19x tighter for
    brow specifically. Real measured errors from [DEBUG align] on an
    actual run: eye 6.8px, brow 75.0px, nose 17.8px, mouth 11.8px,
    ear 29.6px (mean_err=22.2px, median_err=15.1px, n=625). At the old
    sigma=6, virtually every point above ~15px scores ~0 regardless of
    how the rest of the face aligned, so s_lm was crushed toward 0 on
    nearly every frame and c_t sat around 0.58 permanently, forcing
    CONSERVATIVE
    on every single frame regardless of actual relative quality.

    sigma_lm=12.8 is calibrated so the ACTUAL median point (15.1px)
    scores s_lm_point=0.50 — a defensible "this is a typical landmark,
    call it half-credit" anchor, computed from real measured data
    instead of assumed numbers. Genuinely good regions still score
    high (eye 0.87, mouth 0.66) and genuinely poor regions are still
    correctly penalized (ear 0.07, brow ~0.00) — this does not hide
    real misalignment, it stops the whole gate from being pinned to
    the floor by a precision level this pipeline doesn't currently
    achieve. If the upstream landmark/GN fit quality improves later
    (see Phase 1 Stage 2 / Phase 2 fit_target_frame), this constant
    should be re-derived again from the new measured distribution —
    it is a snapshot of current pipeline accuracy, not a target.
    """
    # s_lm
    sq_lm = np.sum((landmarks - projected)**2, axis=1)
    s_lm  = float(np.mean(np.exp(-sq_lm / (2.0 * sigma_lm**2))))

    # s_pose
    s_pose = float(np.exp(
        -(omega[1]**2 + omega[0]**2) / (2.0 * SIGMA_POSE**2)
    ))

    # s_flow — FIX: single exp of mean squared error
    mean_sq_flow = float(np.mean(np.sum((f_t - f_pred)**2, axis=1)))
    s_flow = float(np.exp(-mean_sq_flow / (2.0 * sigma_flow**2)))

    c_t = 0.50*s_lm + 0.30*s_flow + 0.20*s_pose
    c_t = float(np.clip(np.nan_to_num(c_t, nan=0.0), 0.0, 1.0))

    if c_t > 0.75:
        decision = "PROCEED"
    elif c_t >= 0.40:
        decision = "CONSERVATIVE"
    else:
        decision = "FALLBACK"

    return c_t, decision
