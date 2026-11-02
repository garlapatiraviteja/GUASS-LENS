"""
phase1/outputs/stage4_target_neutral.py — Target Neutral Fit

GN solve on first N ≥ 10 relaxed target frames simultaneously.

Energy per frame t:
  Eᵗ(α_tgt, β_tgt^neutral, ωᵗ, tᵗ) = Σᵢ ‖π(Sᵢ(α_tgt,β^neutral);ωᵗ,tᵗ) − Lᵢ^tgt,t‖²

Aggregate:
  E_total = Σ_{t=1}^{N} Eᵗ

GN update:
  Δ[α_tgt, β_neutral] = −(J_tgt^T J_tgt)⁻¹ J_tgt^T e_tgt

Terminate: ‖Δ[α_tgt, β_neutral]‖ < 1e-4 or 10 iterations.

Outputs:
  β_tgt^neutral — target neutral expression, feeds Phase 2 every frame
  α_tgt         — target identity shape, used only for boundary residual in Phase 3

FIXES vs previous version:
1. _build_joint_jacobian — Jacobian blocks now use exact depth-correction
   via dpi_dt instead of approximation (1/Z_i)*K*R*A.
   Per PDF: ∂π/∂α = (∂π/∂t)·R·A_id,i  (depth-corrected)
2. _update_per_frame_pose — dpi_domega now passes omega for exact
   Rodrigues derivative (was missing omega argument → first-order approx).
3. NEW — backtracking added to both _update_per_frame_pose and the shared
   [alpha, beta_neutral] step in fit_target_neutral. Confirmed via logs
   that this stage was diverging silently (alpha_tgt pinned at the
   MAX_ALPHA_NORM clamp, beta_neutral norm in the thousands) — same
   catastrophic-step failure mode found and fixed in stage2's fit_source().
   Any step that pushes vertices behind the camera or increases the
   residual by more than 5% is now rejected/halved, same pattern as
   fit_source()'s fix.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from ..geometry import rodrigues, project_mesh, dpi_dt, dpi_domega, build_dpi_dalpha, build_dpi_dbeta
from ..shape_model import MorphableModel


GN_MAX_ITER  = 10
GN_THRESH    = 1e-4
N_MIN_FRAMES = 10


@dataclass
class TargetNeutralResult:
    alpha_tgt:    np.ndarray   # (80,)
    beta_neutral: np.ndarray   # (64,)


@dataclass
class TargetFrameData:
    """Per-frame data for target neutral fit."""
    landmarks: np.ndarray   # (N_lm, 2)
    lm_vidx:   np.ndarray   # (N_lm,)
    omega:     np.ndarray   # (3,)
    t:         np.ndarray   # (3,)


# ---------------------------------------------------------------------------
# NEW — residual helper (used by the backtracking checks below)
# ---------------------------------------------------------------------------

def _frame_residual(S: np.ndarray, frame: TargetFrameData,
                    K: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Compute the (2*N_lm,) residual vector for one frame at a given shape S,
    using the exact same skip-logic as everywhere else (Z<=0 or NaN -> that
    landmark's rows stay 0, not removed). Also returns n_valid — the count
    of landmarks that projected in front of the camera.
    """
    pi, Z = project_mesh(S, frame.omega, frame.t, K)
    N_lm = frame.landmarks.shape[0]
    e = np.zeros(2 * N_lm, dtype=np.float64)
    n_valid = 0
    for n in range(N_lm):
        vi = frame.lm_vidx[n]
        if Z[vi] <= 0 or np.isnan(Z[vi]):
            continue
        e[2*n:2*n+2] = pi[vi] - frame.landmarks[n]
        n_valid += 1
    return e, n_valid


# ---------------------------------------------------------------------------
# Build joint Jacobian across all N frames
# ---------------------------------------------------------------------------

def _build_joint_jacobian(
        model:        MorphableModel,
        alpha:        np.ndarray,
        beta_neutral: np.ndarray,
        frames:       List[TargetFrameData],
        K:            np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stacked Jacobian and residual over all N frames.

    Shared parameters: [α(80), β_neutral(64)] → 144 columns
    Per-frame pose: [ω(3), t(3)] → updated separately

    FIX: Jacobian blocks now use depth-corrected formula via dpi_dt:
      ∂π/∂α = (∂π/∂t) · R · A_id,i   (exact)
      ∂π/∂β = (∂π/∂t) · R · A_exp,i  (exact)
    Previously used approximation (1/Z_i)*K*R*A which misses depth coupling.
    """
    N_frames = len(frames)
    N_lm     = frames[0].landmarks.shape[0]
    total    = 2 * N_lm * N_frames

    J_joint = np.zeros((total, 144), dtype=np.float64)
    e_joint = np.zeros(total,        dtype=np.float64)

    for f_idx, frame in enumerate(frames):
        R  = rodrigues(frame.omega)
        S  = model.base_shape(alpha, beta_neutral)
        pi, Z = project_mesh(S, frame.omega, frame.t, K)

        for n in range(N_lm):
            vi  = frame.lm_vidx[n]
            Z_i = Z[vi]
            if Z_i <= 0 or np.isnan(Z_i):
                continue

            S_i     = S[vi]
            p_cam_i = R @ S_i + frame.t

            row = f_idx * 2 * N_lm + 2 * n
            e_joint[row:row+2] = pi[vi] - frame.landmarks[n]

            # FIX: negate v-row to match the v-flip applied in project_mesh()
            # (pi_v_actual = 2*cy - raw_v). Every block below chains through
            # dp_dt_i, so this single negation propagates the fix everywhere.
            dp_dt_i = dpi_dt(K, Z_i, p_cam_i)
            dp_dt_i = dp_dt_i.copy()
            dp_dt_i[1, :] *= -1.0

            # ∂π/∂α — (2, 80) — depth-corrected
            A_id_i    = model.A_id[vi]
            dp_dalpha = build_dpi_dalpha(K, Z_i, R, A_id_i, dp_dt_i, p_cam_i)
            J_joint[row:row+2, :80] = dp_dalpha

            # ∂π/∂β_neutral — (2, 64) — depth-corrected
            A_exp_i  = model.A_exp[vi]
            dp_dbeta = build_dpi_dbeta(K, Z_i, R, A_exp_i, dp_dt_i, p_cam_i)
            J_joint[row:row+2, 80:144] = dp_dbeta

    return J_joint, e_joint


# ---------------------------------------------------------------------------
# Per-frame pose update
# ---------------------------------------------------------------------------

def _update_per_frame_pose(
        model:        MorphableModel,
        alpha:        np.ndarray,
        beta_neutral: np.ndarray,
        frame:        TargetFrameData,
        K:            np.ndarray,
) -> TargetFrameData:
    """
    Update each frame's [ω, t] by one GN step with [α, β_neutral] fixed.

    FIX: dpi_domega now passes frame.omega for exact Rodrigues derivative.
    Previously called without omega → used first-order approximation.

    NEW: backtracking — reject/shrink any step that pushes vertices behind
    the camera or increases this frame's own residual by more than 5%,
    same pattern used in stage2_gn_source.py's fit_source().
    """
    S   = model.base_shape(alpha, beta_neutral)
    R   = rodrigues(frame.omega)
    pi, Z = project_mesh(S, frame.omega, frame.t, K)

    N_lm = frame.landmarks.shape[0]
    J6   = np.zeros((2*N_lm, 6), dtype=np.float64)
    e6   = np.zeros(2*N_lm,     dtype=np.float64)

    n_valid_before = 0
    for n in range(N_lm):
        vi  = frame.lm_vidx[n]
        Z_i = Z[vi]
        if Z_i <= 0 or np.isnan(Z_i):
            continue
        n_valid_before += 1

        S_i     = S[vi]
        p_cam_i = R @ S_i + frame.t

        row = 2 * n
        e6[row:row+2] = pi[vi] - frame.landmarks[n]

        dp_dt_i = dpi_dt(K, Z_i, p_cam_i)
        # FIX: negate v-row to match the v-flip applied in project_mesh()
        dp_dt_i = dp_dt_i.copy()
        dp_dt_i[1, :] *= -1.0
        # FIX: pass frame.omega for exact Rodrigues derivative
        dp_dw_i = dpi_domega(dp_dt_i, S_i, omega=frame.omega)

        J6[row:row+2, :3] = dp_dw_i
        J6[row:row+2, 3:] = dp_dt_i

    resid_before = np.linalg.norm(e6)   # NEW

    H6 = J6.T @ J6 + 1e-6 * np.eye(6)
    g6 = J6.T @ e6
    try:
        delta6 = -np.linalg.solve(H6, g6)
    except np.linalg.LinAlgError:
        delta6 = -np.linalg.lstsq(H6, g6, rcond=None)[0]

    # NEW — backtracking on this single frame's pose step
    for _ in range(5):
        trial_omega = frame.omega + delta6[:3]
        trial_t     = frame.t     + delta6[3:]
        trial_frame = TargetFrameData(frame.landmarks, frame.lm_vidx, trial_omega, trial_t)
        e_trial, n_valid_trial = _frame_residual(S, trial_frame, K)
        resid_trial = np.linalg.norm(e_trial)

        if n_valid_trial >= n_valid_before and resid_trial <= resid_before * 1.05:
            return trial_frame
        delta6 *= 0.5

    # no safe step found — hold this frame's pose unchanged
    return TargetFrameData(
        frame.landmarks, frame.lm_vidx,
        frame.omega.copy(),
        frame.t.copy(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# def fit_target_neutral(
#         model:       MorphableModel,
#         K:           np.ndarray,
#         frames:      List[TargetFrameData],
#         alpha_init:  np.ndarray = None,
#         beta_init:   np.ndarray = None,
# ) -> TargetNeutralResult:
#     """
#     Phase-1 Stage-4: joint GN over first N ≥ 10 target frames.
#     Returns β_tgt^neutral and α_tgt.

#     NEW: backtracking on the shared [alpha, beta_neutral] step — same
#     reasoning as fit_source()'s fix. Confirmed via logs this stage was
#     silently diverging (alpha_tgt pinned at MAX_ALPHA_NORM, beta_neutral
#     norm in the thousands) with no safeguard against a bad joint step.
#     """
#     assert len(frames) >= N_MIN_FRAMES, \
#         f"Need at least {N_MIN_FRAMES} frames, got {len(frames)}"


#             if n_valid_trial >= n_valid_before and resid_trial <= resid_before * 1.05:
#                 alpha, beta_neutral = alpha_trial, beta_trial
#                 step_accepted = True
#                 break
#             delta *= 0.5
#         else:
#             print(f"  [S4 iter {iteration}] step rejected — holding alpha/beta")
#             delta = np.zeros_like(delta)


def fit_target_neutral(
        model:       MorphableModel,
        K:           np.ndarray,
        frames:      List[TargetFrameData],
        alpha_init:  np.ndarray = None,
        beta_init:   np.ndarray = None,
) -> TargetNeutralResult:
    """
    Phase-1 Stage-4: joint GN over first N ≥ 10 target frames.
    Returns β_tgt^neutral and α_tgt.

    NEW: backtracking on the shared [alpha, beta_neutral] step — same
    reasoning as fit_source()'s fix. Confirmed via logs this stage was
    silently diverging (alpha_tgt pinned at MAX_ALPHA_NORM, beta_neutral
    norm in the thousands) with no safeguard against a bad joint step.
    """
    assert len(frames) >= N_MIN_FRAMES, \
        f"Need at least {N_MIN_FRAMES} frames, got {len(frames)}"

    alpha        = alpha_init   if alpha_init is not None else np.zeros(80)
    beta_neutral = beta_init    if beta_init  is not None else np.zeros(64)
    frames       = list(frames)

    LAMBDA_ALPHA = 5
    LAMBDA_BETA  = 5

    # FIX: MAX_ALPHA_NORM/MAX_BETA_NEUTRAL_NORM were hardcoded to 300.0,
    # tuned against the original 49-vertex mesh's source alpha norm
    # (~143). With the denser 625-vertex mesh, the identity basis has
    # proportionally more structured modes to fit, and 300.0 was pinning
    # alpha_norm at exactly that value every iteration — backtracking
    # then rejected every step from iteration 6 onward because it could
    # never satisfy both the clipped norm AND the residual-improvement
    # check simultaneously (see [S4] logs: alpha_norm=300.00 identical
    # every iteration, "step rejected" from iter 6-9).
    # Scale the ceiling off the actual source alpha norm instead of a
    # constant tuned for a different mesh resolution.
    src_alpha_norm = float(np.linalg.norm(alpha_init)) if alpha_init is not None else 143.0
    MAX_ALPHA_NORM = 3.0 * src_alpha_norm
    MAX_BETA_NEUTRAL_NORM = 3.0 * src_alpha_norm

    bn = np.linalg.norm(beta_neutral)
    if bn > MAX_BETA_NEUTRAL_NORM:
        beta_neutral = beta_neutral * (MAX_BETA_NEUTRAL_NORM / bn)

    for iteration in range(GN_MAX_ITER):
        # Step 1: update per-frame pose with [α, β_neutral] fixed
        frames = [
            _update_per_frame_pose(model, alpha, beta_neutral, fr, K)
            for fr in frames
        ]

        # Step 2: update shared [α, β_neutral] with all poses fixed
        J, e = _build_joint_jacobian(model, alpha, beta_neutral, frames, K)

        # NEW — snapshot validity/residual BEFORE this step, for backtracking
        n_valid_before = 0
        for fr in frames:
            S_cur = model.base_shape(alpha, beta_neutral)
            _, n_v = _frame_residual(S_cur, fr, K)
            n_valid_before += n_v
        resid_before = np.linalg.norm(e)

        # Block-diagonal regularisation
        reg = np.zeros(144)
        reg[:80]    = LAMBDA_ALPHA
        reg[80:144] = LAMBDA_BETA
        H144 = J.T @ J + np.diag(reg)
        g144 = J.T @ e
        try:
            delta = -np.linalg.solve(H144, g144)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(H144, g144, rcond=None)[0]

        # NEW — backtracking on the shared alpha/beta step
        step_accepted = False
        for _ in range(5):
            alpha_trial = alpha + delta[:80]
            an = np.linalg.norm(alpha_trial)
            if an > MAX_ALPHA_NORM:
                alpha_trial = alpha_trial * (MAX_ALPHA_NORM / an)
            beta_trial = beta_neutral + delta[80:144]

            n_valid_trial = 0
            resid_sq_sum  = 0.0
            S_trial = model.base_shape(alpha_trial, beta_trial)
            for fr in frames:
                e_trial, n_v = _frame_residual(S_trial, fr, K)
                n_valid_trial += n_v
                resid_sq_sum  += float(np.dot(e_trial, e_trial))
            resid_trial = np.sqrt(resid_sq_sum)

            if n_valid_trial >= n_valid_before and resid_trial <= resid_before * 1.05:
                alpha, beta_neutral = alpha_trial, beta_trial
                step_accepted = True
                break
            delta *= 0.5
        else:
            print(f"  [S4 iter {iteration}] step rejected — holding alpha/beta")
            delta = np.zeros_like(delta)

        print(f"  [S4 iter {iteration}] ‖delta‖={np.linalg.norm(delta):.5f}  "
              f"alpha_norm={np.linalg.norm(alpha):.2f}  beta_norm={np.linalg.norm(beta_neutral):.2f}  "
              f"resid_before={resid_before:.1f}")

        # Clamp alpha norm (kept as a second line of defence, same as original)
        an = np.linalg.norm(alpha)
        if an > MAX_ALPHA_NORM:
            alpha = alpha * (MAX_ALPHA_NORM / an)

        if step_accepted and np.linalg.norm(delta) < GN_THRESH:
            print(f"  [S4] converged at iteration {iteration}")
            break

    return TargetNeutralResult(
        alpha_tgt    = alpha,
        beta_neutral = beta_neutral,
    )