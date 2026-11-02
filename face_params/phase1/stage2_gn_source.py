"""
phase1/stage2_gn_source.py — Single Joint Gauss-Newton Fit on Source

AUDIT NOTE: no neural network runs in this solver. Its only image-derived input is the landmark
set (template projection, or SAM 3.1 region masks via perception/); the fit itself is NumPy.

State vector:
  Θ_src = [α(80), β(64), ω(3), t(3), ψ_eye(6), γ_brow(8), δ_nose(3·V_nose)]

Full 3-term energy:
  E(Θ) = E_appearance + E_geometric + E_regularise
  E_appearance : ‖I_src(π(S_i)) - T_i^patch‖²
  E_geometric  : ‖π(S_i; Θ) - x_i‖²
  E_regularise : ‖L_tps · δ_nose‖²

Full Jacobian:
  J = [ ∂I_src/∂x · ∂π/∂Θ ]   ← appearance rows
      [  -I  |  ∂π/∂Θ      ]   ← geometric rows
      [   0  |  L_tps       ]   ← regularise rows

FIXES vs previous version:
1. TPS Jacobian — was diagonal only (L_tps[vi,vi]*I₃).
   Now fills full L_tps rows — all off-diagonal nose-nose couplings.
2. _build_lambda — was diagonal only (LtL[vi,vi]*I₃).
   Now fills full LtL block — correct TPS prior.
3. app_res diagnostic — was wrong slice. Now correct.
4. full_shape — will accept psi_mouth/psi_ear once shape_model updated.
5. dpi_domega — now passes omega for exact Rodrigues derivative.
"""

import numpy as np
from typing import Optional

from ..geometry import *
from ..shape_model import MorphableModel, SourceState, build_tps_laplacian

GN_LAMBDA   = 0.01
GN_MAX_ITER = 10
GN_THRESH   = 1e-4
THETA_OPEN  = 0.15


ALPHA_ACTIVE = 20

BETA_ACTIVE  = 15

# Mouth expression columns inside A_exp (64-dim)
MOUTH_EXP_COLS = np.array([0, 1, 2, 3], dtype=np.intp)

# Column offsets inside Θ
C_ALPHA = 0
C_BETA  = 80
C_OMEGA = 144
C_T     = 147
C_PSI   = 150
C_BROW  = 156
C_NOSE  = 164

# Appearance weight
W_APP = 5


# ---------------------------------------------------------------------------
# Image sampling helpers
# ---------------------------------------------------------------------------

def _sample_bilinear(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear sample of img (H,W,C) at (x,y). Returns zeros if OOB."""
    H, W = img.shape[:2]
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return np.zeros(img.shape[2:] if img.ndim == 3 else (), dtype=np.float64)
    fx = x - x0; fy = y - y0
    v = ((1-fx)*(1-fy)*img[y0,x0] + fx*(1-fy)*img[y0,x1] +
         (1-fx)*fy    *img[y1,x0] + fx*fy    *img[y1,x1])
    return v.astype(np.float64)


def _image_gradient(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """∂I/∂x, ∂I/∂y at (x,y). Returns (2,C)."""
    H, W = img.shape[:2]
    xc = float(np.clip(x, 1, W-2))
    yc = float(np.clip(y, 1, H-2))
    gx = (_sample_bilinear(img, xc+1, yc) - _sample_bilinear(img, xc-1, yc)) * 0.5
    gy = (_sample_bilinear(img, xc, yc+1) - _sample_bilinear(img, xc, yc-1)) * 0.5
    return np.stack([gx, gy], axis=0)


def _build_T_patch(I_src: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """T_i^patch = I_src sampled at detected landmark positions. Returns (N,C)."""
    N = len(landmarks)
    C = I_src.shape[2] if I_src.ndim == 3 else 1
    T = np.zeros((N, C), dtype=np.float64)
    for i in range(N):
        T[i] = _sample_bilinear(I_src, landmarks[i,0], landmarks[i,1])
    return T


# ---------------------------------------------------------------------------
# Main Jacobian + residual builder
# ---------------------------------------------------------------------------

def _build_jacobian_and_residual(model, state, K, landmarks, lm_vidx, I_src):
    """
    Full 3-term Jacobian and residual per your PDF maths.

    Row structure:
      rows 0       .. 2N-1        : E_geometric  (2 rows per landmark)
      rows 2N      .. 2N+C*N-1   : E_appearance (C rows per landmark)
      rows 2N+C*N  .. end         : E_regularise (3 rows per nose vertex)

    FIXES:
    - TPS rows now fill full L_tps row, not just diagonal block
    - dpi_domega now passes omega for exact derivative
    """
    V_nose = len(model.nose_idx)
    dim_Θ  = C_NOSE + 3 * V_nose +4+4
    C_MOUTH = C_NOSE + 3 * V_nose        # NEW — runtime-dependent, NOT a module constant
    C_EAR   = C_MOUTH + 4                 # NEW
    mouth_local = {int(g): l for l, g in enumerate(model.mouth_idx.tolist())}   # NEW
    ear_local   = {int(g): l for l, g in enumerate(model.ear_idx.tolist())}     # NEW

    N      = len(landmarks)
    C      = I_src.shape[2] if I_src.ndim == 3 else 1

    S  = model.full_shape(
    state.alpha, state.beta,
    state.psi_eye, state.gamma_brow, state.delta_nose,
    psi_mouth=state.psi_mouth, gamma_ear=state.gamma_ear
      )
    R  = rodrigues(state.omega)
    pi, Z = project_mesh(S, state.omega, state.t, K)

    rows_geo  = 2 * N
    rows_app  = C * N
    rows_tps  = 3 * V_nose
    total     = rows_geo + rows_app + rows_tps

    J = np.zeros((total, dim_Θ), dtype=np.float64)
    e = np.zeros(total,          dtype=np.float64)

    T_patch = _build_T_patch(I_src, landmarks)

    eye_local  = {int(g): l for l, g in enumerate(model.eye_idx.tolist())}
    brow_local = {int(g): l for l, g in enumerate(model.brow_idx.tolist())}
    nose_local = {int(g): l for l, g in enumerate(model.nose_idx.tolist())}

    for n in range(N):
        vi  = int(lm_vidx[n])
        Z_i = Z[vi]
        if Z_i <= 0 or np.isnan(Z_i):
            continue

        S_i     = S[vi]
        p_cam_i = R @ S_i + state.t

        # ── Geometric rows ────────────────────────────────────────────────
        row_geo = 2 * n
        e[row_geo:row_geo+2] = pi[vi] - landmarks[n]

        dp_dt_i = dpi_dt(K, Z_i, p_cam_i)
        # FIX: negate v-row to match the v-flip applied in project()/project_mesh()
        # (pi_v_actual = 2*cy - raw_v, so d(pi_v_actual)/dtheta = -d(raw_v)/dtheta).
        # Every downstream block below chains through dp_dt_i, so this single
        # negation propagates the fix everywhere, including appearance rows.
        dp_dt_i = dp_dt_i.copy()
        dp_dt_i[1, :] *= -1.0
        # FIX: pass omega for exact Rodrigues derivative
        dp_dw_i = dpi_domega(dp_dt_i, S_i, omega=state.omega)

        J[row_geo:row_geo+2, C_T:C_T+3]         = dp_dt_i
        J[row_geo:row_geo+2, C_OMEGA:C_OMEGA+3] = dp_dw_i

        A_id_i  = model.A_id[vi]
        A_exp_i = model.A_exp[vi]
        J[row_geo:row_geo+2, C_ALPHA:C_ALPHA+80] = build_dpi_dalpha(
            K, Z_i, R, A_id_i, dp_dt_i, p_cam_i)
        J[row_geo:row_geo+2, C_BETA:C_BETA+64]   = build_dpi_dbeta(
            K, Z_i, R, A_exp_i, dp_dt_i, p_cam_i)

        if vi in eye_local:
            li = eye_local[vi]
            B_eye_i = model.B_eye[li]
            J[row_geo:row_geo+2, C_PSI:C_PSI+6] = build_dpi_dpsi_eye(
                K, Z_i, R, B_eye_i, dp_dt_i, p_cam_i)

        if vi in brow_local:
            li = brow_local[vi]
            B_brow_i = model.B_brow[li]
            J[row_geo:row_geo+2, C_BROW:C_BROW+8] = build_dpi_dgamma_brow(
                K, Z_i, R, B_brow_i, dp_dt_i, p_cam_i)

        if vi in mouth_local:
            li = mouth_local[vi]
            B_mouth_i = model.B_mouth[li]
            J[row_geo:row_geo+2, C_MOUTH:C_MOUTH+4] = build_dpi_dpsi_mouth(
                K, Z_i, R, B_mouth_i, dp_dt_i, p_cam_i)
            
        if vi in ear_local:
            li = ear_local[vi]
            B_ear_i = model.B_ear[li]
            J[row_geo:row_geo+2, C_EAR:C_EAR+4] = build_dpi_dgamma_ear(
                K, Z_i, R, B_ear_i, dp_dt_i, p_cam_i)    

        if vi in nose_local:
            li  = nose_local[vi]
            col = C_NOSE + 3 * li
            J[row_geo:row_geo+2, col:col+3] = build_dpi_ddelta_nose(
                K, Z_i, R, V_nose, dp_dt_i, p_cam_i)

        # ── Appearance rows ───────────────────────────────────────────────
        row_app = rows_geo + C * n
        px, py  = float(pi[vi, 0]), float(pi[vi, 1])

        I_curr = _sample_bilinear(I_src, px, py)
        e[row_app:row_app+C] = W_APP * (I_curr - T_patch[n])

        grad_I = _image_gradient(I_src, px, py)   # (2, C)
        J_pi   = J[row_geo:row_geo+2, :]           # (2, dim_Θ)
        for c in range(C):
            J[row_app+c, :] = W_APP * (grad_I[:, c] @ J_pi)

    # ── TPS regularisation rows ───────────────────────────────────────────
    # FIX: fill full L_tps rows, not just diagonal block
    # Previous: J[r0:r0+3, c0:c0+3] = L_tps[vi,vi]*I₃  ← diagonal only
    # Correct:  for each nose vertex vi, fill all vj columns
    L_tps     = build_tps_laplacian(model.nose_idx, model.faces)
    delta_mat = state.delta_nose.reshape(V_nose, 3)
    Ld = L_tps @ delta_mat   # (V_nose, 3)

    for vi in range(V_nose):
        r0 = rows_geo + rows_app + vi * 3
        e[r0:r0+3] = Ld[vi]
        # FIX: fill all vj columns, not just diagonal
        for vj in range(V_nose):
            c0 = C_NOSE + vj * 3
            J[r0:r0+3, c0:c0+3] = L_tps[vi, vj] * np.eye(3)

    # NEW — freeze alpha/beta dims beyond the active count, so alpha+beta's
    # combined free dimensions (35) stay well under the 98 geometric
    # constraints (49 landmarks x 2D), instead of the current 144 > 98
    # imbalance that's letting alpha/beta alias into pose's job.
    
    return J, e


# ---------------------------------------------------------------------------
# GN solver
# ---------------------------------------------------------------------------

def _gn_update(J, e, lam, Lambda):
    H = J.T @ J + lam * Lambda
    g = J.T @ e
    try:
        return -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return -np.linalg.lstsq(H, g, rcond=None)[0]


    
            
#     return Lambda


#     return Lambda


def _build_lambda(dim_Θ, V_nose, L_tps, N_lm,
                  lambda_alpha=50.0,
                  lambda_beta=50.0,
                  lambda_pose=50.0,
                  lambda_eye=250.0,
                  # FIX: brow had the WEAKEST damping of every regional
                  # block (150) despite having MORE free parameters (8)
                  # than eye (6), mouth-corner (2), or ear (4), and being
                  # constrained by only 4 landmark points (brow_idx has
                  # 4 entries) — i.e. the least-constrained region got the
                  # least restraint. That's what let gamma_brow run away
                  # to norm ~10-11 once the output clip was widened; the
                  # clip had been quietly compensating for this
                  # miscalibration rather than the damping doing its job.
                  # Raised above eye's 250, scaled for brow's extra 2
                  # degrees of freedom (250 * 8/6 ≈ 333, rounded up with
                  # margin since brow is also the least-constrained).
                  lambda_brow=400.0,
                  lambda_mouth_corner=250.0,
                  lambda_ear=250.0,
                  lambda_nose_anchor=20.0):
    """
    Block-diagonal prior Λ.

    Every regional block gets damping scaled the same way (value / GN_LAMBDA),
    so the number you pass in means what it says regardless of GN_LAMBDA's
    own scaling in _gn_update(). Nose gets TWO terms: the original TPS
    smoothness term (penalizes roughness BETWEEN neighboring nose vertices)
    plus a NEW absolute anchor term — a graph Laplacian's null space always
    includes the constant vector, so TPS alone applies zero restoring force
    to the whole nose region drifting as one. The anchor fixes that gap.

    FIX: lambda_alpha/lambda_beta were tuned against the original 49-landmark
    mesh (98 geometric residual rows). With a denser mesh (e.g. 625 landmarks,
    1250 rows), the same fixed damping value is proportionally weaker
    relative to the much larger data term (J.T@J scales with row count),
    letting alpha/beta balloon to huge norms (~600 instead of ~150) and
    produce wild per-vertex Z spread across the mesh — confirmed in logs
    (alpha norm=597.79, Z range still spreading iter 0->9 despite pose
    damping being correctly applied). Scale alpha/beta damping by the
    actual landmark count relative to the original baseline so it stays
    proportionally as strong regardless of mesh resolution.
    """
    N_LM_BASELINE = 49
    scale_factor = N_lm / N_LM_BASELINE

    Lambda = np.eye(dim_Θ, dtype=np.float64)

    # --- identity / expression ---
    Lambda[C_ALPHA:C_ALPHA+80, C_ALPHA:C_ALPHA+80] *= (lambda_alpha * scale_factor) / GN_LAMBDA
    Lambda[C_BETA:C_BETA+64,   C_BETA:C_BETA+64]   *= (lambda_beta  * scale_factor) / GN_LAMBDA

    # --- pose (omega, t) ---
    Lambda[C_OMEGA:C_OMEGA+3, C_OMEGA:C_OMEGA+3] *= lambda_pose / GN_LAMBDA
    Lambda[C_T:C_T+3,         C_T:C_T+3]         *= lambda_pose / GN_LAMBDA
    lambda_depth_anchor = 2000.0
    Lambda[C_T+2, C_T+2] += lambda_depth_anchor / GN_LAMBDA

    # --- eye / brow ---
    Lambda[C_PSI:C_PSI+6,   C_PSI:C_PSI+6]   *= lambda_eye  / GN_LAMBDA
    Lambda[C_BROW:C_BROW+8, C_BROW:C_BROW+8] *= lambda_brow / GN_LAMBDA

    # --- nose: TPS smoothness (original) + absolute anchor (NEW) ---
    LtL = L_tps.T @ L_tps
    for vi in range(V_nose):
        for vj in range(V_nose):
            ri = C_NOSE + vi * 3
            rj = C_NOSE + vj * 3
            Lambda[ri:ri+3, rj:rj+3] = LtL[vi, vj] * np.eye(3)

    # NEW — anchor term added on top of the smoothness block above,
    # diagonal-only, doesn't touch the off-diagonal LtL coupling
    for vi in range(V_nose):
        ri = C_NOSE + vi * 3
        Lambda[ri:ri+3, ri:ri+3] += (lambda_nose_anchor / GN_LAMBDA) * np.eye(3)

    # --- mouth: only the two ill-conditioned corner-pull modes ---
    C_MOUTH = C_NOSE + 3 * V_nose
    Lambda[C_MOUTH+1, C_MOUTH+1] *= lambda_mouth_corner / GN_LAMBDA
    Lambda[C_MOUTH+2, C_MOUTH+2] *= lambda_mouth_corner / GN_LAMBDA

    # --- ear ---
    C_EAR = C_MOUTH + 4
    Lambda[C_EAR:C_EAR+4, C_EAR:C_EAR+4] *= lambda_ear / GN_LAMBDA

    return Lambda


#     return Lambda
# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def extract_eye_params(psi_eye: np.ndarray) -> dict:
    """Unpack ψ_eye = [a_L, θ_L^h, θ_L^v, a_R, θ_R^h, θ_R^v]."""
    p = psi_eye.flatten()
    return {
        "aperture_left":  float(p[0]),
        "gaze_h_left":    float(p[1]),
        "gaze_v_left":    float(p[2]),
        "aperture_right": float(p[3]),
        "gaze_h_right":   float(p[4]),
        "gaze_v_right":   float(p[5]),
    }


def compute_mouth_score(beta: np.ndarray,
                        A_exp_mouth_cols: np.ndarray = None) -> float:
    """
    β_mouth = ‖A_exp,mouth^T β‖ — compare to θ_open=0.15.
    """
    if A_exp_mouth_cols is not None:
        beta_mouth = A_exp_mouth_cols.T @ beta
    else:
        beta_mouth = beta[MOUTH_EXP_COLS]
    return float(np.linalg.norm(beta_mouth))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# def fit_source(model, K, landmarks, lm_vidx, I_src, init_state=None):
#     """
#     Phase-1 Stage-2 joint GN fit on source.

#     Full 3-term energy: E_appearance + E_geometric + E_regularise
#     Returns SourceState with all regional parameters populated.
#     """
#     V_nose = len(model.nose_idx)
   
   


#      for it in range(GN_MAX_ITER):
#     #     J, e  = _build_jacobian_and_residual(
#     #         model, state, K, landmarks, lm_vidx, I_src
#     #     )
#     #        # debugging 
#     #     S_dbg = model.full_shape(
#     #         state.alpha, state.beta,
#     #         state.psi_eye, state.gamma_brow, state.delta_nose,
#     #         psi_mouth=state.psi_mouth, gamma_ear=state.gamma_ear
#     #     )
#     #     _, Z_dbg = project_mesh(S_dbg, state.omega, state.t, K)
#     #     print(f"  [GN iter {it}] Z range=[{Z_dbg.min():.1f},{Z_dbg.max():.1f}]  "
#     #           f"n_valid={np.sum((Z_dbg>0)&~np.isnan(Z_dbg))}/{len(Z_dbg)}")
        

#     #     delta = _gn_update(J, e, GN_LAMBDA, Lambda)

#     #     N      = len(landmarks)
#     #     C      = I_src.shape[2] if I_src.ndim == 3 else 1
#     #     rows_geo = 2 * N
#     #     rows_app = C * N

#     #     # FIX: correct app_res slice
#     #     geo_res = np.linalg.norm(e[:rows_geo])
#     #     app_res = np.linalg.norm(e[rows_geo:rows_geo+rows_app])
#     #     print(f"  [GN iter {it}] ‖delta‖={np.linalg.norm(delta):.5f}  "
#     #           f"geo_res={geo_res:.3f}  app_res={app_res:.4f}")

#     #     state = SourceState.from_vec(state.to_vec() + delta, V_nose)

#     #     # Clip eye params to valid range
#     #     state.psi_eye[0] = np.clip(state.psi_eye[0], 0.0, 1.0)
#     #     state.psi_eye[3] = np.clip(state.psi_eye[3], 0.0, 1.0)
#     #     state.psi_eye[[1,2,4,5]] = np.clip(state.psi_eye[[1,2,4,5]], -0.5, 0.5)

#     #     #mouth parameters

#     #     state.psi_mouth = np.clip(state.psi_mouth, -1.0, 1.0)
#     #     print(f"  [GN iter {it}] psi_mouth={state.psi_mouth}  "
#     #                 f"norm={np.linalg.norm(state.psi_mouth):.4f}")  

#     #     if np.linalg.norm(delta) < GN_THRESH:
#     #         print(f"  [GN] converged at iteration {it}")
#     #         break
#         n_valid_before = np.sum((Z_dbg > 0) & ~np.isnan(Z_dbg))   # reuse the value you already compute above

#         delta = _gn_update(J, e, GN_LAMBDA, Lambda)


#         state = SourceState.from_vec(state.to_vec() + delta, V_nose)


#     return state

def fit_source(model, K, landmarks, lm_vidx, I_src, init_state=None):
    """
    Phase-1 Stage-2 joint GN fit on source.

    Full 3-term energy: E_appearance + E_geometric + E_regularise
    Returns SourceState with all regional parameters populated.
    """
    V_nose  = len(model.nose_idx)
    V_mouth = len(model.mouth_idx)
    V_ear   = len(model.ear_idx)
    dim_Θ   = C_NOSE + 3 * V_nose + 4 + 4

    if I_src.dtype != np.float64:
        I_src = I_src.astype(np.float64)
    if I_src.max() > 1.0:
        I_src = I_src / 255.0

    state = init_state or SourceState(
        alpha      = np.zeros(80),
        beta       = np.zeros(64),
        omega      = np.zeros(3),
        t          = np.array([0.0, 0.0, 500.0]),
        psi_eye    = np.zeros(6),
        gamma_brow = np.zeros(8),
        delta_nose = np.zeros(3 * V_nose),
    )

    L_tps  = build_tps_laplacian(model.nose_idx, model.faces)
    Lambda = _build_lambda(dim_Θ, V_nose, L_tps, N_lm=len(landmarks))

    history = []                      # per-iteration residuals (audit trail / demo curve)
    for it in range(GN_MAX_ITER):
        J, e = _build_jacobian_and_residual(
            model, state, K, landmarks, lm_vidx, I_src
        )

        # debugging — geometry health BEFORE this iteration's step
        S_dbg = model.full_shape(
            state.alpha, state.beta,
            state.psi_eye, state.gamma_brow, state.delta_nose,
            psi_mouth=state.psi_mouth, gamma_ear=state.gamma_ear
        )
        _, Z_dbg = project_mesh(S_dbg, state.omega, state.t, K)
        n_valid_before = np.sum((Z_dbg > 0) & ~np.isnan(Z_dbg))

        # track the residual BEFORE this step, for the backtracking check below
        resid_before = np.linalg.norm(e)
        if it == 0:
            resid_initial = resid_before   # NEW — anchor point, set once

        delta = _gn_update(J, e, GN_LAMBDA, Lambda)

        # Backtracking: halve the step until it neither pushes vertices behind
        # the camera NOR makes the residual substantially worse.
        n_valid_trial = n_valid_before
        resid_trial   = resid_before
        step_accepted = False
        for _ in range(5):
            trial_state = SourceState.from_vec(state.to_vec() + delta, V_nose)
            S_trial = model.full_shape(
                trial_state.alpha, trial_state.beta,
                trial_state.psi_eye, trial_state.gamma_brow, trial_state.delta_nose,
                psi_mouth=trial_state.psi_mouth, gamma_ear=trial_state.gamma_ear
            )
            _, Z_trial = project_mesh(S_trial, trial_state.omega, trial_state.t, K)
            n_valid_trial = np.sum((Z_trial > 0) & ~np.isnan(Z_trial))

            # NEW — also re-evaluate the residual at the trial state
            J_trial, e_trial = _build_jacobian_and_residual(
                model, trial_state, K, landmarks, lm_vidx, I_src
            )
            resid_trial = np.linalg.norm(e_trial)

            if n_valid_trial >= n_valid_before and resid_trial <= resid_before:
                step_accepted=True
            
                break
            delta *= 0.5
        else:
            print(f"  [GN iter {it}] step rejected — no safe delta found, holding position")
            delta *= 0.0
            n_valid_trial = n_valid_before
            resid_trial   = resid_before
            step_accpeted = False

        N        = len(landmarks)
        C        = I_src.shape[2] if I_src.ndim == 3 else 1
        rows_geo = 2 * N
        rows_app = C * N

        geo_res = np.linalg.norm(e[:rows_geo])
        app_res = np.linalg.norm(e[rows_geo:rows_geo+rows_app])
        print(f"  [GN iter {it}] ‖delta‖={np.linalg.norm(delta):.5f}  "
              f"geo_res={geo_res:.3f}  app_res={app_res:.4f}  "
              f"resid_before={resid_before:.1f}  resid_trial={resid_trial:.1f}")

        history.append({"iter": it, "residual_before": float(resid_before),
                        "residual_after": float(resid_trial), "step_norm": float(np.linalg.norm(delta)),
                        "accepted": bool(step_accepted)})

        state = SourceState.from_vec(state.to_vec() + delta, V_nose)

        # Clip eye params to valid range
        state.psi_eye[0] = np.clip(state.psi_eye[0], 0.0, 1.0)
        state.psi_eye[3] = np.clip(state.psi_eye[3], 0.0, 1.0)
        state.psi_eye[[1,2,4,5]] = np.clip(state.psi_eye[[1,2,4,5]], -0.5, 0.5)
        # FIX (revised): ±4.0 was tried first based on saturation evidence,
        # but that evidence was misread — with only 4 landmark points
        # constraining 8 free brow parameters, GN happily ran to whatever
        # wall it was given (norm climbed to ~10-11 once widened to ±4,
        # confirmed in logs: gamma_brow=[-4. -4. 2.2 -4. -4. 2.2 4. -4.]).
        # The real fix is the lambda_brow damping increase above (150->400)
        # so GN doesn't want to reach the wall in the first place. This
        # clip is now purely a safety net, pulled back in from ±4 to ±2.5
        # (a little more headroom than the original ±2, not the ±4 that
        # caused the runaway).
        gamma_brow_raw = state.gamma_brow.copy()
        state.gamma_brow = np.clip(state.gamma_brow, -2.5, 2.5)
        n_brow_clipped = int(np.sum(np.abs(gamma_brow_raw) > 2.5))

        # mouth / ear params
        # state.psi_mouth = np.clip(state.psi_mouth, -1.0, 1.0)
        # state.gamma_ear  = np.clip(state.gamma_ear,  -1.0, 1.0)

        # mouth / ear params
        # FIX: corner-pull modes (indices 1,2) were clipped to the same
        # ±1.0 as aperture/protrusion, but GN kept saturating at that wall
        # every iteration — the real mouth width needs more range than a
        # narrow clip allows. Widen just those two dims; leave aperture (0)
        # and protrusion (3) at ±1.0 since they weren't the ones saturating.
        state.psi_mouth[[0, 3]] = np.clip(state.psi_mouth[[0, 3]], -1.0, 1.0)
        state.psi_mouth[[1, 2]] = np.clip(state.psi_mouth[[1, 2]], -2.5, 2.5)
        #state.gamma_ear = np.clip(state.gamma_ear, -1.0, 1.0)
        state.gamma_ear[[0, 3]] = np.clip(state.gamma_ear[[0, 3]], -1.0, 1.0)
        state.gamma_ear[[1, 2]] = np.clip(state.gamma_ear[[1, 2]], -2.5, 2.5)

        print(f"  [GN iter {it}] psi_mouth={state.psi_mouth}  "
              f"norm={np.linalg.norm(state.psi_mouth):.4f}")
        print(f"  [GN iter {it}] gamma_brow={np.array2string(state.gamma_brow, precision=4)}  "
              f"norm={np.linalg.norm(state.gamma_brow):.4f}  "
              f"gamma_brow_clipped={n_brow_clipped}/8")

        # Only declare convergence if the step was genuinely small AND
        # the geometry is fully valid AND the residual didn't just get
        # frozen at a bad value (resid_trial should be finite/sane, not
        # the "held position" fallback repeating forever)
        if (step_accepted and np.linalg.norm(delta) < GN_THRESH
                and n_valid_trial == len(Z_dbg)):
            print(f"  [GN] converged at iteration {it}")
            break

    state.residual_history = history   # read by audit.report; not part of the parameter vector
    return state