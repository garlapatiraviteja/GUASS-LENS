"""
geometry.py — Shared geometric primitives used across all phases.

All math follows the spec exactly:
  - Rodrigues rotation:   R(ω) = I + (sin‖ω‖/‖ω‖)ω̂ + ((1−cos‖ω‖)/‖ω‖²)ω̂²
  - Perspective proj:     π(S; Θ) = (1/Zᵢ) K(R(ω)Sᵢ + t),   Zᵢ = [R(ω)Sᵢ + t]_z
  - Jacobian blocks:      ∂π/∂α, ∂π/∂β, ∂π/∂ω, ∂π/∂t, ∂π/∂ψ_eye, ∂π/∂γ_brow, ∂π/∂δ_nose
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------

def rodrigues(omega: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector ω ∈ ℝ³ to 3×3 rotation matrix.

    R(ω) = I  +  (sin‖ω‖ / ‖ω‖) ω̂  +  ((1 − cos‖ω‖) / ‖ω‖²) ω̂²

    where ω̂ is the skew-symmetric cross-product matrix of ω.
    Handles ‖ω‖ ≈ 0 via Taylor expansion to avoid division by zero.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(omega)

    W = skew(omega)  # ω̂

    if theta < 1e-8:
        # First-order Taylor: R ≈ I + W
        return np.eye(3) + W + 0.5 * W @ W

    s = np.sin(theta) / theta
    c = (1.0 - np.cos(theta)) / (theta ** 2)
    return np.eye(3) + s * W + c * (W @ W)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix [v]× for v ∈ ℝ³."""
    v = v.flatten()
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Intrinsic camera matrix K  (pinhole)
# ---------------------------------------------------------------------------

def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build 3×3 intrinsic matrix K."""
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Perspective projection
# ---------------------------------------------------------------------------

# def project(S_i: np.ndarray, omega: np.ndarray, t: np.ndarray,
#             K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Project a single 3-D point Sᵢ ∈ ℝ³.

#     π(Sᵢ; Θ) = (1/Zᵢ) K (R(ω) Sᵢ + t)
#     Zᵢ       = [R(ω) Sᵢ + t]_z

#     Returns
#     -------
#     pi   : (2,) pixel coordinate [u, v]
#     Z_i  : scalar depth
#     """
#     R = rodrigues(omega)
#     p_cam = R @ S_i + t                    # 3-D camera-space point
#     Z_i = p_cam[2]
#     if Z_i <= 0:
#         return np.array([np.nan, np.nan]), Z_i
#     uv_h = K @ p_cam                       # homogeneous pixel
#     pi = uv_h[:2] / uv_h[2]               # = uv_h[:2] / Z_i  (since K[2,2]=1)
#     return pi, Z_i


def project(S_i: np.ndarray, omega: np.ndarray, t: np.ndarray, K: np.ndarray):
    R = rodrigues(omega)
    p_cam = R @ S_i + t
    Z_i = p_cam[2]
    if Z_i <= 0:
        return np.array([np.nan, np.nan]), Z_i
    uv_h = K @ p_cam
    pi = uv_h[:2] / uv_h[2]
    pi[1] = 2*K[1,2] - pi[1]     # v-flip fix
    return pi, Z_i


def project_mesh(S: np.ndarray, omega: np.ndarray, t: np.ndarray, K: np.ndarray):
    R = rodrigues(omega)
    P_cam = (R @ S.T).T + t
    Z = P_cam[:, 2]
    uv_h = (K @ P_cam.T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        pi = uv_h[:, :2] / Z[:, None]
    pi[:, 1] = 2*K[1,2] - pi[:, 1]   # v-flip fix
    pi[Z <= 0] = np.nan
    return pi, Z

# ---------------------------------------------------------------------------
# Jacobian blocks  ∂πᵢ/∂(·)
# ---------------------------------------------------------------------------

def dpi_dt(K: np.ndarray, Z_i: float, p_cam_i: np.ndarray) -> np.ndarray:
    """
    ∂πᵢ/∂t = (1/Zᵢ) K (I − pᵢ^{cam} eₓ^⊤ / Zᵢ)
              shape: (2, 3)

    pᵢ^{cam} = R(ω) Sᵢ + t
    """
    ez = np.array([0.0, 0.0, 1.0])
    M = np.eye(3) - np.outer(p_cam_i, ez) / Z_i   # 3×3
    J = (1.0 / Z_i) * K @ M                        # 3×3
    return J[:2, :]                                 # (2, 3)


def dRv_domega(omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Exact Jacobian of R(ω)·v w.r.t. ω  — returns (3, 3) matrix.
    Column k = d(R(ω)·v)/dω_k

    Derived from differentiating the Rodrigues formula term-by-term:
      R = I + (sinθ/θ)W + ((1−cosθ)/θ²)W²
    where W = skew(ω), θ = ‖ω‖.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    v     = np.asarray(v,     dtype=np.float64).reshape(3)
    theta = np.linalg.norm(omega)
    W  = skew(omega)

    if theta < 1e-8:
        # Taylor: R ≈ I + W  →  dRv/dω_k ≈ skew(e_k) · v
        J = np.zeros((3, 3))
        for k in range(3):
            ek = np.zeros(3); ek[k] = 1.0
            Wek = skew(ek)
            J[:, k] = (Wek + 0.5 * (Wek @ W + W @ Wek)) @ v
        return J

    W2 = W @ W
    s  = np.sin(theta);  c  = np.cos(theta)
    t2 = theta ** 2;     t3 = theta ** 3

    J = np.zeros((3, 3))
    for k in range(3):
        ek      = np.zeros(3); ek[k] = 1.0
        Wek     = skew(ek)                       # d skew(ω)/dω_k = skew(e_k)
        dth_dk  = omega[k] / theta               # d‖ω‖/dω_k
        dW2_dk  = Wek @ W + W @ Wek             # product rule on W²

        d_sot    = (c / theta - s / t2) * dth_dk          # d(sinθ/θ)/dω_k
        d_coterm = (s / t2  - 2*(1-c) / t3) * dth_dk     # d((1-cosθ)/θ²)/dω_k

        J[:, k] = (d_sot     * (W  @ v)
                 + (s/theta) * (Wek @ v)
                 + d_coterm  * (W2 @ v)
                 + ((1-c)/t2) * (dW2_dk @ v))
    return J


def dpi_domega(dpi_dt_i: np.ndarray, S_i: np.ndarray,
               omega: np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂ω = (∂πᵢ/∂t) · d(R(ω)·Sᵢ)/dω
    shape: (2, 3)

    Uses the exact Rodrigues derivative dRv_domega(ω, Sᵢ).
    omega must be supplied for the exact formula; if None falls back
    to the first-order approximation (ω ≈ 0) which equals -skew(Sᵢ).
    """
    if omega is None:
        # First-order approx (kept for backward compat where ω≈0)
        return dpi_dt_i @ (-skew(S_i))
    return dpi_dt_i @ dRv_domega(omega, S_i)


def dpi_dshape(K: np.ndarray, Z_i: float, R: np.ndarray,
               A_col: np.ndarray) -> np.ndarray:
    """
    Generic shape Jacobian block for one basis column A_col ∈ ℝ³:

    ∂πᵢ/∂θ_k = (1/Zᵢ) K R A_{col,i}
    shape: (2,)
    """
    return ((1.0 / Z_i) * K @ R @ A_col)[:2]


def build_dpi_dalpha(K: np.ndarray, Z_i: float, R: np.ndarray,
                     A_id_i: np.ndarray,
                     dpi_dt_i: np.ndarray = None,
                     p_cam_i:  np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂α = (∂πᵢ/∂t) · R · A_{id,i}    shape: (2, 80)
    Includes depth-correction term via dpi_dt.
    A_id_i: (3, 80)
    """
    if dpi_dt_i is None:
        # Fallback: approx without depth correction (legacy)
        return ((1.0 / Z_i) * K @ R @ A_id_i)[:2, :]
    return dpi_dt_i @ (R @ A_id_i)   # (2,3)@(3,80) = (2,80)


def build_dpi_dbeta(K: np.ndarray, Z_i: float, R: np.ndarray,
                    A_exp_i: np.ndarray,
                    dpi_dt_i: np.ndarray = None,
                    p_cam_i:  np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂β = (∂πᵢ/∂t) · R · A_{exp,i}   shape: (2, 64)
    A_exp_i: (3, 64)
    """
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R @ A_exp_i)[:2, :]
    return dpi_dt_i @ (R @ A_exp_i)   # (2,64)


def build_dpi_dpsi_eye(K: np.ndarray, Z_i: float, R: np.ndarray,
                       B_eye_i: np.ndarray,
                       dpi_dt_i: np.ndarray = None,
                       p_cam_i:  np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂ψ_eye = (∂πᵢ/∂t) · R · B_{eye,i}   shape: (2, 6)
    B_eye_i: (3, 6)
    """
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R @ B_eye_i)[:2, :]
    return dpi_dt_i @ (R @ B_eye_i)   # (2,6)


def build_dpi_dgamma_brow(K: np.ndarray, Z_i: float, R: np.ndarray,
                           B_brow_i: np.ndarray,
                           dpi_dt_i: np.ndarray = None,
                           p_cam_i:  np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂γ_brow = (∂πᵢ/∂t) · R · B_{brow,i}   shape: (2, 8)
    B_brow_i: (3, 8)
    """
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R @ B_brow_i)[:2, :]
    return dpi_dt_i @ (R @ B_brow_i)   # (2,8)

def build_dpi_dpsi_mouth(K, Z_i, R, B_mouth_i, dpi_dt_i=None, p_cam_i=None):
    """d(pi_i)/d(psi_mouth) = (d pi_i/dt) . R . B_mouth,i   shape: (2, 4)"""
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R @ B_mouth_i)[:2, :]
    return dpi_dt_i @ (R @ B_mouth_i)


def build_dpi_dgamma_ear(K, Z_i, R, B_ear_i, dpi_dt_i=None, p_cam_i=None):
    """d(pi_i)/d(gamma_ear) = (d pi_i/dt) . R . B_ear,i   shape: (2, 4)"""
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R @ B_ear_i)[:2, :]
    return dpi_dt_i @ (R @ B_ear_i)    


def build_dpi_ddelta_nose(K: np.ndarray, Z_i: float, R: np.ndarray,
                           V_nose: int,
                           dpi_dt_i: np.ndarray = None,
                           p_cam_i:  np.ndarray = None) -> np.ndarray:
    """
    ∂πᵢ/∂δ_nose = (∂πᵢ/∂t) · R   shape: (2, 3)

    Returns the 2×3 non-zero block for vertex i inside the nose submesh.
    δ_nose is a direct vertex offset so ∂S_i/∂δ_i = I₃, giving (∂π/∂t)·R.
    """
    if dpi_dt_i is None:
        return ((1.0 / Z_i) * K @ R)[:2, :]
    return dpi_dt_i @ R   # (2,3)
