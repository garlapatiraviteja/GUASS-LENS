"""
phase1/outputs/stage3_rasterise.py — UV Texture Bake (Stage 3)

Two sub-passes. Both use identical rasteriser mathematics.

Rasteriser Mathematics — Common To Both Sub-Passes:
  Back-face cull:  n_z = [(p_b-p_a) × (p_c-p_a)]_z < 0 → skip
  Near-plane cull: Z_i < z_near → skip
  Z-buffer:        z_interp(q) = λ₀Za + λ₁Zb + λ₂Zc
  Barycentric:     [λ₁,λ₂]ᵀ = [p_b-p_a | p_c-p_a]⁻¹(q-p_a), λ₀=1-λ₁-λ₂
  Persp-correct UV: u = (λ₀u_a/Za + λ₁u_b/Zb + λ₂u_c/Zc) /
                        (λ₀/Za   + λ₁/Zb   + λ₂/Zc)
  Retinex albedo:  log ρ = log I_src - GaussBlur(log I_src, σ=60)
                   ρ = exp(log ρ)

Sub-Pass A — Neutral Rasterisation:
  Mesh: S^neutral = S̄ + A_id·α_src + A_exp·β_src^neutral
  Camera: R_src, t_src from Stage 2 GN fit.
  Covers: full face except mouth interior U_mouth.

Sub-Pass B — Mouth Interior Rasterisation:
  Identify open-mouth frames: β_mouth = ‖A_exp,mouth^T β_src‖ > θ_open=0.15
  Re-fit β, ω, t on open-mouth frames (α fixed).
  Rasterise with open-mouth mesh → T_uv^src,mouth
  Merge → T_uv^src

Synthesis fallback (no open-mouth frames):
  T_uv^src,synth(u,v) = T_uv^src,neutral(u_lip - α_reflect(u-u_lip), v)
  α_reflect=0.6; w_mouth_max=0.3 (synthesis) vs 1.0 (real)

FIXES vs previous version:
1. subpass_A_neutral — was using neutral_shape(alpha) = S̄+A_id·α (β=0).
   Now uses base_shape(alpha, beta) = S̄+A_id·α+A_exp·β_src^neutral.
   Per PDF: Sub-pass A mesh uses β_src^neutral not zero.
2. refit_open_mouth — fitted omega was returned but ignored in
   subpass_B_mouth. Now uses state_fit.omega for rasterisation.
3. dim_Θ — updated to include psi_mouth/psi_ear once shape_model updated.
4. sigma_retinex — capped to 20% of smaller image dim to prevent
   over-blurring on small images (was fixed at 60).
"""

import numpy as np
from typing import Optional
from ..geometry import rodrigues, project_mesh
from ..shape_model import MorphableModel, SourceState, build_tps_laplacian
from .stage2_gn_source import fit_source, GN_MAX_ITER, GN_THRESH, _gn_update, _build_lambda


Z_NEAR        = 0.1
SIGMA_RETINEX = 60.0
THETA_OPEN    = 0.15
ALPHA_REFLECT = 0.6


# ---------------------------------------------------------------------------
# Core rasteriser
# ---------------------------------------------------------------------------

# def rasterise(
#         S:       np.ndarray,
#         faces:   np.ndarray,
#         uv:      np.ndarray,
#         omega:   np.ndarray,
#         t:       np.ndarray,
#         K:       np.ndarray,
#         I_src:   np.ndarray,
#         img_h:   int,
#         img_w:   int,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Rasterise mesh and bake Retinex albedo into UV texture map.

#     Returns
#     -------
#     T_uv     : (tex_h, tex_w, 3) float   Retinex albedo texture
#     vis_mask : (V,) bool                  which vertices were visible
#     """
#     tex_h, tex_w = img_h, img_w


#         # FIX: removed back-face cull (n_z <= 0 check). This mesh is an
#         # open, flat oval — not a closed 3D shape — so there's no real
#         # "back" to cull. The grid triangulation alternates winding per
#         # quad ([tl,tr,bl] then [tr,br,bl]), which under this projection's
#         # v-flip flipped n_z's sign every other triangle even on a fully
#         # frontal mesh, discarding 61% of source faces (44/72) and leaving
#         # T_uv_src only 15% populated pre-dilation. the renderer() has no
#         # equivalent check and rasterizes ~94% of target faces fine —
#         # matching that behavior here instead.


#     return T_uv, vis_mask

def rasterise(
        S:       np.ndarray,
        faces:   np.ndarray,
        uv:      np.ndarray,
        omega:   np.ndarray,
        t:       np.ndarray,
        K:       np.ndarray,
        I_src:   np.ndarray,
        img_h:   int,
        img_w:   int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterise mesh and bake Retinex albedo into UV texture map.

    Returns
    -------
    T_uv     : (tex_h, tex_w, 3) float   Retinex albedo texture
    vis_mask : (V,) bool                  which vertices were visible

    FIX: bake by GATHER (iterate texture-space texels, sample image) instead
    of SCATTER (iterate image pixels, write to texture). Scatter leaves
    texels unwritten wherever a triangle's UV-space footprint is larger
    than its image-space footprint (stretch) — with this mesh's UV derived
    directly from vertex X/Y and per-row oval tapering, that stretch varies
    row-to-row, producing systematic unwritten texture COLUMNS (visible as
    vertical streaks after dilation smears across them). This got much
    worse with the denser 625-vertex/1152-triangle mesh than the old
    72-triangle one. Gather guarantees every texel inside a triangle's UV
    footprint gets exactly one write, computed via the same barycentric
    weights mapped back into image space (valid since both spaces are
    affine functions of the same triangle).
    """
    tex_h, tex_w = img_h, img_w

    pi, Z = project_mesh(S, omega, t, K)
    R     = rodrigues(omega)

    # DEBUG: confirm the source mesh actually lands inside the source image
    _in_img = ((pi[:,0]>=0)&(pi[:,0]<img_w)&(pi[:,1]>=0)&(pi[:,1]<img_h)
               & np.isfinite(pi).all(axis=1))

    T_uv     = np.zeros((tex_h, tex_w, 3), dtype=np.float64)
    tex_z_buf = np.full((tex_h, tex_w), np.inf, dtype=np.float64)
    vis_mask = np.zeros(S.shape[0], dtype=bool)

    # FIX: was using single-scale Retinex (log_I - log_blur, exponentiated)
    # to extract an "albedo" — but SSR drives values toward exp(~0)=1.0
    # almost everywhere except sharp edges, which strips actual skin-tone
    # color along with illumination. Confirmed: rho_img mean was 0.9034
    # (near-white) when real skin reflectance albedo should be ~0.3-0.6 —
    # this is what caused every raw render this session to look
    # washed-out gray. Illumination is left to any downstream consumer —
    # doing illumination-removal here too was
    # redundant AND actively destroying source identity's color signal.
    # Just use the source image's actual pixel color directly.
    rho_img = np.clip(I_src, 0.0, 1.0)

    _n_faces_skipped_z    = 0
    _n_faces_skipped_bbox = 0
    _n_faces_skipped_det  = 0
    _n_faces_rasterized   = 0

    for face in faces:
        a, b, c = face
        Za, Zb, Zc = Z[a], Z[b], Z[c]
        if Za <= Z_NEAR or Zb <= Z_NEAR or Zc <= Z_NEAR:
            _n_faces_skipped_z += 1
            continue
        if np.isnan(Za) or np.isnan(Zb) or np.isnan(Zc):
            _n_faces_skipped_z += 1
            continue

        pa, pb, pc = pi[a], pi[b], pi[c]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
            _n_faces_skipped_z += 1
            continue

        # FIX: removed back-face cull (n_z <= 0 check). This mesh is an
        # open, flat oval — not a closed 3D shape — so there's no real
        # "back" to cull. The grid triangulation alternates winding per
        # quad ([tl,tr,bl] then [tr,br,bl]), which under this projection's
        # v-flip flipped n_z's sign every other triangle even on a fully
        # frontal mesh, discarding 61% of source faces (44/72) and leaving
        # T_uv_src only 15% populated pre-dilation. the renderer() has no
        # equivalent check and rasterizes ~94% of target faces fine —
        # matching that behavior here instead.

        ua, ub, uc = uv[a], uv[b], uv[c]
        _n_faces_rasterized += 1

        # ── GATHER: iterate texture-space texels within this triangle's
        # UV footprint, sample the source image via the same barycentric
        # weights mapped into image space. ──────────────────────────────
        tex_ua = ua * np.array([tex_w - 1, tex_h - 1])
        tex_ub = ub * np.array([tex_w - 1, tex_h - 1])
        tex_uc = uc * np.array([tex_w - 1, tex_h - 1])

        tu_min = max(0,        int(np.floor(min(tex_ua[0], tex_ub[0], tex_uc[0]))))
        tu_max = min(tex_w-1,  int(np.ceil( max(tex_ua[0], tex_ub[0], tex_uc[0]))))
        tv_min = max(0,        int(np.floor(min(tex_ua[1], tex_ub[1], tex_uc[1]))))
        tv_max = min(tex_h-1,  int(np.ceil( max(tex_ua[1], tex_ub[1], tex_uc[1]))))
        if tu_max < tu_min or tv_max < tv_min:
            _n_faces_skipped_bbox += 1
            continue

        M_tex = np.array([[tex_ub[0]-tex_ua[0], tex_uc[0]-tex_ua[0]],
                          [tex_ub[1]-tex_ua[1], tex_uc[1]-tex_ua[1]]], dtype=np.float64)
        det_tex = M_tex[0,0]*M_tex[1,1] - M_tex[0,1]*M_tex[1,0]
        if abs(det_tex) < 1e-10:
            _n_faces_skipped_det += 1
            continue
        M_tex_inv = np.array([[ M_tex[1,1], -M_tex[0,1]],
                              [-M_tex[1,0],  M_tex[0,0]]], dtype=np.float64) / det_tex

        for tv in range(tv_min, tv_max+1):
            for tu in range(tu_min, tu_max+1):
                q_tex = np.array([tu, tv], dtype=np.float64)
                lam12 = M_tex_inv @ (q_tex - tex_ua)
                lam1, lam2 = lam12
                lam0 = 1.0 - lam1 - lam2
                EDGE_EPS = 1e-6
                if lam0 < -EDGE_EPS or lam1 < -EDGE_EPS or lam2 < -EDGE_EPS:
                    continue

                # Same barycentric weights map back to image space AND
                # depth, since both are affine functions of the same
                # triangle vertices.
                px_interp = lam0*pa + lam1*pb + lam2*pc
                z_interp  = lam0*Za + lam1*Zb + lam2*Zc

                if z_interp >= tex_z_buf[tv, tu]:
                    continue

                pu_img, pv_img = px_interp
                if not (0 <= pu_img < img_w - 1 and 0 <= pv_img < img_h - 1):
                    continue

                # Bilinear sample from source image (sub-pixel accurate,
                # unlike the old nearest-pixel rho_img[pv, pu] lookup)
                x0, y0 = int(np.floor(pu_img)), int(np.floor(pv_img))
                fx, fy = pu_img - x0, pv_img - y0
                rho = (rho_img[y0,   x0]   * (1-fx) * (1-fy) +
                       rho_img[y0,   x0+1] * fx     * (1-fy) +
                       rho_img[y0+1, x0]   * (1-fx) * fy     +
                       rho_img[y0+1, x0+1] * fx     * fy)

                tex_z_buf[tv, tu] = z_interp
                T_uv[tv, tu] = rho
                vis_mask[a] = vis_mask[b] = vis_mask[c] = True

    # Fill empty UV texels via dilation
    T_uv = _dilate_texture(T_uv, iters=2000)

    return T_uv, vis_mask


# ---------------------------------------------------------------------------
# Sub-pass A — Neutral Rasterisation
# ---------------------------------------------------------------------------

def subpass_A_neutral(
        model:   MorphableModel,
        state:   SourceState,
        uv:      np.ndarray,
        K:       np.ndarray,
        I_src:   np.ndarray,
        img_h:   int,
        img_w:   int,
        mouth_uv_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Rasterise neutral mesh → T_uv^src,neutral (excludes U_mouth).

    FIX: was using neutral_shape(alpha) = S̄+A_id·α (β=0).
    Per PDF Sub-pass A: mesh = S̄ + A_id·α_src + A_exp·β_src^neutral.
    Now uses base_shape(alpha, beta) which includes β_src^neutral.

    Parameters
    ----------
    state : SourceState from Stage 2 — state.beta = β_src^neutral
    """
    # FIX: use base_shape(alpha, beta) not neutral_shape(alpha)
    #S_neutral = model.base_shape(state.alpha, state.beta)
    
    S_neutral = model.full_shape(
    state.alpha, state.beta,
    state.psi_eye, state.gamma_brow, state.delta_nose,
    psi_mouth=np.zeros(4),     # mouth intentionally stays neutral — handled separately by subpass B / synthesis_fallback
    gamma_ear=state.gamma_ear,
  )

    T_uv, vis_mask = rasterise(
        S_neutral, model.faces, uv,
        state.omega, state.t, K,
        I_src, img_h, img_w,
    )

    # Zero out mouth UV region — will be filled by subpass B or synthesis
    if mouth_uv_mask is not None:
        T_uv[mouth_uv_mask] = 0.0

    return T_uv


# ---------------------------------------------------------------------------
# Sub-pass B — Mouth Interior Rasterisation
# ---------------------------------------------------------------------------

def detect_open_mouth_frames(
        beta_list:      list,
        A_exp:          np.ndarray,
        mouth_exp_cols: np.ndarray,
        theta_open:     float = THETA_OPEN,
) -> list:
    """
    Identify frames where β_mouth = ‖A_exp,mouth^T β‖ > θ_open.
    Returns list of frame indices.
    """
    open_frames = []
    for i, beta in enumerate(beta_list):
        beta_mouth = beta[mouth_exp_cols]
        score = float(np.linalg.norm(beta_mouth))
        if score > theta_open:
            open_frames.append(i)
    return open_frames


def refit_open_mouth(
        model:      MorphableModel,
        alpha_src:  np.ndarray,
        K:          np.ndarray,
        I_src:      np.ndarray,
        landmarks:  np.ndarray,
        lm_vidx:    np.ndarray,
        beta_init:  np.ndarray,
        omega_init: np.ndarray,
        t_init:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GN re-fit on open-mouth frame holding α_src fixed.
    Returns (omega_fitted, t_fitted, beta_fitted).

    FIX: previously returned R_open but caller used original omega.
    Now returns omega directly so caller can use fitted pose.
    """
    from .stage2_gn_source import (
        C_NOSE, _gn_update, _build_lambda,
        _build_jacobian_and_residual,
    )

    V_nose = len(model.nose_idx)
    dim_Θ  = C_NOSE + 3 * V_nose+4+4

    state_init = SourceState(
        alpha      = alpha_src.copy(),
        beta       = beta_init.copy(),
        omega      = omega_init.copy(),
        t          = t_init.copy(),
        psi_eye    = np.zeros(6),
        gamma_brow = np.zeros(8),
        delta_nose = np.zeros(3 * V_nose),
    )

    from ..shape_model import build_tps_laplacian
    L_tps  = build_tps_laplacian(model.nose_idx, model.faces)
    Lambda = _build_lambda(dim_Θ, V_nose, L_tps)

    state = state_init
    for _ in range(GN_MAX_ITER):
        J, e = _build_jacobian_and_residual(model, state, K, landmarks, lm_vidx, I_src)
        # Freeze alpha — zero its columns
        J[:, :80]     = 0.0   # alpha

        resid_before = np.linalg.norm(e)
        delta = _gn_update(J, e, 0.01, Lambda)
        step_accepted = False
        for _ in range(5):
            trial_state = SourceState.from_vec(state.to_vec() + delta, V_nose)
            S_trial = model.full_shape(
                    trial_state.alpha, trial_state.beta, trial_state.psi_eye,
                    trial_state.gamma_brow, trial_state.delta_nose,
                    psi_mouth=trial_state.psi_mouth, gamma_ear=trial_state.gamma_ear
                )
            _, Z_trial = project_mesh(S_trial, trial_state.omega, trial_state.t, K)
            n_valid_trial = np.sum((Z_trial > 0) & ~np.isnan(Z_trial))
            _, e_trial = _build_jacobian_and_residual(model, trial_state, K, landmarks, lm_vidx, I_src)
            resid_trial = np.linalg.norm(e_trial)
            if resid_trial <= resid_before * 1.05:
                step_accepted = True
                break
            delta *= 0.5
        else:
            delta *= 0.0

        state = SourceState.from_vec(state.to_vec() + delta, V_nose)
        if step_accepted and np.linalg.norm(delta) < GN_THRESH:
            break

    # FIX: return fitted omega, not original
    return state.omega, state.t, state.beta


def subpass_B_mouth(
        model:         MorphableModel,
        alpha_src:     np.ndarray,
        uv:            np.ndarray,
        K:             np.ndarray,
        open_mouth_frames: list,
        mouth_uv_mask: np.ndarray,
        img_h:         int,
        img_w:         int,
) -> Optional[np.ndarray]:
    """
    Rasterise open-mouth mesh → T_uv^src,mouth.
    Returns None if no valid open-mouth frames.

    FIX: uses fitted omega_open from refit_open_mouth,
    not the original omega.
    """
    if not open_mouth_frames:
        return None

    T_acc   = np.zeros((img_h, img_w, 3), dtype=np.float64)
    n_valid = 0

    for frame_info in open_mouth_frames:
        I_src      = frame_info['I_src']
        landmarks  = frame_info['landmarks']
        lm_vidx    = frame_info['lm_vidx']
        beta_init  = frame_info['beta']
        omega_init = frame_info['omega']
        t_init     = frame_info['t']

        # FIX: use fitted omega/t/beta from refit
        omega_open, t_open, beta_open = refit_open_mouth(
            model, alpha_src, K, I_src,
            landmarks, lm_vidx,
            beta_init, omega_init, t_init,
        )

        # Build open-mouth mesh with fitted beta
        S_open = model.base_shape(alpha_src, beta_open)

        T_frame, vis = rasterise(
            S_open, model.faces, uv,
            omega_open, t_open, K,   # FIX: use fitted omega_open
            I_src, img_h, img_w,
        )
        T_acc += T_frame
        n_valid += 1

    if n_valid == 0:
        return None

    # Per-texel median approximation — mean over valid frames
    # (true median needs full frame stack; mean is practical approximation)
    T_mouth = T_acc / n_valid
    return T_mouth


# ---------------------------------------------------------------------------
# Synthesis fallback
# ---------------------------------------------------------------------------

def synthesis_fallback(
        T_neutral:     np.ndarray,
        u_lip:         int,
        mouth_uv_mask: np.ndarray,
        alpha_reflect: float = ALPHA_REFLECT,
) -> np.ndarray:
    """
    Reflect inner lip albedo inward along mesh normal:
      T^synth(u,v) = T^neutral(u_lip - α_reflect·(u-u_lip), v)

    Applied only to U_mouth pixels.
    """
    H, W    = T_neutral.shape[:2]
    T_synth = T_neutral.copy()

    ys, xs = np.where(mouth_uv_mask)
    for y, x in zip(ys, xs):
        u_src = int(round(u_lip - alpha_reflect * (x - u_lip)))
        u_src = np.clip(u_src, 0, W-1)
        T_synth[y, x] = T_neutral[y, u_src]

    return T_synth


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_texture(
        T_neutral:     np.ndarray,
        T_mouth:       Optional[np.ndarray],
        mouth_uv_mask: np.ndarray,
        T_synth:       Optional[np.ndarray],
) -> np.ndarray:
    """
    T_uv^src(u,v) =
      T^neutral   if (u,v) ∉ U_mouth
      T^mouth     if (u,v) ∈ U_mouth and valid frames exist
      T^synth     if (u,v) ∈ U_mouth and no valid frames
    """
    T_out = T_neutral.copy()
    if T_mouth is not None:
        T_out[mouth_uv_mask] = T_mouth[mouth_uv_mask]
    elif T_synth is not None:
        T_out[mouth_uv_mask] = T_synth[mouth_uv_mask]
    # else: stays as neutral (no zeros — mouth region already in neutral)
    return T_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_blur_3c(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on (H,W,3) image."""
    from scipy.ndimage import gaussian_filter
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return out


def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on (H,W) image."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


def _dilate_texture(T_uv: np.ndarray, iters: int = 500) -> np.ndarray:
    """Fill empty UV texels by dilating from neighbours.

    FIX: previous version averaged ALL 4 shifted neighbours
    UNCONDITIONALLY, including zero/still-empty ones — that's
    dilution (heat-diffusion-like blending toward zero), not true
    dilation. A pixel with only 1 real neighbour and 3 empty ones
    got filled to ~25% strength, landed under the 0.05 "now valid"
    threshold, and was never marked filled — so texels topologically
    far from real texture content could survive thousands of
    iterations still at zero. Confirmed: "POST-dilation T_uv
    min=0.0000" appeared in every run this session, and the renderer's
    large black triangular wedges traced directly back to render-mesh
    triangles sampling from these still-unfilled texels (not a
    rendering bug — legitimate sampling of a genuinely-black texel).

    Fixed to only average over ALREADY-VALID neighbours (excluding
    still-empty ones from the average). Each iteration now grows the
    filled region by exactly one full-strength ring, guaranteeing any
    region connected to real texture data gets fully filled within
    its pixel-distance — not dependent on region size/shape vs a
    fixed iteration budget.
    """
    T = T_uv.copy()
    valid = T.sum(axis=2) >= 0.05
    for _ in range(iters):
        mask_zero = ~valid
        if not mask_zero.any():
            break
        T_pad     = np.pad(T,     ((1,1),(1,1),(0,0)), mode='edge')
        valid_pad = np.pad(valid, ((1,1),(1,1)),       mode='edge')

        shifts       = [T_pad[:-2,1:-1],     T_pad[2:,1:-1],
                        T_pad[1:-1,:-2],     T_pad[1:-1,2:]]
        valid_shifts = [valid_pad[:-2,1:-1], valid_pad[2:,1:-1],
                        valid_pad[1:-1,:-2], valid_pad[1:-1,2:]]

        stacked_vals  = np.stack(shifts, axis=0)                      # (4,H,W,3)
        stacked_valid = np.stack(valid_shifts, axis=0).astype(np.float64)  # (4,H,W)

        neighbour_count = stacked_valid.sum(axis=0)                    # (H,W)
        weighted_sum    = np.sum(stacked_vals * stacked_valid[..., None], axis=0)
        safe_count      = np.where(neighbour_count > 0, neighbour_count, 1.0)
        filled_vals     = weighted_sum / safe_count[..., None]

        can_fill = mask_zero & (neighbour_count > 0)
        T[can_fill]     = filled_vals[can_fill]
        valid[can_fill] = True

    # FIX: main loop only fills texels with a path (via 4-connectivity)
    # back to real texture data. Confirmed via logs: mean/min improved
    # hugely after the main-loop fix (mean 0.31->0.59, eye region min
    # 0.0065->0.0196) but min stayed exactly 0.0000 — a small number of
    # texels are topologically disconnected islands (fully enclosed by
    # other empty texels, no valid neighbour ever), which no amount of
    # correct ring-growth can reach. Catch any stragglers here with a
    # global-mean fallback so zero pure-black texels can survive.
    still_zero = ~valid
    if still_zero.any() and valid.any():
        global_mean = T[valid].mean(axis=0)
        T[still_zero] = global_mean

    return T