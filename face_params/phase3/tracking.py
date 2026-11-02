"""
phase3/tracking.py — temporal tracking helpers (from the old blend_and_track.py).
Kept: gn_optical_flow (GN Lucas-Kanade per landmark), velocity_update (Newton state velocities).
Removed: mask building, Laplacian-pyramid blend.
"""
import numpy as np
from typing import Optional

FLOW_MAX_ITER   = 5
FLOW_THRESH     = 0.01    # px
WINDOW_SIZE     = 7       # optical-flow patch half-window


def gn_optical_flow(
        I_curr:   np.ndarray,
        I_prev:   np.ndarray,
        points:   np.ndarray,
        win_half: int   = WINDOW_SIZE // 2,
        max_iter: int   = FLOW_MAX_ITER,
        thresh:   float = FLOW_THRESH,
) -> np.ndarray:
    """
    GN optical flow per tracked point.

    Per PDF:
      Iter 1 (LK): J = ∇I_prev(x),         e = I_curr(x) - I_prev(x)
      Iter 2+ (GN): J = ∇I_prev(x+f_curr),  e = I_curr(x) - I_prev(x+f_curr)
      Δf = -(J^T J)^{-1} J^T e
      Terminate: ‖Δf‖ < 0.01px or 5 iters.

    FIX: window coords — cx/cy are floats, np.arange on floats
    causes rounding issues. Now rounds to int first.

    Returns f : (N_pts, 2)
    """
    N_pts = points.shape[0]
    H, W  = I_curr.shape[:2]
    f     = np.zeros((N_pts, 2), dtype=np.float64)

    I_c_gray = _to_gray(I_curr)
    I_p_gray = _to_gray(I_prev)

    for pt_idx in range(N_pts):
        cx, cy = points[pt_idx]
        f_pt   = np.zeros(2, dtype=np.float64)

        # FIX: round to int before building window
        cx_i = int(round(float(cx)))
        cy_i = int(round(float(cy)))

        xs = np.arange(cx_i - win_half, cx_i + win_half + 1)
        ys = np.arange(cy_i - win_half, cy_i + win_half + 1)
        xs = xs[(xs >= 0) & (xs < W)].astype(int)
        ys = ys[(ys >= 0) & (ys < H)].astype(int)

        if len(xs) == 0 or len(ys) == 0:
            continue

        xx, yy  = np.meshgrid(xs, ys)
        coords  = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float64)
        n_px    = coords.shape[0]

        for it in range(max_iter):
            J_flow = np.zeros((n_px, 2), dtype=np.float64)
            e_flow = np.zeros(n_px,      dtype=np.float64)

            if it == 0:
                # Iteration 1 — Lucas-Kanade init
                # J = ∇I_prev(x),  e = I_curr(x) - I_prev(x)
                for pi_idx, (px, py) in enumerate(coords):
                    Ix = _image_grad_x(I_p_gray, float(px), float(py), H, W)
                    Iy = _image_grad_y(I_p_gray, float(px), float(py), H, W)
                    J_flow[pi_idx] = [Ix, Iy]
                    e_flow[pi_idx] = (_sample_gray(I_c_gray, px, py, H, W)
                                     - _sample_gray(I_p_gray, px, py, H, W))
            else:
                # Subsequent — full GN: J = ∇I_prev(x + f_current)
                # e = I_curr(x) - I_prev(x + f_current)
                for pi_idx, (px, py) in enumerate(coords):
                    xf = px + f_pt[0]
                    yf = py + f_pt[1]
                    Ix = _image_grad_x(I_p_gray, xf, yf, H, W)
                    Iy = _image_grad_y(I_p_gray, xf, yf, H, W)
                    J_flow[pi_idx] = [Ix, Iy]
                    e_flow[pi_idx] = (_sample_gray(I_c_gray, px, py, H, W)
                                     - _sample_gray(I_p_gray, xf, yf, H, W))

            H2 = J_flow.T @ J_flow + 1e-6*np.eye(2)
            g2 = J_flow.T @ e_flow
            try:
                delta_f = -np.linalg.solve(H2, g2)
            except np.linalg.LinAlgError:
                break
            f_pt += delta_f

            if np.linalg.norm(delta_f) < thresh:
                break

        f[pt_idx] = f_pt

    return f


def velocity_update(
        q_curr: np.ndarray, q_prev: np.ndarray,
        t_curr: np.ndarray, t_prev: np.ndarray,
        dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    q̇_t = (q_t − q_{t-1}) / Δt
    ṫ_t = (t_t − t_{t-1}) / Δt
    """
    return (q_curr - q_prev) / dt, (t_curr - t_prev) / dt


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """
    Gaussian pyramid for single-channel image.
    Each level: blur previous level then downsample.
    """
    pyr = [img]
    for _ in range(levels - 1):
        # Blur previous level (pyr[-1]), then downsample
        pyr.append(_downsample(_gaussian_blur_2d(pyr[-1], sigma=1.0)))
    return pyr


def _downsample(img: np.ndarray) -> np.ndarray:
    """2× downsample single-channel."""
    return img[::2, ::2]


def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur on 2-D array."""
    radius = max(1, int(round(3.0 * sigma)))
    x      = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-x**2 / (2.0 * sigma**2))
    kernel /= kernel.sum()

    H, W = img.shape
    r    = len(kernel) // 2

    # Horizontal
    padh = np.pad(img, r, mode='reflect')
    out  = np.zeros_like(img)
    for i, kv in enumerate(kernel):
        out += kv * padh[r:-r, i:i+W]

    # Vertical
    padv = np.pad(out, r, mode='reflect')
    out2 = np.zeros_like(img)
    for i, kv in enumerate(kernel):
        out2 += kv * padv[i:i+H, r:-r]
    return out2


def _to_gray(img: np.ndarray) -> np.ndarray:
    """(H,W,3) → (H,W) luminance."""
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]


def _sample_gray(img: np.ndarray, x: float, y: float, H: int, W: int) -> float:
    x0 = int(np.clip(np.floor(x), 0, W-1))
    x1 = int(np.clip(x0+1,       0, W-1))
    y0 = int(np.clip(np.floor(y), 0, H-1))
    y1 = int(np.clip(y0+1,       0, H-1))
    fx = x-x0; fy = y-y0
    return float(
        (1-fx)*(1-fy)*img[y0,x0] + fx*(1-fy)*img[y0,x1]
        + (1-fx)*fy*img[y1,x0]   + fx*fy*img[y1,x1]
    )


def _image_grad_x(img: np.ndarray, x: float, y: float, H: int, W: int) -> float:
    """∂I/∂x via central difference."""
    return (_sample_gray(img, x+1, y, H, W) - _sample_gray(img, x-1, y, H, W)) / 2.0


def _image_grad_y(img: np.ndarray, x: float, y: float, H: int, W: int) -> float:
    """∂I/∂y via central difference."""
    return (_sample_gray(img, x, y+1, H, W) - _sample_gray(img, x, y-1, H, W)) / 2.0
