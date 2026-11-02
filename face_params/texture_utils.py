"""
texture_utils.py — UV texture sampling helpers kept from the original project
(use them to read the baked texture T_uv_src).

  _uv_to_px          : (u,v) in [0,1] → pixel coords in a (tex_h, tex_w) texture
  _sample_bilinear_3c: bilinear sample of an (H,W,3) image
  _texture_gradient  : finite-difference gradient (dT/du, dT/dv) at a texel
Public aliases without the underscore are provided.
"""
import numpy as np


def _uv_to_px(u: float, v: float, tex_h: int, tex_w: int):
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    return u*(tex_w-1), v*(tex_h-1)


def _sample_bilinear_3c(img: np.ndarray, x: float, y: float,
                         H: int, W: int) -> np.ndarray:
    """Bilinear sample of (H,W,3) at (x,y)."""
    x0 = int(np.floor(x)); x1 = min(x0+1, W-1)
    y0 = int(np.floor(y)); y1 = min(y0+1, H-1)
    x0 = max(0, x0); y0 = max(0, y0)
    fx = x-x0; fy = y-y0
    return ((1-fx)*(1-fy)*img[y0,x0] + fx*(1-fy)*img[y0,x1] +
            (1-fx)*fy    *img[y1,x0] + fx*fy    *img[y1,x1])


def _texture_gradient(T: np.ndarray, x: float, y: float,
                       H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference gradient of texture T at (x,y). Returns (dT/du, dT/dv)."""
    eps  = 0.5
    T_xp = _sample_bilinear_3c(T, x+eps, y,   H, W)
    T_xm = _sample_bilinear_3c(T, x-eps, y,   H, W)
    T_yp = _sample_bilinear_3c(T, x,   y+eps, H, W)
    T_ym = _sample_bilinear_3c(T, x,   y-eps, H, W)
    return (T_xp-T_xm)/(2*eps), (T_yp-T_ym)/(2*eps)


uv_to_px           = _uv_to_px
sample_bilinear_3c = _sample_bilinear_3c
texture_gradient   = _texture_gradient
