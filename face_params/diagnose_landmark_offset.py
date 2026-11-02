"""
diagnose_landmark_offset.py

Purpose: make the ~75px brow offset visible directly on the target image,
instead of inferring it from the [DEBUG align] numbers. This isolates two
competing hypotheses:

  (A) The base pose/identity (omega, t, beta) already places the forehead
      region wrong, BEFORE gamma_brow ever gets a chance to correct it —
      in which case the fix belongs in pose/beta fitting, not in brow's
      own damping/clip (which we already tuned without moving the number).

  (B) gamma_brow's own basis just can't reach far enough to close the gap
      even with pose/beta correct — a genuine brow-model capacity limit.

  (C) The detected 2D landmark itself is wrong (e.g. hair/bangs occluding
      the eyebrow confusing the detector) — in which case no amount of
      3D-side tuning fixes it, because the target being fit against is
      itself incorrect.

How to read the output image:
  - GREEN dots  = detected 2D landmarks (frame.landmarks) — what the
                  detector thinks the brow/eye/nose/mouth/ear points are.
  - RED dots    = projected 3D mesh points using the FITTED state
                  (tgt_state, with whatever gamma_brow GN converged to).
  - YELLOW dots = projected brow points with gamma_brow forced to ZERO
                  (pose/beta/everything else held at the fitted values).
                  If yellow sits almost exactly where red sits, gamma_brow
                  isn't doing meaningful work either way, and the gap
                  between green and red/yellow is a pose/beta bias (A) or
                  a bad detection (C), NOT a brow-basis capacity limit (B).
                  If yellow is far from red but red still doesn't reach
                  green, gamma_brow is working but doesn't have enough
                  range (B).
  - Brow points are drawn larger (radius 7) than the other regions
    (radius 3) so they stand out.

Usage — add near the top of pipeline.py:
    from diagnose_landmark_offset import draw_landmark_overlay
    import os, tempfile

Then right after the fit_target_frame(...) call (pipeline.py ~line 439),
add:
    debug_dir = os.path.join(tempfile.gettempdir(), "face_params_debug")
    os.makedirs(debug_dir, exist_ok=True)
    draw_landmark_overlay(
        I_tgt=frame.I_tgt, landmarks=frame.landmarks, model=model,
        tgt_state=tgt_state, alpha_tgt=session.alpha_tgt, K=K,
        out_path=os.path.join(debug_dir, f"debug_landmarks_frame{state.frame_idx}.png"),
    )

This writes a PNG you can open directly and eyeball the offset.
"""
import numpy as np
from PIL import Image, ImageDraw

from .geometry import project_mesh



def _to_uint8_image(I):
    """I_tgt in this codebase is float [0,1] HxWx3 — convert for PIL."""
    arr = np.clip(I, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _draw_dots(draw, points, color, radius):
    for (x, y) in points:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=color, width=2)


def draw_landmark_overlay(I_tgt, landmarks, model, tgt_state, alpha_tgt, K,
                           out_path, region_idx_groups=None):
    """
    Save a PNG overlaying:
      - detected 2D landmarks (green)
      - projected fitted-state mesh points (red)
      - projected brow points with gamma_brow=0 (yellow), to isolate
        whether pose/beta already carries the offset independent of any
        brow-specific deformation.

    Parameters mirror what's already in scope right after the
    fit_target_frame(...) call in pipeline.py — pass them through as-is,
    no extra computation needed on the caller's side.
    """
    # GUARD: this fires when the caller passes frame.I_tgt / frame.landmarks
    # before a target photo has actually been captured/loaded — e.g. this
    # diagnostic call was placed too early in the flow (before the capture
    # step populates `frame`), or it's being called in a session where no
    # image was ever loaded into `frame`. Fail with a clear message instead
    # of a cryptic AttributeError/TypeError deep inside PIL/numpy.
    if I_tgt is None:
        raise ValueError(
            "draw_landmark_overlay: I_tgt is None — no target image is "
            "loaded yet. Make sure a photo has actually been captured/"
            "uploaded and `frame.I_tgt` is populated BEFORE this call. If "
            "you're calling this right after fit_target_frame(...) in "
            "pipeline.py, check that the capture/upload step earlier in "
            "the flow actually ran and didn't silently skip (e.g. the "
            "person needs to take or upload the photo first)."
        )
    if landmarks is None:
        raise ValueError(
            "draw_landmark_overlay: landmarks is None — landmark detection "
            "hasn't produced anything for this frame yet (frame.landmarks "
            "is None). This usually means detection was skipped or failed "
            "silently upstream — check whatever populates frame.landmarks "
            "ran on an actual captured photo, not a placeholder/empty frame."
        )

    img = Image.fromarray(_to_uint8_image(I_tgt))
    draw = ImageDraw.Draw(img)

    if region_idx_groups is None:
        region_idx_groups = {
            "eye":   (model.eye_idx,   3),
            "brow":  (model.brow_idx,  7),   # bigger — the region under test
            "nose":  (model.nose_idx,  3),
        }
        if hasattr(model, "mouth_idx"):
            region_idx_groups["mouth"] = (model.mouth_idx, 3)
        if hasattr(model, "ear_idx"):
            region_idx_groups["ear"] = (model.ear_idx, 3)

    # --- RED: projected mesh at the fitted state (whatever GN converged to) ---
    S_fitted = model.full_shape(
        alpha_tgt, tgt_state.beta,
        tgt_state.psi_eye, tgt_state.gamma_brow,
        np.zeros(3 * len(model.nose_idx)),
        psi_mouth=tgt_state.psi_mouth, gamma_ear=tgt_state.gamma_ear,
    )
    pi_fitted, _ = project_mesh(S_fitted, tgt_state.omega, tgt_state.t, K)

    # --- YELLOW: same, but gamma_brow forced to zero ---
    S_brow0 = model.full_shape(
        alpha_tgt, tgt_state.beta,
        tgt_state.psi_eye, np.zeros_like(tgt_state.gamma_brow),
        np.zeros(3 * len(model.nose_idx)),
        psi_mouth=tgt_state.psi_mouth, gamma_ear=tgt_state.gamma_ear,
    )
    pi_brow0, _ = project_mesh(S_brow0, tgt_state.omega, tgt_state.t, K)

    for name, (idx, radius) in region_idx_groups.items():
        _draw_dots(draw, landmarks[idx],   (0, 255, 0),   radius)   # detected
        _draw_dots(draw, pi_fitted[idx],   (255, 0, 0),   radius)   # fitted
        if name == "brow":
            _draw_dots(draw, pi_brow0[idx], (255, 255, 0), radius)  # gamma_brow=0

    # Print the actual offsets so it's not just eyeballing
    brow_idx = model.brow_idx
    err_fitted = np.linalg.norm(pi_fitted[brow_idx] - landmarks[brow_idx], axis=1)
    err_brow0  = np.linalg.norm(pi_brow0[brow_idx]  - landmarks[brow_idx], axis=1)
    print(f"[diagnose] brow err WITH fitted gamma_brow: "
          f"{err_fitted.mean():.1f}px (per-point: {np.round(err_fitted, 1)})")
    print(f"[diagnose] brow err WITH gamma_brow FORCED TO ZERO: "
          f"{err_brow0.mean():.1f}px (per-point: {np.round(err_brow0, 1)})")
    print(f"[diagnose] gamma_brow's own contribution: "
          f"{(err_brow0.mean() - err_fitted.mean()):.1f}px of the gap closed "
          f"(out of {err_brow0.mean():.1f}px total)")
    if err_fitted.mean() > 0.7 * err_brow0.mean():
        print("[diagnose] gamma_brow barely moved the error -> look at "
              "pose/beta (hypothesis A) or detector quality (hypothesis C), "
              "not brow basis capacity.")
    else:
        print("[diagnose] gamma_brow closed most of the gap but still falls "
              "short -> may be a genuine brow-basis capacity limit "
              "(hypothesis B).")

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"[diagnose] saved overlay to {out_path}")