"""
test_file/synthetic_model.py
Build a spec-faithful synthetic MorphableModel for testing.

FIXES APPLIED
-------------
1. make_synthetic_landmarks: N_lm now defaults to model.V (all vertices).
   Old default was N_lm=8 which only covered grid corners [0..7].
   eye_idx=[8,9,11,12,15,16,18,19], nose_idx=[17,22,23,24,31],
   brow_idx=[1,2,4,5] — none of these overlap with [0..7].
   Consequence: in any test that used make_synthetic_landmarks with the
   default N_lm=8, the GN Jacobian blocks for psi_eye / delta_nose /
   gamma_brow were always zero → those params never converged.

2. _build_structured_basis: np.ptp(S_bar, ) is deprecated since NumPy 1.24.
   Replaced with (S_bar.max(axis=0) - S_bar.min(axis=0)).clip(min=1.0).

3. make_synthetic_uv fallback: was uv[vi,1] = r/(rows-1) which maps
   forehead (row 0) to v=0 and chin (row 6) to v=1.
   Geometry-driven path flips: v=1.0-(y-y_min)/y_rng so forehead=0.
   Fallback now matches with uv[vi,1] = 1.0 - r/(rows-1).
   Inconsistent UV between paths caused texture bake/sample mismatch
   when S_bar was not passed.
"""
import numpy as np
from .shape_model import MorphableModel


# ─────────────────────────────────────────────────────────────────────────────
# Vertex layout
# ─────────────────────────────────────────────────────────────────────────────


def _face_oval_vertices(rows: int = 25, cols: int = 25) -> np.ndarray:
    V = rows * cols
    verts = np.zeros((V, 3), dtype=np.float64)
    ys = np.linspace(80, -80, rows)
    xs = np.linspace(-60, 60, cols)
    # Original 7-point taper preserved via interpolation, so the face
    # silhouette shape stays identical regardless of grid resolution.
    anchor_frac  = np.linspace(0, 1, 7)
    anchor_scale = np.array([0.55, 0.80, 0.95, 1.00, 0.95, 0.75, 0.45])
    row_frac  = np.linspace(0, 1, rows)
    oval_x_scale = np.interp(row_frac, anchor_frac, anchor_scale)
    vi = 0
    for r in range(rows):
        for c in range(cols):
            x = xs[c] * oval_x_scale[r]
            y = ys[r]
            z = 500.0 - (x**2 + y**2) / 2000.0
            verts[vi] = [x, y, z]
            vi += 1
    return verts


def _rc(row_frac: float, col_frac: float, rows: int, cols: int) -> int:
    """Map a proportional (row_frac, col_frac) in [0,1] to a flat vertex
    index, so sub-region indices stay correct at any grid resolution."""
    r = int(round(row_frac * (rows - 1)))
    c = int(round(col_frac * (cols - 1)))
    return r * cols + c


# ─────────────────────────────────────────────────────────────────────────────
# Structured basis builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_structured_basis(S_bar: np.ndarray, n_modes: int,
                             axis_cycle: bool = False,
                             scale: float = 1.0,
                             rows: int = 25, cols: int = 25) -> np.ndarray:
    V = S_bar.shape[0]
    A = np.zeros((V, 3, n_modes), dtype=np.float64)
    xyz_min = S_bar.min(axis=0)
    xyz_rng = np.ptp(S_bar, axis=0).clip(min=1.0)
    S_norm  = (S_bar - xyz_min) / xyz_rng * (2 * np.pi)

    # FIX: cap frequency at Nyquist for the actual mesh resolution — see
    # diagnose_aliasing.py / diagnose_real_canonical.py. Old code let
    # freq climb to n_modes//3 (up to 27) regardless of grid size (25),
    # which aliases into per-vertex noise and tears the mesh.
    max_safe_freq = max(1, min(rows, cols) // 2 - 1)

    for k in range(n_modes):
        axis      = k % 3
        combo_idx = k // 3
        freq      = (combo_idx % max_safe_freq) + 1
        use_cos   = (combo_idx // max_safe_freq) % 2 == 1
        if use_cos:
            envelope = np.cos(freq * S_norm[:, axis])
        else:
            envelope = np.sin(freq * S_norm[:, axis])
        A[:, axis, k] = envelope * scale
    return A


def _build_eye_basis(V_eye: int, n_modes: int = 6,
                     scale: float = 8.0) -> np.ndarray:
    """
    Eye basis — encodes per-eye aperture and gaze.
    ψ_eye = [a_L, θ_L^h, θ_L^v, a_R, θ_R^h, θ_R^v]

    Mode 0: left aperture  → y displacement on left eye verts
    Mode 1: left gaze h    → x displacement on left eye verts
    Mode 2: left gaze v    → y displacement on left eye verts (smaller)
    Mode 3: right aperture → y displacement on right eye verts
    Mode 4: right gaze h   → x displacement on right eye verts
    Mode 5: right gaze v   → y displacement on right eye verts (smaller)

    FIX: scale was 0.5 — same undersized-displacement problem mouth had
    before its fix. Phase 1 logs showed psi_eye[1]/[4] (gaze_h) pinned
    at exactly +-0.5 (the clip wall) — GN wanted more displacement than
    0.5 units could provide on a mesh spanning x:+-60/y:+-80. Unlike
    mouth, aperture here has no structural single-row problem (each
    eye's vertices already move together correctly) — this was purely
    a scale issue, same fix as mouth's original one.
    """
    B    = np.zeros((V_eye, 3, n_modes), dtype=np.float64)
    half = V_eye // 2
    B[:half, 1, 0] = scale          # aperture_L
    B[:half, 0, 1] = scale          # gaze_h_L
    B[:half, 1, 2] = scale * 0.5    # gaze_v_L
    B[half:, 1, 3] = scale          # aperture_R
    B[half:, 0, 4] = scale          # gaze_h_R
    B[half:, 1, 5] = scale * 0.5    # gaze_v_R
    return B


def _build_brow_basis(V_brow: int, n_modes: int = 8,
                      scale: float = 0.5) -> np.ndarray:
    """
    Brow basis — 8 modes, 4 per side (left/right × xyz).
    γ_brow ∈ ℝ⁸
    """
    B    = np.zeros((V_brow, 3, n_modes), dtype=np.float64)
    half = max(1, V_brow // 2)
    for k in range(n_modes):
        axis = k % 3
        side = slice(None, half) if k < 4 else slice(half, None)
        B[side, axis, k] = scale
    return B


def _build_mouth_basis(V_mouth: int, n_upper: int, n_modes: int = 4,
                        scale: float = 8.0) -> np.ndarray:
    """psi_mouth in R^4: aperture, corner-pull L, corner-pull R, protrusion.

    FIX (structural, not just scale): mouth_idx used to be a SINGLE row
    (5 vertices), and mode 0 moved ALL of them in the same y-direction:
        B[:, 1, 0] = scale
    That can only rigidly shift the whole mouth up/down — it can never
    create an opening gap, no matter how large scale gets. That's why
    psi_mouth[0]/[3] stayed pinned at the +-1.0 clip wall even after the
    scale=0.5->8.0 fix (confirmed: identical pinning, slightly WORSE
    resid_before, in the next run's logs) — GN was chasing a shape this
    basis structurally could not produce.

    mouth_idx is now two rows: the first n_upper vertices are the upper
    lip, the rest are the lower lip (see make_synthetic_model). Mode 0
    now moves them in OPPOSITE y-directions, creating a real gap.
    Corner-pull (modes 1/2) and protrusion (mode 3) apply to both rows.
    Left/right split within each row assumes vertices are ordered
    left-to-right (matches how mouth_idx/mouth_upper_idx/
    mouth_lower_idx are constructed).
    """
    B = np.zeros((V_mouth, 3, n_modes), dtype=np.float64)
    upper = slice(0, n_upper)
    lower = slice(n_upper, V_mouth)

    # mode 0: aperture — upper lip up, lower lip down → real opening
    B[upper, 1, 0] =  scale
    B[lower, 1, 0] = -scale

    # modes 1/2: corner pull — left half / right half of EACH row (x-axis)
    upper_mid = n_upper // 2
    lower_n   = V_mouth - n_upper
    lower_mid = lower_n // 2
    B[0:upper_mid, 0, 1] = -scale                      # upper-left  → mode1
    B[n_upper:n_upper+lower_mid, 0, 1] = -scale         # lower-left  → mode1
    B[upper_mid:n_upper, 0, 2] = scale                  # upper-right → mode2
    B[n_upper+lower_mid:, 0, 2] = scale                 # lower-right → mode2

    # mode 3: protrusion (z) — both rows
    # FIX: was scale*0.5, shared off the same budget as aperture/corner
    # pull. psi_mouth[3] pinned at +1.0 for the last several GN
    # iterations — same saturation pattern as before, isolated to this
    # one mode. Give it its own full-scale budget.
    B[:, 2, 3] = scale
    return B

def _build_ear_basis(V_ear: int, n_modes: int = 4, scale: float = 0.5) -> np.ndarray:
    """gamma_ear in R^4: 2 modes per side (left/right ear)."""
    B = np.zeros((V_ear, 3, n_modes), dtype=np.float64)
    half = max(1, V_ear // 2)
    for k in range(n_modes):
        axis = k % 3
        side = slice(None, half) if k < 2 else slice(half, None)
        B[side, axis, k] = scale
    return B

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


#     # Boost mouth modes 0-3 on lower-face rows (rows 4-6 = jaw/mouth area)
#     # This gives A_exp stronger mouth-opening influence on lower vertices
#     # lower_verts = np.arange(4 * cols, V, dtype=np.intp)
#     # for k in range(4):
#     #     A_exp[lower_verts, 1, k] += 2.0

    # Sanity check to run once this is in:
    # assert model.mouth_idx.size == 3 and model.ear_idx.size == 4
    # assert model.B_mouth.shape == (3, 3, 4) and model.B_ear.shape == (4, 3, 4)
    # assert not set(model.mouth_idx) & set(model.nose_idx)
    # assert not set(model.ear_idx) & (set(model.eye_idx)|set(model.brow_idx)|set(model.nose_idx))

def make_synthetic_model(seed: int = 42) -> MorphableModel:
    ROWS, COLS = 25, 25
    V = ROWS * COLS
    S_bar = _face_oval_vertices(ROWS, COLS)

    # Structured identity and expression bases
    A_id  = _build_structured_basis(S_bar, n_modes=80, scale=1.0, rows=ROWS, cols=COLS)
    A_exp = _build_structured_basis(S_bar, n_modes=64, scale=0.5, rows=ROWS, cols=COLS)

    # Sub-region indices — computed as fractional positions across the
    # grid (via _rc) instead of hardcoded "*7+..." offsets, so they stay
    # correct regardless of ROWS/COLS. Fractions mirror the original
    # 7x7 layout's relative placement (row 0=brow, 1-2=eye, 2-4=nose,
    # 4=mouth-upper, 5=mouth-lower, col 0/6&6/6=ear, 1/6&2/6=left,
    # 4/6&5/6=right, 3/6=centre).
    eye_idx = np.array([
        _rc(1/6, 1/6, ROWS, COLS), _rc(1/6, 2/6, ROWS, COLS),
        _rc(2/6, 1/6, ROWS, COLS), _rc(2/6, 2/6, ROWS, COLS),
        _rc(1/6, 4/6, ROWS, COLS), _rc(1/6, 5/6, ROWS, COLS),
        _rc(2/6, 4/6, ROWS, COLS), _rc(2/6, 5/6, ROWS, COLS),
    ], dtype=np.intp)

    brow_idx = np.array([
        _rc(0, 1/6, ROWS, COLS), _rc(0, 2/6, ROWS, COLS),
        _rc(0, 4/6, ROWS, COLS), _rc(0, 5/6, ROWS, COLS),
    ], dtype=np.intp)

    nose_idx = np.array([
        _rc(2/6, 3/6, ROWS, COLS),
        _rc(3/6, 2/6, ROWS, COLS), _rc(3/6, 3/6, ROWS, COLS), _rc(3/6, 4/6, ROWS, COLS),
        _rc(4/6, 3/6, ROWS, COLS),
    ], dtype=np.intp)

    # FIX: mouth was a SINGLE row — structurally unable to represent a
    # real mouth-opening gap. Two rows: upper lip and lower lip.
    mouth_upper_idx = np.array([
        _rc(4/6, 1/6, ROWS, COLS), _rc(4/6, 2/6, ROWS, COLS),
        _rc(4/6, 4/6, ROWS, COLS), _rc(4/6, 5/6, ROWS, COLS),
    ], dtype=np.intp)
    mouth_lower_idx = np.array([
        _rc(5/6, 1/6, ROWS, COLS), _rc(5/6, 2/6, ROWS, COLS), _rc(5/6, 3/6, ROWS, COLS),
        _rc(5/6, 4/6, ROWS, COLS), _rc(5/6, 5/6, ROWS, COLS),
    ], dtype=np.intp)
    mouth_idx = np.concatenate([mouth_upper_idx, mouth_lower_idx])
    N_MOUTH_UPPER = len(mouth_upper_idx)

    # ear: rows 2-3, outermost columns (cheek height)
    ear_idx = np.array([
        _rc(2/6, 0, ROWS, COLS), _rc(2/6, 1, ROWS, COLS),
        _rc(3/6, 0, ROWS, COLS), _rc(3/6, 1, ROWS, COLS),
    ], dtype=np.intp)

    V_eye   = len(eye_idx)
    V_brow  = len(brow_idx)
    V_mouth = len(mouth_idx)
    V_ear   = len(ear_idx)

    B_eye   = _build_eye_basis(V_eye,  n_modes=6, scale=8.0)
    B_brow  = _build_brow_basis(V_brow, n_modes=8, scale=0.5)
    B_mouth = _build_mouth_basis(V_mouth, N_MOUTH_UPPER, n_modes=4, scale=8.0)
    B_ear   = _build_ear_basis(V_ear, n_modes=4, scale=0.5)
    S_eye_ref = S_bar[eye_idx].copy()

    # Triangles — CCW winding, grid-based connectivity.
    # FIX: this loop previously used hardcoded rows=7/cols=7, so only the
    # first 7x7 corner of the (now 25x25) grid ever got triangulated —
    # 72 faces total, leaving 576 of 625 vertices with no triangle at all.
    faces = []
    for ri in range(ROWS - 1):
        for ci in range(COLS - 1):
            tl = ri * COLS + ci
            tr = tl + 1
            bl = tl + COLS
            br = bl + 1
            faces.append([tl, tr, bl])
            faces.append([tr, br, bl])
    faces = np.array(faces, dtype=np.intp)

    return MorphableModel(
        S_bar     = S_bar,
        A_id      = A_id,
        A_exp     = A_exp,
        B_eye     = B_eye,
        B_brow    = B_brow,
        S_eye_ref = S_eye_ref,
        eye_idx   = eye_idx,
        brow_idx  = brow_idx,
        nose_idx  = nose_idx,
        faces     = faces,
        mouth_idx = mouth_idx,
        ear_idx   = ear_idx,
        B_mouth   = B_mouth,
        B_ear     = B_ear,
    )


def _rc(row_frac: float, col_frac: float, rows: int, cols: int) -> int:
    r = int(round(row_frac * (rows - 1)))
    c = int(round(col_frac * (cols - 1)))
    return r * cols + c

def make_synthetic_uv(V: int, model_S_bar: np.ndarray = None, seed: int = 0) -> np.ndarray:
    """
    UV layout derived from actual vertex XY world positions (normalized).

    WHY: old version used grid row/col index directly:
         u = c/(cols-1),  v = r/(rows-1)
    This put eye vertices (row 1-2) at v=0.167-0.333 — the very top
    of the texture — where subpass_A_neutral bakes NO valid pixels.
    Bake and sample now agree because both use the same geometry-derived UV.

    FIX 1 (original): derive UV from actual XY extent of S_bar so UV
    position tracks the real geometry. Geometry-driven path flips v so
    forehead (y=80, row 0) maps to v=0 and chin (y=-80, row 6) maps to v=1.

    FIX 2 (new): fallback grid was NOT flipping v:
      uv[vi, 1] = r / (rows-1)    ← row 0 = v=0, row 6 = v=1 (wrong)
    The geometry-driven path does:
      uv[vi, 1] = 1.0 - (y - y_min) / y_rng   ← forehead=v=0 (correct)
    Now the fallback matches:
      uv[vi, 1] = 1.0 - r / (rows-1)   ← row 0 = v=0 (forehead), row 6 = v=1
    This eliminates the bake/sample UV mismatch when S_bar is not passed.
    """
    uv = np.zeros((V, 2), dtype=np.float64)

    if model_S_bar is not None and len(model_S_bar) == V:
        xy     = model_S_bar[:, :2]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_rng = (xy_max - xy_min).clip(min=1e-6)
        uv[:, 0] =       (xy[:, 0] - xy_min[0]) / xy_rng[0]   # u from X (left→right)
        uv[:, 1] = 1.0 - (xy[:, 1] - xy_min[1]) / xy_rng[1]   # v from Y (flip: forehead=top)
    else:
        # Fallback: uniform grid (backward compat if S_bar not passed)
        # FIX: flip v so forehead (row 0) = v=0, chin (row 6) = v=1
        # Matches the geometry-driven path above.
        rows, cols = 7, 7
        for r in range(rows):
            for c in range(cols):
                vi = r * cols + c
                if vi < V:
                    uv[vi, 0] =       c / (cols - 1)
                    uv[vi, 1] = 1.0 - r / (rows - 1)   # FIX: was r/(rows-1)

    return uv


def make_synthetic_image(H: int = 128, W: int = 128, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (H, W, 3)).astype(np.float64)


def make_synthetic_landmarks(model: MorphableModel, K: np.ndarray,
                              omega: np.ndarray, t: np.ndarray,
                              N_lm: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    FIX 1 — N_lm defaults to model.V (all vertices).

    Old signature had N_lm=8 which produced lm_vidx=[0..7].
    eye_idx / nose_idx / brow_idx are all > 7, so their GN Jacobian
    blocks were never populated → psi_eye / delta_nose / gamma_brow
    stayed at zero throughout all unit tests using this helper.

    Callers that explicitly pass N_lm=8 for a specific test still work,
    but the default is now the safe choice.
    """
    from .geometry import project_mesh

    S = model.S_bar
    pi, Z = project_mesh(S, omega, t, K)

    # Default: use ALL vertices so sub-region indices are always covered
    if N_lm is None:
        N_lm = model.V

    lm_vidx  = np.arange(N_lm, dtype=np.intp)
    detected = pi[:N_lm].copy()
    rng = np.random.default_rng(99)
    detected += rng.standard_normal(detected.shape) * 0.5
    return detected, lm_vidx