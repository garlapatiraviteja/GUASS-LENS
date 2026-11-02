"""
run.py — Face-Parameter Extraction: entry point (CLI)

  python run.py --image face.png
  python run.py --image face.png --out_dir out/
  python run.py --image face.png --model bfm.npz
  python run.py --image small.png --no_strict        # don't stop on a Stage-1 rejection

One face image in. Pipeline:
  Phase 1  S2 Gauss-Newton fit → S1 validation → S3 UV texture bake → S4 identity
  Phase 2  pose / eye / brow / nose / mouth / ear fit + quality gate
  Phase 3  motion-state update

Outputs (in --out_dir, default ./extract_out):
  phase1_params.npz   α β ω t ψ_eye γ_brow δ_nose ψ_mouth γ_ear from the Phase-1 fit
  phase2_params.npz   the same parameters after the Phase-2 frame fit (+ quality score)
  texture.png         baked UV colour texture (Stage 3)
  overlay.png         projected model landmarks drawn on the image (coloured by region)
  summary.json        head pose, eyes, mouth openness, quality, per-region px error

The model loader, symmetry pairs and session builder below are the original project code.
"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_params.pipeline                     import FaceParamPipeline, FrameData
from face_params.shape_model                  import MorphableModel
from face_params.geometry                     import make_K
from face_params.face_recogniser              import FaceRecogniser
from face_params.phase1.stage4_target_neutral import TargetFrameData
from face_params.phase1.stage2_gn_source      import (
    extract_eye_params, compute_mouth_score, THETA_OPEN
)
from face_params.io_utils import *
from face_params.synthetic_model    import make_synthetic_model, make_synthetic_uv


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path, img_h, img_w):
    """
    Load 3DMM model and build camera intrinsics for given image dims.
    FIX: mouth_mask is empty for synthetic model — derived from
    mouth vertex UV positions when real mouth_idx available.
    """
    fx = fy = max(img_h, img_w) * 1.2
    K  = make_K(fx, fy, img_w / 2.0, img_h / 2.0)

    if path is not None and os.path.isfile(path):
        print(f"[model] Loading 3DMM from {path}")
        d     = np.load(path, allow_pickle=False)
        model = MorphableModel(
            S_bar     = d["S_bar"],
            A_id      = d["A_id"],
            A_exp     = d["A_exp"],
            B_eye     = d["B_eye"],
            B_brow    = d["B_brow"],
            S_eye_ref = d["S_eye_ref"],
            eye_idx   = d["eye_idx"].astype(np.intp),
            brow_idx  = d["brow_idx"].astype(np.intp),
            nose_idx  = d["nose_idx"].astype(np.intp),
            faces     = d["faces"].astype(np.intp),

             # NEW — .get-style fallback so OLDER .npz files without these keys don't crash:
            ear_idx   = d["ear_idx"].astype(np.intp)   if "ear_idx"   in d else np.array([], dtype=np.intp),
            mouth_idx = d["mouth_idx"].astype(np.intp) if "mouth_idx" in d else np.array([], dtype=np.intp),
            B_ear     = d["B_ear"]   if "B_ear"   in d else np.zeros((0,3,4)),
            B_mouth   = d["B_mouth"] if "B_mouth" in d else np.zeros((0,3,4)),
            
        )
        uv         = d["uv"]
        K          = d.get("K", K)
        mouth_mask = d.get("mouth_mask",
                           np.zeros((img_h, img_w), dtype=bool)).astype(bool)
    else:
        if path:
            print(f"[model] {path} not found — using synthetic model")
        else:
            print("[model] No 3DMM supplied — using synthetic model")
        model      = make_synthetic_model()
        uv         = make_synthetic_uv(model.V, model_S_bar=model.S_bar)
        # FIX: mouth_mask is all-False for synthetic model.
        # Mouth UV region is derived from mouth vertex UVs in pipeline.
        # Never set a hardcoded pixel region — it corrupts T_uv_src.
        mouth_mask = np.zeros((img_h, img_w), dtype=bool)

    return model, uv, K, mouth_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# def _sharp_ref(H: int, W: int) -> np.ndarray:
#     """
#     Synthetic sharp reference image for s_blur denominator.

#     FIX: was returning grayscale of source image:
#       gray = np.mean(I_src, axis=2)
#     This made s_blur = ||∇²I_src|| / ||∇²I_src|| = 1.0 always.

#     Per PDF: I_ref must be a genuinely sharp reference.
#     This synthetic pattern has maximum high-frequency content —
#     any real image will score < 1.0 relative to it.
#     """
#     x  = np.linspace(0, 4*np.pi, W)
#     y  = np.linspace(0, 4*np.pi, H)
#     XX, YY = np.meshgrid(x, y)
#     return (np.sin(10*XX) * np.cos(10*YY) * 0.5 + 0.5)


def _sharp_ref(H: int, W: int) -> np.ndarray:
    CYCLES_PER_PIXEL = 0.25   # tune this constant, not the span
    x = np.arange(W) * CYCLES_PER_PIXEL
    y = np.arange(H) * CYCLES_PER_PIXEL
    XX, YY = np.meshgrid(x, y)
    return np.sin(2*np.pi*XX) * np.cos(2*np.pi*YY) * 0.5 + 0.5


def _make_lm_vidx(model) -> np.ndarray:
    """Use ALL model.V vertices as landmark indices."""
    return np.arange(model.V, dtype=np.intp)


def _make_sym_pairs(lm_vidx, model):
    """
    Build anatomically correct left/right symmetry index pairs
    from 3D geometry — works for any model (synthetic or real BFM/FLAME).
    """
    S = model.S_bar[lm_vidx]
    N = len(lm_vidx)

    x_extent   = float(np.max(np.abs(S[:,0]))) if N else 0.0
    midline_tol = max(1e-6, 0.02 * x_extent)
    is_midline  = np.abs(S[:,0]) <= midline_tol

    left_list, right_list = [], []
    used = set(int(i) for i in np.where(is_midline)[0])

    for i in range(N):
        if i in used:
            continue
        mirror = S[i].copy()
        mirror[0] *= -1.0
        dists = np.linalg.norm(S - mirror, axis=1)
        dists[i] = np.inf
        for u in used:
            dists[u] = np.inf
        same_side = (np.sign(S[:,0]) == np.sign(S[i,0]))
        dists[same_side] = np.inf

        j = int(np.argmin(dists))
        if dists[j] < np.inf:
            if S[i,0] <= S[j,0]:
                left_list.append(i); right_list.append(j)
            else:
                left_list.append(j); right_list.append(i)
            used.add(i); used.add(j)

    n = min(len(left_list), len(right_list))
    left_sym  = np.array(left_list[:n], dtype=np.intp)
    right_sym = np.array(right_list[:n], dtype=np.intp)

    if n == 0:
        half      = N // 2
        left_sym  = np.arange(0,    half, dtype=np.intp)
        right_sym = np.arange(half, half*2, dtype=np.intp)

    return left_sym, right_sym


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------

def build_session(pipe, recogniser, I_src, tgt_images, model, lm_vidx,
                  strict_validation=True):
    """
    Initialise Phase 1 session.

    FIX: I_ref_gray removed as argument — pipeline.py generates it
    internally using _sharp_ref(H,W). Prevents passing wrong value.
    FIX: target frames use per-image t_init from recogniser (not src t_init).
    FIX: pads to 10 frames with ±0.5px noise (not pixel-identical copies).
    """
    H, W = I_src.shape[:2]

    # Source recognition — GN fit
    src_r = recogniser.process(I_src, lm_vidx)

    # Symmetry pairs from 3D geometry
    left_sym, right_sym = _make_sym_pairs(lm_vidx, model)

    # Target frames — each uses its own t_init
    tgt_frames       = []
    recognised_tgts  = []
    for img in tgt_images[:50]:
        r = recogniser.process(img, lm_vidx)
        recognised_tgts.append(r)
        tgt_frames.append(TargetFrameData(
            landmarks = r.landmarks,
            lm_vidx   = lm_vidx,
            omega     = np.zeros(3),
            #t         = r.t_init,   # per-image t_init — not src_r.t_init

            t=r.t_fitted,
        ))

    # Pad to at least 10 frames with small landmark noise
    # Breaks rank-degeneracy in Stage 4 joint Jacobian
    rng_pad  = np.random.default_rng(seed=0)
    base_frames = list(tgt_frames)
    n_base   = max(1, len(base_frames))
    while len(tgt_frames) < 10:
        src_frame = base_frames[len(tgt_frames) % n_base]
        noise     = rng_pad.uniform(-0.5, 0.5, src_frame.landmarks.shape)
        tgt_frames.append(TargetFrameData(
            landmarks = src_frame.landmarks + noise,
            lm_vidx   = src_frame.lm_vidx,
            omega     = src_frame.omega.copy(),
            t         = src_frame.t.copy(),
        ))

    # FIX: I_ref_gray NOT passed — pipeline.py generates it internally
    session = pipe.init_session(
        I_src            = I_src.astype(np.float64),
        src_landmarks    = src_r.landmarks,
        src_lm_vidx      = lm_vidx,
        src_lm_left_idx  = left_sym,
        src_lm_right_idx = right_sym,
        tgt_frames       = tgt_frames,
        src_init_state   = getattr(src_r, 'fitted_state', None),
        K_tgt            = (recognised_tgts[0].K
                            if recognised_tgts and hasattr(recognised_tgts[0], 'K')
                            else None),
        src_face_box     = getattr(src_r, 'box', None),
        skip_refit       = True,
        strict_validation = strict_validation,
        lm_sigma_px       = getattr(src_r, 'lm_sigma_px', None),
    )
    session._recognised_tgts = recognised_tgts
    return session, src_r


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExtractResult:
    """Per-frame extraction result. Attribute names are what app.py's display code reads."""
    def __init__(self, frame_result, src_state, I_overlay, texture):
        fr = frame_result
        self.params         = fr.params
        self.quality        = fr.quality
        self.decision       = fr.decision
        self.delta_beta     = fr.delta_beta
        self.region_error   = fr.region_error
        self.eye_params     = extract_eye_params(src_state.psi_eye)
        self.mouth_score    = compute_mouth_score(src_state.beta)
        self.mouth_open     = self.mouth_score > THETA_OPEN
        self.src_state      = src_state
        self.tgt_eye_params = extract_eye_params(fr.params.psi_eye)
        self.tgt_mouth_score = fr.params.mouth_openness
        self.I_overlay      = I_overlay    # frame with projected landmarks drawn
        self.texture        = texture      # baked UV albedo (S3)

    def __repr__(self):
        return (f"ExtractResult(q={self.quality:.3f} | {self.decision} | "
                f"mouth={'open' if self.mouth_open else 'closed'} "
                f"({self.mouth_score:.3f}) | "
                f"eye_L={self.eye_params['aperture_left']:.2f} "
                f"eye_R={self.eye_params['aperture_right']:.2f})")


_REGION_COLOURS = {"eye": (0, 200, 255), "brow": (255, 200, 0), "nose": (0, 255, 120),
                   "mouth": (255, 60, 120), "ear": (180, 120, 255)}


def draw_overlay(I, params):
    """Draw each region's projected vertices on the image (float [0,1] RGB)."""
    out = np.clip(np.asarray(I, dtype=np.float64), 0, 1).copy()
    H, W = out.shape[:2]
    r = max(2, int(max(H, W) / 200))
    for name, pts in params.region_points_2d.items():
        col = np.array(_REGION_COLOURS.get(name, (255, 255, 255))) / 255.0
        for x, y in pts:
            if np.isfinite(x) and np.isfinite(y):
                xi, yi = int(round(x)), int(round(y))
                out[max(0, yi-r):min(H, yi+r+1), max(0, xi-r):min(W, xi+r+1)] = col
    return out


# ---------------------------------------------------------------------------
# Per-frame processor
# ---------------------------------------------------------------------------

def process_frame(pipe, session, recogniser, I_tgt, fstate, src_state, lm_vidx,
                  use_flow=False):
    """Phase 2+3 on one frame. Uses the tracked face's OWN landmarks."""
    tgt_r = recogniser.process(I_tgt, lm_vidx)
    frame = FrameData(I_tgt.astype(np.float64), tgt_r.landmarks, lm_vidx)
    result, new_fstate = pipe.process_frame(session, frame, fstate, use_flow=use_flow)
    overlay = draw_overlay(I_tgt, result.params)
    return ExtractResult(result, src_state, overlay, session.T_uv_src), new_fstate


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_outputs(out_dir, session, pipe, res, sam_measures=None):
    import json
    os.makedirs(out_dir, exist_ok=True)
    pipe.source_params(session).save(os.path.join(out_dir, "phase1_params.npz"))
    res.params.save(os.path.join(out_dir, "phase2_params.npz"))
    save_image(np.clip(session.T_uv_src, 0, 1).astype(np.float32),
               os.path.join(out_dir, "texture.png"))
    save_image(res.I_overlay.astype(np.float32), os.path.join(out_dir, "overlay.png"))
    p = res.params
    summary = {
        "quality": float(res.quality), "decision": res.decision,
        "head_pose_deg": p.head_pose_deg,
        "eye": {k: float(v) for k, v in p.eye.items()},
        "mouth_openness": float(p.mouth_openness), "mouth_open": bool(p.mouth_open),
        "region_error_px": res.region_error,
    }
    if sam_measures:
        summary["sam_measures"] = sam_measures   # mask-measured, not fitted
        # Eyes and mouth: report what the masks measured; the fitted values are poorly determined from one
        # image, so they are kept only under an explicit "_fitted_unreliable" name.
        if "eye_aspect_L" in sam_measures or "eye_aspect_R" in sam_measures:
            summary["eye_fitted_unreliable"] = summary.pop("eye")
            summary["eye"] = {"aspect_left": sam_measures.get("eye_aspect_L"),
                              "aspect_right": sam_measures.get("eye_aspect_R"),
                              "source": "SAM mask height/width"}
        if "mouth_open" in sam_measures:
            summary["mouth_open"] = bool(sam_measures["mouth_open"])
            summary["mouth_aspect"] = sam_measures["mouth_aspect"]
            summary["mouth_openness_fitted_unreliable"] = summary.pop("mouth_openness")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _emit_verdict(out_dir, validation, session, res, sam_measures, missing, mode):
    """report.json + verdict.txt; 'template' is free, 'muse' makes at most one (cached) live call."""
    if mode == "off":
        return
    import json
    from face_params.audit import build_report, make_verdict
    report = build_report(session, res, sam_measures, missing, validation=validation)
    text, source = make_verdict(report, mode)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "verdict.txt"), "w", encoding="utf-8") as f: f.write(text + "\n")
    print(f"\n  VERDICT [{source}]: {text}")


def _report_cost(out_dir):
    """Print and save what this run cost (estimates from the price table; cached answers are free)."""
    import json
    try:
        from face_params.perception import sam_client as _sc
        sam = _sc.stats()
    except Exception:
        sam = {"live": 0, "cached": 0, "est_usd": 0.0}
    from face_params.audit import verdict as _vd
    mu = _vd.stats()
    cost = {"sam_live_calls": sam["live"], "sam_cached_calls": sam["cached"], "sam_usd": sam["est_usd"],
            "muse_live_calls": mu["live"], "muse_cached_calls": mu["cached"], "muse_usd": mu["usd"],
            "total_usd": round(sam["est_usd"] + mu["usd"], 5)}
    print(f"  [cost] SAM {cost['sam_live_calls']} live / {cost['sam_cached_calls']} cached = ${cost['sam_usd']:.4f}  |  "
          f"Muse {cost['muse_live_calls']} live / {cost['muse_cached_calls']} cached = ${cost['muse_usd']:.5f}  |  "
          f"total ${cost['total_usd']:.4f}  (estimate; cached answers are free)")
    f = os.path.join(out_dir, "summary.json")
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8")); d["cost_usd"] = cost
        json.dump(d, open(f, "w", encoding="utf-8"), indent=2)


def extract(image_path, out_dir="extract_out", model_path=None, verbose=True,
            strict_validation=True, landmarks="model", verdict="template"):
    I = load_image(image_path)
    if I is None:
        print(f"[error] Cannot load: {image_path}"); sys.exit(1)

    H, W = I.shape[:2]
    model, uv, K, mouth_mask = load_model(model_path, H, W)
    lm_vidx    = _make_lm_vidx(model)
    provider   = None
    sam_measures = None
    sam_missing  = []
    if landmarks == "sam":
        # Real landmarks: SAM 3.1 masks -> calibrated template -> region-vertex landmarks.
        from face_params.perception import SamLandmarks
        if verbose: print("[Perception] SAM 3.1 regions…")
        sam        = SamLandmarks(image_path, W, H)
        from face_params.perception import sam_client as _sc
        if verbose: print(f"[Perception] SAM calls this run: {_sc.stats()}  (cached answers are free)")
        model      = sam.calibrate(model)
        lmset      = sam.landmark_set(model, K)
        lm_vidx    = lmset.vidx
        provider   = lambda img: lmset
        sam_measures = lmset.measures
        sam_missing  = lmset.missing
        if verbose:
            print(f"[Perception] {len(lm_vidx)} landmarks; missing regions: {lmset.missing or 'none'}")
    recogniser = FaceRecogniser(model=model, K=K, landmark_provider=provider)
    pipe       = FaceParamPipeline(model, K, uv, mouth_mask)

    if verbose: print("[Phase 1] Session init…")
    try:
        session, _ = build_session(pipe, recogniser, I, [I], model, lm_vidx,
                                   strict_validation=strict_validation)
    except ValueError as e:
        if "rejected by validation" not in str(e):
            raise
        print(f"\n[Stage 1] {e}\n"
              f"          The face must be ≥192 px wide, sharp, frontal and well fitted.\n"
              f"          Use a better image, or re-run with --no_strict to continue anyway.")
        _emit_verdict(out_dir, getattr(e, 'validation', None), None, None, sam_measures, sam_missing, verdict)
        _report_cost(out_dir)
        sys.exit(1)

    r      = recogniser.process(I, lm_vidx)
    frame  = FrameData(I.astype(np.float64), r.landmarks, lm_vidx)
    fstate = pipe.init_frame_state(frame, session)

    if verbose: print("[Phase 2+3] Fitting frame…")
    res, _ = process_frame(pipe, session, recogniser, I, fstate,
                           session.source_state, lm_vidx)

    p = res.params
    if sam_measures and 'mouth_open' in sam_measures:      # mask-measured, not β-fitted
        mouth_txt = f"{'open' if sam_measures['mouth_open'] else 'closed'} (lips h/w {sam_measures['mouth_aspect']:.2f}, from SAM mask)"
    else:
        mouth_txt = f"{'open' if p.mouth_open else 'closed'} ({p.mouth_openness:.3f})"
    print(f"\n  quality={res.quality:.3f}  {res.decision}  |  "
          f"mouth={mouth_txt}  |  "
          f"yaw={p.head_pose_deg['yaw']:+.1f}°  pitch={p.head_pose_deg['pitch']:+.1f}°  "
          f"roll={p.head_pose_deg['roll']:+.1f}°")

    if sam_measures:
        print(f"  measured from SAM masks: {sam_measures}")
    _save_outputs(out_dir, session, pipe, res, sam_measures)
    _emit_verdict(out_dir, None, session, res, sam_measures, sam_missing, verdict)
    _report_cost(out_dir)
    print(f"[done] → {out_dir}/")
    return session, res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Geometric face-parameter extraction — the 3D solve is pure NumPy Gauss-Newton with no neural network inside it (--landmarks sam adds SAM 3.1 perception before the solve).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --image face.png
  python run.py --image face.png --out_dir ./out/
  python run.py --image face.png --model bfm.npz
        """)
    ap.add_argument("--image", "--src", dest="image", required=True,
                    help="face image (PNG / BMP)")
    ap.add_argument("--out_dir", default="extract_out")
    ap.add_argument("--model",   default=None, help="3DMM .npz (omit → synthetic model)")
    ap.add_argument("--landmarks", choices=["model", "sam"], default="model",
                    help="landmark source: 'model' = template projection (original, circular); "
                         "'sam' = real landmarks from SAM 3.1 masks (needs MODEL_API_KEY)")
    ap.add_argument("--verdict", choices=["off", "template", "muse"], default="template",
                    help="plain-language verdict: 'template' = free deterministic text (default); "
                         "'muse' = Muse Spark (<=1 live call, cached, falls back to template)")
    ap.add_argument("--no_strict", action="store_true",
                    help="keep going even if Stage-1 validation rejects the image")
    args = ap.parse_args()
    extract(args.image, args.out_dir, args.model, strict_validation=not args.no_strict,
            landmarks=args.landmarks, verdict=args.verdict)


if __name__ == "__main__":
    main()
