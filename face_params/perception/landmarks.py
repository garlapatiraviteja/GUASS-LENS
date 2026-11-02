"""
perception/landmarks.py
=======================
Turns SAM 3.1 region masks into REAL image landmarks for the geometric solver.

Why this exists: the original pipeline fitted the model to landmarks that were
the model's own projection, so every validation score was circular.  Here the
landmarks come from the photo (SAM masks) and the 3D solve stays network-free.

Pipeline
  1. collect_regions : one SAM call per phrase ("face","eye","eyebrow","nose","lips","ear")
  2. measured bboxes : per region instance, split left/right by x-position
  3. calibrate       : warp the template's proportions to this face (anchors below)
  4. landmark_set    : each region vertex -> affine image of its template position
                       from the template region bbox onto the measured mask bbox
  5. init pose       : closed-form translation/depth from the landmark spread
"""
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from . import sam_client as sc

# 5 live calls per new image (ears are not used as landmarks, so they are not requested).
REGION_PROMPTS = {"face": "face", "eye": "eye", "brow": "eyebrow",
                  "nose": "nose", "mouth": "lips"}
MIN_LANDMARKS = 8
MOUTH_OPEN_ASPECT = 0.5     # lips-mask height/width above this -> mouth open (tunable)


# ------------------------------------------------------------------ data
@dataclass
class Box:
    l: float; t: float; r: float; b: float
    @property
    def w(self): return self.r - self.l
    @property
    def h(self): return self.b - self.t
    @property
    def cx(self): return 0.5 * (self.l + self.r)
    @property
    def cy(self): return 0.5 * (self.t + self.b)


def _box_of(inst) -> Optional[Box]:
    if inst.mask is not None and inst.mask.any():
        ys, xs = np.nonzero(inst.mask)
        return Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    l, t, r, b = inst.box
    return Box(l, t, r, b) if r > l and b > t else None


SPLIT_REGIONS = ("eye", "brow")   # paired organs: SAM may return both as ONE merged mask
MIN_BLOB_FRAC = 0.15              # ignore specks smaller than this fraction of the largest blob


def _boxes_of(inst, split: bool = False) -> List[Box]:
    """Boxes for one SAM instance.  With split=True a mask made of several separate blobs
    (e.g. both eyebrows returned as one instance) yields one Box per blob."""
    if split and inst.mask is not None and inst.mask.any():
        from scipy import ndimage
        lab, n = ndimage.label(inst.mask)
        if n > 1:
            sizes = ndimage.sum(inst.mask, lab, range(1, n + 1))
            out = []
            for k, sz in enumerate(sizes, start=1):
                if sz >= MIN_BLOB_FRAC * sizes.max():
                    ys, xs = np.nonzero(lab == k)
                    out.append(Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
            if out:
                return out
    b = _box_of(inst)
    return [b] if b is not None else []


@dataclass
class LandmarkSet:
    xy: np.ndarray        # (N,2) image px
    vidx: np.ndarray      # (N,)  model vertex ids
    names: List[str]
    box: object           # FaceBox from the SAM face mask
    sigma_lm_px: float    # tolerance for the landmark-consistency score
    t_init: np.ndarray    # (3,) closed-form translation warm start
    missing: List[str]    # regions SAM did not return
    measures: dict = None # directly measured from masks (auditable), e.g. mouth_aspect


# ------------------------------------------------------------------ measuring
class SamLandmarks:
    """Runs/caches SAM for one image and exposes calibrate() + landmark_set()."""

    def __init__(self, image_path: str, W: int, H: int,
                 segment: Optional[Callable] = None, prompts: Dict[str, str] = None):
        from PIL import Image
        self.W, self.H = int(W), int(H)
        W0, H0 = Image.open(image_path).size
        self.sx, self.sy = self.W / W0, self.H / H0      # SAM pixels -> pipeline pixels
        seg = segment or sc.segment
        self.regions: Dict[str, List[Box]] = {}
        self.confidence: Dict[str, float] = {}           # best SAM mask score per region (audit only)
        for key, phrase in (prompts or REGION_PROMPTS).items():
            inst = seg(image_path, phrase)[0]
            boxes = [b for i in inst for b in _boxes_of(i, split=key in SPLIT_REGIONS)]
            self.confidence[key] = max([float(i.confidence) for i in inst if i.confidence is not None], default=0.0)
            self.regions[key] = [Box(b.l * self.sx, b.t * self.sy, b.r * self.sx, b.b * self.sy)
                                 for b in boxes]
        self.face = max(self.regions.get("face", []), key=lambda b: b.w * b.h, default=None)
        if self.face is None:
            raise ValueError("SAM found no face in the image — cannot extract landmarks.")
        self.missing = [k for k in self.regions if k != "face" and not self.regions[k]]

    # left/right as seen in the image (smaller x = image-left = model x<0)
    def _lr(self, key):
        bs = sorted(self.regions.get(key, []), key=lambda b: b.cx)
        mid = self.face.cx
        left = [b for b in bs if b.cx < mid]; right = [b for b in bs if b.cx >= mid]
        pick = lambda lst: max(lst, key=lambda b: b.w * b.h) if lst else None
        return pick(left), pick(right)

    def _single(self, key):
        bs = self.regions.get(key, [])
        return max(bs, key=lambda b: b.w * b.h) if bs else None

    # ------------------------------------------------------------ template groups
    @staticmethod
    def _groups(model):
        S = model.S_bar
        def split(idx):
            idx = np.asarray(idx, dtype=np.intp)
            if len(idx) == 0: return idx, idx
            return idx[S[idx, 0] < 0], idx[S[idx, 0] >= 0]
        eL, eR = split(model.eye_idx); bL, bR = split(model.brow_idx)
        aL, aR = split(model.ear_idx)
        return {"eye": (eL, eR), "brow": (bL, bR), "ear": (aL, aR),
                "nose": (np.asarray(model.nose_idx, np.intp),),
                "mouth": (np.asarray(model.mouth_idx, np.intp),)}

    def _pairs(self, model):
        """[(name, vertex_ids, measured Box)] for every region SAM found."""
        g = self._groups(model); out = []
        for key in ("eye", "brow", "ear"):
            L, R = self._lr(key)
            for side, ids, box in (("L", g[key][0], L), ("R", g[key][1], R)):
                if box is not None and len(ids): out.append((f"{key}_{side}", ids, box))
        for key in ("nose", "mouth"):
            b = self._single(key)
            if b is not None and len(g[key][0]): out.append((key, g[key][0], b))
        return out

    # ------------------------------------------------------------ calibration
    def calibrate(self, model):
        """Warp S_bar (x scale + piecewise-linear y) so region heights/widths match
        this face.  Anchors: brow, eye, nose, mouth row centres and the chin."""
        S = model.S_bar.copy(); pairs = self._pairs(model)
        tpl_y, img_y = [], []
        for key in ("brow", "eye", "nose", "mouth"):
            sel = [(ids, b) for n, ids, b in pairs if n.split("_")[0] == key]
            if sel:
                tpl_y.append(float(np.mean([S[i, 1] for ids, _ in sel for i in ids])))
                img_y.append(float(np.mean([b.cy for _, b in sel])))
        chin_v = int(np.argmin(S[:, 1] + 1e-3 * np.abs(S[:, 0])))
        tpl_y.append(float(S[chin_v, 1])); img_y.append(self.face.b)
        order = np.argsort(tpl_y)[::-1]                    # template top -> bottom
        tpl_y = np.array(tpl_y)[order]; img_y = np.array(img_y)[order]
        keep = [0]
        for k in range(1, len(img_y)):                     # image y must increase downward
            if img_y[k] > img_y[keep[-1]] + 1.0: keep.append(k)
        tpl_y, img_y = tpl_y[keep], img_y[keep]
        if len(tpl_y) < 3:
            return model
        top, bot = tpl_y[0], tpl_y[-1]
        r = (img_y - img_y[0]) / (img_y[-1] - img_y[0])    # 0 at brow row .. 1 at chin
        new_y = top + r * (bot - top)
        S[:, 1] = np.interp(S[:, 1], tpl_y[::-1], new_y[::-1])
        eyes = [b for n, _, b in pairs if n.startswith("eye")]
        if len(eyes) == 2:                                  # inter-ocular / brow-chin ratio
            g = self._groups(model)
            tx = abs(model.S_bar[g["eye"][1]][:, 0].mean() - model.S_bar[g["eye"][0]][:, 0].mean())
            ty = abs(model.S_bar[:, 1][chin_v] - np.mean([model.S_bar[i, 1] for n, ids, _ in pairs
                                                          if n.startswith("brow") for i in ids] or [top]))
            meas = abs(eyes[1].cx - eyes[0].cx) / max(img_y[-1] - img_y[0], 1.0)
            sxs = np.clip(meas / max(tx / max(ty, 1e-6), 1e-6), 0.6, 2.2)
            S[:, 0] *= sxs
        # The synthetic grid is nearly flat (4 units of relief) and inverted (nose farther
        # than cheeks).  Give it a face-like convex depth so yaw/pitch are observable.
        z0 = float(S[:, 2].mean()); a = np.abs(S[:, 0]).max() * 1.15
        y0, hy = float(S[:, 1].mean()), (S[:, 1].max() - S[:, 1].min()) * 0.6
        dome = 0.38 * a * np.sqrt(np.clip(1 - (S[:, 0] / a) ** 2 - ((S[:, 1] - y0) / hy) ** 2, 0, None))
        nose_c = S[model.nose_idx, :2].mean(axis=0) if len(model.nose_idx) else np.zeros(2)
        bump = 0.17 * a * np.exp(-(((S[:, 0] - nose_c[0]) / (0.22 * a)) ** 2
                                   + ((S[:, 1] - nose_c[1]) / (0.30 * a)) ** 2))
        S[:, 2] = z0 - dome - bump
        eye_ref = model.S_eye_ref.copy()
        if len(eye_ref):
            eye_ref[:] = S[model.eye_idx]
        return replace(model, S_bar=S, S_eye_ref=eye_ref)

    # ------------------------------------------------------------ landmarks
    def landmark_set(self, model, K) -> LandmarkSet:
        """Centre-aligned binding: every region keeps the template's own internal geometry
        and is translated so its centre sits on the measured mask centre.  (Forcing template
        corners onto mask corners made the expression coefficients run away, because the
        coarse template regions are not the size of real eyes/lips.)  Ears are left out: they
        are usually hidden by hair, and stay unobserved at their defaults."""
        from ..face_recogniser import FaceBox
        S = model.S_bar
        reg = [(n, ids, b) for n, ids, b in self._pairs(model) if not n.startswith("ear")]
        chin_v = int(np.argmin(S[:, 1] + 1e-3 * np.abs(S[:, 0])))
        # region centres (template 3D point  <->  measured image point) -> pose & scale
        P = [np.r_[S[ids, :2].mean(0), S[ids, 2].mean()] for _, ids, _ in reg] + [S[chin_v]]
        X = [(b.cx, b.cy) for _, _, b in reg] + [(self.face.cx, self.face.b)]
        P, X = np.asarray(P), np.asarray(X, float)
        t0 = init_translation(P, K, np.arange(len(P)), X)
        px_per_unit = K[0, 0] / (P[:, 2].mean() + t0[2])
        xy, vid, names = [], [], []
        for name, ids, box in reg:
            c = S[ids, :2].mean(0)
            for i in ids:
                xy.append((box.cx + px_per_unit * (S[i, 0] - c[0]),
                           box.cy - px_per_unit * (S[i, 1] - c[1])))
                vid.append(int(i)); names.append(name)
        xy.append((self.face.cx, self.face.b)); vid.append(chin_v); names.append("chin")
        if len(xy) < MIN_LANDMARKS:
            raise ValueError(f"Only {len(xy)} landmarks from SAM (regions missing: {self.missing}).")
        xy = np.asarray(xy, float); vid = np.asarray(vid, np.intp)
        fb = FaceBox(int(self.face.l), int(self.face.t), int(self.face.w), int(self.face.h))
        return LandmarkSet(xy, vid, names, fb, max(3.0, 0.02 * self.face.w),
                           init_translation(S, K, vid, xy), list(self.missing), self.measures())

    def measures(self) -> dict:
        """Quantities read straight off the masks (no fitting, so they cannot run away)."""
        out = {}   # region_confidence left out on purpose: the API's mask score does not reach the parser yet (it reads 0)
        L, R = self._lr("eye")
        for side, b in (("L", L), ("R", R)):
            if b is not None: out[f"eye_aspect_{side}"] = round(b.h / max(b.w, 1e-6), 3)
        m = self._single("mouth")
        if m is not None:
            out["mouth_aspect"] = round(m.h / max(m.w, 1e-6), 3)
            out["mouth_open"] = bool(out["mouth_aspect"] > MOUTH_OPEN_ASPECT)
        return out


def init_translation(S_bar, K, vidx, xy):
    """Closed-form t (omega=0) from landmark spread:  Z = f*sigma_model/sigma_image."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    P = np.asarray(S_bar)[vidx]; u, v = xy[:, 0], xy[:, 1]
    sm = np.sqrt(P[:, 0].var() + P[:, 1].var()); si = np.sqrt(u.var() + v.var())
    Z = f * sm / max(si, 1e-6)
    tx = (u.mean() - cx) * Z / f - P[:, 0].mean()
    ty = (cy - v.mean()) * Z / f - P[:, 1].mean()
    return np.array([tx, ty, Z - P[:, 2].mean()])
