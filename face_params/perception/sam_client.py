"""
sam_client.py  SAM 3.1 through the Meta Model API.

One call = one concept phrase ("eye", "nose" ...). Commas are NOT a documented
multi-label interface, so we make one call per region and cache every answer on
disk so each (image, prompt) pair is only ever paid for once.

Endpoint : POST https://api.meta.ai/v1/responses   model="sam-3.1"
Decoding : meta-sam-parser (Meta's own parser for the special-token text lane)
"""
import base64, hashlib, mimetypes, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import meta_sam_parser as msp

MODEL = "sam-3.1"
# Credit guard: at most this many LIVE (uncached) SAM calls per process.  Cached answers are free.
# Meta pricing (dev.meta.ai/docs/pricing-rate-limits): SAM 3.1 = $2.50 per 1,000 images; we assume 1 call = 1 image.
SAM_USD_PER_IMAGE = 0.0025
MAX_LIVE_CALLS = int(os.environ.get("FACE_PARAMS_MAX_SAM_CALLS", "40"))   # 40 calls ~ $0.10
_calls = {"live": 0, "cached": 0}


def stats() -> dict:
    """{"live": n billed-attempt calls, "cached": n free answers} so far in this process."""
    return {**_calls, "est_usd": round(_calls["live"] * SAM_USD_PER_IMAGE, 4)}
CACHE = Path(os.environ.get("FACE_PARAMS_SAM_CACHE", "cache/sam"))
CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- API call
def _client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"],
                  timeout=90.0, max_retries=1)


MAX_SIDE = int(os.environ.get("FACE_PARAMS_SAM_MAX_SIDE", "2048"))   # longest side sent to the API
MAX_BYTES = 4_000_000                                                # larger files are re-encoded as JPEG


def sent_size(W0: int, H0: int):
    """Pixel size of the image the API sees (pure function of the original size)."""
    s = min(1.0, MAX_SIDE / max(W0, H0))
    return max(1, round(W0 * s)), max(1, round(H0 * s))


def _data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    W0, H0 = Image.open(path).size
    if sent_size(W0, H0) == (W0, H0) and len(raw) <= MAX_BYTES:       # send the file untouched
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode()
    import io
    im = Image.open(path).convert("RGB")
    if sent_size(W0, H0) != (W0, H0):
        im = im.resize(sent_size(W0, H0), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cache_file(path: str, prompt: str) -> Path:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return CACHE / f"{h}__{prompt.replace(' ', '_')}.txt"


def segment_raw(path: str, prompt: str) -> str:
    """Return SAM's raw special-token text for (image, prompt). Cached."""
    f = _cache_file(path, prompt)
    if f.exists():
        _calls["cached"] += 1
        return f.read_text()
    if _calls["live"] >= MAX_LIVE_CALLS:
        raise RuntimeError(f"SAM credit guard: {MAX_LIVE_CALLS} live calls already made in this run. "
                           "Raise FACE_PARAMS_MAX_SAM_CALLS if you really want more.")
    _calls["live"] += 1          # counted before the request: a failed call may still be billed
    resp = _client().responses.create(
        model=MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _data_url(path)},
        ]}],
    )
    status = getattr(resp, "status", None)
    if status not in (None, "completed"):                      # never cache a failed/incomplete answer
        raise RuntimeError(f"SAM response status={status!r}: {getattr(resp, 'error', None)}")
    text = getattr(resp, "output_text", "") or ""
    if not text:                                               # SDK property empty -> walk the output list
        text = "".join(getattr(b, "text", "") or ""
                       for item in (getattr(resp, "output", None) or [])
                       for b in (getattr(item, "content", None) or [])
                       if getattr(b, "type", "") == "output_text")
    f.write_text(text)                                         # a completed call with no match is cached too
    return text


# ---------------------------------------------------------------- decoding
@dataclass
class Instance:
    object_id: str
    confidence: Optional[float]
    box: tuple                 # (left, top, right, bottom) in image pixels
    mask: Optional[np.ndarray] # (H, W) bool, full-image size, or None

    @property
    def centroid(self):
        if self.mask is not None and self.mask.any():
            ys, xs = np.nonzero(self.mask)
            return float(xs.mean()), float(ys.mean())
        l, t, r, b = self.box
        return (l + r) / 2, (t + b) / 2


def _to_px(bounds, W, H):
    """Bounds may arrive normalised (0..1) or in pixels; normalise to pixels."""
    l, t, r, b = bounds
    if max(l, t, r, b) <= 1.0001:
        return l * W, t * H, r * W, b * H
    return l, t, r, b


def place_mask(mask, bounds, W, H):
    """Paste a cropped mask raster into a full-size boolean image."""
    raster = msp.decode_mask_to_raster(mask)
    small = np.frombuffer(raster, np.uint8).reshape(mask.height, mask.width)
    l, t, r, b = [int(round(v)) for v in _to_px(bounds, W, H)]
    w, h = max(r - l, 1), max(b - t, 1)
    big = np.array(Image.fromarray(small * 255).resize((w, h), Image.NEAREST)) > 127
    full = np.zeros((H, W), bool)
    y0, x0 = max(t, 0), max(l, 0)
    y1, x1 = min(t + h, H), min(l + w, W)
    if y1 > y0 and x1 > x0:
        full[y0:y1, x0:x1] = big[y0 - t:y1 - t, x0 - l:x1 - l]
    return full


def parse(raw: str, W: int, H: int):
    """Raw SAM text -> (list[Instance], diagnostics)."""
    parser = msp.image_segmentation_format().create_parser()
    parser.push(raw)
    result = parser.finish(msp.CompletedOutcome()).result
    boxes, masks = {}, {}
    for rec in result.records:
        if rec.kind == "box":
            boxes[rec.object_id] = rec
        elif rec.kind == "mask":
            masks[rec.object_id] = rec
    out = []
    for oid in sorted(set(boxes) | set(masks), key=str):
        mrec, brec = masks.get(oid), boxes.get(oid)
        if mrec is not None:
            bd = (mrec.bounds.left, mrec.bounds.top, mrec.bounds.right, mrec.bounds.bottom)
            m = place_mask(mrec.mask, bd, W, H)
            conf = mrec.confidence
        else:
            bd = (brec.left, brec.top, brec.right, brec.bottom)
            m, conf = None, brec.confidence
        if brec is not None and conf is None:
            conf = brec.confidence
        out.append(Instance(str(oid), conf, tuple(_to_px(bd, W, H)), m))
    return out, list(result.diagnostics)


def _rescale(inst: "Instance", W0: int, H0: int, Ws: int, Hs: int) -> "Instance":
    sx, sy = W0 / Ws, H0 / Hs
    l, t, r, b = inst.box
    mask = None
    if inst.mask is not None:
        mask = np.array(Image.fromarray(inst.mask.astype(np.uint8) * 255).resize((W0, H0), Image.NEAREST)) > 127
    return Instance(inst.object_id, inst.confidence, (l * sx, t * sy, r * sx, b * sy), mask)


def segment(path: str, prompt: str):
    """-> (instances in ORIGINAL image pixels, parser diagnostics, raw text)."""
    W0, H0 = Image.open(path).size
    Ws, Hs = sent_size(W0, H0)
    raw = segment_raw(path, prompt)
    inst, diag = parse(raw, Ws, Hs)
    if (Ws, Hs) != (W0, H0):
        inst = [_rescale(i, W0, H0, Ws, Hs) for i in inst]
    return inst, diag, raw
