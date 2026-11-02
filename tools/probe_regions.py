"""
probe_regions.py  -  which facial regions does SAM 3.1 find, and how well?

    python tools/probe_regions.py face1.jpg face2.jpg ...

Writes, for every image:  out/sam_probe/<name>_overlay.png
And once:                 out/sam_probe/hit_rate.csv   (image, prompt, n_found, best_conf, area_frac)
Prints a per-prompt hit-rate table at the end. Cached calls cost nothing on re-runs.
"""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from face_params.perception import sam_client as sc

# one concept phrase per call; expected instance counts on a frontal face
# default = the 5 phrases the pipeline uses.  --extra also tries "mouth" and "ear" (2 more calls/image).
PROMPTS = {"face": 1, "eye": 2, "eyebrow": 2, "nose": 1, "lips": 1}
EXTRA = {"mouth": 1, "ear": 2}
COLORS = {"face": (120, 120, 120), "eye": (0, 170, 255), "eyebrow": (255, 190, 0),
          "nose": (0, 220, 120), "lips": (255, 60, 130), "mouth": (200, 40, 200),
          "ear": (160, 90, 255)}
OUT = Path("out/sam_probe"); OUT.mkdir(parents=True, exist_ok=True)


def main(paths, extra=False, dry=False):
    if extra: PROMPTS.update(EXTRA)
    todo = sum(1 for p in paths for ph in PROMPTS if not sc._cache_file(p, ph).exists())
    print(f"live SAM calls needed: {todo}  (cached: {len(paths) * len(PROMPTS) - todo}; guard limit {sc.MAX_LIVE_CALLS})"
          f"  ~ ${todo * sc.SAM_USD_PER_IMAGE:.4f} at $2.50 per 1,000 images")
    if dry or todo > sc.MAX_LIVE_CALLS: return
    rows = []
    for p in paths:
        img = Image.open(p).convert("RGB"); W, H = img.size
        over = np.array(img, dtype=np.float32)
        d_marks = []
        print(f"\n== {p}  ({W}x{H})")
        for prompt, expect in PROMPTS.items():
            try:
                inst, diag, raw = sc.segment(p, prompt)
            except Exception as e:            # show the real API error, don't hide it
                print(f"  {prompt:8s} ERROR: {type(e).__name__}: {e}")
                rows.append([p, prompt, "error", "", ""]); continue
            best = max([i.confidence or 0.0 for i in inst], default=0.0)
            area = sum(float(i.mask.mean()) for i in inst if i.mask is not None)
            flag = "" if len(inst) == expect else f"  (expected {expect})"
            print(f"  {prompt:8s} found={len(inst)}  best_conf={best:.2f}  area={area:.3f}{flag}")
            if diag: print("     parser diagnostics:", [(d.code, d.message[:60]) for d in diag][:3])
            if not inst and raw.strip(): print("     raw (first 160 chars):", raw[:160].replace("\n", " "))
            rows.append([p, prompt, len(inst), f"{best:.3f}", f"{area:.4f}"])
            col = np.array(COLORS[prompt], np.float32)
            for i in inst:
                if i.mask is not None:
                    over[i.mask] = over[i.mask] * 0.45 + col * 0.55
                d_marks.append((i.centroid, col))
        pil = Image.fromarray(over.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(pil)
        for (cx, cy), col in d_marks:
            dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=tuple(int(c) for c in col), outline="white")
        pil.save(OUT / f"{Path(p).stem}_overlay.png")

    with open(OUT / "hit_rate.csv", "w", newline="") as f:
        csv.writer(f).writerows([["image", "prompt", "n_found", "best_conf", "area_frac"]] + rows)

    print("\n== hit rate (image has at least one match)")
    for prompt in PROMPTS:
        r = [x for x in rows if x[1] == prompt and x[2] != "error"]
        hits = sum(1 for x in r if x[2] > 0)
        print(f"  {prompt:8s} {hits}/{len(r)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/probe_regions.py [--dry-run] [--extra] image1.jpg [...]")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args, extra="--extra" in sys.argv, dry="--dry-run" in sys.argv)
    print("SAM calls this run:", sc.stats())
