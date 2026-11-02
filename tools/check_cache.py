"""
check_cache.py - verify that REAL cached SAM answers parse correctly with this project's decoder.

    python tools/check_cache.py            # checks every file in cache/sam/

Run it after your first live SAM calls.  It makes no API calls and costs nothing.  For each cached answer it
prints the instance count, the box, and how much of the box the mask fills.  Anything odd (parser diagnostics,
masks that do not match their box, empty answers) is flagged so a request/decoding problem shows up immediately.
"""
import os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from face_params.perception import sam_client as sc

HDR = re.compile(r"w=(\d+);h=(\d+)")


def main():
    files = sorted(sc.CACHE.glob("*.txt"))
    if not files:
        print(f"no cached SAM answers in {sc.CACHE} yet"); return 0
    bad = 0
    for f in files:
        raw = f.read_text()
        m = HDR.search(raw)
        if not m:
            print(f"{f.name}: no box in the answer ({len(raw)} chars) -> SAM found nothing for this phrase"); continue
        W, H = int(m.group(1)), int(m.group(2))           # frame size the API reported
        inst, diag = sc.parse(raw, W, H)
        flags = []
        if diag: flags.append("parser diagnostics: " + ", ".join(d.code for d in diag)); bad += 1
        for i in inst:
            l, t, r, b = i.box; area = max((r - l) * (b - t), 1)
            fill = float(i.mask.sum()) / area if i.mask is not None else float("nan")
            if i.mask is not None and not (0.05 < fill <= 1.0):
                flags.append(f"mask fills {fill:.2f} of its box"); bad += 1
            print(f"{f.name}: id={i.object_id} box=({l:.0f},{t:.0f})-({r:.0f},{b:.0f}) frame={W}x{H} mask fill={fill:.2f}")
        if not inst: print(f"{f.name}: 0 instances"); bad += 1
        for fl in flags: print("   !!", fl)
    print("\nOK: every cached answer decodes cleanly." if not bad else f"\n{bad} problem(s) above.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
