"""python examples/run_demo.py [image.png] [--model model.npz]   """
import sys, argparse
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from face_params import FaceParamExtractor, load_image
from face_params.synthetic_model import make_synthetic_image

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?")
ap.add_argument("--model", default=None, help="3DMM .npz (omit -> synthetic demo model)")
a = ap.parse_args()

img = load_image(a.image) if a.image else make_synthetic_image(128, 128)
ex = FaceParamExtractor(model_path=a.model)
p = ex.fit_image(img)

print("head pose (deg):", p.head_pose_deg)
print("eyes:           ", p.eye)
print("mouth openness: ", round(p.mouth_openness, 3), "open" if p.mouth_open else "closed")
print("brow gamma:     ", p.gamma_brow.round(3))
print("ear gamma:      ", p.gamma_ear.round(3))
for name, pts in p.region_points_2d.items():
    print(f"{name:5s} 2D points:", pts.shape)
p.save("face_params_out.npz")
if p.texture is not None:
    from face_params import save_image
    import numpy as np
    save_image(np.clip(p.texture, 0, 1).astype(np.float32), "face_texture_out.png")
    print("texture (Stage 3) saved -> face_texture_out.png")
