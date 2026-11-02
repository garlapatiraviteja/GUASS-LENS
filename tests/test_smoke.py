import sys, numpy as np; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from face_params import FaceParamExtractor
from face_params.synthetic_model import make_synthetic_image

def test_fit_image_and_own_landmarks():
    ex  = FaceParamExtractor()
    img = make_synthetic_image(128, 128)
    p   = ex.fit_image(img)
    assert p.alpha.shape == (80,) and p.beta.shape == (64,)
    assert p.texture is not None and p.texture.shape == (128, 128, 3)   # Stage-3 texture
    assert p.normals().shape == (ex.model.V, 3)
    assert np.isfinite(p.project()).all()
    # user-supplied landmarks path
    p2 = ex.fit_image(img, landmarks=p.project(), lm_vidx=ex.lm_vidx)
    assert np.isfinite(p2.omega).all()

if __name__ == "__main__":
    test_fit_image_and_own_landmarks(); print("ok")
