"""Full pipeline test: Phase 1 (S2→S1→S3→S4) + Phase 2/3 tracking on a synthetic 512px face."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import contextlib, io
from run import load_model, build_session, process_frame, _make_lm_vidx
from face_params.pipeline import FaceParamPipeline, FrameData
from face_params.face_recogniser import FaceRecogniser
from face_params.synthetic_model import make_synthetic_image

def test_full_pipeline():
    img = make_synthetic_image(512, 512)
    with contextlib.redirect_stdout(io.StringIO()):
        model, uv, K, mm = load_model(None, 512, 512)
        lm = _make_lm_vidx(model)
        rec = FaceRecogniser(model=model, K=K)
        pipe = FaceParamPipeline(model, K, uv, mm)
        session, _ = build_session(pipe, rec, img, [img], model, lm)
        r = rec.process(img, lm)
        fs = pipe.init_frame_state(FrameData(img.astype(np.float64), r.landmarks, lm), session)
        res, fs = process_frame(pipe, session, rec, img, fs, session.source_state, lm)
    assert session.validation.accepted
    assert session.T_uv_src.shape[:2] == (512, 512)
    assert res.decision in ("PROCEED", "CONSERVATIVE", "FALLBACK")
    assert np.isfinite(res.params.project()).all()

if __name__ == "__main__":
    test_full_pipeline(); print("ok")
