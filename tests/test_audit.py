"""Offline tests for the verdict step (Muse Spark is replaced by fakes; no network, no credits)."""
import contextlib, io, json, os, sys, tempfile, types
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run
import test_sam_landmarks as ts
from face_params.audit import verdict as vd, template_verdict, numbers_ok

REJ = {"decision": "REJECTED", "measured_from_masks": {}, "regions_missing": [],
       "validation": {"accepted": False, "accept_threshold": 0.6, "landmark_consistency": 0.98,
                      "symmetry": 1.0, "resolution": 0.781, "sharpness": 0.058, "pose": 1.0,
                      "weakest": "sharpness"}}
OK = {"decision": "PROCEED", "quality": 0.97, "head_pose_deg": {"yaw": 0.2, "pitch": 0.4, "roll": 0.9},
      "region_error_px": {"eye": 1.7}, "measured_from_masks": {"mouth_aspect": 0.312, "mouth_open": False},
      "regions_missing": ["ear"],
      "validation": {"accepted": True, "accept_threshold": 0.6, "landmark_consistency": 0.85, "symmetry": 1.0,
                     "resolution": 1.0, "sharpness": 1.0, "pose": 1.0, "weakest": "landmark_consistency"}}


SEEN = {}


def _fake_client(text=None, boom=False):
    def create(**kw):
        SEEN["messages"] = kw.get("messages")
        if boom: raise TimeoutError("down")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])
    return lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def _isolated():
    d = tempfile.mkdtemp(); vd.CACHE = vd.Path(d); vd._calls.update(live=0, cached=0); vd.MAX_LIVE_CALLS = 5


def test_template_verdicts_cite_real_numbers():
    t = template_verdict(REJ); assert "blurry" in t and "0.058" in t and "0.6" in t
    t = template_verdict(OK);  assert "PROCEED" in t and "yaw 0.2" in t and "closed" in t and "ear" in t


def test_residual_curve_is_cited_but_kept_out_of_the_muse_prompt():
    r = {**OK, "residual_curve": {"start": 133.4, "end": 11.3, "reduction_pct": 91.5, "iterations": 10,
                                  "max_iterations": 10, "converged": False},
         "residual_history": [{"iter": 0, "residual_before": 133.393, "residual_after": 31.407,
                               "step_norm": 8.799, "accepted": True}]}
    t = template_verdict(r)
    assert "133.4 to 11.3" in t and "91.5% lower" in t and "cap" in t
    _isolated(); vd._client = _fake_client("Residual fell from 133.4 to 11.3, so the fit is good.")
    assert vd.muse_verdict(r)[1] == "muse-spark"
    sent = SEEN["messages"][1]["content"]
    assert "residual_curve" in sent and "residual_history" not in sent and "31.407" not in sent


def test_contributor_models_are_blocked_and_cost_is_tracked():
    _isolated(); old = vd.MUSE_MODEL
    try:
        vd.MUSE_MODEL = "muse-spark-1.3-contributor"; vd._client = _fake_client("never used")
        t, src = vd.muse_verdict(OK)
        assert "blocked" in src and vd.stats()["live"] == 0
        vd.MUSE_MODEL = "muse-spark-1.3"; good = "Proceeded with quality 0.97."
        usage = types.SimpleNamespace(prompt_tokens=500, completion_tokens=100)
        def create(**kw):
            return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=good))])
        vd._client = lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        vd.muse_verdict(OK)
        assert abs(vd.stats()["usd"] - (500 * 1.25e-6 + 100 * 4.25e-6)) < 1e-6     # = $0.00105
    finally:
        vd.MUSE_MODEL = old


def test_muse_steps_down_parameters_when_the_endpoint_rejects_them():
    class BadRequestError(Exception): pass                                     # same class name the SDK uses
    _isolated(); vd._good = 0; seen = []
    def create(**kw):
        seen.append({k: v for k, v in kw.items() if k not in ("messages", "model")})
        if "max_completion_tokens" in kw: raise BadRequestError("unsupported parameter")
        return types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="Proceeded with quality 0.97."))])
    vd._client = lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    t, src = vd.muse_verdict(OK)
    assert src == "muse-spark" and seen[0] == {"reasoning_effort": "minimal", "max_completion_tokens": 1500}
    assert seen[1] == {"reasoning_effort": "minimal"} and vd._good == 1 and vd.stats()["live"] == 2
    _isolated(); seen.clear()                                                   # remembered: next call goes straight to the working set
    vd.muse_verdict({**OK, "quality": 0.97, "decision": "PROCEED", "x": 1}); assert seen == [{"reasoning_effort": "minimal"}]
    _isolated(); vd._good = 0
    vd._client = _fake_client("")                                               # reasoning ate the budget -> empty reply
    assert "empty reply" in vd.muse_verdict(OK)[1]


def test_number_check_rejects_invented_values():
    assert numbers_ok("Quality 0.97, yaw 0.2 degrees, mouth closed.", OK)
    assert not numbers_ok("Quality is 0.88.", OK)
    assert not numbers_ok("Yaw is 12 degrees.", OK)


def test_muse_path_cache_guard_and_fallbacks():
    _isolated(); good = "Proceeded with quality 0.97; the face is frontal and the mouth is closed."
    vd._client = _fake_client(good)
    t, src = vd.muse_verdict(OK); assert src == "muse-spark" and t == good and vd.stats()["live"] == 1
    vd._client = _fake_client(boom=True)                       # cache hit must not touch the client
    t, src = vd.muse_verdict(OK); assert src == "muse-spark (cached)" and vd.stats()["live"] == 1
    _isolated(); vd._client = _fake_client("Quality is 0.12 and yaw 40.")   # invented numbers
    assert vd.muse_verdict(OK)[1].startswith("template") and not (vd.CACHE / "x").exists()
    _isolated(); vd._client = _fake_client(boom=True)                        # API failure
    assert "unavailable" in vd.muse_verdict(OK)[1]
    _isolated(); vd.MAX_LIVE_CALLS = 0; vd._client = _fake_client("never used")
    assert "credit guard" in vd.muse_verdict(OK)[1] and vd.stats()["live"] == 0


def test_run_writes_report_and_verdict_for_good_and_rejected():
    ts._install(ts.REGIONS)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ok"); ts._extract(ts._photo(d), out)
        rep = json.load(open(os.path.join(out, "report.json")))
        assert rep["decision"] in ("PROCEED", "CONSERVATIVE")
        assert rep["residual_curve"]["iterations"] == len(rep["residual_history"]) >= 1
        assert rep["residual_curve"]["end"] < rep["residual_curve"]["start"]
        assert open(os.path.join(out, "verdict.txt")).read().strip()
        # smooth gradient = blurry -> rejected, and the verdict must say why
        yy, xx = np.mgrid[0:512, 0:512]
        blurry = os.path.join(d, "blur.png")
        Image.fromarray((np.stack([120 + 40*np.sin(xx/40.), 110 + 30*np.cos(yy/50.), 100 + 0*xx], -1)).astype(np.uint8)).save(blurry)
        out2 = os.path.join(d, "rej")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run.extract(blurry, out2, None, strict_validation=True, landmarks="sam", verdict="template")
        except SystemExit: pass
        assert "blurry" in open(os.path.join(out2, "verdict.txt")).read()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok  ", name)


def test_forced_run_does_not_say_validation_passed():
    r = {**OK, "decision": "CONSERVATIVE", "quality": 0.572,
         "validation": {**OK["validation"], "accepted": False, "landmark_consistency": 0.307}}
    t = template_verdict(r)
    assert "Validation FAILED" in t and "0.307" in t and "passed" not in t
