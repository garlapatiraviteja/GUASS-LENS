"""Offline tests for the SAM-landmark path (no network: SAM is replaced by fixed region boxes)."""
import contextlib, io, os, struct, sys, tempfile, types, zlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run
from face_params import io_utils
from face_params.perception import sam_client as sc

REGIONS = {"face": [(160, 70, 360, 470)], "eye": [(195, 215, 245, 235), (275, 215, 325, 235)],
           "eyebrow": [(190, 190, 250, 205), (270, 190, 330, 205)], "nose": [(235, 250, 285, 320)],
           "lips": [(220, 350, 300, 378)], "ear": []}


def _install(regions):
    sc.segment = lambda path, prompt: ([sc.Instance(str(k), 0.9, b, None)
                                        for k, b in enumerate(regions.get(prompt, []))], [], "")


def _photo(tmp):
    yy, xx = np.mgrid[0:512, 0:512]
    noise = np.random.default_rng(1).normal(0, 35, (512, 512, 1))          # sharp texture (blur gate)
    img = np.stack([120 + 60 * np.sin(xx / 40.), 110 + 50 * np.cos(yy / 50.), 100 + 0 * xx], -1) + noise
    path = os.path.join(tmp, "face.png"); Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(path)
    return path


def _extract(path, out):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return run.extract(path, out, None, strict_validation=True, landmarks="sam")
    except SystemExit:
        raise AssertionError("rejected: " + " ".join(l for l in buf.getvalue().splitlines() if "Stage 1" in l))


def test_png_loader_handles_all_filter_types():
    rng = np.random.default_rng(0)
    img = (rng.random((40, 57, 3)) * 255).astype(np.uint8)
    img[:, :20] = np.linspace(0, 255, 20, dtype=np.uint8)[None, :, None]     # smooth block -> Sub/Up/Paeth rows
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png"); Image.fromarray(img).save(p, optimize=True)
        import builtins; real = builtins.__import__
        def no_pil(n, *a, **k):
            if n == "PIL" or n.startswith("PIL."): raise ImportError
            return real(n, *a, **k)
        builtins.__import__ = no_pil
        try: mine = io_utils._load_png(p)                    # pure-NumPy fallback
        finally: builtins.__import__ = real
    assert np.array_equal((mine * 255).round().astype(np.uint8), img)


def test_sam_path_runs_and_reports_measured_values():
    _install(REGIONS)
    with tempfile.TemporaryDirectory() as d:
        session, res = _extract(_photo(d), os.path.join(d, "out"))
        assert res.decision in ("PROCEED", "CONSERVATIVE")
        assert max(res.region_error.values()) < 15.0, res.region_error
        import json
        m = json.load(open(os.path.join(d, "out", "summary.json")))["sam_measures"]
        assert m["mouth_open"] is False and 0.2 < m["mouth_aspect"] < 0.4


def test_no_face_raises_clear_error():
    _install({**REGIONS, "face": []})
    with tempfile.TemporaryDirectory() as d:
        try: _extract(_photo(d), os.path.join(d, "out"))
        except ValueError as e: assert "no face" in str(e)
        else: raise AssertionError("expected ValueError")


def test_missing_regions_still_fit():
    _install({**REGIONS, "eyebrow": [], "lips": []})
    with tempfile.TemporaryDirectory() as d:
        _, res = _extract(_photo(d), os.path.join(d, "out"))
        assert res.decision in ("PROCEED", "CONSERVATIVE", "FALLBACK")


def test_credit_guard_blocks_live_calls_but_allows_cached():
    with tempfile.TemporaryDirectory() as d:
        sc.CACHE = sc.Path(d) / "cache"; sc.CACHE.mkdir()
        real_client = sc._client
        sc._client = lambda: (_ for _ in ()).throw(AssertionError("network must not be touched"))
        img = _photo(d); old = (sc.MAX_LIVE_CALLS, dict(sc._calls))
        try:
            sc._cache_file(img, "eye").write_text("cached-answer")
            sc.MAX_LIVE_CALLS = 0
            assert sc.segment_raw(img, "eye") == "cached-answer"          # free
            try: sc.segment_raw(img, "nose")                              # would be a live call
            except RuntimeError as e: assert "credit guard" in str(e)
            else: raise AssertionError("guard did not trigger")
        finally:
            sc._client = real_client; sc.MAX_LIVE_CALLS = old[0]; sc._calls.update(old[1])


def _fake_responses(output_text="", status="completed", output=None):
    seen = {}
    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(status=status, output_text=output_text, output=output or [], error=None)
    client = types.SimpleNamespace(responses=types.SimpleNamespace(create=create))
    return seen, (lambda: client)


def test_request_matches_metas_working_client_and_handles_odd_responses():
    with tempfile.TemporaryDirectory() as d:
        sc.CACHE = sc.Path(d) / "cache"; sc.CACHE.mkdir(); img = _photo(d)
        real_client, old = sc._client, (sc.MAX_LIVE_CALLS, dict(sc._calls)); sc.MAX_LIVE_CALLS = 50
        try:
            seen, sc._client = _fake_responses(output_text="")                    # SDK property empty
            sc._client = _fake_responses(output=[types.SimpleNamespace(content=[
                types.SimpleNamespace(type="output_text", text="<0f>fallback")])])[1]
            assert sc.segment_raw(img, "eye") == "<0f>fallback"                   # walked the output list
            seen, sc._client = _fake_responses(output_text="ok")
            sc.segment_raw(img, "nose")
            item = seen["input"][0]
            assert seen["model"] == "sam-3.1" and item["type"] == "message" and item["role"] == "user"
            assert item["content"][0] == {"type": "input_text", "text": "nose"}
            assert item["content"][1]["type"] == "input_image" and item["content"][1]["image_url"].startswith("data:image/")
            sc._client = _fake_responses(output_text="partial", status="failed")[1]  # failed: raise, never cache
            try: sc.segment_raw(img, "lips")
            except RuntimeError as e: assert "failed" in str(e)
            else: raise AssertionError("a failed response must raise")
            assert not sc._cache_file(img, "lips").exists()
            sc._client = _fake_responses(output_text="")[1]                          # completed, no match: cached
            sc.segment_raw(img, "ear"); assert sc._cache_file(img, "ear").exists()
        finally:
            sc._client = real_client; sc.MAX_LIVE_CALLS = old[0]; sc._calls.update(old[1])


def test_big_images_are_downscaled_and_results_mapped_back():
    old = sc.MAX_SIDE
    try:
        sc.MAX_SIDE = 2048
        assert sc.sent_size(1024, 1024) == (1024, 1024) and sc.sent_size(4096, 3072) == (2048, 1536)
        m = np.zeros((100, 200), bool); m[20:60, 50:150] = True                    # mask in the SENT frame (200x100)
        inst = sc._rescale(sc.Instance("0", None, (50, 20, 150, 60), m), 800, 400, 200, 100)   # original is 4x
        assert inst.box == (200, 80, 600, 240) and inst.mask.shape == (400, 800)
        assert int(inst.mask.sum()) == 16 * int(m.sum())
        with tempfile.TemporaryDirectory() as d:                                   # data URL for a huge image is capped
            Image.fromarray(np.zeros((3000, 4000, 3), np.uint8)).save(os.path.join(d, "big.png"))
            url = sc._data_url(os.path.join(d, "big.png"))
            import base64, io
            sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            assert url.startswith("data:image/jpeg") and sent.size == (2048, 1536)
    finally:
        sc.MAX_SIDE = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok  ", name)


def test_merged_brow_mask_is_split_into_left_and_right():
    from face_params.perception.landmarks import SamLandmarks
    m = np.zeros((512, 512), bool); m[190:205, 190:250] = True; m[190:205, 270:330] = True   # one mask, two blobs
    faces = {"face": [sc.Instance("0", .9, (160, 70, 360, 470), None)],
             "eye": [sc.Instance("0", .9, (195, 215, 245, 235), None), sc.Instance("1", .9, (275, 215, 325, 235), None)],
             "eyebrow": [sc.Instance("0", .9, (190, 190, 330, 205), m)]}
    with tempfile.TemporaryDirectory() as d:
        path = _photo(d)
        s = SamLandmarks(path, 512, 512, segment=lambda p, ph: (faces.get(ph, []), [], ""))
        L, R = s._lr("brow")
        assert L is not None and R is not None and L.cx < 256 < R.cx
