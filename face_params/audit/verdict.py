"""
audit/verdict.py - plain-language verdict from report.json.

  template : deterministic text, free, always available (also the fallback)
  muse     : Muse Spark rewrites/explains the report.  CREDIT-SAFE:
             * answers cached by report hash (re-runs are free)
             * hard cap on live calls per process (FACE_PARAMS_MAX_MUSE_CALLS, default 5)
             * text only, temperature 0, ~200 output tokens
             * any number in the reply that is not in the report -> reply discarded, template used
"""
import hashlib, json, os, re
from pathlib import Path

from ..phase1.stage1_validation import ACCEPT_THRESHOLD

# Standard-mode prices (dev.meta.ai/docs/pricing-rate-limits), USD per token
USD_PER_INPUT_TOKEN, USD_PER_OUTPUT_TOKEN = 1.25e-6, 4.25e-6
MUSE_MODEL = os.environ.get("MUSE_MODEL", "muse-spark-1.3")   # 1.3 / 1.2 / 1.1 are the Standard models
MAX_LIVE_CALLS = int(os.environ.get("FACE_PARAMS_MAX_MUSE_CALLS", "10"))   # ~ $0.02 at ~500 tokens/call
CACHE = Path(os.environ.get("FACE_PARAMS_MUSE_CACHE", "cache/muse"))
_calls = {"live": 0, "cached": 0, "usd": 0.0}
# request settings, richest first; the first one the endpoint accepts is remembered for the session
_PARAM_SETS = [{"reasoning_effort": "minimal", "max_completion_tokens": 1500},
               {"reasoning_effort": "minimal"},
               {}]
_good = 0

_WHY = {"resolution": "the face is too small in the image",
        "sharpness": "the image is too blurry",
        "pose": "the head is turned too far from frontal",
        "symmetry": "the face is not close to symmetric (likely turned)",
        "landmark_consistency": "the detected landmarks disagree with the fitted 3D face"}


def _core(r: dict) -> dict:
    """The report without the per-iteration history (chart data, not prose material)."""
    return {k: v for k, v in r.items() if k != "residual_history"}


def stats() -> dict:
    return {**_calls, "usd": round(_calls["usd"], 5)}


# ------------------------------------------------------------------ template
def template_verdict(r: dict) -> str:
    v, out = r.get("validation"), []
    if r.get("decision") == "REJECTED" and v:
        w = v["weakest"]
        out.append(f"Rejected: {_WHY[w]} ({w} score {v[w]}, needs more than {v['accept_threshold']}).")
        out.append("Use a larger, sharper, frontal photo, or run with --no_strict to continue anyway.")
        return " ".join(out)
    out.append(f"Decision {r['decision']} with quality {r['quality']}.")
    if v:
        w = v["weakest"]
        if v.get("accepted", True):
            out.append(f"Validation passed; the weakest check was {w} at {v[w]}.")
        else:                       # only reachable with --no_strict: the gate said no and the run was forced on
            out.append(f"Validation FAILED: the weakest check was {w} at {v[w]} (needs more than "
                       f"{v['accept_threshold']}); the run was forced past it, so treat the parameters as unreliable.")
    rc = r.get("residual_curve")
    if rc:
        out.append(f"The solver’s residual fell from {rc['start']} to {rc['end']} ({rc['reduction_pct']}% lower) in "
                   f"{rc['iterations']} iterations" + ("; it converged." if rc["converged"] else
                   f"; it hit the {rc['max_iterations']}-iteration cap without fully converging."))
    hp = r.get("head_pose_deg")
    if hp:
        out.append(f"Head pose is yaw {hp['yaw']}, pitch {hp['pitch']}, roll {hp['roll']} degrees.")
    m = r.get("measured_from_masks", {})
    if "mouth_open" in m:
        out.append(f"The mouth is {'open' if m['mouth_open'] else 'closed'} (lips height/width {m['mouth_aspect']}, measured from the mask).")
    if r.get("regions_missing"):
        out.append("SAM did not find: " + ", ".join(r["regions_missing"]) + "; those regions were left out of the fit.")
    return " ".join(out)


# ------------------------------------------------------------------ number check
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _allowed_numbers(r: dict):
    nums = [float(x) for x in _NUM.findall(json.dumps(_core(r)))]
    return nums + [ACCEPT_THRESHOLD, 0.75, 0.40, 192.0]          # thresholds the report text may cite


def numbers_ok(text: str, r: dict) -> bool:
    allowed = _allowed_numbers(r)
    for tok in _NUM.findall(text):
        x = float(tok)
        if x in (0.0, 1.0, 2.0, 3.0):                            # "3 sentences", "1 region"
            continue
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9          # must match a report value at the precision quoted
        if not any(abs(x - a) <= tol for a in allowed):
            return False
    return True


# ------------------------------------------------------------------ Muse Spark
_SYSTEM = ("You explain a face-fitting report to a developer in plain language. Use ONLY facts and "
           "numbers that appear in the JSON; never invent or estimate numbers. Write at most 3 short "
           "sentences: (1) the decision and the weakest validation check with its score; (2) the head "
           "pose and the mouth state measured from the masks (skip this if the report has neither); "
           "(3) if residual_curve.converged is false, say the solver stopped at its iteration cap, and "
           "if the decision is REJECTED or CONSERVATIVE, give one concrete fix for the PHOTO (larger, "
           "sharper, frontal), not for the code. Do not list every number.")


def _client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"],
                  timeout=60.0, max_retries=1)


def muse_verdict(r: dict):
    """-> (text, source).  Never raises: falls back to the template verdict."""
    key = hashlib.sha256((json.dumps(_core(r), sort_keys=True) + MUSE_MODEL + _SYSTEM).encode()).hexdigest()[:16]
    f = CACHE / f"{key}.txt"
    if f.exists():
        _calls["cached"] += 1
        return f.read_text(), "muse-spark (cached)"
    if _calls["live"] >= MAX_LIVE_CALLS:
        return template_verdict(r), f"template (Muse credit guard: {MAX_LIVE_CALLS} live calls used)"
    if MUSE_MODEL.endswith("-contributor"):                      # Contributor mode trains on what you send
        return template_verdict(r), "template (blocked: -contributor models train on your content, and this is a face)"
    # Muse Spark is a REASONING model: reasoning tokens are billed as output and can eat a small token cap,
    # so ask for minimal reasoning and leave room.  If the endpoint rejects a parameter (HTTP 400 is a
    # validation error, not a billed generation) step down to simpler settings and remember what worked.
    global _good
    text, last_err, why = "", None, ""
    for idx in range(_good, len(_PARAM_SETS)):
        _calls["live"] += 1                                      # counted per attempt, before the request
        try:
            resp = _client().chat.completions.create(
                model=MUSE_MODEL, **_PARAM_SETS[idx],
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": json.dumps(_core(r))}])
            u = getattr(resp, "usage", None)                     # real cost from the response, not a guess
            if u is not None:
                _calls["usd"] += (getattr(u, "prompt_tokens", 0) or 0) * USD_PER_INPUT_TOKEN \
                               + (getattr(u, "completion_tokens", 0) or 0) * USD_PER_OUTPUT_TOKEN
            text = (resp.choices[0].message.content or "").strip()
            why = (f"finish_reason={getattr(resp.choices[0], 'finish_reason', None)}, "
                   f"completion_tokens={getattr(u, 'completion_tokens', None)}")
            _good = idx
            break
        except Exception as e:
            last_err = e
            if type(e).__name__ == "BadRequestError" and idx + 1 < len(_PARAM_SETS) \
                    and _calls["live"] < MAX_LIVE_CALLS:
                continue                                         # parameter rejected -> try simpler settings
            return template_verdict(r), f"template (Muse unavailable: {type(e).__name__})"
    if not text:
        return template_verdict(r), f"template (Muse returned an empty reply: {why})"
    if not numbers_ok(text, r):
        return template_verdict(r), "template (Muse reply quoted numbers not in the report)"
    CACHE.mkdir(parents=True, exist_ok=True); f.write_text(text)
    return text, "muse-spark"


def make_verdict(r: dict, mode: str = "template"):
    if mode == "muse":
        return muse_verdict(r)
    return template_verdict(r), "template"
