"""
app.py — GAUSS LENS · Face-Parameter Extraction (Streamlit UI)

Run:  streamlit run app.py

Same UI as the original app (hero, startup screen, loader, About panel,
curated log panel) but the pipeline is now extraction-only:
  One image in → Phase 1  fit (S2) · validate (S1) · UV texture bake (S3) · identity (S4)
               → Phase 2  pose / eye / brow / nose / mouth / ear fit + quality gate
Shown: head pose, eye and mouth openness (measured from the SAM masks), nose / brow / ear parameters,
per-region alignment error, landmark overlay, baked UV texture, curated log.
"""

import os, sys, io, re, contextlib, tempfile, traceback, base64, math, random, html, json, time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NOTE: the heavy imports (run / pipeline / face_recogniser / phase1) now happen
# further down, behind the startup screen — see "boot sequence".


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

TITLE   = "GAUSS LENS"
TAGLINE = ("Neural nets find the face parts; pure geometry reads the 3D — a Gauss-Newton "
           "solver with no network inside it fits the face, then grades its own work.")

st.set_page_config(page_title=TITLE, page_icon="🎭", layout="centered", initial_sidebar_state="collapsed")

try:                                   # MODEL_API_KEY / MUSE_MODEL from .env (SAM + Muse Spark only)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _html_frame(doc: str, height: int) -> None:
    """Render a complete HTML document inside an isolated iframe.

    Replaces the deprecated st.components.v1.html(). st.iframe treats any
    string that isn't a URL / file path as raw HTML (srcdoc), so <svg>, SMIL
    animations, <style> and <script> all work exactly as written. Pass an
    explicit int height — the documents below are fixed-height cards with
    overflow:hidden, so auto-measuring ("content") isn't wanted.
    """
    if hasattr(st, "iframe"):                       # newer Streamlit
        st.iframe(doc, height=height)
    else:                                           # older Streamlit: no st.iframe -> use components.html
        import streamlit.components.v1 as _components
        _components.html(doc, height=height)


# ---------------------------------------------------------------------------
# Sound effects — every sound is SYNTHESISED in the browser with the Web Audio API
# (no audio files to ship). One engine lives in the main page (window.__sfx) and is
# shared with the iframes (intro, loader, hero, About panel).
#   • clicks      : any mouse click ticks; buttons thump; ▶ Run punches; READY rings the bell;
#                   tabs, expanders, toggles/checkboxes, file upload/drop and downloads each
#                   have their own sound; a disabled button buzzes; hovering controls = faint tick
#   • from Python : model connected / error · "round" bells + K.O. while the pipeline runs ·
#                   win / ok / lose jingle for PROCEED / CONSERVATIVE / FALLBACK-or-rejected
#   • a floating 🔊/🔇 button (bottom-right) mutes everything; the choice is remembered.
# Browsers block audio until the first click/tap on the page, so the opening cinematic is
# silent on a cold first visit; everything after the first click (READY ▶) is audible.
# Set GAUSS_SOUND=0 to remove sound entirely.
# ---------------------------------------------------------------------------

SOUND_ENABLED = os.environ.get("GAUSS_SOUND", "1") != "0"

_SFX_JS = r"""(function () {
  if (window.__sfx) return;
  var KEY = 'gl_sfx_muted';
  var ctx = null, master = null, noiseBuf = null;
  var muted = false, gesture = false, last = {};
  try { muted = window.localStorage.getItem(KEY) === '1'; } catch (e) {}

  /* ---------- audio context (created lazily, only after a user gesture) ---------- */
  function ensure() {
    if (ctx) return ctx;
    var C = window.AudioContext || window.webkitAudioContext;
    if (!C) return null;
    try { ctx = new C(); } catch (e) { return null; }
    master = ctx.createGain();
    master.gain.value = 0.6;
    var comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14; comp.ratio.value = 6;
    master.connect(comp); comp.connect(ctx.destination);
    var n = ctx.sampleRate, b = ctx.createBuffer(1, n, ctx.sampleRate), d = b.getChannelData(0);
    for (var i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    noiseBuf = b;
    return ctx;
  }
  function activated() {
    try { if (navigator.userActivation && navigator.userActivation.hasBeenActive) return true; } catch (e) {}
    return gesture;
  }

  /* ---------- synth primitives ---------- */
  function env(g, t0, a, d, peak) {
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(peak, t0 + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + a + d);
  }
  // pitched tone: f Hz for d seconds. o: {type, to, v, a, at, lp}
  function tone(f, d, o) {
    o = o || {};
    var c = ctx, t0 = c.currentTime + (o.at || 0), a = o.a || 0.004;
    var osc = c.createOscillator(), g = c.createGain();
    osc.type = o.type || 'sine';
    osc.frequency.setValueAtTime(f, t0);
    if (o.to) osc.frequency.exponentialRampToValueAtTime(o.to, t0 + a + d);
    env(g, t0, a, d, o.v || 0.2);
    osc.connect(g);
    var out = g;
    if (o.lp) {
      var fl = c.createBiquadFilter();
      fl.type = 'lowpass'; fl.frequency.value = o.lp;
      g.connect(fl); out = fl;
    }
    out.connect(master);
    osc.start(t0); osc.stop(t0 + a + d + 0.05);
  }
  // filtered noise burst. o: {type, f, to, q, v, a, at}
  function hit(d, o) {
    o = o || {};
    var c = ctx, t0 = c.currentTime + (o.at || 0), a = o.a || 0.003;
    var src = c.createBufferSource();
    src.buffer = noiseBuf; src.loop = true;
    var fl = c.createBiquadFilter();
    fl.type = o.type || 'bandpass';
    fl.frequency.setValueAtTime(o.f || 1000, t0);
    if (o.to) fl.frequency.exponentialRampToValueAtTime(o.to, t0 + a + d);
    fl.Q.value = o.q || 1;
    var g = c.createGain();
    env(g, t0, a, d, o.v || 0.15);
    src.connect(fl); fl.connect(g); g.connect(master);
    src.start(t0, Math.random() * 0.4); src.stop(t0 + a + d + 0.05);
  }
  // inharmonic partials = a boxing-ring bell
  function bell(f, v, at) {
    var P = [[1, 1, 2.0], [2.76, 0.5, 1.4], [5.4, 0.28, 0.9], [8.93, 0.15, 0.5]];
    for (var i = 0; i < P.length; i++)
      tone(f * P[i][0], P[i][2], { v: v * P[i][1], a: 0.002, at: at || 0 });
  }
  function jit() { return 1 + (Math.random() - 0.5) * 0.08; }   // tiny pitch jitter: no machine-gun repetition

  /* ---------- the sound bank ---------- */
  var S = {
    /* mouse / UI */
    click:   function () { var j = jit(); tone(1500 * j, 0.035, { type: 'triangle', to: 700 * j, v: 0.10 }); hit(0.02, { type: 'highpass', f: 4000, v: 0.04 }); },
    press:   function () { var j = jit(); tone(260 * j, 0.09, { to: 120, v: 0.28 }); tone(1300 * j, 0.03, { type: 'triangle', v: 0.08 }); hit(0.03, { f: 2500, q: 0.8, v: 0.06 }); },
    primary: function () { tone(190, 0.16, { to: 60, v: 0.42 }); hit(0.14, { f: 600, to: 3200, q: 1.2, v: 0.13 }); tone(900, 0.06, { type: 'triangle', v: 0.08, at: 0.03 }); },
    go:      function () { tone(330, 0.28, { type: 'square', v: 0.07, lp: 1800 }); tone(495, 0.28, { type: 'square', v: 0.06, lp: 1800, at: 0.05 }); hit(0.35, { f: 300, to: 3000, q: 0.9, v: 0.10, a: 0.12 }); bell(988, 0.14, 0.06); },
    deny:    function () { tone(200, 0.12, { type: 'square', to: 150, v: 0.09, lp: 900 }); tone(160, 0.14, { type: 'square', v: 0.09, lp: 900, at: 0.1 }); },
    hover:   function () { tone(2200 * jit(), 0.018, { v: 0.025 }); },
    tab:     function () { var j = jit(); tone(880 * j, 0.05, { type: 'triangle', v: 0.14 }); tone(1320 * j, 0.06, { type: 'triangle', v: 0.08, at: 0.035 }); },
    on:      function () { tone(520, 0.08, { type: 'triangle', to: 780, v: 0.16 }); tone(1040, 0.10, { v: 0.08, at: 0.06 }); },
    off:     function () { tone(780, 0.08, { type: 'triangle', to: 480, v: 0.14 }); tone(390, 0.10, { v: 0.07, at: 0.05 }); },
    open:    function () { hit(0.18, { f: 300, to: 2600, q: 1.4, v: 0.09 }); tone(500, 0.12, { type: 'triangle', to: 900, v: 0.07 }); },
    close:   function () { hit(0.16, { f: 2400, to: 300, q: 1.4, v: 0.08 }); tone(800, 0.10, { type: 'triangle', to: 420, v: 0.07 }); },
    upload:  function () { [523, 659, 784, 1047].forEach(function (f, i) { tone(f, 0.14, { type: 'triangle', v: 0.13, at: i * 0.06 }); }); },
    download:function () { tone(440, 0.09, { type: 'triangle', to: 880, v: 0.15 }); tone(880, 0.12, { v: 0.10, at: 0.08 }); },

    /* status */
    connect: function () { tone(880, 0.5, { v: 0.14, a: 0.006 }); tone(1318, 0.6, { v: 0.12, at: 0.09 }); tone(1760, 0.5, { v: 0.06, at: 0.18 }); },
    error:   function () { [0, 0.17].forEach(function (at) { tone(140, 0.15, { type: 'sawtooth', to: 110, v: 0.2, lp: 700, at: at }); }); },
    toast:   function () { tone(1175, 0.25, { v: 0.10 }); tone(1568, 0.30, { v: 0.08, at: 0.07 }); },
    ready:   function () { tone(784, 0.3, { v: 0.10 }); tone(1046, 0.3, { v: 0.10, at: 0.10 }); tone(1318, 0.5, { v: 0.10, at: 0.20 }); },
    tick:    function (p) { tone(500 + (p || 0) * 10, 0.03, { type: 'triangle', v: 0.05 }); },

    /* verdicts */
    win:     function () { [523, 659, 784, 1047].forEach(function (f, i) { tone(f, i === 3 ? 0.6 : 0.16, { type: 'square', v: 0.09, lp: 2400, at: i * 0.1 }); }); bell(1568, 0.08, 0.35); hit(0.7, { f: 1800, q: 0.4, v: 0.05, a: 0.2, at: 0.25 }); },
    ok:      function () { tone(659, 0.2, { type: 'triangle', v: 0.15 }); tone(784, 0.35, { type: 'triangle', v: 0.15, at: 0.14 }); },
    lose:    function () { [392, 330, 262].forEach(function (f, i) { tone(f, 0.28, { type: 'sawtooth', v: 0.12, lp: 900, at: i * 0.2 }); }); tone(220, 0.6, { type: 'sawtooth', to: 90, v: 0.12, lp: 700, at: 0.62 }); },

    /* fight-card / cinematic */
    whoosh:  function () { hit(0.5, { f: 250, to: 3200, q: 0.8, v: 0.16, a: 0.18 }); },
    punch:   function () { tone(180, 0.2, { to: 45, v: 0.5 }); hit(0.12, { type: 'lowpass', f: 2000, to: 300, v: 0.35 }); hit(0.04, { type: 'highpass', f: 3000, v: 0.15 }); },
    boom:    function () { tone(120, 0.7, { to: 30, v: 0.55 }); hit(0.45, { type: 'lowpass', f: 900, to: 120, v: 0.35 }); },
    ping:    function (k) { tone(k ? 1400 : 1000, 0.4, { v: 0.08 }); },
    siren:   function () { for (var i = 0; i < 3; i++) tone(420, 0.4, { type: 'sawtooth', to: 640, v: 0.07, lp: 1200, at: 0.2 + i * 0.55 }); },
    title:   function () { [0.2, 0.45, 0.72].forEach(function (at) { tone(150, 0.3, { to: 50, v: 0.4, at: at }); hit(0.1, { type: 'lowpass', f: 1500, v: 0.2, at: at }); }); bell(988, 0.12, 1.0); },
    curtain: function () { hit(0.8, { f: 200, to: 3500, q: 0.9, v: 0.16, a: 0.25 }); tone(110, 0.8, { to: 55, v: 0.2, at: 0.12 }); bell(740, 0.10, 0.5); },
    round:   function (n) { bell(740 + (n || 0) * 120, 0.2, 0); },
    bell2:   function () { bell(880, 0.2, 0); bell(880, 0.2, 0.32); },
    ko:      function () { S.boom(); bell(660, 0.2, 0.2); bell(660, 0.2, 0.5); bell(660, 0.2, 0.8); }
  };
  var GAP = { hover: 70, click: 30, press: 30, tab: 30, tick: 60 };   // min ms between repeats

  function play(name, arg, delay) {
    if (muted) return;
    if (delay) { setTimeout(function () { play(name, arg, 0); }, delay); return; }
    var fn = S[name]; if (!fn) return;
    var now = Date.now();
    if (GAP[name] && last[name] && now - last[name] < GAP[name]) return;
    var c = ensure(); if (!c) return;
    if (c.state === 'running') { last[name] = now; try { fn(arg || 0); } catch (e) {} return; }
    if (!activated()) return;                       // autoplay policy: stay silent until the user has clicked once
    try {
      c.resume().then(function () {
        if (c.state === 'running' && !muted) { last[name] = Date.now(); try { fn(arg || 0); } catch (e) {} }
      });
    } catch (e) {}
  }

  /* ---------- what a click means ---------- */
  function classify(t) {
    if (!t || !t.closest) return 'click';
    if (t.closest('#gl-sfx-btn')) return null;                       // the mute button makes its own sound
    if (t.closest('.st-key-fo_ready')) return 'go';                  // READY ▶ on the title page
    var btn = t.closest('button');
    var wrap = t.closest('[data-testid="stButton"],[data-testid="stDownloadButton"]');
    if (!btn && wrap) btn = wrap.querySelector('button');
    if (btn && btn.disabled) return 'deny';                          // e.g. Run before an image is uploaded
    if (t.closest('[data-testid="stDownloadButton"]')) return 'download';
    if (btn && (btn.getAttribute('data-testid') || '').indexOf('primary') > -1) return 'primary';   // ▶ Run Extraction
    if (t.closest('[role="tab"]')) return 'tab';
    if (t.closest('[data-testid="stToggle"],[data-testid="stCheckbox"]')) return null;              // 'on'/'off' comes from the change event
    var sum = t.closest('summary');
    if (sum) { var det = sum.parentElement; return (det && det.open) ? 'close' : 'open'; }
    if (btn) return 'press';
    return 'click';
  }
  function onDown(e) {
    gesture = true;
    if (e.button) return;                                            // left button / touch / pen only
    var k = classify(e.target);
    if (k) play(k);
  }
  function onKey(e) {                                                // Enter / Space on a focused control
    if (e.key !== 'Enter' && e.key !== ' ') return;
    var t = e.target;
    if (t && t.closest && t.closest('button,[role="tab"],summary')) {
      gesture = true;
      var k = classify(t); if (k) play(k);
    }
  }
  function onChange(e) {
    var t = e.target; if (!t || !t.type) return;
    if (t.type === 'file') play('upload');
    else if (t.type === 'checkbox') play(t.checked ? 'on' : 'off');
  }
  function onDrop(e) {
    if (e.target && e.target.closest && e.target.closest('[data-testid="stFileUploaderDropzone"]')) play('upload');
  }
  var lastHover = null, hoverOK = true;
  try { hoverOK = window.matchMedia('(hover: hover)').matches; } catch (e) {}
  function onOver(e) {
    if (!hoverOK) return;
    var t = e.target && e.target.closest && e.target.closest('button,[role="tab"],summary,[data-testid="stFileUploaderDropzone"]');
    if (!t || t.disabled || t.id === 'gl-sfx-btn') { lastHover = null; return; }
    if (t === lastHover) return;
    lastHover = t; play('hover');
  }

  /* ---------- wiring: main page + every same-origin iframe (hero, About panel, flowchart…) ---------- */
  function attach(doc, top) {
    if (!doc || doc.__glSfx) return;
    doc.__glSfx = 1;
    doc.addEventListener('pointerdown', onDown, true);
    doc.addEventListener('keydown', onKey, true);
    if (top) {
      doc.addEventListener('change', onChange, true);
      doc.addEventListener('drop', onDrop, true);
      doc.addEventListener('mouseover', onOver, true);
    }
  }
  function scan() {
    var fr = document.getElementsByTagName('iframe');
    for (var i = 0; i < fr.length; i++) {
      try { var d = fr[i].contentDocument; if (d && d.body && !d.__glSfx) attach(d, false); } catch (e) {}
    }
  }
  attach(document, true);
  setInterval(scan, 800);

  /* ---------- floating mute button (state remembered in localStorage) ---------- */
  function mkBtn() {
    if (!document.body || document.getElementById('gl-sfx-btn')) return;
    var b = document.createElement('button');
    b.id = 'gl-sfx-btn'; b.type = 'button';
    b.setAttribute('aria-label', 'Toggle sound effects');
    var sm = false;
    try { sm = window.matchMedia('(max-width:640px)').matches; } catch (e) {}
    b.style.cssText = 'position:fixed;right:' + (sm ? '10px' : '16px') + ';bottom:' + (sm ? 'calc(1.7rem + 10px)' : '16px') +
      ';z-index:2147483000;width:' + (sm ? '40px' : '46px') + ';height:' + (sm ? '40px' : '46px') + ';' +
      'border:3px solid #111;background:#fff;color:#111;font-size:' + (sm ? '19px' : '22px') + ';line-height:1;cursor:pointer;' +
      'box-shadow:4px 4px 0 #e3122b;padding:0;border-radius:0;touch-action:manipulation';
    function paint() {
      b.textContent = muted ? '\uD83D\uDD07' : '\uD83D\uDD0A';
      b.title = muted ? 'Sound effects: OFF (click to turn on)' : 'Sound effects: ON (click to mute)';
      b.style.opacity = muted ? '0.75' : '1';
    }
    b.addEventListener('click', function () {
      muted = !muted;
      try { window.localStorage.setItem(KEY, muted ? '1' : '0'); } catch (e) {}
      paint(); gesture = true;
      if (!muted) play('on');
    });
    paint(); document.body.appendChild(b);
  }
  mkBtn(); setInterval(mkBtn, 1500);

  window.__sfx = { play: play, list: Object.keys(S), isMuted: function () { return muted; } };
})();
"""

_SFX_CSS = """<style>
/* hidden 1px hosts for the sound-trigger iframes (kept inside the viewport so browsers still load them) */
[class*="st-key-sfxh_"] { position: fixed !important; right: 0; bottom: 0; width: 1px !important; height: 1px !important;
  min-height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: hidden; opacity: 0; pointer-events: none; z-index: -1; }
[class*="st-key-sfxh_"] iframe { width: 1px !important; height: 1px !important; border: 0 !important; }
</style>"""


def _sfx_host(key: str):
    try:
        return st.container(key=key)
    except TypeError:                       # older Streamlit: no keyed containers
        return st.container()


def _sfx_install() -> None:
    """Install the sound engine into the main page (once per browser tab; later reruns are no-ops)."""
    if not SOUND_ENABLED:
        return
    st.markdown(_SFX_CSS, unsafe_allow_html=True)
    doc = ("<!DOCTYPE html><html><body><script>(function(){try{var P=window.parent;"
           "if(P.__sfx||P.document.getElementById('gl-sfx-js'))return;"
           "var s=P.document.createElement('script');s.id='gl-sfx-js';s.text=__JS__;"
           "P.document.head.appendChild(s);}catch(e){}})();</script></body></html>"
           ).replace("__JS__", json.dumps(_SFX_JS).replace("</", "<\\/"))
    with _sfx_host("sfxh_install"):
        _html_frame(doc, height=1)


def _sfx(name: str, arg: float = 0, delay: int = 0) -> None:
    """Play one named sound in the browser (see the sound bank in _SFX_JS). Silent if muted / disabled."""
    if not SOUND_ENABLED:
        return
    n = st.session_state["_sfx_n"] = st.session_state.get("_sfx_n", 0) + 1     # unique key -> the iframe always reloads
    doc = ("<!DOCTYPE html><html><body><script>try{var S=window.parent.__sfx;"
           "if(S)S.play(__NAME__,__ARG__,__DELAY__);}catch(e){}</script><!--__N__--></body></html>"
           ).replace("__NAME__", json.dumps(name)).replace("__ARG__", str(float(arg))
           ).replace("__DELAY__", str(int(delay))).replace("__N__", str(n))
    with _sfx_host(f"sfxh_{n}"):
        _html_frame(doc, height=1)


# ---------------------------------------------------------------------------
# Look & feel — black-and-white pencil manga (boxing-manga energy), borrowed
# from demo_app.py: ink panel frames, screentone dots, pencil hatching in the
# margins, speed lines, SFX lettering. Content always sits on plain white
# "panels" so it stays readable.
# ---------------------------------------------------------------------------

def _b64(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _speed_lines(cx: int = 1060, cy: int = 175) -> str:
    """Radial 'focus lines' burst, centred behind the robot (cx, cy = burst centre)."""
    rnd = random.Random(5)
    W, H = 1200, 340
    polys = []
    for _ in range(170):
        a = rnd.uniform(0, 2 * math.pi)
        inner, outer = rnd.uniform(125, 235), rnd.uniform(420, 1100)
        w = rnd.uniform(.004, .016)
        pts = [(cx + inner * math.cos(a), cy + inner * math.sin(a)),
               (cx + outer * math.cos(a - w), cy + outer * math.sin(a - w)),
               (cx + outer * math.cos(a + w), cy + outer * math.sin(a + w))]
        polys.append("<polygon points='" + " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
                     + "' fill='#111'/>")
    return _b64(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
                f"viewBox='0 0 {W} {H}'>" + "".join(polys) + "</svg>")


# fine graphite grain for the paper
_GRAIN = _b64(
    "<svg xmlns='http://www.w3.org/2000/svg' width='260' height='260'>"
    "<filter id='g'><feTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' "
    "stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .15  0 0 0 0 .15  0 0 0 0 .15  "
    "0 0 0 .35 -.06'/></filter><rect width='100%' height='100%' filter='url(#g)'/></svg>")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Dela+Gothic+One&family=Zen+Kaku+Gothic+New:wght@400;500;700;900&display=swap');

:root { --ink: #111; --paper: #eee9d8; --red: #e3122b; }

/* ---- Paper -------------------------------------------------------------- */
[data-testid="stApp"] {
  background-color: var(--paper);
  background-image: url("__GRAIN__");
  font-family: 'Zen Kaku Gothic New', 'Segoe UI', sans-serif;
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] { position: relative; z-index: 1; }

/* pencil hatching in the side gutters */
[data-testid="stApp"]::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: repeating-linear-gradient(135deg, rgba(17,17,17,.55) 0 1.4px, transparent 1.4px 6px);
  -webkit-mask-image: linear-gradient(90deg, #000 0, transparent 9%, transparent 91%, #000 100%);
          mask-image: linear-gradient(90deg, #000 0, transparent 9%, transparent 91%, #000 100%);
}
/* screentone dots fading in from the corners */
[data-testid="stApp"]::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: radial-gradient(circle, #111 1.5px, transparent 2px),
                    radial-gradient(circle, #111 1.5px, transparent 2px);
  background-size: 9px 9px; background-position: 0 0, 4.5px 4.5px;
  -webkit-mask-image: radial-gradient(ellipse at 50% 45%, transparent 58%, #000 118%);
          mask-image: radial-gradient(ellipse at 50% 45%, transparent 58%, #000 118%);
  opacity: .8;
}

[data-testid="stMainBlockContainer"], .block-container {
  background: transparent; max-width: 1180px; padding: 2.2rem 1.5rem 3rem !important;
}
[data-testid="stMainBlockContainer"] p, [data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] label, [data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4, [data-testid="stMainBlockContainer"] h5,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #333; }

h2, h3 {
  font-family: 'Bangers', 'Zen Kaku Gothic New', sans-serif !important; font-weight: 400 !important;
  letter-spacing: .06em; font-size: 1.9rem !important;
  display: inline-block; border-bottom: 4px solid var(--ink); padding-right: 1.2rem;
}

/* ---- Tabs = manga panels ------------------------------------------------ */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 8px; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
  display: none !important; height: 0 !important; background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
  background: #fff !important; border: 3px solid var(--ink) !important; border-bottom: none !important;
  border-radius: 0 !important; padding: .35rem 1.1rem !important; height: auto !important;
}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {
  color: var(--ink) !important; font-weight: 900 !important;
}
.stTabs [data-baseweb="tab"]:hover { background: #ffe3e6 !important; }
.stTabs [aria-selected="true"] { background: var(--ink) !important; }
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
  color: #fff !important;
}

[data-baseweb="tab-panel"] {
  background-color: #fff; background-image: url("__GRAIN__");
  border: 4px solid var(--ink); outline: 2px solid var(--ink); outline-offset: 5px;
  padding: 1.6rem 1.8rem 2rem !important; margin: 0 0 1.4rem;
}

/* ---- Cards / images / alerts: inked frames ------------------------------- */
[data-testid="stMetric"], .step-card {
  background: #fff; border: 3px solid var(--ink); border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink); color: var(--ink);
}
[data-testid="stMetric"] { padding: 10px 14px; }
[data-testid="stMetricLabel"] * { text-transform: uppercase; letter-spacing: .04em; font-weight: 700; }
[data-testid="stMetricValue"] { font-family: 'Bangers', sans-serif; letter-spacing: .05em; }
.step-card { padding: 14px 16px; height: 100%; }
.step-tag {
  display: inline-block; background: var(--ink); color: #fff; padding: 1px 9px; margin-bottom: 6px;
  font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; transform: skewX(-10deg);
}
[data-testid="stAlert"], [data-testid="stExpander"] {
  border: 3px solid var(--ink) !important; border-radius: 0 !important; background: #fff;
}
[data-testid="stImage"] img { border: 4px solid var(--ink); border-radius: 0; box-shadow: 6px 6px 0 var(--ink); }
/* ---- File uploader: white manga panel + ink text (also fixes Streamlit's dark theme).
   Newer Streamlit renamed the test-id to stFileUploaderDropzone, so both are targeted. -- */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"],
[data-testid="stFileUploader"] section {
  background: #fff !important; border: 3px dashed var(--ink) !important; border-radius: 0 !important;
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploadDropzone"] *,
[data-testid="stFileUploader"] section * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] *, [data-testid="stFileUploader"] section small {
  color: #222 !important; font-weight: 700;
}
/* the Upload / Browse button: solid ink with white text */
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button {
  background: var(--ink) !important; border: 3px solid var(--ink) !important; border-radius: 0 !important;
  box-shadow: 3px 3px 0 var(--red);
}
[data-testid="stFileUploaderDropzone"] button *, [data-testid="stFileUploader"] section button * {
  color: #fff !important; fill: #fff !important; font-weight: 900;
}
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploader"] section button:hover {
  background: var(--red) !important;
}
/* rows for already-uploaded files (name / size / delete) */
[data-testid="stFileUploaderFile"] {
  background: #fff; border: 3px solid var(--ink); border-radius: 0; padding: .3rem .6rem; margin-top: .5rem;
}
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderFileName"] { font-weight: 900; }

/* ---- Buttons -------------------------------------------------------------- */
.stButton > button, [data-testid^="stBaseButton"] {
  border: 3px solid var(--ink); border-radius: 0; font-weight: 900;
  transition: transform .08s, box-shadow .08s;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--ink); box-shadow: 4px 4px 0 var(--red);
  font-family: 'Bangers', sans-serif; letter-spacing: .08em; font-size: 1.2rem;
}
[data-testid="stBaseButton-primary"] p, .stButton > button[kind="primary"] p {
  color: #fff !important; font-family: inherit; font-size: 1.2rem;
}
.stButton > button:hover { transform: translate(-2px,-2px); box-shadow: 6px 6px 0 var(--red); }
.stButton > button:active { transform: translate(3px,3px); box-shadow: 1px 1px 0 var(--red); }

/* ---- "About" pill — an inline element styled + behaving like a closable
   browser tab, sitting flush on top of the panel it controls ------------- */
.st-key-about_tabbar {
  display: flex; align-items: center;
}
.st-key-about_tabbar [data-testid="stHorizontalBlock"] {
  align-items: center; gap: 0;
}
.about-pill {
  display: inline-block; background: var(--ink); color: #fff;
  font-family: 'Bangers', sans-serif; font-size: 1.05rem; letter-spacing: .05em;
  padding: .38rem 1.1rem .3rem; margin: 0 0 -3px 6px;
  border: 3px solid var(--ink); border-bottom: none;
  border-radius: 10px 10px 0 0; box-shadow: 4px -4px 0 var(--red);
  transform: skewX(-6deg);
}
.about-pill * { transform: skewX(6deg); display: inline-block; }
.st-key-about_close, .st-key-about_open { display: flex; justify-content: flex-end; }
.st-key-about_close button, .st-key-about_open button {
  border-radius: 50% !important; width: 30px !important; height: 30px !important;
  min-width: 30px !important; padding: 0 !important; margin-bottom: -3px;
  font-weight: 900 !important; line-height: 1 !important;
}
.st-key-about_close button { background: #fff !important; }
.st-key-about_open button { background: var(--ink) !important; color: #fff !important; }
.st-key-about_open button p { color: #fff !important; }

/* ---- Loading screen = full-window overlay (only while it holds an iframe) -- */
.st-key-fo_loader:has(iframe), .st-key-fo_splash:has(iframe) {
  position: fixed !important; inset: 0; width: 100vw; height: 100vh; z-index: 500;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: rgba(17,17,17,.62); backdrop-filter: blur(3px);
}
.st-key-fo_loader:has(iframe) [data-testid="stVerticalBlock"],
.st-key-fo_splash:has(iframe) [data-testid="stVerticalBlock"] { align-items: center !important; width: 100%; }
.st-key-fo_loader:has(iframe) [data-testid="stElementContainer"],
.st-key-fo_loader:has(iframe) iframe,
.st-key-fo_splash:has(iframe) [data-testid="stElementContainer"],
.st-key-fo_splash:has(iframe) iframe { width: min(740px, 94vw) !important; max-width: 94vw; }
/* ---- Gauss Lens boot screen: true full-window cinematic (intro -> title/loader) -- */
.st-key-fo_boot:has(iframe) {
  position: fixed !important; inset: 0; width: 100vw; height: 100vh; z-index: 9999;
  background: #000; margin: 0 !important; padding: 0 !important;
}
.st-key-fo_boot:has(iframe) [data-testid="stVerticalBlock"] { gap: 0 !important; width: 100%; height: 100%; }
.st-key-fo_boot:has(iframe) [data-testid="stElementContainer"],
.st-key-fo_boot:has(iframe) iframe { width: 100vw !important; max-width: 100vw !important; height: 100vh !important; border: 0 !important; }
/* READY button: a real Streamlit button floating over the title page; the app waits for the click */
.st-key-fo_ready { position: fixed !important; right: 3vw; top: calc(8.5vh + 3.5vh); z-index: 10000; width: auto !important; }
.st-key-fo_ready button { background: var(--rdy, #e3122b) !important; border: .5vh solid #0d0d0d !important; border-radius: 0 !important;
  padding: .5em 1.4em !important; box-shadow: .5vh .6vh 0 #0d0d0d; rotate: -2deg; min-height: 0 !important; animation: fo-nud .7s ease-in-out infinite; }
.st-key-fo_ready button p { font-family: 'Bangers', sans-serif !important; font-size: min(3.2vw, 5vh) !important; letter-spacing: .14em; line-height: 1 !important;
  color: #fff !important; -webkit-text-stroke: .04em #0d0d0d; paint-order: stroke fill; }
@keyframes fo-nud { 50% { transform: translateY(-.8vh); } }
</style>
"""

st.markdown(_CSS.replace("__SPEED__", _speed_lines()).replace("__GRAIN__", _GRAIN),
            unsafe_allow_html=True)


def _accent():
    """(main, alt) colourway for this session: red/blue or blue/red, same as the boot screen."""
    st.session_state.setdefault("_boot_acc", random.choice([0, 1]))
    return ("#e3122b", "#1f4fd8") if st.session_state["_boot_acc"] == 1 else ("#1f4fd8", "#e3122b")


def _gdiv_uri(alt: str) -> str:
    """A hand-inked bell curve (hatched under the curve in the alt colour) used as a section divider."""
    pts, hat = [], []
    for i in range(101):
        x = i * 10
        y = 56 - 50 * math.exp(-.5 * ((x - 500) / 130) ** 2)
        pts.append(f"{x},{y:.1f}")
        if i % 2 == 0 and y < 54:
            hat.append(f"<path d='M{x} {y:.1f}V58' stroke='{alt}' stroke-width='2' opacity='.75'/>")
    return _b64("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 60' preserveAspectRatio='none'>"
                + "".join(hat) + "<polyline points='" + " ".join(pts)
                + "' fill='none' stroke='#0d0d0d' stroke-width='3' vector-effect='non-scaling-stroke'/>"
                "<line x1='0' y1='58' x2='1000' y2='58' stroke='#0d0d0d' stroke-width='3' "
                "vector-effect='non-scaling-stroke'/></svg>")


_ACC, _ALT = _accent()
st.markdown(f"""<style>
:root {{ --red: {_ACC}; --alt: {_ALT}; }}
.gdiv {{ height: 54px; margin: 1.2rem 0 1rem; background: url("{_gdiv_uri(_ALT)}") center / 100% 100% no-repeat; }}
h2::before, h3::before {{ content: '◢ '; color: var(--red); font-size: .75em; vertical-align: .12em; }}
.stButton > button[kind="primary"]:hover {{ background: var(--red) !important; }}
</style>""", unsafe_allow_html=True)


_CINE_CSS = """<style>
/* ===== CINEMATIC LAYER — letterbox bars, scroll-driven reveals, shutter-open ===== */
:root { --fo-bar: 2.1rem; }

/* top letterbox = Streamlit's header, restyled like the boot screen's black bar */
[data-testid="stHeader"] { background:#000 !important; height:var(--fo-bar) !important; min-height:0 !important; z-index:60; }
[data-testid="stHeader"]::before { content:'GAUSS LENS  ·  FACE · PARAMETER · EXTRACTION'; position:absolute; left:3vw; top:50%; translate:0 -50%;
  font:700 11px/1 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.22em; color:#d8d8d8; pointer-events:none; white-space:nowrap; }
[data-testid="stToolbar"], [data-testid="stToolbar"] * { color:#d8d8d8 !important; }
[data-testid="stToolbar"] { top:0 !important; height:var(--fo-bar) !important; }
/* bottom letterbox */
[data-testid="stAppViewContainer"]::after { content:'●  REC          N(μ, σ²)'; position:fixed; left:0; right:0; bottom:0; height:1.7rem; z-index:60; background:#000;
  display:flex; align-items:center; padding:0 3vw; font:700 10px/1 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.22em; color:#d8d8d8; pointer-events:none; white-space:pre; }
[data-testid="stMainBlockContainer"], .block-container { padding-top:3.4rem !important; padding-bottom:4.2rem !important; }
@media (max-width:640px) { [data-testid="stHeader"]::before { font-size:9px; letter-spacing:.12em; } }

/* hide the letterbox header/footer while the full-window boot / loader screen is showing
   (stMain has its own stacking context, so the header at z-index 60 would otherwise sit on top of it) */
[data-testid="stApp"]:has(.st-key-fo_boot iframe) [data-testid="stHeader"],
[data-testid="stApp"]:has(.st-key-fo_boot iframe) [data-testid="stToolbar"] { display:none !important; }
[data-testid="stApp"]:has(.st-key-fo_boot iframe) [data-testid="stAppViewContainer"]::after { display:none !important; }

/* shutter-open after READY: two black halves slide away, with a white flash */
.fo-curtain { position:fixed; inset:0; z-index:9000; pointer-events:none; visibility:hidden; animation:fo-show 1.5s linear both; }
/* fail-safe: hidden by default, only visible while the shutter animation runs - if animations are
   disabled/blocked the curtain can never get stuck covering the page */
.fo-curtain::before, .fo-curtain::after { content:''; position:absolute; left:0; right:0; height:50.5%; background:#000; }
.fo-curtain::before { top:0; animation:fo-up .9s .15s cubic-bezier(.7,0,.2,1) both; }
.fo-curtain::after { bottom:0; animation:fo-dn .9s .15s cubic-bezier(.7,0,.2,1) both; }
.fo-curtain i { position:absolute; inset:0; background:#fff; opacity:0; animation:fo-fl .6s .1s ease-out; }
@keyframes fo-up { to { translate:0 -101%; } } @keyframes fo-dn { to { translate:0 101%; } }
@keyframes fo-fl { 0% { opacity:.9; } 100% { opacity:0; } } @keyframes fo-gone { to { visibility:hidden; } } @keyframes fo-show { 0%,99% { visibility:visible; } 100% { visibility:hidden; } }

/* ---- reveal keyframes ---- */
@keyframes fo-rise  { from { opacity:.3; translate:0 46px; } to { opacity:1; translate:0 0; } }
@keyframes fo-stamp { from { opacity:.3; scale:1.5; rotate:-3deg; } 60% { opacity:1; } to { opacity:1; scale:1; rotate:0deg; } }
@keyframes fo-ink   { from { clip-path:inset(0 70% 0 0); } to { clip-path:inset(0 0 0 0); } }
@keyframes fo-film  { from { opacity:.3; clip-path:inset(0 70% -12px 0); scale:1.05; } to { opacity:1; clip-path:inset(-8px -14px -14px -8px); scale:1; } }
@keyframes fo-pop   { from { opacity:.3; scale:.82; translate:0 24px; } to { opacity:1; scale:1; translate:0 0; } }
@keyframes fo-in    { from { opacity:.3; } to { opacity:1; } }

/* scroll-driven (Chromium / Safari 26+): every block animates in as it enters the viewport */
@supports (animation-timeline: view()) {
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:not(.st-key-fo_boot):not(.st-key-fo_ready):not(:has(.fo-curtain)):not(:has(iframe)) {
    animation:fo-rise linear both; animation-timeline:view(); animation-range:entry 0% entry 38%; }
  [data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 {
    animation:fo-stamp linear both; animation-timeline:view(); animation-range:entry 0% entry 55%; transform-origin:left center; }
  .gdiv { animation:fo-ink linear both; animation-timeline:view(); animation-range:entry 0% entry 80%; }
  [data-testid="stImage"] img { animation:fo-film linear both; animation-timeline:view(); animation-range:entry 0% entry 70%; }
  [data-testid="stMetric"], .step-card { animation:fo-pop linear both; animation-timeline:view(); animation-range:entry 0% entry 55%; }
  [data-testid="stExpander"], [data-testid="stAlert"] { animation:fo-rise linear both; animation-timeline:view(); animation-range:entry 0% entry 45%; }
}
/* fallback: simple fade on load */
@supports not (animation-timeline: view()) {
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:not(.st-key-fo_boot):not(.st-key-fo_ready):not(:has(.fo-curtain)) { animation:fo-in .7s ease-out both; }
}

/* READY button must never be hidden/animated by the page-wide reveal rules */
.st-key-fo_ready { animation:none !important; translate:none !important; opacity:1 !important; visibility:visible !important; }
/* ---- richer interaction ---- */
.stButton > button[kind="primary"]:not(:disabled), [data-testid="stBaseButton-primary"]:not(:disabled) { animation:fo-beat 1.6s ease-in-out infinite; }
@keyframes fo-beat { 0%,100% { box-shadow:4px 4px 0 var(--red); } 50% { box-shadow:7px 7px 0 var(--red); } }
[data-testid="stMetric"]:hover, .step-card:hover { translate:-3px -3px; box-shadow:7px 7px 0 var(--red); transition:translate .1s, box-shadow .1s; }
[data-testid="stImage"] img { transition:box-shadow .15s, translate .15s; }
[data-testid="stImage"] img:hover { translate:-3px -3px; box-shadow:9px 9px 0 var(--red); }
@media (prefers-reduced-motion: reduce) {
  [data-testid="stMainBlockContainer"] *, .fo-curtain, .fo-curtain::before, .fo-curtain::after { animation:none !important; }
  .fo-curtain { display:none !important; }   /* BUGFIX: with animations off the black shutter halves never slid away -> permanent black screen */
}
</style>"""
st.markdown(_CINE_CSS, unsafe_allow_html=True)


_LIVE_CSS = """<style>
/* ===== LIVE LAYER: ticker, step tracker, drifting screentone, pulsing upload ===== */
@keyframes lv-marq { to { transform:translateX(-50%); } }
@keyframes lv-drift { to { background-position:36px 18px, 40.5px 22.5px; } }
@keyframes lv-pulse { 0%,100% { box-shadow:0 0 0 0 rgba(227,18,43,.0); } 50% { box-shadow:0 0 0 7px rgba(227,18,43,.28); } }
@keyframes lv-blink { 50% { opacity:.25; } }
@keyframes lv-wob { 0%,100% { transform:skewX(-9deg) translateY(0); } 50% { transform:skewX(-9deg) translateY(-3px); } }

/* calmer manga background: fainter hatching, dots, speed lines and skyline so text stays readable */
[data-testid="stApp"]::before {
  background-image: repeating-linear-gradient(135deg, rgba(17,17,17,.16) 0 1.4px, transparent 1.4px 6px) !important; }
[data-testid="stApp"]::after { display: none !important; content: none !important; opacity: .18 !important;
  -webkit-mask-image: linear-gradient(90deg, #000 0, transparent 8%, transparent 92%, #000 100%) !important;
          mask-image: linear-gradient(90deg, #000 0, transparent 8%, transparent 92%, #000 100%) !important; }
/* small labels / captions: full-strength ink, a bit bolder */
[data-testid="stCaptionContainer"] { opacity: 1 !important; }
[data-testid="stCaptionContainer"] *, [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label p, [data-testid="stToggle"] label p { color: #111 !important; font-weight: 700 !important; }
[data-testid="stMain"]::before { opacity: .5 !important;
  background:
    radial-gradient(circle at 91% 24%, transparent 0 16.4vmin, #0d0d0d 16.5vmin 17.1vmin, transparent 17.2vmin),
    radial-gradient(circle at 91% 24%, var(--red) 0 16.4vmin, transparent 16.5vmin),
    radial-gradient(circle at 7% 74%, transparent 0 10.9vmin, #0d0d0d 11vmin 11.6vmin, transparent 11.7vmin),
    radial-gradient(circle at 7% 74%, var(--alt) 0 10.9vmin, transparent 11vmin) !important; }
[data-testid="stAppViewContainer"]::before { opacity: .06 !important; }

/* hero banner: full-bleed, edge to edge of the screen */
[data-testid="stMain"] { overflow-x: clip; }
.st-key-hero_wide { width: 100vw !important; max-width: 100vw !important; position: relative; left: 50%; margin-left: -50vw; margin-top: -1rem; }
.st-key-hero_wide [data-testid="stElementContainer"], .st-key-hero_wide iframe { width: 100vw !important; max-width: 100vw !important; }
/* pull the hero up under the top bar: less empty space above it */
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.1rem !important; }
/* the <style>-only markdown blocks are invisible but each still adds a row of flex gap above the hero: collapse them */
[data-testid="stElementContainer"]:has(style) { display: none !important; }

/* ---- motion layer: only in the side margins, the reading area stays calm ---- */
@keyframes bg-pulse { 0%,100% { opacity: .12; } 50% { opacity: .27; } }
@keyframes bg-float { 0%,100% { translate: 0 0; } 50% { translate: 0 -1.8vh; } }
@keyframes fx-rise {
  0%   { transform: translateY(105vh) scale(.6); opacity: 0; }
  12%  { opacity: .85; } 88% { opacity: .85; }
  100% { transform: translateY(-15vh) scale(1.15); opacity: 0; } }
/* speed lines: visible (and breathing) only in the left/right margins */
[data-testid="stMain"]::after {
  -webkit-mask: linear-gradient(90deg,#000 0,#000 5%,transparent 15%,transparent 85%,#000 95%,#000 100%) center / 100vw 130vh no-repeat !important;
          mask: linear-gradient(90deg,#000 0,#000 5%,transparent 15%,transparent 85%,#000 95%,#000 100%) center / 100vw 130vh no-repeat !important;
  animation: bg-boil .35s steps(1) infinite, bg-pulse 5s ease-in-out infinite !important; }
/* the two inked suns breathe and drift gently */
[data-testid="stMain"]::before { animation: bg-throb 2.6s ease-in-out infinite, bg-float 8s ease-in-out infinite !important; }
/* rising ink bubbles in the margins */
.fxb { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }
.fxb span { position: absolute; top: 0; border-radius: 50%; border: 3px solid #0d0d0d; background: transparent;
  opacity: 0; animation: fx-rise linear infinite; }
.fxb span.f { background: var(--red); }
.fxb span.f:nth-child(even) { background: var(--alt); }

/* ---- animated skyline (replaces the old static one): drifts slowly, windows flicker, fades out behind the text column ---- */
[data-testid="stAppViewContainer"]::before { display: none !important; }
@keyframes sky-drift { to { transform: translateX(-50%); } }
@keyframes win-blink { 0%,100% { opacity: 1; } 50% { opacity: .08; } }
.sky { position: fixed; left: 0; right: 0; bottom: 1.7rem; height: 24vh; z-index: -1; pointer-events: none; overflow: hidden; opacity: .34;
  -webkit-mask: linear-gradient(90deg,#000 0,#000 10%,rgba(0,0,0,.3) 24%,rgba(0,0,0,.3) 76%,#000 90%,#000 100%);
          mask: linear-gradient(90deg,#000 0,#000 10%,rgba(0,0,0,.3) 24%,rgba(0,0,0,.3) 76%,#000 90%,#000 100%); }
.sky svg { display: block; width: 200%; height: 100%; animation: sky-drift 140s linear infinite; }
.sky rect.w { fill: #fff; animation: win-blink 4s steps(1) infinite; }
.sky rect.w.r { fill: var(--red); }

/* ---- the ABOUT block as a whole: pill floats + shines, tabs lift, frame breathes, tab content rises in ---- */
@keyframes pill-in { from { opacity: 0; translate: -50px 0; rotate: -7deg; } to { opacity: 1; translate: 0 0; rotate: 0deg; } }
@keyframes pill-bob { 0%,100% { translate: 0 0; rotate: -1.4deg; } 50% { translate: 0 -5px; rotate: 1.4deg; } }
@keyframes pill-glow { 0%,100% { box-shadow: 4px -4px 0 var(--red); } 50% { box-shadow: 10px -9px 0 var(--red); } }
@keyframes tab-hop { 0%,100% { translate: 0 0; } 50% { translate: 0 -3px; } }
@keyframes tab-glow { 0%,100% { box-shadow: 0 3px 0 var(--red); } 50% { box-shadow: 0 6px 0 var(--red); } }
@keyframes tape { to { background-position: 22.6px 0; } }
@keyframes x-ring { 0% { box-shadow: 0 0 0 0 rgba(227,18,43,.55); } 100% { box-shadow: 0 0 0 10px rgba(227,18,43,0); } }
@keyframes pill-shine { 0% { left: -60%; } 60%,100% { left: 130%; } }
@keyframes frame-breathe { 0%,100% { outline-offset: 5px; } 50% { outline-offset: 10px; } }
@keyframes tp-in { from { opacity: 0; translate: 0 18px; } to { opacity: 1; translate: 0 0; } }
.about-pill { position: relative; overflow: hidden;
  animation: pill-in .6s cubic-bezier(.2,1.4,.3,1) both, pill-bob 2.4s .6s ease-in-out infinite, pill-glow 1.8s ease-in-out infinite; }
.about-pill::after { content: ''; position: absolute; top: 0; bottom: 0; width: 35%; left: -60%; pointer-events: none;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.5), transparent); animation: pill-shine 3.6s ease-in-out infinite; }
:is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]) { transition: translate .12s, background .15s; }
:is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]) { animation: tab-hop 2.6s ease-in-out infinite; }
:is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]):nth-child(2) { animation-delay: -.9s; }
:is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]):nth-child(3) { animation-delay: -1.8s; }
.stTabs [aria-selected="true"] { animation: tab-hop 2.6s ease-in-out infinite, tab-glow 1.4s ease-in-out infinite; }
:is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]):hover { animation-play-state: paused; translate: 0 -3px; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) { position: relative; animation: frame-breathe 3.2s ease-in-out infinite; }
/* moving hazard-tape strip along the top edge of the panel */
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"])::before { content: ''; position: absolute; left: -4px; right: -4px; top: -4px; height: 9px; pointer-events: none;
  background: repeating-linear-gradient(135deg, var(--ink) 0 8px, var(--red) 8px 16px); background-size: 22.6px 100%;
  animation: tape 1s linear infinite; }
/* the close / open button pulses a ring and spins on hover */
.st-key-about_close button, .st-key-about_open button { animation: x-ring 1.6s ease-out infinite; transition: rotate .25s; }
.st-key-about_close button:hover, .st-key-about_open button:hover { rotate: 90deg; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"] { animation: tp-in .5s ease-out both; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"]:nth-child(2) { animation-delay: .08s; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"]:nth-child(3) { animation-delay: .16s; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"]:nth-child(4) { animation-delay: .24s; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"]:nth-child(5) { animation-delay: .32s; }
:is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"]:nth-child(n+6) { animation-delay: .4s; }

/* ---- animated section headers: a red ink streak runs along the underline, the ◢ marker sways ---- */
@keyframes hd-run  { from { left: 0; } to { left: 62%; } }
@keyframes hd-sway { from { transform: rotate(-10deg) translateY(0); } to { transform: rotate(10deg) translateY(-2px); } }
@keyframes btn-shine { 0% { left: -45%; } 55%, 100% { left: 135%; } }
h2, h3 { position: relative; }
h2::after, h3::after {
  content: ''; position: absolute; left: 0; bottom: -4px; width: 38%; height: 4px; background: var(--red);
  animation: hd-run 2.8s ease-in-out infinite alternate; }
h2::before, h3::before { display: inline-block; transform-origin: 40% 70%; animation: hd-sway 1.6s ease-in-out infinite alternate; }
h2:hover::after, h3:hover::after { width: 100%; animation: none; left: 0; transition: width .25s; }
/* the Run button gets a diagonal shine sweep */
.stButton > button[kind="primary"]:not(:disabled) { position: relative; overflow: hidden; }
.stButton > button[kind="primary"]:not(:disabled)::after {
  content: ''; position: absolute; top: 0; bottom: 0; width: 28%; left: -45%; pointer-events: none;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.42), transparent); transform: skewX(-14deg);
  animation: btn-shine 3.4s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) {
  .fxb { display: none; }
  h2::before, h3::before, h2::after, h3::after, .stButton > button::after { animation: none !important; }
  .sky svg, .sky rect.w { animation: none !important; }
  .about-pill, .about-pill::after, :is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]), :is([data-baseweb="tab-panel"], [data-testid="stTabPanel"])::before, :is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"] { animation: none !important; }
  :is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]), .stTabs [aria-selected="true"], .st-key-about_close button, .st-key-about_open button { animation: none !important; }
  [data-testid="stMain"]::before, [data-testid="stMain"]::after { animation: none !important; }
}

/* screentone dots in the corners slowly drift */
[data-testid="stApp"]::after { animation:lv-drift 6s linear infinite; }

/* scrolling ticker tape */
.ticker { overflow:hidden; background:var(--ink); border:3px solid var(--ink); box-shadow:5px 5px 0 var(--red);
  margin:.2rem 0 1.1rem; transform:skewX(-2deg); }
.ticker .track { display:inline-flex; white-space:nowrap; width:max-content; animation:lv-marq 28s linear infinite; }
.ticker:hover .track { animation-play-state:paused; }
.ticker span { font-family:'Bangers',sans-serif; letter-spacing:.12em; font-size:1.05rem; color:#fff; padding:.28rem 1.1rem; }
.ticker span b { color:var(--red); font-weight:400; margin-left:2.2rem; }

/* step tracker */
.stepper { display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; margin:0 0 1.2rem; }
.stepper .st { font-family:'Bangers',sans-serif; letter-spacing:.07em; font-size:1.05rem; padding:.18rem .95rem;
  border:3px solid var(--ink); background:#fff; color:var(--ink); transform:skewX(-9deg); transition:all .25s; }
.stepper .st.done { background:var(--ink); color:#fff; }
.stepper .st.cur { background:#fff; color:var(--ink); animation:lv-wob .9s ease-in-out infinite; }
.stepper .ar { font-family:'Bangers',sans-serif; color:var(--ink); }
.stepper .hint { margin-left:auto; font-weight:900; font-size:.85rem; }
.stepper .hint i { font-style:normal; color:var(--red); animation:lv-blink 1s steps(1) infinite; }

/* empty uploader asks for attention until a file is chosen */
[data-testid="stFileUploader"]:not(:has([data-testid="stFileUploaderFile"])) [data-testid="stFileUploaderDropzone"] {
  animation:lv-pulse 1.8s ease-in-out infinite; }
[data-testid="stFileUploaderDropzone"]:hover { background:#ffe3e6 !important; }

/* toggles + expanders react to the mouse */
[data-testid="stToggle"] { transition:translate .12s; } [data-testid="stToggle"]:hover { translate:3px 0; }
[data-testid="stExpander"] { transition:box-shadow .15s, translate .15s; }
[data-testid="stExpander"]:hover { box-shadow:5px 5px 0 var(--red); translate:-2px -2px; }
h2:hover, h3:hover { border-bottom-color:var(--red); transition:border-color .2s; }
@media (prefers-reduced-motion: reduce) {
  [data-testid="stApp"]::after, .ticker .track, .stepper .st.cur, [data-testid="stFileUploaderDropzone"] { animation:none !important; }
}
</style>"""
st.markdown(_LIVE_CSS, unsafe_allow_html=True)


# ---- page background: same visual language as the boot screen (burst lines, suns, skyline) ----
def _skyline_uri() -> str:
    rnd, x, parts = random.Random(21), 0.0, []
    while x < 100:
        w, t = 2 + rnd.random() * 5, 8 + rnd.random() * 26
        parts.append(f"<rect x='{x:.1f}' y='{40 - t:.1f}' width='{w + .2:.1f}' height='{t:.1f}' fill='#0d0d0d'/>")
        if rnd.random() < .4:
            parts.append(f"<rect x='{x + w / 2 - .3:.1f}' y='{40 - t - 4:.1f}' width='.6' height='4' fill='#0d0d0d'/>")
        x += w
    return _b64("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 40' preserveAspectRatio='none'>"
                + "".join(parts) + "</svg>")


_BG_CSS = """<style>
@keyframes bg-boil { 0% { rotate:0deg; } 34% { rotate:.9deg; scale:1.015; } 67% { rotate:-.8deg; } }
@keyframes bg-throb { 0%,100% { scale:1; } 50% { scale:1.05; } }

/* radiating speed lines behind everything (mostly seen in the side gutters) */
[data-testid="stMain"]::after {
  content:''; position:fixed; left:50%; top:42%; width:170vmax; height:170vmax; translate:-50% -50%;
  z-index:-2; pointer-events:none; opacity:.17;
  background:repeating-conic-gradient(from 3deg,#0d0d0d 0 1.1deg,transparent 1.1deg 3.9deg,#0d0d0d 3.9deg 4.4deg,transparent 4.4deg 9deg);
  -webkit-mask:radial-gradient(circle,transparent 9%,#000 30%); mask:radial-gradient(circle,transparent 9%,#000 30%);
  animation:bg-boil .35s steps(1) infinite;
}
/* two inked suns with halftone shading, one in each gutter */
[data-testid="stMain"]::before {
  content:''; position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.9;
  background:
    radial-gradient(circle at 91% 24%, transparent 0 16.4vmin, #0d0d0d 16.5vmin 17.1vmin, transparent 17.2vmin),
    radial-gradient(circle at 91% 24%, #0d0d0d 1.4px, transparent 2px) 0 0 / 9px 9px,
    radial-gradient(circle at 91% 24%, var(--red) 0 16.4vmin, transparent 16.5vmin),
    radial-gradient(circle at 7% 74%, transparent 0 10.9vmin, #0d0d0d 11vmin 11.6vmin, transparent 11.7vmin),
    radial-gradient(circle at 7% 74%, var(--alt) 0 10.9vmin, transparent 11vmin);
  -webkit-mask:linear-gradient(90deg,#000 0,#000 22%,transparent 30%,transparent 70%,#000 78%,#000 100%);
          mask:linear-gradient(90deg,#000 0,#000 22%,transparent 30%,transparent 70%,#000 78%,#000 100%);
  animation:bg-throb 1.6s ease-in-out infinite;
}
/* skyline silhouette along the bottom */
[data-testid="stAppViewContainer"]::before {
  content:''; position:fixed; left:0; right:0; bottom:1.7rem; height:24vh; z-index:0; pointer-events:none; opacity:.2;
  background:url("__SKY__") bottom / 100% 100% no-repeat;
}
@media (max-width:900px) { [data-testid="stMain"]::before { opacity:.35; } }
@media (prefers-reduced-motion: reduce) { [data-testid="stMain"]::before, [data-testid="stMain"]::after { animation:none !important; } }
</style>"""
st.markdown(_BG_CSS.replace("__SKY__", _skyline_uri()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Mobile layer — phones / small tablets. Layout, tap targets, touch (no hover),
# dynamic-viewport overlays and lighter animation. Desktop is unchanged.
# ---------------------------------------------------------------------------
_MOBILE_CSS = """<style>
html { -webkit-text-size-adjust: 100%; }
[data-testid="stApp"] { overflow-x: hidden; }

/* full-window overlays: 100vh is taller than the visible area on phone browsers (URL bar) -> use dvh */
@supports (height: 100dvh) {
  .st-key-fo_loader:has(iframe), .st-key-fo_splash:has(iframe), .st-key-fo_boot:has(iframe) { height: 100dvh; }
  .st-key-fo_boot:has(iframe) [data-testid="stElementContainer"],
  .st-key-fo_boot:has(iframe) iframe { height: 100dvh !important; }
}

/* touch screens: no sticky hover lifts */
@media (hover: none) {
  .stButton > button:hover { transform: none; }
  .stButton > button:not([kind="primary"]):hover { box-shadow: none; }
  .stButton > button[kind="primary"]:hover { box-shadow: 4px 4px 0 var(--red); }
  [data-testid="stMetric"]:hover, .step-card:hover { translate: none; box-shadow: 4px 4px 0 var(--ink); }
  [data-testid="stImage"] img:hover { translate: none; box-shadow: 6px 6px 0 var(--ink); }
  [data-testid="stExpander"]:hover { translate: none; box-shadow: none; }
  [data-testid="stToggle"]:hover { translate: none; }
  .st-key-about_close button:hover, .st-key-about_open button:hover { rotate: none; }
}

@media (max-width: 768px) {
  /* lighter page: drop the heavy ambient animation (bubbles, skyline, rotating speed lines) */
  .fxb, .sky { display: none !important; }
  [data-testid="stMain"]::before, [data-testid="stMain"]::after { animation: none !important; }
  :is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]),
  :is([data-baseweb="tab-panel"], [data-testid="stTabPanel"]) [data-testid="stElementContainer"],
  :is(.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]), .stTabs [aria-selected="true"],
  .about-pill, .about-pill::after { animation: none !important; }
  h2::after, h3::after { animation: none !important; width: 38%; }

  [data-testid="stMainBlockContainer"], .block-container {
    padding-left: 1rem !important; padding-right: 1rem !important; padding-bottom: 5rem !important; }
  h2, h3 { font-size: 1.5rem !important; padding-right: .8rem; }

  /* About tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 5px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .stTabs [data-baseweb="tab"] { padding: .3rem .7rem !important; white-space: nowrap; }
  [data-baseweb="tab-panel"], [data-testid="stTabPanel"] { padding: 1rem .8rem 1.2rem !important; outline-offset: 3px; }
  .katex-display { overflow-x: auto; overflow-y: hidden; padding-bottom: 4px; }
  .katex { font-size: .95em; }

  /* cards / metrics */
  [data-testid="stMetric"] { padding: 8px 10px; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
  [data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; }

  /* tap targets */
  .stButton > button, [data-testid^="stBaseButton"] { min-height: 2.8rem; }
  .stButton > button[kind="primary"], [data-testid="stDownloadButton"] button { width: 100%; }
  .st-key-about_close button, .st-key-about_open button { width: 40px !important; height: 40px !important; min-width: 40px !important; min-height: 40px !important; }
  [data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] section button { min-height: 2.8rem; }

  /* step tracker + ticker */
  .stepper { gap: .4rem; }
  .stepper .st { font-size: .9rem; padding: .15rem .6rem; }
  .stepper .hint { margin-left: 0; width: 100%; }
  .ticker span { font-size: .9rem; }
  .about-pill { font-size: .95rem; padding: .3rem .8rem .25rem; }
}

@media (max-width: 640px) {
  /* columns: one per row (metrics: two per row); the About pill + its close button stay on one line */
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .75rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"], [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width: 100% !important; flex: 1 1 100% !important; width: 100% !important; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="column"] {
    min-width: calc(50% - .5rem) !important; flex: 1 1 calc(50% - .5rem) !important; width: calc(50% - .5rem) !important; }
  .st-key-about_tabbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0 !important; }
  .st-key-about_tabbar [data-testid="stColumn"]:first-child, .st-key-about_tabbar [data-testid="column"]:first-child {
    min-width: 0 !important; flex: 1 1 auto !important; width: auto !important; }
  .st-key-about_tabbar [data-testid="stColumn"]:last-child, .st-key-about_tabbar [data-testid="column"]:last-child {
    min-width: 44px !important; flex: 0 0 44px !important; width: 44px !important; }

  /* READY button on the title page: big enough to tap */
  .st-key-fo_ready button p { font-size: max(18px, min(3.2vw, 5vh)) !important; }
  .st-key-fo_ready button { min-height: 44px !important; }

  /* About / flowchart frames: wrapper and iframe must be the same height or the iframe spills over the next block */
  .st-key-about_frame [data-testid="stElementContainer"], .st-key-about_frame iframe { height: 1500px !important; min-height: 1500px !important; }
  .st-key-flow_frame [data-testid="stElementContainer"], .st-key-flow_frame iframe { height: 600px !important; }
  .st-key-about_frame, .st-key-flow_frame { overflow: hidden; }
}

@media (max-width: 480px) {
  [data-testid="stHeader"]::before { content: 'GAUSS LENS  ·  FACE EXTRACTION'; font-size: 9px; letter-spacing: .1em; }
}
</style>"""
st.markdown(_MOBILE_CSS, unsafe_allow_html=True)


# iframe auto-height for narrow screens: the About / flowchart cards are fixed-height on desktop, but on a
# phone their content wraps and gets taller. Runs inside the iframe (same-origin); no-op on wide screens.
_AUTOFIT_JS = """<script>
(function () {
  function fit() {
    try {
      var f = window.frameElement, w = document.body && document.body.firstElementChild;
      if (!f || !w || window.innerWidth >= 700) return;
      var h = Math.ceil(w.getBoundingClientRect().bottom) + 24, old = f.getBoundingClientRect().height;
      if (Math.abs(old - h) < 3) return;
      /* Streamlit locks the iframe's wrapper(s) to the original height too: grow them as well, or the
         taller iframe spills over whatever comes next (the overlap bug). Only ancestors that match the old height. */
      var a = f.parentElement, n = 0;
      while (a && n++ < 5 && a.tagName !== 'BODY') {
        if (Math.abs(a.getBoundingClientRect().height - old) < 4) {
          a.style.setProperty('height', h + 'px', 'important');
          a.style.setProperty('min-height', h + 'px', 'important');
        }
        a = a.parentElement;
      }
      f.style.setProperty('height', h + 'px', 'important'); f.height = h;
    } catch (e) {}
  }
  window.addEventListener('load', fit); window.addEventListener('resize', fit);
  try { if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit); } catch (e) {}
  try { new ResizeObserver(fit).observe(document.body.firstElementChild); } catch (e) {}
  setTimeout(fit, 60); setTimeout(fit, 400); setTimeout(fit, 1200);
})();
</script>"""


def _ticker() -> None:
    items = ["SAM 3.1 · FIND THE FACE PARTS", "27 LANDMARKS", "GAUSS-NEWTON · NO NETWORK INSIDE",
             "HEAD POSE · EYES · BROWS · NOSE · MOUTH", "QUALITY GATE c_t", "MUSE SPARK · AUDIT VERDICT",
             "PROCEED / CONSERVATIVE / FALLBACK"]
    one = "".join(f"<span>{html.escape(t)}<b>◢</b></span>" for t in items)
    st.markdown(f"<div class='ticker'><div class='track'>{one}{one}</div></div>", unsafe_allow_html=True)


def _stepper_html(step: int) -> str:
    names = ["UPLOAD", "CONNECT", "RUN", "RESULT"]
    hints = ["Drop a face photo below", "Switch on SAM 3.1 + Muse Spark", "Hit ▶ Run Extraction", "Scroll down for the breakdown"]
    out = []
    for i, n in enumerate(names):
        cls = "done" if i < step else "cur" if i == step else ""
        out.append(f"<span class='st {cls}'>{'✓ ' if i < step else ''}{i + 1} · {n}</span>")
        if i < len(names) - 1:
            out.append("<span class='ar'>▶</span>")
    return f"<div class='stepper'>{''.join(out)}<span class='hint'><i>●</i> {hints[step]}</span></div>"


def _gdiv() -> None:
    """Plain breathing space (the old bell-curve divider was removed)."""
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Robot mascot — inked boxer bot (screentone shading, red gloves). Instead of
# sitting still in the corner, the whole bot steps/shuffles left-to-right —
# a boxer shifting its weight — via an outer translate animation. Animated
# with SMIL so it works inside an <img>.
# ---------------------------------------------------------------------------

_SK = "stroke='#111' stroke-width='4.5' stroke-linejoin='round' stroke-linecap='round'"
_SK3 = "stroke='#111' stroke-width='3' stroke-linejoin='round' stroke-linecap='round'"


_BOT_DEFS = """<defs>
<pattern id='dots' width='5' height='5' patternUnits='userSpaceOnUse'><circle cx='2.5' cy='2.5' r='1.15' fill='#111'/></pattern>
<pattern id='hatch' width='4.5' height='4.5' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'><line x1='0' y1='0' x2='0' y2='4.5' stroke='#111' stroke-width='1.3'/></pattern>
<clipPath id='hc'><rect x='36' y='38' width='108' height='74' rx='22'/></clipPath>
<clipPath id='bc'><rect x='54' y='112' width='72' height='48' rx='14'/></clipPath>
</defs>"""

_BOT_BODY = f"""
<ellipse cx='90' cy='178' rx='48' ry='6' fill='url(#hatch)' opacity='.6'/>
<path d='M80 13 L74 7 M100 13 L106 7 M90 11 L90 4' stroke='#111' stroke-width='3' stroke-linecap='round'>
  <animate attributeName='opacity' values='1;.2;1' dur='.7s' repeatCount='indefinite'/></path>
<line x1='90' y1='26' x2='90' y2='38' {_SK}/>
<circle cx='90' cy='20' r='6' fill='#fff' {_SK}/>
<rect x='66' y='158' width='18' height='14' rx='4' fill='#fff' {_SK}/>
<rect x='98' y='158' width='18' height='14' rx='4' fill='#fff' {_SK}/>
<rect x='62' y='169' width='26' height='8' rx='3' fill='#111'/>
<rect x='94' y='169' width='26' height='8' rx='3' fill='#111'/>
<line x1='58' y1='130' x2='44' y2='114' stroke='#111' stroke-width='9' stroke-linecap='round'/>
<rect x='54' y='112' width='72' height='48' rx='14' fill='#fff' {_SK}/>
<g clip-path='url(#bc)'>
  <path d='M104 112 H126 V160 H96 Z' fill='url(#dots)'/>
  <rect x='54' y='148' width='72' height='7' fill='#111'/>
</g>
<circle cx='90' cy='130' r='9' fill='#111'/>
<circle cx='90' cy='130' r='4' fill='#fff'>
  <animate attributeName='r' values='3;5;3' dur='.9s' repeatCount='indefinite'/></circle>
<rect x='26' y='64' width='12' height='28' rx='4' fill='#fff' {_SK3}/>
<rect x='142' y='64' width='12' height='28' rx='4' fill='#fff' {_SK3}/>
<rect x='36' y='38' width='108' height='74' rx='22' fill='#fff' {_SK}/>
<g clip-path='url(#hc)'><path d='M112 38 H144 V112 H100 Q118 78 112 38Z' fill='url(#hatch)'/></g>
<rect x='46' y='48' width='88' height='54' rx='14' fill='#111'/>
<path d='M52 54 L72 52 L58 66Z' fill='#fff' opacity='.14'/>
<ellipse cx='68' cy='78' rx='10' ry='9' fill='#fff'>
  <animate attributeName='ry' values='9;9;1;9;9' keyTimes='0;0.9;0.94;0.98;1' dur='3.6s' repeatCount='indefinite'/></ellipse>
<ellipse cx='112' cy='78' rx='10' ry='9' fill='#fff'>
  <animate attributeName='ry' values='9;9;1;9;9' keyTimes='0;0.9;0.94;0.98;1' dur='3.6s' repeatCount='indefinite'/></ellipse>
<circle cx='70' cy='79' r='5' fill='#111'/><circle cx='110' cy='79' r='5' fill='#111'/>
<circle cx='68' cy='76' r='2' fill='#fff'>
  <animate attributeName='r' values='2;2;0;2;2' keyTimes='0;0.9;0.94;0.98;1' dur='3.6s' repeatCount='indefinite'/></circle>
<circle cx='108' cy='76' r='2' fill='#fff'>
  <animate attributeName='r' values='2;2;0;2;2' keyTimes='0;0.9;0.94;0.98;1' dur='3.6s' repeatCount='indefinite'/></circle>
<path d='M53 60 L80 65 M127 60 L100 65' stroke='#fff' stroke-width='5' stroke-linecap='round'/>
<rect x='78' y='91' width='24' height='7' rx='2' fill='#fff'/>
<path d='M84 91 V98 M90 91 V98 M96 91 V98' stroke='#111' stroke-width='1.5'/>
<path d='M152 44 Q160 58 152 64 Q144 58 152 44Z' fill='#fff' {_SK3.replace("width='3'", "width='2.5'")}>
  <animateTransform attributeName='transform' type='translate' values='0 0;0 10;0 10' keyTimes='0;.7;1' dur='1.6s' repeatCount='indefinite'/>
  <animate attributeName='opacity' values='1;1;0' keyTimes='0;.7;1' dur='1.6s' repeatCount='indefinite'/></path>
"""


def _glove(cx, cy, r=16):
    return (f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='#e3122b' {_SK}/>"
            f"<path d='M{cx - r + 5} {cy - 6} Q{cx - 2} {cy - r + 1} {cx + 6} {cy - r + 3}' "
            f"stroke='#fff' stroke-width='3' stroke-linecap='round' fill='none' opacity='.9'/>")


# left glove always guards the face
_BOT_LEFT = f"""
<g><animateTransform attributeName='transform' type='translate' values='0 0;0 -4;0 0' dur='.9s' repeatCount='indefinite'/>
  {_glove(40, 108)}
  <rect x='30' y='120' width='20' height='8' rx='3' fill='#fff' {_SK3}/></g>
"""

# "input" robot: full guard, both gloves up
_BOT_GUARD = f"""
<line x1='122' y1='130' x2='136' y2='114' stroke='#111' stroke-width='9' stroke-linecap='round'/>
<g><animateTransform attributeName='transform' type='translate' values='0 0;0 -4;0 0' dur='.9s' begin='.45s' repeatCount='indefinite'/>
  {_glove(140, 108)}
  <rect x='130' y='120' width='20' height='8' rx='3' fill='#fff' {_SK3}/></g>
"""

def _bot_svg(suffix: str) -> str:
    """Inline SVG markup for one bot (NOT a base64 <img> data URI).

    This has to be inlined directly into the page's HTML rather than loaded
    via <img src="data:image/svg+xml;..."> — browsers render an SVG that is
    referenced from an <img> (or a CSS background-image) in a restricted
    "static image" mode that disables SMIL animation (<animate>,
    <animateTransform>). That's why the blink / glove-bounce / antenna-flick
    baked into _BOT_BODY / _BOT_LEFT / _BOT_GUARD never played: as an <img>,
    every one of those <animate> tags was silently ignored, so the bot itself
    only ever showed a single static frame. Inlining the <svg> puts it in the
    real DOM, where SMIL runs normally.

    `suffix` renames every id (`dots`, `hatch`, `hc`, `bc`) and every url(#…)
    reference that points at them, so the two copies placed side by side in
    the hero panel don't collide on the same ids (an SVG `url(#id)` always
    resolves to the *first* matching id in the document, so two unpatched
    copies would both end up cloning the first bot's clip-path/pattern).
    """
    body = _BOT_DEFS + _BOT_BODY + _BOT_LEFT + _BOT_GUARD
    for _id in ("dots", "hatch", "hc", "bc"):
        body = body.replace(f"id='{_id}'", f"id='{_id}-{suffix}'")
        body = body.replace(f"url(#{_id})", f"url(#{_id}-{suffix})")
    # Explicit width/height (matching the viewBox exactly) are required in
    # addition to the CSS width below: without them, a bare <svg> with only
    # a viewBox can fall back to a tiny host-page default box (Streamlit's
    # own styles reset inline svg sizing like an icon), and since the SVG
    # root defaults to overflow:hidden, anything outside that shrunken box
    # — the gloves, the lower body — gets silently clipped instead of
    # scaled. Setting both gives every browser an unambiguous 230x190
    # intrinsic size to scale down from.
    return (f"<svg class='hero-bot bot-{suffix}' viewBox='0 0 230 190' "
            f"width='230' height='190' preserveAspectRatio='xMidYMid meet' "
            f"xmlns='http://www.w3.org/2000/svg'>{body}</svg>")


_HERO_EYE = (
    "<svg viewBox='0 -50 640 350' xmlns='http://www.w3.org/2000/svg'><defs><clipPath id='heo'><path d='M44 172C132 92 232 60 320 60S508 92 596 172C516 226 420 250 320 250S124 226 44 172Z'/></clipPath></defs>"
    "<path d='M44 172C132 92 232 60 320 60S508 92 596 172C516 226 420 250 320 250S124 226 44 172Z' transform='translate(11 13)' fill='#111'/>"
    "<path d='M44 172C132 92 232 60 320 60S508 92 596 172C516 226 420 250 320 250S124 226 44 172Z' fill='#fff'/>"
    "<g clip-path='url(#heo)'><g transform='translate(320 156)'><circle r='112' fill='#111'/>"
    "<g stroke='#fff' stroke-width='3.2' stroke-linecap='round'><line x1='46' y1='0' x2='46' y2='92'/><line x1='0' y1='46' x2='-92' y2='46'/><line x1='-46' y1='0' x2='-46' y2='-92'/><line x1='0' y1='-46' x2='92' y2='-46'/>"
    "<animateTransform attributeName='transform' type='rotate' from='0' to='360' dur='18s' repeatCount='indefinite'/></g>"
    "<circle r='48' fill='#111' stroke='#fff' stroke-width='3'/><circle cy='-20' r='7' fill='__MAIN__' stroke='#fff' stroke-width='2'><animate attributeName='r' values='6;9;6' dur='1.1s' repeatCount='indefinite'/></circle>"
    "<circle cx='-48' cy='-52' r='19' fill='#fff'/></g></g>"
    "<path d='M20 176C112 56 232 22 320 22S528 56 620 176L596 172C508 92 408 60 320 60S132 92 44 172Z' fill='#111'/>"
    "<path d='M44 172C132 92 232 60 320 60S508 92 596 172C516 226 420 250 320 250S124 226 44 172Z' fill='none' stroke='#111' stroke-width='6'/></svg>")

_HERO_TPL = r"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Zen+Kaku+Gothic+New:wght@700;900&display=swap');
:root { --ink:#0d0d0d; --paper:#eee9d8; --red:__MAIN__; --alt:__ALT__; --mx:0; --my:0; }
html, body { margin:0; padding:0; height:100%; background:transparent; overflow:hidden; }
* { box-sizing:border-box; }
.film { position:absolute; inset:0 0 8px 0; background:#000; border:5px solid var(--ink); border-width:5px 0; box-shadow:0 7px 0 var(--ink);
  display:flex; flex-direction:column; overflow:hidden; font-family:'Zen Kaku Gothic New',sans-serif; }
/* letterbox bars (same language as the boot screen) */
.bar { flex:0 0 27px; z-index:30; background:#000; color:#d8d8d8; display:flex; align-items:center; justify-content:space-between; padding:0 18px;
  font:700 11px/1 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.22em; text-transform:uppercase; white-space:nowrap; overflow:hidden; }
.bar .rec { color:#fff; } .bar .rec::first-letter { color:var(--red); animation:blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity:0; } }
.film > .bar:first-child { display:none; }   /* top bar (GAUSS LENS / timecode / REC) removed */
#sub { flex:1; text-align:center; letter-spacing:.06em; text-transform:none; font:700 13px/1 'Zen Kaku Gothic New',sans-serif; color:#fff; transition:opacity .35s; }
.stage { position:relative; flex:1; overflow:hidden; background:var(--paper); display:flex; align-items:center; justify-content:center; gap:5vw; padding:0 4%; animation:sk .45s .62s linear; }
@keyframes sk { 12%{translate:-7px 4px} 26%{translate:6px -5px} 40%{translate:-5px -3px} 55%{translate:4px 4px} 72%{translate:-2px 1px} }
.burst { position:absolute; z-index:-1; left:50%; top:50%; width:max(2600px,240vw); height:max(2600px,240vw); translate:calc(-50% + var(--mx)*-16px) calc(-50% + var(--my)*-10px);
  animation:boil .3s steps(1) infinite, bin 1.1s .3s cubic-bezier(.2,.8,.2,1) both;
  background:repeating-conic-gradient(from 3deg,var(--ink) 0 1.1deg,transparent 1.1deg 3.9deg,var(--ink) 3.9deg 4.4deg,transparent 4.4deg 9deg);
  -webkit-mask:radial-gradient(circle,transparent 5%,#000 17%); mask:radial-gradient(circle,transparent 5%,#000 17%); }
@keyframes boil { 0% { rotate:0deg; } 34% { rotate:.9deg; scale:1.015; } 67% { rotate:-.8deg; } }
@keyframes bin { from { opacity:0; scale:.2; } to { opacity:1; scale:1; } }
.tone { position:absolute; inset:0; z-index:25; pointer-events:none; opacity:.6; background-image:radial-gradient(circle,#0d0d0d 1.4px,transparent 1.9px); background-size:8px 8px;
  -webkit-mask:radial-gradient(ellipse at 40% 50%,transparent 58%,#000 130%); mask:radial-gradient(ellipse at 40% 50%,transparent 58%,#000 130%); }
.vig { position:absolute; inset:0; z-index:26; pointer-events:none; background:radial-gradient(ellipse at 45% 50%,transparent 55%,rgba(0,0,0,.55) 100%); }
.grain { position:absolute; inset:-20px; z-index:27; pointer-events:none; opacity:.09; mix-blend-mode:multiply; animation:gr .5s steps(1) infinite;
  background-image:url("__GRAIN__"); }
@keyframes gr { 0%{translate:0 0} 25%{translate:-9px 6px} 50%{translate:7px -8px} 75%{translate:-4px -3px} }
.sweep { position:absolute; top:0; bottom:0; width:22%; z-index:24; pointer-events:none; left:-30%; skewX:-12deg; transform:skewX(-14deg);
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent); animation:sw 1.3s 1.05s ease-in-out both, sw 1.3s 9s ease-in-out infinite; }
@keyframes sw { from { left:-30%; } to { left:130%; } }
.flash { position:absolute; inset:0; z-index:28; background:#fff; opacity:0; pointer-events:none; animation:fl .55s .6s ease-out; }
@keyframes fl { 0%{opacity:1} 100%{opacity:0} }
.cover { position:absolute; inset:0; z-index:40; background:#000; pointer-events:none; animation:cv .7s .1s ease-in both; }
@keyframes cv { to { opacity:0; visibility:hidden; } }

.eyebox { position:relative; z-index:2; flex:0 0 min(38%,360px); height:100%; display:flex; align-items:center; justify-content:center;
  translate:calc(var(--mx)*9px) calc(var(--my)*6px); }
.sun { position:absolute; width:min(270px,80%); aspect-ratio:1; border-radius:50%; background:var(--red); border:5px solid var(--ink); overflow:hidden;
  animation:pop .5s .45s cubic-bezier(.2,1.4,.3,1) both, throb 1.1s 1s ease-in-out infinite; }
.sun::after { content:''; position:absolute; inset:0; background-image:radial-gradient(circle,#0d0d0d 1.5px,transparent 2.1px); background-size:8px 8px;
  -webkit-mask:linear-gradient(155deg,transparent 30%,#000 100%); mask:linear-gradient(155deg,transparent 30%,#000 100%); }
@keyframes pop { from { scale:0; } to { scale:1; } }
@keyframes throb { 0%,100% { scale:1; } 50% { scale:1.06; } }
.ring { position:absolute; left:50%; top:50%; width:60px; aspect-ratio:1; translate:-50% -50%; border-radius:50%; border:9px solid var(--ink); opacity:0; animation:rg .9s calc(.55s + var(--t)) ease-out both; }
@keyframes rg { from { opacity:1; scale:.2; } to { opacity:0; scale:12; } }
.eye { position:relative; width:min(100%,340px); animation:eo .45s .7s cubic-bezier(.2,1.3,.3,1) both; }
@keyframes eo { from { transform:scaleY(.03); opacity:0; } 60% { opacity:1; } to { transform:none; opacity:1; } }
.eye svg { width:100%; display:block; overflow:visible; filter:drop-shadow(0 4px 0 rgba(13,13,13,.25)); }

.txt { position:relative; z-index:3; flex:0 1 auto; width:min(640px,50%); text-align:center; min-width:0; }
.title { margin:0 0 .6rem; line-height:.92; font-family:'Bangers',sans-serif; font-weight:400; font-size:clamp(2.6rem,8.2vw,5.6rem); letter-spacing:.05em; color:#fff;
  -webkit-text-stroke:.075em var(--ink); paint-order:stroke fill; filter:drop-shadow(.05em .06em 0 var(--ink)) drop-shadow(.1em .11em 0 var(--alt)); }
.title .w { display:inline-block; margin:0 .12em; white-space:nowrap; }
.title i { display:inline-block; font-style:normal; rotate:var(--r); opacity:0; animation:slam .4s calc(.5s + var(--d)) cubic-bezier(.2,1.5,.4,1) both; }
@keyframes slam { from { opacity:0; translate:0 -260px; scale:2.6; rotate:calc(var(--r)*5); } to { opacity:1; translate:0 0; scale:1; rotate:var(--r); } }
.strip { display:inline-flex; gap:.7rem; align-items:center; background:var(--paper); border:3px solid var(--ink); box-shadow:5px 5px 0 var(--alt);
  padding:.12rem 1rem; transform:skewX(-6deg); font-family:'Bangers',sans-serif; font-size:1.3rem; letter-spacing:.09em; }
.strip span { opacity:0; animation:rise .45s calc(1.35s + var(--c)) ease-out both; } .strip b { color:var(--alt); font-weight:400; }
@keyframes rise { from { opacity:0; translate:0 16px; } to { opacity:1; translate:0 0; } }
.cap { display:block; max-width:470px; margin:.8rem auto 0; padding:.4rem .8rem; background:#fff; border:3px solid var(--ink); box-shadow:4px 4px 0 var(--ink);
  font-weight:700; font-size:.8rem; line-height:1.3; text-align:left; transform:rotate(-1deg); animation:rise .5s 1.75s ease-out both; }
.note { display:block; max-width:470px; margin:.7rem auto 0; padding:.45rem .9rem; background:#fff; border:3px solid var(--ink); border-radius:20px 24px 20px 16px;
  box-shadow:4px 4px 0 var(--red); font-weight:700; font-size:.82rem; line-height:1.3; text-align:left; animation:rise .5s 2.05s ease-out both; }
.sfx { position:absolute; left:14px; top:8px; z-index:4; rotate:-7deg; font-family:'Bangers',sans-serif; font-size:2.6rem; letter-spacing:.04em; line-height:1; color:#fff;
  -webkit-text-stroke:.09em var(--ink); paint-order:stroke fill; text-shadow:.05em .06em 0 var(--ink);
  animation:stamp .35s 1.0s cubic-bezier(.2,1.5,.4,1) both, shv .14s 1.4s steps(1) infinite; }
@keyframes stamp { from { opacity:0; scale:3.2; } to { opacity:1; scale:1; } }
@keyframes shv { 0%{rotate:-7deg} 50%{rotate:-5.4deg} }
@media (max-width:620px) { .stage { flex-direction:column; justify-content:center; gap:0; padding:0 4%; } .eyebox { flex:0 0 38%; height:auto; width:60%; } .txt { width:100%; }
  .cap, .bar .dim { display:none; } .title { font-size:3rem; } }
@media (prefers-reduced-motion:reduce) { * { animation-duration:.01s !important; animation-delay:0s !important; animation-iteration-count:1 !important; } }
</style></head><body>
<div class='film'>
  <div class='bar'><span>GAUSS LENS</span><span class='dim' id='tc'>00:00:00:00</span><span class='rec'>● REC</span></div>
  <div class='stage' id='stage'>
    <div class='tone'></div>
    <div class='eyebox'><div class='burst'></div><div class='ring' style='--t:0s'></div><div class='ring' style='--t:.18s'></div><div class='sun'></div><div class='eye'>__EYE__</div></div>
    <div class='txt'><h1 class='title'>__TITLE__</h1>
      <div class='strip'><span style='--c:0s'>PERCEPTION</span><span style='--c:.1s'><b>/</b></span><span style='--c:.2s'>FIT</span><span style='--c:.3s'><b>/</b></span><span style='--c:.4s'>AUDIT</span></div>
      <div class='cap'>__TAG__</div>__NOTE__</div>
    <div class='sfx'>__SFX__</div>
    <div class='sweep'></div><div class='vig'></div><div class='grain'></div><div class='flash'></div><div class='cover'></div>
  </div>
  <div class='bar'><span id='sub'></span></div>
</div>
<script>
(function(){
  var st=document.getElementById('stage'),pup=document.getElementById('pup'),tc=document.getElementById('tc'),sub=document.getElementById('sub');
  // ---- eye + parallax follow the cursor; idles with random glances when the mouse is elsewhere ----
  var tx=0,ty=0,cx=0,cy=0,idle=true;
  window.addEventListener('mousemove',function(e){var r=st.getBoundingClientRect();
    tx=Math.max(-1,Math.min(1,(e.clientX-r.left)/r.width*2-1));ty=Math.max(-1,Math.min(1,(e.clientY-r.top)/r.height*2-1));idle=false;});
  document.addEventListener('mouseleave',function(){idle=true});
  setInterval(function(){if(idle){tx=(Math.random()*2-1)*.9;ty=(Math.random()*2-1)*.6}},1700);
  (function loop(){cx+=(tx-cx)*.09;cy+=(ty-cy)*.09;
    st.style.setProperty('--mx',cx.toFixed(3));st.style.setProperty('--my',cy.toFixed(3));
    if(pup)pup.setAttribute('transform','translate('+(320+cx*44).toFixed(1)+' '+(156+cy*26).toFixed(1)+')');
    requestAnimationFrame(loop)})();
  // ---- running timecode (24 fps) ----
  var t0=performance.now();
  setInterval(function(){var t=(performance.now()-t0)/1000,f=Math.floor((t%1)*24),s=Math.floor(t)%60,m=Math.floor(t/60)%60,h=Math.floor(t/3600),p=function(n){return(n<10?'0':'')+n};
    tc.textContent=p(h)+':'+p(m)+':'+p(s)+':'+p(f)},41);
  // ---- rolling subtitle, same story as the boot intro ----
  var L=['One photo. Hundreds of unknown face parameters.','Perception: finding the face in the noise…','Fit: Gauss-Newton shrinks the residual, iteration by iteration.','Audit: every parameter checked against the evidence.'],i=0;
  function say(){sub.style.opacity=0;setTimeout(function(){sub.textContent=L[i++%L.length];sub.style.opacity=1},350)}
  setTimeout(say,1300);setInterval(say,3600);
})();
</script>
</body></html>"""


def _hero_title() -> str:
    """Title as per-letter <i> spans (staggered slam-in, same recipe as the boot title page)."""
    rot, d, out = [1.5, -1, 2, -1.5, -2], 0.0, []
    for w in TITLE.split():
        out.append("<span class='w'>")
        for i, c in enumerate(w):
            d += .055
            out.append(f"<i style='--d:{d:.3f}s;--r:{rot[i % 5]}deg'>{html.escape(c)}</i>")
        out.append("</span> ")
    return "".join(out)


def _hero(sfx, note_msg=None):
    """Main-page hero: a cinematic title card that continues the boot screen — letterbox bars + timecode,
    burst lines, throbbing sun, an eye whose pupil follows the cursor, slam-in title, rolling subtitle.
    Rendered in an iframe via _html_frame() so the SVG eye, CSS animation and JS survive Streamlit's sanitizer."""
    main, alt = _accent()
    eye = _HERO_EYE.replace("__MAIN__", main).replace("<g transform='translate(320 156)'>",
                                                    "<g id='pup' transform='translate(320 156)'>")
    rep = {"__MAIN__": main, "__ALT__": alt, "__SFX__": html.escape(sfx), "__TITLE__": _hero_title(),
           "__EYE__": eye, "__TAG__": html.escape(TAGLINE), "__GRAIN__": _GRAIN,
           "__NOTE__": f"<div class='note'>{note_msg}</div>" if note_msg else ""}
    out = _HERO_TPL
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


# ---------------------------------------------------------------------------
# Loading screen — a "fight card" shown while the pipeline runs. Two boxer bots
# trade punches around a VS! burst, a striped bar (with a bouncing red glove at
# its tip) fills as the stages complete, and ROUND chips tick off:
#   ROUND 1 model  ->  ROUND 2 session init  ->  ROUND 3 track + score  ->  K.O.!
#
# Like the hero it is rendered in a real <iframe> (via _html_frame / st.iframe)
# so the inline <svg> / SMIL animation, its <style> and its <script> are
# isolated from the app and aren't stripped by Streamlit's sanitizer.
# Each progress_cb() call re-renders the iframe (which restarts its JS), so the
# tracker below passes in where the bar currently *is* — the bar keeps creeping
# toward the next milestone while a long stage runs, and a fresh render resumes
# from that same spot instead of jumping back.
# ---------------------------------------------------------------------------

_LOAD_MILESTONES = [0, 5, 15, 75, 100]
_LOAD_SFX   = ["GRRR…", "FIT!!", "AUDIT!!", "K.O.!!"]
_LOAD_CHIPS = ["MODEL", "SESSION INIT", "TRACK + SCORE"]
_LOAD_QUIPS = [
    ["Waking up the 3DMM…", "Loading vertices, normals and UVs…",
     "Every face starts life as a mesh of numbers."],
    ["Fitting the source face, vertex by vertex…",
     "Gauss-Newton is iterating — no network inside the solver.",
     "Checking the source's eyes, mouth and pose…"],
    ["Tracking pose, eyes, brows and mouth…", "Projecting the fitted mesh onto the frame…",
     "Measuring alignment error per region.",
     "Grading its own work before you see it…"],
    ["Parameters extracted — enjoy the result!"],
]

# Fixed iframe height for the loader card: .load is 372px tall, plus room for
# its 2px outline + 5px offset so the bottom edge isn't clipped.
_LOADER_HEIGHT = 392


def _load_stage(pct: int) -> int:
    return 0 if pct < 15 else 1 if pct < 75 else 2 if pct < 100 else 3


def _load_ceiling(pct: int, milestones=None) -> int:
    """How far the bar may creep while waiting for the next milestone."""
    nxt = [m for m in (milestones or _LOAD_MILESTONES) if m > pct]
    return (nxt[0] - 1) if nxt else 100


def _load_creep(pct: float, ceil: float, since: float) -> float:
    """Eased bar position `since` seconds after reaching `pct` (mirrors the JS)."""
    if pct >= 100:
        return 100.0
    tau = max(6.0, (ceil - pct) * 0.5)
    return pct + (ceil - pct) * (1 - math.exp(-since / tau))


def _star_clip(n: int = 15, r_out: float = 50, r_in: float = 33) -> str:
    """CSS clip-path polygon for a jagged manga impact burst."""
    rnd = random.Random(11)
    pts = []
    for i in range(n * 2):
        r = (r_out if i % 2 == 0 else r_in) * rnd.uniform(.93, 1.0)
        a = math.pi * i / n - math.pi / 2
        pts.append(f"{50 + r * math.cos(a):.1f}% {50 + r * math.sin(a):.1f}%")
    return "polygon(" + ",".join(pts) + ")"


_LOADER_TPL = """<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Dela+Gothic+One&family=Zen+Kaku+Gothic+New:wght@700;900&display=swap');
:root { --ink:#111; --red:#e3122b; }
html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
* { box-sizing:border-box; }

.load {
  position:relative; height:372px; padding:14px 18px; display:flex; flex-direction:column; gap:8px;
  font-family:'Zen Kaku Gothic New', sans-serif;
  background:#fff url("__SPEED__") center / cover no-repeat;
  border:5px solid var(--ink); outline:2px solid var(--ink); outline-offset:5px; overflow:hidden;
}
.sfx {
  position:absolute; left:14px; top:2px; z-index:3; transform:rotate(-7deg);
  font-family:'Dela Gothic One', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', sans-serif; font-size:2.3rem; color:#fff; letter-spacing:.04em;
  text-shadow:-3px -3px 0 #111, 3px -3px 0 #111, -3px 3px 0 #111, 3px 3px 0 #111, 0 4px 0 #111;
}
.round {
  position:absolute; right:16px; top:14px; z-index:3; background:var(--ink); color:#fff;
  padding:.15rem .9rem; transform:skewX(-9deg); box-shadow:5px 5px 0 var(--red);
  font-family:'Bangers', sans-serif; font-size:1.25rem; letter-spacing:.08em;
}

/* ---- arena: bot  VS!  bot ------------------------------------------------ */
.arena { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; gap:18px; padding-top:10px; }
.fighter { width:140px; height:116px; flex:none; filter:drop-shadow(3px 3px 0 rgba(17,17,17,.2)); }
.fighter svg { width:100%; height:100%; display:block; }
.fighter.l { animation:lunge-l 1.2s ease-in-out infinite; }
.fighter.r { animation:lunge-r 1.2s ease-in-out infinite .6s; }
.fighter.r svg { transform:scaleX(-1); }
@keyframes lunge-l { 0%,58%,100% { transform:translate(0,0); } 67% { transform:translate(-9px,2px); }
                     76% { transform:translate(32px,-5px); } 88% { transform:translate(0,0); } }
@keyframes lunge-r { 0%,58%,100% { transform:translate(0,0); } 67% { transform:translate(9px,2px); }
                     76% { transform:translate(-32px,-5px); } 88% { transform:translate(0,0); } }

.burst { position:relative; width:128px; height:128px; flex:none; animation:thump .6s ease-out infinite; }
.burst .o, .burst .i { position:absolute; inset:0; clip-path:__STAR__; }
.burst .o { background:var(--ink); }
.burst .i { inset:9px; background:#fff; }
.burst.ko .i { background:var(--red); }
.burst b {
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-family:'Bangers', sans-serif; font-size:2.2rem; letter-spacing:.05em; color:var(--ink);
  transform:rotate(-6deg);
}
.burst.ko b { color:#fff; text-shadow:2px 2px 0 var(--ink); }
@keyframes thump { 0%,30%,100% { transform:scale(1) rotate(0); } 45% { transform:scale(1.22) rotate(6deg); }
                   60% { transform:scale(.96) rotate(-3deg); } }

/* ---- HUD panel ------------------------------------------------------------ */
.hud { position:relative; z-index:2; background:#fff; border:4px solid var(--ink);
       box-shadow:6px 6px 0 var(--ink); padding:10px 14px 8px; }
.row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
.bubble {
  flex:1; padding:.4rem .9rem; background:#fff; color:var(--ink); border:3px solid var(--ink);
  border-radius:20px 24px 20px 16px; box-shadow:4px 4px 0 var(--red);
  font-weight:900; font-size:.9rem; line-height:1.25;
}
.pct {
  min-width:98px; text-align:right; font-family:'Bangers', sans-serif; font-size:2.7rem; line-height:1;
  color:#fff; letter-spacing:.03em;
  text-shadow:-2px -2px 0 #111, 2px -2px 0 #111, -2px 2px 0 #111, 2px 2px 0 #111, 0 3px 0 #111;
}
.barwrap { position:relative; margin:0 20px 12px 8px; }
.bar { position:relative; height:28px; border:4px solid var(--ink); overflow:hidden; background-color:#fff;
       background-image:radial-gradient(#111 1px, transparent 1.4px); background-size:6px 6px; }
.fill { height:100%; width:0; border-right:4px solid var(--ink);
        background:linear-gradient(45deg,#111 25%,#e3122b 25%,#e3122b 50%,#111 50%,#111 75%,#e3122b 75%);
        background-size:28px 28px; animation:scroll .7s linear infinite; }
@keyframes scroll { to { background-position:28px 0; } }
.glove { position:absolute; top:50%; left:0; width:40px; height:40px; margin:-20px 0 0 -20px; z-index:3; }
.glove svg { width:100%; height:100%; display:block; animation:hop .5s ease-in-out infinite; }
@keyframes hop { 0%,100% { transform:translateY(0) rotate(-6deg); } 50% { transform:translateY(-5px) rotate(6deg); } }

.chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.chip { padding:.1rem .7rem; transform:skewX(-9deg); border:3px solid var(--ink); background:#fff; color:var(--ink);
        font-family:'Bangers', sans-serif; font-size:.98rem; letter-spacing:.07em; }
.chip.done { background:var(--ink); color:#fff; }
.chip.cur  { background:var(--red); color:#fff; animation:pulse .8s ease-in-out infinite; }
@keyframes pulse { 0%,100% { transform:skewX(-9deg) scale(1); } 50% { transform:skewX(-9deg) scale(1.07); } }
.foot { display:flex; justify-content:space-between; gap:10px; font-weight:700; font-size:.82rem; color:var(--ink); }
.quip.in { animation:pop .35s ease-out; }
@keyframes pop { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }
.timer { font-family:'Bangers', sans-serif; letter-spacing:.06em; font-size:1rem; font-variant-numeric:tabular-nums; }

@media (max-width:560px) {
  .load { padding:10px 10px; gap:6px; }
  .fighter { width:84px; height:69px; } .burst { width:84px; height:84px; } .burst b { font-size:1.4rem; }
  .arena { gap:6px; padding-top:22px; }
  .sfx { font-size:1.3rem; left:8px; top:0; } .round { font-size:.95rem; right:8px; top:8px; padding:.1rem .6rem; }
  .hud { padding:8px 10px 6px; } .row { margin-bottom:8px; gap:8px; }
  .bubble { font-size:.78rem; padding:.3rem .7rem; } .pct { font-size:2rem; min-width:70px; }
  .barwrap { margin-bottom:8px; } .chips { gap:5px; margin-bottom:6px; }
  .chip { font-size:.85rem; padding:.05rem .5rem; } .chip:not(.cur) .nm { display:none; }
  .quip { display:none; } .foot { justify-content:flex-end; }
}
</style>
</head>
<body>
<div class='load'>
  <div class='sfx'>__SFX__</div>
  <div class='round'>__ROUND__</div>
  <div class='arena'>
    <div class='fighter l'>__BOT_L__</div>
    <div class='burst __KO__'><div class='o'></div><div class='i'></div><b>__BURST__</b></div>
    <div class='fighter r'>__BOT_R__</div>
  </div>
  <div class='hud'>
    <div class='row'><div class='bubble'>__MSG__</div><div class='pct' id='pct'>0%</div></div>
    <div class='barwrap'>
      <div class='bar' id='bar' role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='0'>
        <div class='fill' id='fill'></div></div>
      <div class='glove' id='glove'><svg viewBox='0 0 40 40'>__GLOVE__</svg></div>
    </div>
    <div class='chips'>__CHIPS__</div>
    <div class='foot'><span class='quip' id='quip'></span><span class='timer'>⏱ <span id='timer'>00:00</span></span></div>
  </div>
</div>
<script>
(function () {
  var PCT = __PCT__, START = __START__, CEIL = __CEIL__, ELAPSED = __ELAPSED__, QUIPS = __QUIPS__;
  var fill = document.getElementById('fill'), glove = document.getElementById('glove'),
      pctEl = document.getElementById('pct'), bar = document.getElementById('bar'),
      tm = document.getElementById('timer'), q = document.getElementById('quip');
  var tau = Math.max(6, (CEIL - PCT) * 0.5);           // same easing as _load_creep()
  var shown = START, last = performance.now(), t0 = last;
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function frame(now) {
    var dt = Math.min((now - last) / 1000, 0.1); last = now;
    var since = (now - t0) / 1000;
    var target = PCT >= 100 ? 100 : PCT + (CEIL - PCT) * (1 - Math.exp(-since / tau));
    if (target < shown) target = shown;                // never slide backwards
    shown += (target - shown) * (1 - Math.exp(-dt / 0.2));
    var p = Math.max(0, Math.min(100, shown));
    fill.style.width = p + '%'; glove.style.left = p + '%';
    pctEl.textContent = Math.round(p) + '%'; bar.setAttribute('aria-valuenow', Math.round(p));
    var s = Math.floor(ELAPSED + since); tm.textContent = pad(Math.floor(s / 60)) + ':' + pad(s % 60);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  if (__SCROLL__) { try { window.frameElement.scrollIntoView({block: 'center'}); } catch (e) {} }
  var qi = Math.floor(Math.random() * QUIPS.length);
  function setQuip() {
    q.classList.remove('in'); void q.offsetWidth;
    q.textContent = QUIPS[qi % QUIPS.length]; q.classList.add('in'); qi++;
  }
  setQuip(); if (QUIPS.length > 1) setInterval(setQuip, 3200);
})();
</script>
</body></html>"""


def _loader_html(pct: int, msg: str, start: float, elapsed: float, scroll: bool = False) -> str:
    stage = _load_stage(pct)
    ko = stage == 3
    chips = "".join(
        f"<span class='chip {'done' if stage > i else 'cur' if stage == i else ''}'>"
        f"{'✓ ' if stage > i else ''}R{i + 1}<span class='nm'> · {name}</span></span>"
        for i, name in enumerate(_LOAD_CHIPS))
    rep = {
        "__SPEED__": _speed_lines(600, 175),
        "__STAR__":  _star_clip(),
        "__SFX__":   _LOAD_SFX[stage],
        "__ROUND__": "FINISH!" if ko else f"ROUND {stage + 1}/3",
        "__BOT_L__": _bot_svg("l"),
        "__BOT_R__": _bot_svg("r"),
        "__KO__":    "ko" if ko else "",
        "__BURST__": "K.O.!" if ko else "VS!",
        "__MSG__":   html.escape(msg),
        "__GLOVE__": _glove(20, 20, 15),
        "__CHIPS__": chips,
        "__PCT__":   str(int(pct)),
        "__START__": f"{start:.2f}",
        "__CEIL__":  str(_load_ceiling(pct)),
        "__ELAPSED__": f"{elapsed:.1f}",
        "__QUIPS__": json.dumps(_LOAD_QUIPS[stage]),
        "__SCROLL__": "true" if scroll else "false",
    }
    out = _LOADER_TPL
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


class _LoaderTracker:
    """Drives the loading screen from progress_cb(pct, msg) calls."""

    def __init__(self, slot, render=None, milestones=None, height=None):
        self.slot, self.pct, self.first = slot, 0, True
        self.render     = render or _loader_html          # (pct, msg, start, elapsed, scroll) -> html
        self.milestones = milestones or _LOAD_MILESTONES
        self.height     = height or _LOADER_HEIGHT
        self.t0 = self.tmark = time.time()

    def update(self, pct: int, msg: str):
        now = time.time()
        # where the bar had crept to in the previous iframe, so this one resumes there
        start = _load_creep(self.pct, _load_ceiling(self.pct, self.milestones), now - self.tmark)
        elapsed = now - self.t0
        self.pct, self.tmark = pct, now
        with self.slot.container():
            # st.iframe (NOT st.html): the loader is a full document with its own
            # <style>/<script>, so it must live in an isolated iframe. That also
            # makes the `.st-key-fo_loader:has(iframe)` overlay CSS match.
            _html_frame(
                self.render(pct, msg, start, elapsed, scroll=self.first),
                height=self.height,
            )
        self.first = False


# ---------------------------------------------------------------------------
# Startup screen — the GAUSS LENS boot screen (manga intro -> title page whose bell
# curve is the real boot progress bar), shown ONCE per browser session while the
# heavy modules (solver, pipeline, face recogniser) are imported.
# ---------------------------------------------------------------------------

SHOW_STARTUP_SCREEN = os.environ.get("GAUSS_SKIP_BOOT", "0") != "1"   # set GAUSS_SKIP_BOOT=1 to skip the intro entirely
STARTUP_STAGE_MIN   = 0.8    # min seconds each stage stays on screen
STARTUP_FINISH_HOLD = 1.6    # how long the final READY frame lingers
_SPLASH_MILESTONES  = [0, 10, 45, 70, 100]


# ---- Gauss Lens boot screen ---------------------------------------------------
# _BOOT_TPL below (self-contained, no extra file needed) = manga intro (blackout -> slide ->
# fist -> collision -> title) that lands on the title page, whose bell curve is the REAL
# boot progress bar.
INTRO_SECONDS    = 8.2     # how long the opening cinematic plays before real progress frames take over
_BOOT_TPL = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>GAUSS LENS — Intro</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Dela+Gothic+One&family=Zen+Kaku+Gothic+New:wght@700;900&display=swap');
:root{--ink:#0d0d0d;--paper:#eee9d8;--red:#e3122b;--bar:8.5vh;color-scheme:dark}
html,body{margin:0;height:100%;background:#000;overflow:hidden}
*{box-sizing:border-box}
.cine{position:relative;width:100vw;height:100vh;background:#000;overflow:hidden;color:var(--ink);font-family:'Zen Kaku Gothic New','Hiragino Sans','Yu Gothic',sans-serif;cursor:pointer}
.bar{position:absolute;left:0;right:0;height:var(--bar);z-index:30;background:#000;color:#d8d8d8;display:flex;align-items:center;justify-content:space-between;padding:0 3vw;font:700 clamp(10px,1.05vw,14px)/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.22em;text-transform:uppercase}
.bar.top{top:0}.bar.bot{bottom:0}
.bar.top span{opacity:0}
.cine.t .bar.top span{opacity:1;transition:opacity .5s 1.3s}
#sub{flex:1;text-align:center;letter-spacing:.06em;text-transform:none;color:#fff;font:700 clamp(12px,1.6vw,22px)/1.2 'Zen Kaku Gothic New',sans-serif}
.stage{position:absolute;left:0;right:0;top:var(--bar);bottom:var(--bar);overflow:hidden;background:var(--paper)}
.sc{position:absolute;inset:0;visibility:hidden;overflow:hidden}
.sc.on{visibility:visible}
.fx{position:absolute;inset:0;pointer-events:none}
/* burst speed lines */
.burst{position:absolute;left:50%;top:50%;width:170vmax;height:170vmax;translate:-50% -50%;
 background:repeating-conic-gradient(from 3deg,var(--ink) 0 1.1deg,transparent 1.1deg 3.9deg,var(--ink) 3.9deg 4.4deg,transparent 4.4deg 9deg);
 -webkit-mask:radial-gradient(circle,transparent 9%,#000 30%);mask:radial-gradient(circle,transparent 9%,#000 30%);animation:boil .3s steps(1) infinite}
.burst.w{background:repeating-conic-gradient(from 3deg,var(--paper) 0 1deg,transparent 1deg 3.6deg,var(--paper) 3.6deg 4.1deg,transparent 4.1deg 8deg)}
@keyframes boil{0%{rotate:0deg}34%{rotate:.9deg;scale:1.015}67%{rotate:-.8deg}}
.tone{position:absolute;inset:0;pointer-events:none;z-index:25;opacity:.6;background-image:radial-gradient(circle,#0d0d0d 1.4px,transparent 1.9px);background-size:8px 8px;-webkit-mask:radial-gradient(ellipse at 50% 45%,transparent 60%,#000 130%);mask:radial-gradient(ellipse at 50% 45%,transparent 60%,#000 130%)}
.vig{position:absolute;inset:0;pointer-events:none;z-index:26;background:radial-gradient(ellipse at 50% 45%,transparent 52%,rgba(0,0,0,.65) 100%)}
.flash{position:absolute;inset:0;z-index:28;background:#fff;opacity:0;pointer-events:none}
.flash.fire{animation:fl .5s ease-out}
@keyframes fl{0%{opacity:1}100%{opacity:0}}
.shake{animation:sk .45s linear}
@keyframes sk{12%{translate:-1.2vw .8vh}26%{translate:1vw -.9vh}40%{translate:-.8vw -.6vh}55%{translate:.7vw .7vh}72%{translate:-.4vw .3vh}}
.sfx{position:absolute;z-index:6;font-family:'Bangers','Impact',sans-serif;line-height:1;color:#fff;letter-spacing:.04em;-webkit-text-stroke:.09em var(--ink);paint-order:stroke fill;text-shadow:.05em .06em 0 var(--ink)}
.jp{font-family:'Bangers','Impact',sans-serif;letter-spacing:.04em;color:var(--red);-webkit-text-stroke:.06em var(--ink);paint-order:stroke fill;text-shadow:.05em .06em 0 var(--ink);position:absolute;z-index:6;line-height:1}
.on .pop{animation:stamp .35s var(--t,0s) cubic-bezier(.2,1.5,.4,1) both}
@keyframes stamp{from{opacity:0;scale:3.2}to{opacity:1;scale:1}}
.on .shiv{animation:stamp .35s var(--t,0s) cubic-bezier(.2,1.5,.4,1) both,shv .14s calc(var(--t,0s) + .4s) steps(1) infinite}
@keyframes shv{0%{rotate:var(--r,0deg)}50%{rotate:calc(var(--r,0deg) + 1.6deg)}}
.sun{position:absolute;left:50%;top:50%;translate:-50% -50%;border-radius:50%;background:var(--red);border:.55vh solid var(--ink);overflow:hidden}
.sun::after{content:'';position:absolute;inset:0;background-image:radial-gradient(circle,#0d0d0d 1.5px,transparent 2.1px);background-size:8px 8px;-webkit-mask:linear-gradient(155deg,transparent 30%,#000 100%);mask:linear-gradient(155deg,transparent 30%,#000 100%)}

/* S1 — blackout / siren */
#s1{background:#000}
#s1 .siren{position:absolute;left:50%;top:50%;width:min(60vh,60vw);aspect-ratio:1;translate:-50% -50%;border-radius:50%;background:radial-gradient(circle,var(--red) 0 25%,transparent 70%);opacity:0}
#s1.on .siren{animation:sir .55s .2s ease-in-out 3}
@keyframes sir{0%,100%{opacity:0;scale:.6}50%{opacity:.9;scale:1.15}}
#s1 .jp{left:50%;top:50%;translate:-50% -50%;font-size:min(14vw,22vh);color:#fff;-webkit-text-stroke:0;text-shadow:0 0 .35em var(--red);white-space:nowrap}
#s1 .mono{position:absolute;left:6%;bottom:12%;color:#d8d8d8;font:700 min(1.6vw,2.6vh)/1.5 ui-monospace,Menlo,monospace;letter-spacing:.3em}

/* S2 — Akira: skyline, sun, slide */
#s2 .sun{width:min(70vh,70vw);top:46%}
#s2 .sky{position:absolute;left:0;right:0;bottom:0;height:46%;width:100%}
#s2 .road{position:absolute;left:0;right:0;bottom:0;height:16%;background:var(--ink)}
.bike{position:absolute;left:0;bottom:11%;z-index:5;width:26vw;translate:-40vw 0;rotate:-4deg}
#s2.on .bike{animation:slide 1.5s .35s cubic-bezier(.15,.7,.3,1) both}
@keyframes slide{from{translate:-40vw 0}to{translate:112vw 0}}
.bike svg{width:100%;display:block;overflow:visible}
.trail{position:absolute;left:0;bottom:12.5%;height:.7vh;background:var(--ink);box-shadow:0 2.2vh 0 var(--ink),0 -2vh 0 var(--ink),0 4.4vh 0 var(--red);width:0;z-index:4}
#s2.on .trail{animation:tr 1.5s .35s ease-out both}
@keyframes tr{from{width:0;left:-10vw}60%{width:90vw;left:0}to{width:60vw;left:60vw;opacity:0}}
#s2 .jp.r{right:4%;top:6%;writing-mode:vertical-rl;font-size:min(9vw,14vh);letter-spacing:.06em}

/* S3 — Ippo: fist */
#s3{background:var(--ink)}
#s3 .fist{position:absolute;left:50%;top:50%;width:min(80vh,70vw);translate:-50% -50%;z-index:4;scale:.04;rotate:-8deg}
#s3.on .fist{animation:punch .5s .45s cubic-bezier(.6,0,1,.6) both,hold .12s 1s steps(1) infinite}
@keyframes punch{from{scale:.04}to{scale:1.35}}
@keyframes hold{0%{rotate:-8deg}50%{rotate:-6.4deg}}
#s3 .sfx.a{right:4%;top:5%;font-size:min(13vw,22vh);rotate:6deg;--r:6deg}
#s3 .jp.b{left:3%;bottom:4%;font-size:min(14vw,24vh);rotate:-6deg;--r:-6deg}
#s3 .bell{left:4%;top:5%;font-size:min(7vw,11vh);rotate:-10deg;color:var(--paper)}
.drop{position:absolute;z-index:5;width:2.4vw;aspect-ratio:.7;background:#fff;border:.4vh solid var(--ink);border-radius:50% 50% 50% 50%/65% 65% 35% 35%;opacity:0}
#s3.on .drop{animation:dr .7s calc(var(--t) + 1s) ease-out both}
@keyframes dr{from{opacity:1;translate:0 0}to{opacity:0;translate:var(--x) var(--y)}}

/* S4 — collision: shockwave + eye */
#s4 .sun{width:min(46vh,50vw);top:42%;scale:0}
#s4.on .sun{animation:pop .5s .1s cubic-bezier(.2,1.4,.3,1) both,thr .55s .7s ease-in-out infinite}
@keyframes pop{to{scale:1}}
@keyframes thr{0%,100%{scale:1}50%{scale:1.07}}
.ring{position:absolute;left:50%;top:42%;width:20vh;aspect-ratio:1;translate:-50% -50%;border-radius:50%;border:1.6vh solid var(--ink);opacity:0}
#s4.on .ring{animation:rg .9s calc(var(--t)) ease-out both}
@keyframes rg{from{opacity:1;scale:.2}to{opacity:0;scale:14}}
.eye{position:absolute;left:50%;top:42%;width:min(56vw,66vh);translate:-50% -50%;z-index:4}
.eye svg{width:100%;display:block;overflow:visible;filter:drop-shadow(0 .6vh 0 rgba(13,13,13,.25))}
.on .eye{animation:eo .45s .35s cubic-bezier(.2,1.3,.3,1) both}
@keyframes eo{from{transform:scaleY(.03);opacity:0}60%{opacity:1}to{transform:none;opacity:1}}
#s4 .sfx.c{left:50%;bottom:6%;translate:-50% 0;font-size:min(8vw,13vh);white-space:nowrap;color:#fff}
#s4 .jp.d{left:3%;top:4%;font-size:min(8vw,13vh);rotate:-8deg;--r:-8deg}

/* S5 — title */
#s5 .sun{width:min(40vh,44vw);top:27%;animation:thr .55s ease-in-out infinite}
#s5 .eye{top:27%}
#s5 .burst{top:27%}
.title{position:absolute;left:0;right:0;top:61%;translate:0 -50%;margin:0;text-align:center;z-index:7;line-height:.92;font-family:'Bangers','Impact',sans-serif;font-weight:400;font-size:min(11vw,17vh);letter-spacing:.05em;color:#fff;-webkit-text-stroke:.075em var(--ink);paint-order:stroke fill;filter:drop-shadow(.05em .06em 0 var(--ink)) drop-shadow(.1em .11em 0 var(--alt,#1f4fd8))}
.title .w{display:inline-block;margin:0 .13em;white-space:nowrap}
.title i{display:inline-block;font-style:normal;rotate:var(--r);opacity:0}
.on .title i{animation:slam .4s calc(.15s + var(--d)) cubic-bezier(.2,1.5,.4,1) both}
@keyframes slam{from{opacity:0;translate:0 -42vh;scale:2.6;rotate:calc(var(--r)*5)}to{opacity:1;translate:0 0;scale:1;rotate:var(--r)}}
.caps{position:absolute;left:0;right:0;bottom:17%;z-index:8;display:flex;justify-content:center;font-family:'Bangers',sans-serif;font-size:min(4.4vw,7vh);line-height:1;letter-spacing:.09em}
.strip{display:flex;align-items:center;gap:min(2.2vw,3.4vh);background:var(--paper);border:.6vh solid var(--ink);padding:1vh 2.6vw;box-shadow:.8vh .9vh 0 var(--alt,#1f4fd8);transform:skewX(-6deg)}
.gauss{position:absolute;left:3%;right:3%;bottom:1.5%;height:13%;z-index:7}
.gauss svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
#gc{fill:none;stroke:var(--ink);stroke-width:5;stroke-linejoin:round;vector-effect:non-scaling-stroke}
#gg{fill:none;stroke:var(--ink);stroke-width:2;opacity:.28;stroke-dasharray:6 6;vector-effect:non-scaling-stroke}
#gh{stroke:var(--alt,#1f4fd8);stroke-width:2;opacity:.85}
.gdot{position:absolute;left:0;top:100%;width:min(2.2vw,3.6vh);aspect-ratio:1;translate:-50% -50%;border-radius:50%;background:var(--red);border:.35vh solid var(--ink);box-shadow:0 0 0 .5vh rgba(227,18,43,.25);animation:thr .55s ease-in-out infinite}
.pct{position:absolute;left:0;bottom:30%;font-family:'Bangers',sans-serif;font-size:min(7vw,11.5vh);line-height:1;color:#fff;letter-spacing:.03em;-webkit-text-stroke:.07em var(--ink);paint-order:stroke fill;text-shadow:.04em .05em 0 var(--ink);font-variant-numeric:tabular-nums}.pct small{font-size:.45em}
.gf{position:absolute;right:0;bottom:30%;font:900 min(1.9vw,3vh) ui-monospace,Menlo,Consolas,monospace;letter-spacing:.1em;background:var(--paper);padding:0 .5em;border:.3vh solid var(--ink)}
.caps span{opacity:0}.caps b{color:var(--alt,#1f4fd8);font-weight:400;transform:skewX(-14deg)}
.on .caps span{animation:rise .45s calc(1.3s + var(--c)) ease-out both}
@keyframes rise{from{opacity:0;translate:0 18px}to{opacity:1;translate:0 0}}
.go{position:absolute;right:3%;top:4%;z-index:9;border:.5vh solid var(--ink);background:var(--red);color:#fff;padding:.5em 1.4em;font:400 min(3.2vw,5vh)/1 'Bangers',sans-serif;letter-spacing:.14em;-webkit-text-stroke:.04em var(--ink);paint-order:stroke fill;box-shadow:.5vh .6vh 0 var(--ink);rotate:-2deg;opacity:0;pointer-events:none}
.cine.rdy .go{animation:rise .4s ease-out both,nud .7s .5s ease-in-out infinite}
@keyframes nud{50%{translate:0 -.8vh}}
.caps span{position:relative}
.caps span.done::before{content:'✓';position:absolute;left:50%;top:-.75em;translate:-50% 0;font:900 .4em 'Zen Kaku Gothic New',sans-serif;color:var(--red)}
.caps span.cur::after{content:'';position:absolute;left:-4%;right:-4%;bottom:-.28em;height:.14em;background:var(--red);transform:skewX(-20deg)}
.cine.static .title i,.cine.static .caps span{opacity:1!important;animation:none!important}
.cine.static .eye{animation:none!important}
.cine.t.static .bar.top span{transition:none}
@media (max-aspect-ratio:1/1){.title{font-size:min(19vw,12vh);top:56%}.title .w{display:block;margin:0}.caps{font-size:5.4vw}.bar .dim{display:none}}
@media (prefers-reduced-motion:reduce){.burst,.sun{animation:none!important}}
</style></head>
<body>
<div class="cine" id="cine">
  <div class="bar top"><span>GAUSS LENS</span><span class="dim">FACE · PARAMETER · EXTRACTION</span></div>
  <div class="stage" id="stage">

    <!-- 1: blackout -->
    <div class="sc" id="s1"><div class="siren"></div>
      <div class="jp pop">INPUT</div></div>

    <!-- 2: Akira — the slide -->
    <div class="sc" id="s2" style="background:var(--paper)">
      <div class="burst" style="top:46%"></div><div class="sun"></div>
      <svg class="sky" id="sky" viewBox="0 0 100 40" preserveAspectRatio="none"></svg>
      <div class="road"></div><div class="trail"></div>
      <div class="bike"><svg viewBox="0 0 300 90"><path d="M10 52C10 26 50 12 120 12H250C285 12 296 34 290 52C284 70 250 74 190 74H60C28 74 10 68 10 52Z" style="fill:var(--red)" stroke="#0d0d0d" stroke-width="7"/><path d="M60 24H170" stroke="#fff" stroke-width="9" stroke-linecap="round"/><circle cx="56" cy="76" r="15" fill="#0d0d0d"/><circle cx="244" cy="76" r="15" fill="#0d0d0d"/><path d="M270 80l30 6M270 70l36 -2M276 60l28 -10" stroke="#0d0d0d" stroke-width="4" stroke-linecap="round"/></svg></div>
      <div class="jp r pop" style="--t:.5s">ERROR</div>
      <div class="sfx pop" style="--t:.9s;left:5%;top:6%;font-size:min(9vw,15vh);rotate:-7deg">ITERATE!!</div></div>

    <!-- 3: Ippo — the fist -->
    <div class="sc" id="s3">
      <div class="burst w" style="animation-duration:.14s"></div>
      <svg class="fist" viewBox="0 0 200 200"><g stroke="#0d0d0d" stroke-width="6" stroke-linejoin="round" stroke-linecap="round">
        <path d="M30 70C30 50 50 42 100 42S170 50 172 72L176 130C176 168 150 184 104 184S32 170 30 130Z" fill="#eee9d8"/>
        <g fill="#eee9d8"><circle cx="58" cy="70" r="22"/><circle cx="93" cy="62" r="23"/><circle cx="130" cy="63" r="23"/><circle cx="159" cy="76" r="19"/></g>
        <path d="M44 112C60 126 120 128 166 112" fill="none"/><path d="M60 128C84 150 130 150 160 128" fill="#eee9d8"/>
        <path d="M70 140l12-8M100 146l12-8M130 142l12-8M48 100l60 6" stroke-width="3" fill="none"/></g></svg>
      <div class="drop" style="left:28%;top:30%;--t:0s;--x:-18vw;--y:-14vh"></div><div class="drop" style="left:70%;top:26%;--t:.05s;--x:16vw;--y:-16vh"></div>
      <div class="drop" style="left:20%;top:62%;--t:.1s;--x:-20vw;--y:10vh"></div><div class="drop" style="left:76%;top:64%;--t:.02s;--x:18vw;--y:12vh"></div>
      <div class="sfx a shiv" style="--t:1s">FIT IT!!</div>
      <div class="jp b shiv" style="--t:1.1s">MINIMIZE!!</div></div>

    <!-- 4: collision -->
    <div class="sc" id="s4" style="background:var(--paper)">
      <div class="burst" style="top:42%"></div>
      <div class="ring" style="--t:0s"></div><div class="ring" style="--t:.18s"></div><div class="ring" style="--t:.36s"></div>
      <div class="sun"></div><div class="eye" id="eye4"></div>
      <div class="jp d pop" style="--t:.7s">LOOK!!</div>
      <div class="sfx c shiv" style="--t:.9s">SEE THROUGH THE NOISE</div></div>

    <!-- 5: title -->
    <div class="sc" id="s5" style="background:var(--paper)">
      <div class="burst"></div><div class="sun"></div><div class="eye" id="eye5"></div>
      <h1 class="title" id="ttl"></h1>
      <div class="caps"><div class="strip"><span style="--c:0s">PERCEPTION</span><span style="--c:.1s"><b>/</b></span><span style="--c:.2s">FIT</span><span style="--c:.3s"><b>/</b></span><span style="--c:.4s">AUDIT</span></div></div>
      <div class="gauss"><svg viewBox="0 0 1000 120" preserveAspectRatio="none"><defs><clipPath id="cl"><rect id="clr" x="0" y="0" width="0" height="120"/></clipPath></defs><path id="gg"/><g id="gh" clip-path="url(#cl)"></g><path id="gc"/></svg><div class="gdot" id="dot"></div><div class="pct"><span id="pct">0</span><small>%</small></div><div class="gf">N(μ, σ²)</div></div>
      <div class="go" id="go">READY ▶</div></div>

    <div class="tone"></div><div class="vig"></div><div class="flash" id="fl"></div>
  </div>
  <div class="bar bot"><span id="sub"></span></div>
</div>
<script>
(function(){
  var $=function(i){return document.getElementById(i)};
  // ---- sound: plays through the main page's engine (window.parent.__sfx); timers are cancellable on skip ----
  var sfxT=[];
  var sfx=function(n,a,d){try{var S=window.parent.__sfx;if(!S)return;if(d)sfxT.push(setTimeout(function(){S.play(n,a)},d));else S.play(n,a)}catch(e){}};
  var SCN={s1:[['siren',0,0]],s2:[['whoosh',0,350]],s3:[['whoosh',0,450],['punch',0,950]],s4:[['boom',0,0],['ping',0,180],['ping',1,360]],s5:[['title',0,0]]};
  // eye (same language as the title page)
  var P='M44 172C132 92 232 60 320 60S508 92 596 172C516 226 420 250 320 250S124 226 44 172Z';
  var eye='<svg viewBox="0 -50 640 350"><defs><clipPath id="eo"><path d="'+P+'"/></clipPath></defs><path d="'+P+'" transform="translate(11 13)" fill="#111"/><path d="'+P+'" fill="#fff"/><g clip-path="url(#eo)"><g transform="translate(320 156)"><circle r="112" fill="#111"/><g stroke="#fff" stroke-width="3.2" stroke-linecap="round"><line x1="46" y1="0" x2="46" y2="92"/><line x1="0" y1="46" x2="-92" y2="46"/><line x1="-46" y1="0" x2="-46" y2="-92"/><line x1="0" y1="-46" x2="92" y2="-46"/><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="18s" repeatCount="indefinite"/></g><circle r="48" fill="#111" stroke="#fff" stroke-width="3"/><circle cy="-20" r="7" style="fill:var(--red)" stroke="#fff" stroke-width="2"><animate attributeName="r" values="6;9;6" dur="1.1s" repeatCount="indefinite"/></circle><circle cx="-48" cy="-52" r="19" fill="#fff"/></g></g><path d="M20 176C112 56 232 22 320 22S528 56 620 176L596 172C508 92 408 60 320 60S132 92 44 172Z" fill="#111"/><path d="'+P+'" fill="none" stroke="#111" stroke-width="6"/></svg>';
  $('eye4').innerHTML=eye;$('eye5').innerHTML=eye.replace('id="eo"','id="eo2"').replace('url(#eo)','url(#eo2)');
  // title letters
  var h='',d=0,r=[1.5,-1,2,-1.5,-2];
  'GAUSS LENS'.split(' ').forEach(function(w){h+='<span class="w">';w.split('').forEach(function(c,i){d+=.055;h+='<i style="--d:'+d.toFixed(3)+'s;--r:'+r[i%5]+'deg">'+c+'</i>'});h+='</span> '});
  $('ttl').innerHTML=h;
  // skyline
  var s='',x=0;while(x<100){var w=2+Math.random()*5,t=8+Math.random()*26;s+='<rect x="'+x.toFixed(1)+'" y="'+(40-t).toFixed(1)+'" width="'+(w+.2).toFixed(1)+'" height="'+t.toFixed(1)+'" fill="#0d0d0d"/>';if(Math.random()<.4)s+='<rect x="'+(x+w/2-.3).toFixed(1)+'" y="'+(40-t-4).toFixed(1)+'" width=".6" height="4" fill="#0d0d0d"/>';x+=w}
  $('sky').innerHTML=s;
  var g='',p='';for(var i=0;i<=100;i++){var gx=i*10,gy=112-100*Math.exp(-.5*Math.pow((gx-500)/150,2));p+=(i?'L':'M')+gx+' '+gy.toFixed(1);if(i%2===0&&gy<110)g+='<path d="M'+gx+' '+gy.toFixed(1)+'V120"/>'}
  $('gc').setAttribute('d',p);$('gg').setAttribute('d',p);$('gh').innerHTML=g;
  // ---- boot wiring (values injected by app.py) ----
  var MODE='__MODE__',PCT=__PCT__,START=__START__,CEIL=__CEIL__,ACC=__ACC__,MSG=__MSG__,STG=__STG__;
  var CA='#e3122b',CB='#1f4fd8';
  var seq=[[0,'s1','One photo. Hundreds of unknown face parameters.'],[1800,'s2','Perception: finding the face in the noise…'],[3800,'s3','Fit: Gauss-Newton shrinks the residual, iteration by iteration.'],[5500,'s4','Audit: every parameter checked against the evidence.'],[7000,'s5',MSG]],tm=[];
  function show(id,sub,hit,snd){
    document.querySelectorAll('.sc').forEach(function(e){e.classList.remove('on','shake')});
    var e=$(id),c=id==='s5'?ACC===1:Math.random()<.5;
    e.style.setProperty('--red',c?CA:CB);e.style.setProperty('--alt',c?CB:CA);
    void e.offsetWidth;e.classList.add('on');
    if(snd)(SCN[id]||[]).forEach(function(p){sfx(p[0],p[1],p[2])});
    $('sub').textContent=sub;$('cine').classList.toggle('t',id==='s5');
    if(hit){var f=$('fl');f.classList.remove('fire');void f.offsetWidth;f.classList.add('fire');e.classList.add('shake')}
  }
  function skip(){tm.forEach(clearTimeout);sfxT.forEach(clearTimeout);sfxT=[];show('s5',MSG,false,true)}
  $('cine').addEventListener('click',function(){if(MODE==='intro')skip()});
  if(MODE==='intro'){tm=seq.map(function(q,i){return setTimeout(function(){show(q[1],q[2],i>0,true)},q[0])})}
  else{if(MODE!=='runin')$('cine').classList.add('static');show('s5',MSG,false,MODE==='runin')}
  var cs=document.querySelectorAll('.caps .strip span');[0,2,4].forEach(function(k,i){if(STG>i)cs[k].classList.add('done');else if(STG===i)cs[k].classList.add('cur')});
  if(MODE.indexOf('run')===0){$('go').textContent=PCT>=100?'EXTRACTED ✓':'ANALYZING…';$('cine').classList.add('rdy')}
  // ---- real progress: the bell curve inks itself in as the modules load ----
  var gp=$('gc'),clr=$('clr'),dot=$('dot'),pctEl=$('pct'),L=gp.getTotalLength();
  gp.style.strokeDasharray=L;
  var tau=Math.max(6,(CEIL-PCT)*0.5),shown=START,last=performance.now(),t0=last;   // same easing as _load_creep()
  function frame(now){
    var dt=Math.min((now-last)/1000,0.1);last=now;
    var since=(now-t0)/1000,target=PCT>=100?100:PCT+(CEIL-PCT)*(1-Math.exp(-since/tau));
    if(target<shown)target=shown;
    shown+=(target-shown)*(1-Math.exp(-dt/0.2));
    var p=Math.max(0,Math.min(100,shown)),pt=gp.getPointAtLength(L*p/100);
    gp.style.strokeDashoffset=L*(1-p/100);clr.setAttribute('width',pt.x);
    dot.style.left=(pt.x/10)+'%';dot.style.top=(pt.y/120*100)+'%';
    pctEl.textContent=Math.round(p);requestAnimationFrame(frame);
  }
  var PLAY=__PLAY__;PLAY.forEach(function(p){sfx(p[0],p[1],p[2])});
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


def _frame_sounds(pct, mode, stage):
    """Sounds for one boot / pipeline progress frame -> [[name, arg, delay_ms], ...] (played by the iframe)."""
    if not SOUND_ENABLED:
        return []
    ss = st.session_state
    if mode == "load":                                   # boot progress: rising ticks, chime when ready
        return [["tick", int(pct), 0]] + ([["ready", 0, 150]] if pct >= 100 else [])
    if mode == "runin":                                  # pipeline starts: ding-ding (title slam is in the template)
        ss["_snd_stage"], ss["_snd_pct"] = stage, pct
        return [["bell2", 0, 0]]
    if mode == "run":                                    # new ROUND -> bell · K.O. -> boom + bells · else a soft tick
        prev, ppct = ss.get("_snd_stage", -1), ss.get("_snd_pct", -1)
        ss["_snd_stage"], ss["_snd_pct"] = stage, pct
        if stage != prev:
            return [["ko", 0, 0]] if stage >= 3 else [["round", int(stage), 0]]
        if pct != ppct:
            return [["tick", int(pct), 0]]
    return []


def _gl_screen(pct, msg, start, mode, milestones, stage=-1):
    """Fill the Gauss Lens template. mode: intro (full cinematic) | load (boot progress frame)
    | runin (first pipeline frame, title slams in) | run (pipeline progress frame, no replayed entrance)."""
    st.session_state.setdefault("_boot_acc", random.choice([0, 1]))   # same colourway on every frame
    rep = {
        "__MODE__":  mode,
        "__PCT__":   str(int(pct)),
        "__START__": f"{start:.2f}",
        "__CEIL__":  str(_load_ceiling(pct, milestones)),
        "__ACC__":   str(st.session_state["_boot_acc"]),
        "__MSG__":   json.dumps(msg, ensure_ascii=False).replace("</", "<\\/"),
        "__STG__":   str(stage),
        "__PLAY__":  json.dumps(_frame_sounds(pct, mode, stage)),
    }
    out = _BOOT_TPL
    for k, v in rep.items():
        out = out.replace(k, v)
    return out


def _boot_screen_html(pct: int, msg: str, start: float, elapsed: float, scroll: bool = False) -> str:
    """Boot: only the first frame plays the intro; later frames jump to the title page (no replay)."""
    return _gl_screen(pct, msg, start, "intro" if scroll else "load", _SPLASH_MILESTONES)


def _run_screen_html(pct: int, msg: str, start: float, elapsed: float, scroll: bool = False) -> str:
    """Loading page while the pipeline runs: the same title page, with PERCEPTION / FIT / AUDIT
    ticking off as the real stages complete."""
    return _gl_screen(pct, msg, start, "runin" if scroll else "run", _LOAD_MILESTONES, _load_stage(pct))


def _overlay_container(key: str):
    """Keyed container -> picks up the full-window overlay CSS (Streamlit >= 1.39)."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


# ---- boot sequence ----------------------------------------------------------
# The heavy imports live HERE (not at the top of the file) so the splash is
# already on screen while they run. They're only *used* later, inside run_extract().

_sfx_install()                      # sound engine goes in first so the boot screen can use it

_first_visit = SHOW_STARTUP_SCREEN and not st.session_state.get("_booted", False)
_boot_slot = _splash = None
_boot_last = time.time()

if _first_visit:
    with _overlay_container("fo_boot"):
        _boot_slot = st.empty()
    _splash = _LoaderTracker(_boot_slot, render=_boot_screen_html,
                             milestones=_SPLASH_MILESTONES, height=900)
    _splash.update(0, "Loading the Gauss-Newton solver…")
    # hold the first frame so the whole intro plays before progress frames replace it
    _boot_last = time.time() + INTRO_SECONDS - STARTUP_STAGE_MIN


def _boot(pct: int, msg: str) -> None:
    """Advance the startup screen, keeping each stage visible for STARTUP_STAGE_MIN."""
    global _boot_last
    if _splash is None:
        return
    wait = STARTUP_STAGE_MIN - (time.time() - _boot_last)
    if wait > 0:
        time.sleep(wait)
    _splash.update(pct, msg)
    _boot_last = time.time()


_await_ready = False
try:
    _boot(10, "Loading the Gauss-Newton solver…")
    from run import (
        load_model, build_session, process_frame,
        _make_lm_vidx,
    )
    _boot(45, "Assembling the extraction pipeline…")
    from face_params.pipeline                import FaceParamPipeline, FrameData
    _boot(70, "Calibrating the face recogniser…")
    from face_params.face_recogniser         import FaceRecogniser
    from face_params.phase1.stage2_gn_source import extract_eye_params, compute_mouth_score, THETA_OPEN
    if _splash is not None:
        _boot(100, "Ready — press READY to enter")
        _await_ready = True                      # hold on the title page until READY is clicked
finally:
    # failed import: never leave the overlay covering the traceback.
    # success: keep the title page up, the READY button below dismisses it.
    if _boot_slot is not None and not _await_ready:
        _boot_slot.empty()

if _await_ready:
    st.session_state["_booted"] = True           # the click's rerun skips the boot screen
    st.session_state["_curtain"] = True          # ...and plays the shutter-open once
    _rdy = "#e3122b" if st.session_state.get("_boot_acc", 1) == 1 else "#1f4fd8"
    st.markdown(f"<style>:root{{--rdy:{_rdy};}}</style>", unsafe_allow_html=True)
    st.button("READY ▶", key="fo_ready")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACCEPTED_FORMATS = ["png", "jpg", "jpeg", "bmp", "webp"]


MAX_SIDE = int(os.environ.get("GAUSS_MAX_SIDE", "2560"))     # phone photos are 12-48 MP; 0 disables the cap


def _prepare_upload(raw: bytes, name: str):
    """Make an uploaded photo safe for the pipeline: honour the EXIF rotation (phones store portrait shots
    sideways + a flag, which PIL ignores) and cap extreme sizes. Re-encodes (lossless PNG) only if it changed
    something, so the pipeline and the preview always see the same pixels. Returns (PIL RGB, bytes, suffix)."""
    suffix = os.path.splitext(name)[1] or ".png"
    pil = Image.open(io.BytesIO(raw))
    changed = False
    try:
        if pil.getexif().get(0x0112, 1) not in (0, 1):
            pil, changed = ImageOps.exif_transpose(pil), True
    except Exception:
        pass
    pil = pil.convert("RGB")
    w, h = pil.size
    if MAX_SIDE > 0 and max(w, h) > MAX_SIDE:
        k = MAX_SIDE / max(w, h)
        pil, changed = pil.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS), True
    if changed:
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        raw, suffix = buf.getvalue(), ".png"
    return pil, raw, suffix


def _pil_to_ndarray(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def _ndarray_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(img*255, 0, 255).astype(np.uint8))


def _ndarray_to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    _ndarray_to_pil(img).save(buf, format="PNG")
    return buf.getvalue()


def _score_color(score: float) -> str:
    if score >= 0.75: return "normal"
    if score >= 0.50: return "off"
    return "inverse"


# ---------------------------------------------------------------------------
# Pipeline log curation
#
# The solver's stdout is extremely verbose (every Gauss-Newton iteration of
# every fit, every jacobian-conditioning dump). None of that is meant for an
# audience. For the UI we pull out only the checkpoint lines each stage prints
# for itself, grouped by the SAME labels the pipeline uses:
#   Phase 1  S1 validation · S2 source fit · S3 UV texture · S4 identity fit
#   Phase 2  S1 motion prior · S2 pose & expression fit · S3 quality gate
#   Phase 3  S1 optical flow · S2 motion state
# Exact-duplicate lines within a section are dropped (a couple of checkpoints
# print more than once per run, e.g. the source fit is echoed once during
# recognition and once during session build).
# ---------------------------------------------------------------------------

_LOG_SECTIONS = [
    ("Model & input", [
        r"^\[input\].*",
        r"^\[model\].*",
        r"^\[DEBUG\] model loaded.*",
        r"^\[DEBUG\] eye_idx:.*",
        r"^\[DEBUG\] nose_idx:.*",
        r"^\[DEBUG\] brow_idx:.*",
        r"^\[DEBUG\] image=.*",
        r"^\[DEBUG\] lm_vidx:.*",
    ]),
    ("Perception — SAM landmarks", [
        r"^\[Perception\].*",
        r"^\[SAM.*",
        r"^\[sam.*",
    ]),
    ("Phase 1 · S1 — validation", [
        r"^\[Phase1/S1\] ValidationResult.*",
    ]),
    ("Phase 1 · S2 — source fit", [
        r"^\[Phase1/S2\].*",
    ]),
    ("Phase 1 · S3 — UV texture", [
        r"^\[Phase1/S3[AB]?\].*",
    ]),
    ("Phase 1 · S4 — identity fit", [
        r"^\[Phase1/S4\].*",
    ]),
    ("Phase 2 · S1 — motion prior", [
        r"^\[P2/S1\].*",
    ]),
    ("Phase 2 · S2 — pose & expression fit", [
        r"^\[P2/S2 fit\].*",
        r"^\[diagnose\].*",
        r"^\[P2/S2\].*",
    ]),
    ("Phase 2 · alignment check", [
        r"^\[DEBUG align\].*",
    ]),
    ("Phase 2 · S3 — quality gate", [
        r"^\[P2/S3\].*",
    ]),
    ("Phase 3 · S1–S2 — motion state", [
        r"^\[P3/S1\].*",
        r"^\[P3/S2\].*",
    ]),
    ("Result summary", [
        r"^\[result\].*",
        r"^\[verdict\].*",
    ]),
]


_ITER_NOISE = re.compile(r"\b(iter(ation)?|it)\s*[=:#]?\s*\d+\b", re.I)   # "[GN iter 3]", "[S4 iter 0]" ...
_TAG_LINE   = re.compile(r"^\[[^\]]+\]")
_OTHER_CAP  = 150
_DROP       = [re.compile(r"^\[Phase [0-9+]+\] ")]     # "[Phase 1] Session init…" markers: progress noise
_GN_LINE = re.compile(r"^\[GN iter (\d+)\]\s+‖delta‖=(\S+)\s+geo_res=(\S+)\s+app_res=(\S+)"
                      r"\s+resid_before=(\S+)\s+resid_trial=(\S+)")
_S4_LINE = re.compile(r"^\[S4 iter (\d+)\]\s+‖delta‖=(\S+)\s+alpha_norm=(\S+)\s+beta_norm=(\S+)"
                      r"\s+resid_before=(\S+)")
_LONG_FLOAT = re.compile(r"(?<![\w.])-?\d+\.\d{5,}")
_EST_USD    = re.compile(r",?\s*'est_usd':\s*[\d.]+")


def _join_wrapped(raw_lines):
    """numpy prints long arrays over several lines; glue the continuation back onto its tagged line."""
    out = []
    for ln in raw_lines:
        s = ln.strip()
        if out and s and ln[:1] in (" ", "\t") and out[-1].count("[") > out[-1].count("]"):
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def _pretty(s: str) -> str:
    """Public-friendly display form: no cost figure, tidy spacing, numbers to 4 decimals.
    (The downloadable raw log keeps everything at full precision.)"""
    s = _EST_USD.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    return _LONG_FLOAT.sub(lambda m: f"{float(m.group()):.4f}", s)   # rounded, not truncated


def _skip(s: str) -> bool:
    return any(d.match(s) for d in _DROP) or bool(_ITER_NOISE.search(s))


def _summarise_runs(lines, rx, fmt):
    """Collapse '[GN iter 0..9]' style per-iteration spam into one line per distinct run (xN if repeated)."""
    runs, cur = [], None
    for s in lines:
        m = rx.match(s)
        if not m:
            continue
        if int(m.group(1)) == 0 or cur is None:
            cur = []
            runs.append(cur)
        cur.append(m.groups())
    groups = {}
    for r in runs:
        groups.setdefault(tuple(r), [r, 0])[1] += 1
    out = []
    for r, n in groups.values():
        try:
            txt = fmt(r)
        except Exception:
            continue
        out.append(txt + (f"  (ran ×{n}, identical)" if n > 1 else ""))
    return out


def _gn_text(g):
    return (f"Landmark fit (Gauss-Newton): {len(g)} iterations · residual {float(g[0][4]):.1f} → "
            f"{float(g[-1][5]):.1f} · last step ‖Δ‖ {float(g[-1][1]):.3f}")


def _s4_text(g):
    return (f"Identity fit (S4): {len(g)} iterations · residual {float(g[0][4]):.1f} → {float(g[-1][4]):.1f} · "
            f"‖α‖ {float(g[0][2]):.2f} → {float(g[-1][2]):.2f} · ‖β‖ {float(g[0][3]):.2f} → {float(g[-1][3]):.2f}")


def _curate_log(raw: str):
    """Group raw stdout into named, deduplicated checkpoint sections -> [(title, [line, ...])].

      1. the named stage sections in _LOG_SECTIONS (per-iteration lines replaced by one summary line)
      2. "Other checkpoints" - any [tag] line no section claimed, so new log lines never vanish
      3. "Error - traceback" - the Python traceback, if the run crashed
    """
    raw_lines = raw.splitlines()
    lines = _join_wrapped(raw_lines)
    extras = {r"^\[Phase1/S2\].*": _summarise_runs(lines, _GN_LINE, _gn_text),
              r"^\[Phase1/S4\].*": _summarise_runs(lines, _S4_LINE, _s4_text)}

    sections, claimed = [], set()
    for title, patterns in _LOG_SECTIONS:
        compiled = [re.compile(p) for p in patterns]
        seen, picked = set(), []
        for s in lines:
            if not s or s in seen or _skip(s):
                continue
            if any(c.match(s) for c in compiled):
                seen.add(s)
                claimed.add(s)
                picked.append(_pretty(s))
        picked = extras.get(patterns[0], []) + picked
        if picked:
            sections.append((title, picked))

    other, seen = [], set()
    for s in lines:
        if s.startswith("Traceback (most recent call last)"):
            break
        if not s or s in claimed or s in seen or _skip(s):
            continue
        if _TAG_LINE.match(s):
            seen.add(s)
            other.append(_pretty(s))
    if other:
        extra = len(other) - _OTHER_CAP
        sections.append(("Other checkpoints",
                         other[:_OTHER_CAP] + ([f"... {extra} more - see the raw log"] if extra > 0 else [])))

    for i, ln in enumerate(raw_lines):
        if ln.startswith("Traceback (most recent call last)"):
            sections.append(("Error - traceback", [l.rstrip() for l in raw_lines[i:] if l.strip()][:80]))
            break
    return sections


# ---------------------------------------------------------------------------
# "About this project" — rendered as a manga panel (same iframe technique as
# _hero()/the loader): a comic strip of four numbered panels that mirror the pipeline's
# phases, instead of a plain markdown wall of text.
# ---------------------------------------------------------------------------

_ABOUT_PANELS = [
    ("01", "👁", "PERCEPTION · SAM 3.1",
     "Meta's SAM 3.1 is asked for five things: face, eyes, eyebrows, nose and lips. The masks "
     "become 27 named landmarks. This is the only step that looks at the photo with a neural net."),
    ("02", "📐", "SOLVE · GAUSS-NEWTON",
     "No network inside. S2 fits identity (α), expression (β), pose and the eye / brow / nose / mouth "
     "blocks to those landmarks; S1 scores resolution, sharpness, pose, symmetry and landmark fit "
     "and rejects the photo if any is 0.60 or lower; S3 bakes the UV texture and S4 fixes identity."),
    ("03", "✅", "GATE · PROCEED / CONSERVATIVE / FALLBACK",
     "The fitted mesh is projected back onto the photo and the error is measured per region. "
     "S3 scores landmarks and pose into c_t and decides PROCEED, CONSERVATIVE or FALLBACK. "
     "(Phase 3 motion state only matters for video; a single photo has no previous frame.)"),
    ("04", "💬", "AUDIT · MUSE SPARK",
     "Muse Spark reads only the solver's report.json and writes a three-sentence verdict. Any reply "
     "that quotes a number not in the report is thrown away and a template verdict is used instead."),
]

_ABOUT_HEIGHT = 700

_FLOW_HEIGHT = 600


def _box(x, y, w, h, fill, kind, title, lines):
    """One flowchart box: ink outline, hard shadow, a small tag (NEURAL / NO NETWORK / AUDIT), title, text lines."""
    t = "".join(f"<text x='{x + w/2}' y='{y + 52 + 17*i}' text-anchor='middle' class='bt'>{html.escape(l)}</text>"
                for i, l in enumerate(lines))
    return (f"<g class='bx' style='animation-delay:-{((x * 7 + y * 3) % 30) / 10:.1f}s'>"
            f"<rect x='{x+6}' y='{y+6}' width='{w}' height='{h}' fill='#0d0d0d'/>"
            f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{fill}' stroke='#0d0d0d' stroke-width='3'/>"
            f"<rect x='{x}' y='{y}' width='{w}' height='20' fill='#0d0d0d'/>"
            f"<text x='{x+10}' y='{y+15}' class='tag'>{html.escape(kind)}</text>"
            f"<text x='{x + w/2}' y='{y + 40}' text-anchor='middle' class='bh'>{html.escape(title)}</text>{t}</g>")


def _flow_html() -> str:
    main, alt = _accent()
    N, G, A = "#dbe7f3", "#e3f1e1", "#f6e8d2"      # neural / geometry / audit
    W, H = 250, 92
    X = (30, 355, 680)
    Y = (30, 215, 400)
    b = []
    b.append(_box(X[0], Y[0], W, H, "#fff", "INPUT", "Face photo", ["one image, frontal, sharp"]))
    b.append(_box(X[1], Y[0], W, H, N, "NEURAL · META SAM 3.1", "Find the face parts",
                  ["5 prompts: face · eye · eyebrow", "nose · lips  →  masks"]))
    b.append(_box(X[2], Y[0], W, H, G, "NO NETWORK", "Masks → 27 landmarks",
                  ["split left/right, centre on", "each mask, σ = 0.02 · face width"]))
    b.append(_box(X[2], Y[1], W, H, G, "NO NETWORK · GAUSS-NEWTON", "Fit the 3D face (Phase 1)",
                  ["Θ = α β ω t ψ γ δ  ·  10 steps", "damped, with backtracking"]))
    b.append(_box(X[1], Y[1], W, H, G, "NO NETWORK · QUALITY GATE", "5 scores, each > 0.60",
                  ["resolution · sharpness · pose", "symmetry · landmark fit"]))
    b.append(_box(X[0], Y[1], W, H, G, "NO NETWORK · PHASE 2", "Frame fit + c_t gate",
                  ["PROCEED  >0.75  ·  CONSERVATIVE", "0.40–0.75  ·  FALLBACK  <0.40"]))
    b.append(_box(X[0], Y[2], W, H, "#fff", "THE EVIDENCE", "report.json",
                  ["scores · residual per iteration", "per-region error · pose"]))
    b.append(_box(X[1], Y[2], W, H, A, "NEURAL · META MUSE SPARK", "3-sentence verdict",
                  ["reads numbers only (no photo);", "a reply with any new number is dropped"]))
    b.append(_box(X[2], Y[2], W, H, "#fff", "THIS APP", "Overlay · curve · verdict",
                  ["parameters, log, downloads"]))

    def arrow(d, dash=False, label=None, lx=0, ly=0):
        da = " stroke-dasharray='8 6'" if dash else ""
        cls = " class='fb'" if dash else ""
        pk = "" if dash else f"<path d='{d}' class='pk'/>"          # a red 'data packet' that travels along the arrow
        txt = f"<text x='{lx}' y='{ly}' class='al'>{html.escape(label)}</text>" if label else ""
        return f"<path d='{d}' fill='none' stroke='#0d0d0d' stroke-width='3'{da}{cls} marker-end='url(#ah)'/>{pk}{txt}"

    r0, r1, r2 = Y[0] + H/2, Y[1] + H/2, Y[2] + H/2
    a = [arrow(f"M{X[0]+W} {r0} H{X[1]-4}"),
         arrow(f"M{X[1]+W} {r0} H{X[2]-4}"),
         arrow(f"M{X[2]+W/2} {Y[0]+H} V{Y[1]-4}"),
         arrow(f"M{X[2]} {r1} H{X[1]+W+4}"),
         arrow(f"M{X[1]} {r1} H{X[0]+W+4}", label="accepted", lx=X[0]+W+37, ly=r1-9),
         arrow(f"M{X[0]+W/2} {Y[1]+H} V{Y[2]-4}"),
         arrow(f"M{X[0]+W} {r2} H{X[1]-4}"),
         arrow(f"M{X[1]+W} {r2} H{X[2]-4}"),
         arrow(f"M{X[1]+60} {Y[1]+H} L{X[0]+W-40} {Y[2]-4}", dash=True, label="rejected", lx=X[1]-30, ly=Y[1]+H+40),
         arrow(f"M{X[1]+W/2} {Y[2]+H} V{Y[2]+H+34}", dash=True)]
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Zen+Kaku+Gothic+New:wght@700;900&display=swap');
html,body{{margin:0;background:transparent}}
.wrap{{background:#eee9d8;border:5px solid #0d0d0d;outline:2px solid #0d0d0d;outline-offset:5px;padding:10px 8px 6px;font-family:'Zen Kaku Gothic New',sans-serif}}
svg{{width:100%;height:auto;display:block}}
.bh{{font-family:'Bangers',sans-serif;font-size:19px;letter-spacing:.04em;fill:#0d0d0d}}
.bt{{font-size:12.5px;font-weight:700;fill:#0d0d0d}}
.pk{{fill:none;stroke:{main};stroke-width:5;stroke-linecap:round;stroke-dasharray:1 46;animation:flow 1.6s linear infinite}}
.fb{{animation:march 1s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-47}}}}
@keyframes march{{to{{stroke-dashoffset:-14}}}}
.bx{{animation:bxfloat 3.4s ease-in-out infinite}}
@keyframes bxfloat{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-3px)}}}}
.tag{{animation:tagblink 2.2s steps(1) infinite}}
@keyframes tagblink{{0%,92%{{opacity:1}}96%{{opacity:.35}}}}
@media (prefers-reduced-motion:reduce){{.pk,.fb,.bx,.tag{{animation:none}}}}
.tag{{font-family:'Bangers',sans-serif;font-size:12px;letter-spacing:.07em;fill:#fff}}
.al{{font-family:'Bangers',sans-serif;font-size:12px;letter-spacing:.04em;fill:{main};text-anchor:middle}}
.lg{{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;font-weight:900;font-size:12px;margin:2px 0 4px}}
.lg i{{display:inline-block;width:14px;height:14px;border:2px solid #0d0d0d;margin-right:5px;vertical-align:-3px}}
.hint{{display:none;font-weight:900;font-size:12px;margin:0 0 4px;color:#0d0d0d}}
@media (max-width:760px){{
  .wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch;outline-offset:3px}}
  svg{{min-width:760px}} .hint{{display:block}} .lg{{position:sticky;left:0}}
}}
</style></head><body><div class='wrap'>
<div class='hint'>↔ swipe sideways to see the whole chart</div>
<svg viewBox='0 0 960 560' xmlns='http://www.w3.org/2000/svg'>
<defs><marker id='ah' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0 0L10 5L0 10Z' fill='#0d0d0d'/></marker></defs>
{''.join(a)}{''.join(b)}
<text x='{X[1]+W/2}' y='{Y[2]+H+54}' text-anchor='middle' class='bt'>if Muse is unreachable or its reply is rejected → free template verdict from the same report</text>
</svg>
<div class='lg'><span><i style='background:{N}'></i>neural (Meta SAM 3.1)</span><span><i style='background:{G}'></i>geometry — no neural network</span>
<span><i style='background:{A}'></i>neural, text only (Meta Muse Spark)</span></div>
</div>{_AUTOFIT_JS}</body></html>"""



def _about_panel_html() -> str:
    rot = [-1.1, 0.9, -0.7, 1.2]
    cards = "".join(
        f"""<div class='panel' style='transform:rotate({rot[i]}deg);--d:{i * 0.15:.2f}s'>
              <div class='pnum'>{num}</div>
              <div class='picon'>{icon}</div>
              <div class='ptitle'>{html.escape(title)}</div>
              <div class='pdesc'>{html.escape(desc)}</div>
            </div>"""
        for i, (num, icon, title, desc) in enumerate(_ABOUT_PANELS)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Dela+Gothic+One&family=Zen+Kaku+Gothic+New:wght@500;700;900&display=swap');
:root {{ --ink:#111; --red:#e3122b; }}
html, body {{ margin:0; padding:0; background:transparent; }}
* {{ box-sizing:border-box; }}
.wrap {{
  font-family:'Zen Kaku Gothic New',sans-serif; background:#fff;
  border:5px solid var(--ink); outline:2px solid var(--ink); outline-offset:5px;
  padding:1.2rem 1.3rem 1.4rem;
}}
.sfx {{
  font-family:'Dela Gothic One',sans-serif; font-size:1.7rem; color:#fff; letter-spacing:.03em;
  text-shadow:-3px -3px 0 #111,3px -3px 0 #111,-3px 3px 0 #111,3px 3px 0 #111,0 4px 0 #111;
  transform:rotate(-6deg); display:inline-block; margin-bottom:.4rem;
}}
.tag {{
  font-family:'Bangers',sans-serif; font-size:1.85rem; letter-spacing:.05em;
  background:var(--ink); color:#fff; display:inline-block; padding:.15rem 1.1rem;
  transform:skewX(-9deg); box-shadow:6px 6px 0 var(--red); margin-bottom:1rem;
}}
.tag span {{ display:inline-block; transform:skewX(9deg); }}
.sub {{ font-weight:700; font-size:.9rem; max-width:680px; margin:.2rem 0 1.15rem; }}
.grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:1.1rem; }}
.panel {{
  background:#fff; border:3px solid var(--ink); box-shadow:5px 5px 0 var(--ink);
  padding:1rem 1.05rem 1.05rem; position:relative;
}}
.pnum {{
  position:absolute; top:-15px; left:-15px; background:var(--red); color:#fff;
  font-family:'Bangers',sans-serif; font-size:1.25rem; width:34px; height:34px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; border:3px solid var(--ink);
}}
.picon {{ font-size:1.8rem; margin-bottom:.2rem; }}
.ptitle {{ font-family:'Bangers',sans-serif; font-size:1.15rem; letter-spacing:.04em; margin-bottom:.35rem; }}
.pdesc {{ font-size:.82rem; font-weight:600; line-height:1.42; }}
/* ---- motion: panels pop in one by one, then float; icons wiggle, badges and GO!! pulse ---- */
@keyframes panel-in {{ from {{ opacity:0; scale:.85; translate:0 34px; }} to {{ opacity:1; scale:1; translate:0 0; }} }}
@keyframes bob {{ 0%,100% {{ translate:0 0; }} 50% {{ translate:0 -7px; }} }}
@keyframes icon-wig {{ 0%,100% {{ rotate:-9deg; scale:1; }} 50% {{ rotate:9deg; scale:1.18; }} }}
@keyframes badge-pulse {{ 0%,100% {{ scale:1; }} 50% {{ scale:1.16; }} }}
@keyframes sfx-pulse {{ 0%,100% {{ scale:1; }} 50% {{ scale:1.1; }} }}
.panel {{ animation: panel-in .6s var(--d) cubic-bezier(.2,1.2,.3,1) both, bob 5s calc(var(--d) + .7s) ease-in-out infinite; }}
.panel:hover {{ animation-play-state:paused; box-shadow:8px 8px 0 var(--red); }}
.picon {{ display:inline-block; animation: icon-wig 2.2s ease-in-out infinite; }}
.panel:nth-child(2) .picon {{ animation-delay:-.6s; }} .panel:nth-child(3) .picon {{ animation-delay:-1.2s; }} .panel:nth-child(4) .picon {{ animation-delay:-1.8s; }}
.pnum {{ animation: badge-pulse 1.6s ease-in-out infinite; }}
.sfx {{ animation: sfx-pulse 1.1s ease-in-out infinite; }}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none !important; }} }}
.foot {{
  margin-top:1.15rem; text-align:center; font-family:'Bangers',sans-serif; font-size:1.02rem;
  letter-spacing:.04em; border-top:3px dashed var(--ink); padding-top:.7rem;
}}

@media (max-width:640px) {{
  .wrap {{ padding:1rem .9rem 1.2rem; outline-offset:3px; }}
  .grid {{ grid-template-columns:1fr; gap:1.2rem; }}
  .panel {{ transform:none !important; }}
  .pnum {{ top:-12px; left:-8px; }}
  .tag {{ font-size:1.5rem; }} .sfx {{ font-size:1.4rem; }}
  .sub {{ font-size:.85rem; }}
}}
</style></head>
<body><div class='wrap'>
  <div class='sfx'>GO!!</div><br>
  <div class='tag'><span>HOW IT WORKS</span></div>
  <div class='sub'>GAUSS LENS is 3D analysis-by-synthesis, built from scratch —
  perception can be neural, the 3D solve is auditable. Meta's SAM 3.1 finds the
  face regions; a Gauss-Newton solver with no network inside it fits the morphable mesh;
  Meta's Muse Spark then writes the verdict from the solver's own report; every number below comes from that solver, phase by phase:</div>
  <div class='grid'>{cards}</div>
  <div class='foot'>NEURAL PERCEPTION &middot; NETWORK-FREE 3D SOLVE &middot; AUDITABLE VERDICT</div>
</div>{_AUTOFIT_JS}</body></html>"""


# ---------------------------------------------------------------------------
# Pipeline log panel — a "match report" styled the same way as the loader
# (real <iframe>, ink borders, Bangers headers). Each curated section is a
# collapsible "round" (plain <details>/<summary>, no JS needed); the raw log
# download is a data-URI link styled like the app's own primary button.
# ---------------------------------------------------------------------------

_LOG_PANEL_HEIGHT = 560
_DECISION_COLOR = {"PROCEED": "#1a7f37", "CONSERVATIVE": "#b8860b", "FALLBACK": "#e3122b"}
_DECISION_ICON  = {"PROCEED": "✅", "CONSERVATIVE": "⚠️", "FALLBACK": "❌"}
_LOG_DEFAULT_OPEN = {"Phase 2 · S3 — quality gate", "Phase 2 · alignment check",
                     "Result summary", "Error - traceback", "Phase 1 · S1 — validation"}
_DECISION_COLOR.update({"REJECTED": "#e3122b", "ERROR": "#e3122b"})
_DECISION_ICON.update({"REJECTED": "\u26d4", "ERROR": "\U0001f4a5"})


def _log_panel_html(curated, quality: float, decision: str, raw_log: str) -> str:
    color = _DECISION_COLOR.get(decision, "#111")
    icon  = _DECISION_ICON.get(decision, "❔")

    stamp_txt = f"{icon} {decision}" + (f" \u00b7 c_t={quality:.3f}" if quality is not None else "")

    rounds = []
    for i, (title, lines) in enumerate(curated, 1):
        open_attr = " open" if title in _LOG_DEFAULT_OPEN else ""
        body = "".join(f"<div class='logline'>{html.escape(l)}</div>" for l in lines)
        rounds.append(f"""<details class='round'{open_attr}>
          <summary><span class='rn'>R{i:02d}</span>{html.escape(title)}
            <span class='cnt'>{len(lines)}</span></summary>
          <div class='roundbody'>{body}</div>
        </details>""")
    rounds_html = "".join(rounds) if rounds else (
        "<div class='empty'>No checkpoint lines matched — nothing to show.</div>")

    b64 = base64.b64encode(raw_log.encode("utf-8", errors="replace")).decode()
    dl_href = f"data:text/plain;charset=utf-8;base64,{b64}"

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Zen+Kaku+Gothic+New:wght@500;700;900&family=Space+Mono&display=swap');
:root {{ --ink:#111; --red:#e3122b; }}
html, body {{ margin:0; padding:0; background:transparent; }}
* {{ box-sizing:border-box; }}
.wrap {{
  font-family:'Zen Kaku Gothic New',sans-serif; background:#fff;
  background-image: radial-gradient(circle, #111 1px, transparent 1.6px);
  background-size: 8px 8px;
  border:5px solid var(--ink); outline:2px solid var(--ink); outline-offset:5px;
  padding:1.05rem 1.15rem 1.2rem; height:{_LOG_PANEL_HEIGHT - 16}px; overflow-y:auto;
}}
.head {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem;
  flex-wrap:wrap; margin-bottom:.8rem; }}
.head h2 {{
  font-family:'Bangers',sans-serif; font-size:1.55rem; letter-spacing:.05em; margin:0;
  background:var(--ink); color:#fff; padding:.15rem 1rem; transform:skewX(-9deg);
  box-shadow:5px 5px 0 var(--red); display:inline-block;
}}
.head h2 span {{ display:inline-block; transform:skewX(9deg); }}
.stamp {{
  font-family:'Bangers',sans-serif; font-size:1.05rem; letter-spacing:.05em; color:{color};
  border:3px solid {color}; border-radius:6px; padding:.2rem .8rem; transform:rotate(-5deg);
  background:#fff;
}}
.badge {{
  font-family:'Zen Kaku Gothic New',sans-serif; font-weight:900; font-size:.68rem;
  letter-spacing:.04em; text-transform:uppercase; color:#7a5b00;
  background:#fff8dc; border:2px solid #d7a900; border-radius:4px; padding:.2rem .55rem;
}}
.note {{
  font-size:.78rem; font-weight:600; color:#555; margin:-.3rem 0 .75rem;
  border-left:3px solid #d7a900; padding-left:.55rem;
}}
.round {{ background:#fff; border:3px solid var(--ink); margin-bottom:.55rem; box-shadow:3px 3px 0 var(--ink); }}
.round summary {{
  cursor:pointer; list-style:none; display:flex; align-items:center; gap:.5rem;
  padding:.4rem .75rem; font-weight:900; font-size:.88rem; background:#fff; user-select:none;
}}
.round summary::-webkit-details-marker {{ display:none; }}
.round summary:before {{ content:'▸'; transition:transform .15s; }}
.round[open] summary:before {{ transform:rotate(90deg); }}
.round .rn {{
  background:var(--red); color:#fff; font-family:'Bangers',sans-serif; font-size:.82rem;
  padding:1px 7px; transform:skewX(-9deg); flex-shrink:0;
}}
.round .cnt {{ margin-left:auto; font-size:.72rem; color:#666; font-weight:700; }}
.roundbody {{ border-top:2px dashed var(--ink); padding:.45rem .75rem .65rem; }}
.logline {{
  font-family:'Space Mono',monospace; font-size:.7rem; line-height:1.55; white-space:pre-wrap;
  word-break:break-word; padding:1px 0 1px 8px; border-left:3px solid #ddd;
}}
.logline:nth-child(odd) {{ background:#faf9f5; }}
.empty {{ font-weight:700; padding:.6rem 0; }}
.dlbtn {{
  display:inline-block; margin-top:.75rem; background:var(--ink); color:#fff !important;
  text-decoration:none; font-family:'Bangers',sans-serif; letter-spacing:.06em; font-size:.95rem;
  padding:.45rem 1rem; border:3px solid var(--ink); box-shadow:4px 4px 0 var(--red);
}}
.dlbtn:hover {{ transform:translate(-2px,-2px); box-shadow:6px 6px 0 var(--red); }}
@media (max-width:640px) {{
  .wrap {{ padding:.8rem .75rem 1rem; outline-offset:3px; }}
  .head h2 {{ font-size:1.25rem; }} .dlbtn {{ display:block; text-align:center; }}
}}
</style></head>
<body><div class='wrap'>
  <div class='head'>
    <h2><span>PIPELINE LOG</span></h2>
    <div class='badge'>SUMMARY · NOT FULL LOG</div>
    <div class='stamp'>{html.escape(stamp_txt)}</div>
  </div>
  <div class='note'>Curated excerpt: per-iteration lines are summarised in one line per fit (&times;N = the same fit ran N times with identical results), and numbers are rounded to 4 decimals. The raw download below keeps everything at full precision.</div>
  {rounds_html}
  <a class='dlbtn' href='{dl_href}' download='face_params_pipeline_log.txt'>⬇ DOWNLOAD FULL RAW LOG</a>
</div></body></html>"""


def _render_log_section(quality, decision):
    """Curated log panel. Works for success, Stage-1 rejection and crashed runs."""
    st.subheader("🧾 Pipeline log — summary")
    st.caption(
        "A **curated summary** of the solver's trace for this run — per-iteration Gauss-Newton "
        "lines are collapsed and exact duplicates removed. Unrecognised tagged lines land under "
        "*Other checkpoints* so nothing is silently dropped. Use **⬇ DOWNLOAD FULL RAW LOG** "
        "inside the panel for the complete, unfiltered trace."
    )
    raw_log = st.session_state.get("extract_log", "")
    if raw_log:
        _html_frame(_log_panel_html(_curate_log(raw_log), quality, decision, raw_log),
                    height=_LOG_PANEL_HEIGHT)
    else:
        st.info("No log was captured for this run.")


def _short(e) -> str:
    """One line, max ~140 chars: never dump a wall of text onto the page."""
    first = (str(e).strip().splitlines() or [""])[0]
    return f"{type(e).__name__}: {first[:140]}"


def _check_sam():
    """None when SAM 3.1 is ready; otherwise ONE short line (full traceback kept in session_state for 'details')."""
    st.session_state["_detail_sam"] = ""
    if not os.environ.get("MODEL_API_KEY"):
        return "MODEL_API_KEY not found (put it in a .env file next to app.py)."
    try:
        from face_params.perception import sam_client, SamLandmarks   # noqa: F401
    except (Exception, SystemExit) as e:
        st.session_state["_detail_sam"] = traceback.format_exc()
        return _short(e)
    return None


def _check_muse():
    """None when Muse Spark is ready; otherwise ONE short line."""
    st.session_state["_detail_muse"] = ""
    if not os.environ.get("MODEL_API_KEY"):
        return "MODEL_API_KEY not found (put it in a .env file next to app.py)."
    try:
        from face_params.audit import make_verdict                    # noqa: F401
    except (Exception, SystemExit) as e:
        st.session_state["_detail_muse"] = traceback.format_exc()
        return _short(e)
    return None


def _single_verdict(report):
    """The verdict is written by Muse Spark. If it does not answer, raise (no silent template text)."""
    from face_params.audit import make_verdict
    text, source = make_verdict(report, "muse")
    print(f"[verdict] source={source}")
    if "muse" not in str(source).lower():
        raise RuntimeError(f"Muse Spark did not answer (got '{source}'). Check the connection and try again.")
    return text, source


def _is_rejection(e) -> bool:
    """Stage-1 rejection, however the pipeline raises it (SourceRejected or run.py's ValueError)."""
    try:
        from face_params.pipeline import SourceRejected
        if isinstance(e, SourceRejected):
            return True
    except Exception:
        pass
    return isinstance(e, ValueError) and "rejected by validation" in str(e)


def _summary_dict(result) -> dict:
    """Same structure run.py writes to summary.json: when SAM masks measured the eyes / mouth,
    those numbers are the headline and the fitted values are kept under *_fitted_unreliable."""
    p = result.params
    summary = {
        "quality": float(result.quality), "decision": result.decision,
        "head_pose_deg": p.head_pose_deg,
        "eye": {k: float(v) for k, v in p.eye.items()},
        "mouth_openness": float(p.mouth_openness), "mouth_open": bool(p.mouth_open),
        "region_error_px": result.region_error,
    }
    sm = getattr(result, "sam_measures", None)
    if sm:
        summary["sam_measures"] = sm
        if "eye_aspect_L" in sm or "eye_aspect_R" in sm:
            summary["eye_fitted_unreliable"] = summary.pop("eye")
            summary["eye"] = {"aspect_left": sm.get("eye_aspect_L"),
                              "aspect_right": sm.get("eye_aspect_R"),
                              "source": "SAM mask height/width"}
        if "mouth_open" in sm:
            summary["mouth_open"] = bool(sm["mouth_open"])
            summary["mouth_aspect"] = sm["mouth_aspect"]
            summary["mouth_openness_fitted_unreliable"] = summary.pop("mouth_openness")
    return summary


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_extract(I_src, model_path, progress_cb, strict=True, landmarks="sam",
                src_bytes=None, src_suffix=".png"):
    """Single image in: Phase 1 on it, then Phase 2 fit on the same frame."""
    I_tgt = I_src
    Hs, Ws = I_src.shape[:2]

    progress_cb(5, "Loading 3DMM model…")
    print(f"[input] image={Ws}x{Hs}px  landmarks={landmarks}  strict={strict}")
    model, uv, K, mouth_mask = load_model(model_path, Hs, Ws)
    is_synthetic = (model_path is None) or (not os.path.isfile(model_path))
    if is_synthetic:
        st.warning("⚠️ No 3DMM model (.npz) supplied — running with the built-in "
                   "synthetic model. Results will be approximate.")

    lm_vidx = _make_lm_vidx(model)

    provider, sam_measures, sam_missing = None, None, []
    if landmarks == "sam":
        # Real landmarks from SAM 3.1 masks (<= 5 live calls; cached answers are free).
        from face_params.perception import SamLandmarks
        progress_cb(8, "SAM 3.1 — finding face regions…")
        tmp_img = _write_tmp(src_bytes, src_suffix)
        try:
            print("[Perception] SAM 3.1 regions…")
            from face_params.perception import sam_client as _sc
            _b = dict(_sc.stats())            # sam_client counters live for the whole server process, so log the delta
            sam = SamLandmarks(tmp_img, Ws, Hs)
            _a = _sc.stats()
            print(f"[Perception] SAM calls this run: {{'live': {_a['live'] - _b['live']}, "
                  f"'cached': {_a['cached'] - _b['cached']}}}  (cached answers are free)")
            model = sam.calibrate(model)
            lmset = sam.landmark_set(model, K)
        finally:
            os.unlink(tmp_img)           # kept until calibrate/landmark_set are done with the file
        lm_vidx = lmset.vidx
        provider = lambda img: lmset
        sam_measures, sam_missing = lmset.measures, lmset.missing
        print(f"[Perception] {len(lm_vidx)} landmarks from SAM; missing regions: {sam_missing or 'none'}")

    recogniser = FaceRecogniser(model=model, K=K, landmark_provider=provider)
    pipe       = FaceParamPipeline(model, K, uv, mouth_mask)

    progress_cb(15, "Phase 1 — session init…")
    print("[Phase 1] Session init…")
    try:
        session, src_r = build_session(
            pipe, recogniser, I_src, [I_tgt], model, lm_vidx,
            strict_validation=strict,
        )
    except Exception as e:
        if not _is_rejection(e):          # a Stage-1 rejection is an outcome, not a crash: no traceback in the log
            print(traceback.format_exc())
        else:
            print(f"[Phase1/S1] REJECTED: {e}")
        raise

    progress_cb(75, "Phase 2+3 — tracking pose, eyes, brows, mouth…")
    print("[Phase 2+3] Fitting frame…")
    try:
        tgt_r  = recogniser.process(I_tgt, lm_vidx)
        frame  = FrameData(I_tgt.astype(np.float64), tgt_r.landmarks, lm_vidx)
        fstate = pipe.init_frame_state(frame, session)

        res, _ = process_frame(
            pipe, session, recogniser, I_tgt, fstate,
            session.source_state, lm_vidx
        )
    except Exception:
        print(traceback.format_exc())
        raise

    p = res.params
    if sam_measures and "mouth_open" in sam_measures:
        mouth_txt = (f"{'open' if sam_measures['mouth_open'] else 'closed'} "
                     f"(lips h/w {sam_measures['mouth_aspect']:.2f}, from SAM mask)")
    else:
        mouth_txt = f"{'open' if p.mouth_open else 'closed'} ({p.mouth_openness:.3f})"
    print(f"[result] quality={res.quality:.3f}  {res.decision}  |  mouth={mouth_txt}  |  "
          f"yaw={p.head_pose_deg['yaw']:+.1f}  pitch={p.head_pose_deg['pitch']:+.1f}  "
          f"roll={p.head_pose_deg['roll']:+.1f}")
    if sam_measures:
        print(f"[result] measured from SAM masks: {sam_measures}")
    progress_cb(92, "Muse Spark — writing the verdict…")
    from face_params.audit import build_report as _br
    res.report  = _br(session, res, sam_measures, sam_missing)
    try:                                               # part of the run: no second block, no button
        res.verdict, res.verdict_error = _single_verdict(res.report), None
    except Exception as _ve:
        res.verdict, res.verdict_error = None, str(_ve)
        print(f"[verdict] FAILED: {_ve}")
    progress_cb(100, "Complete ✓")
    res.session, res.landmark_mode = session, landmarks
    res.sam_measures, res.sam_missing = sam_measures, sam_missing
    return res


def _write_tmp(raw: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    return path


def _residual_chart(report: dict, key: str):
    """The solver's residual at every Gauss-Newton iteration; the replay button redraws it step by step."""
    hist = report.get("residual_history")
    if not hist:
        return
    import time as _t
    import pandas as pd
    vals = [hist[0]["residual_before"]] + [h["residual_after"] for h in hist]
    df = pd.DataFrame({"residual": vals}, index=pd.Index(range(len(vals)), name="Gauss-Newton iteration"))
    rc = report.get("residual_curve", {})
    st.markdown("**Gauss-Newton residual per iteration**")
    slot = st.empty()
    slot.line_chart(df)
    st.caption(f"{rc.get('start')} → {rc.get('end')} ({rc.get('reduction_pct')}% lower) in {rc.get('iterations')} iterations — "
               + ("converged." if rc.get("converged") else f"stopped at the {rc.get('max_iterations')}-iteration cap."))
    if st.button("▶ Replay the solve", key=f"replay_{key}"):
        for i in range(1, len(df) + 1):
            slot.line_chart(df.iloc[:i])
            _t.sleep(0.35)


def _verdict_panel(report: dict, verdict=None, error=None):
    """ONE verdict block per run, written by Muse Spark during the run. No fallback text: show the error."""
    if not verdict:
        st.error(f"Muse Spark verdict unavailable — {error or 'no response'}")
        return
    text, _source = verdict[0], verdict[1]
    (st.warning if report.get("decision") == "REJECTED" else st.info)(f"**Muse Spark** — {text}")
    st.caption("Written by Meta's Muse Spark from the solver's report; it only quotes numbers from that report.")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key in ("src_arr", "extract_result", "src_bytes", "src_suffix", "reject_report", "reject_msg", "reject_verdict", "reject_verdict_error", "muse_out"):
    st.session_state.setdefault(key, None)


# ---------------------------------------------------------------------------
# Header  (title panel, centered; two robots cross behind it)
# ---------------------------------------------------------------------------

def _hero_frame(doc: str, height: int = 470) -> None:
    """Hero banner inside a keyed container that CSS stretches to the full screen width."""
    with _overlay_container("hero_wide"):
        _html_frame(doc, height=height)


_result_for_hero = st.session_state.get("extract_result")
if st.session_state.pop("_curtain", False):
    st.markdown("<div class='fo-curtain'><i></i></div>", unsafe_allow_html=True)
    _sfx("curtain")
if _result_for_hero is None:
    _hero_frame(
        _hero("READY!!", "Upload a <b>face</b> photo, switch on <b>SAM</b> + <b>Muse</b>, "
                          "then hit <b>Run</b>!"),
        height=470,
    )
else:
    _icon = {"PROCEED": "✅", "CONSERVATIVE": "⚠️", "FALLBACK": "❌"}.get(_result_for_hero.decision, "")
    _hero_frame(
        _hero("EXTRACTED!!", f"<b>EXTRACTED!</b><br>Quality {_result_for_hero.quality:.2f} — "
                          f"{_icon} {_result_for_hero.decision}. Scroll down for the full "
                          f"breakdown."),
        height=470,
    )
_ticker()
_step_slot = st.empty()


# ---------------------------------------------------------------------------
# About this project — an inline panel styled like a closable browser tab
# (dark tab pill + ✕), sitting in the normal page scroll — not a real
# st.tabs() page. Click ✕ to collapse it to just the pill; click the pill
# again (▸) to reopen. State lives in st.session_state.show_about.
# ---------------------------------------------------------------------------

st.session_state.setdefault("show_about", True)

with st.container(key="about_tabbar"):
    bar_l, bar_r = st.columns([0.9, 0.1])
    with bar_l:
        st.markdown("<div class='about-pill'>ℹ️ ABOUT · GAUSS LENS</div>",
                    unsafe_allow_html=True)
    with bar_r:
        if st.session_state.show_about:
            if st.button("✕", key="about_close", help="Close"):
                st.session_state.show_about = False
                st.rerun()
        else:
            if st.button("▸", key="about_open", help="Show About"):
                st.session_state.show_about = True
                st.rerun()

if st.session_state.show_about:
    _tab_how, _tab_flow, _tab_math = st.tabs(["How it works", "Flowchart", "Math in brief"])
    with _tab_how:
        with _overlay_container("about_frame"):
            _html_frame(_about_panel_html(), height=_ABOUT_HEIGHT)
    with _tab_flow:
        st.caption("Where the neural models sit, and where they stop. Dashed arrows are the fallback paths.")
        with _overlay_container("flow_frame"):
            _html_frame(_flow_html(), height=_FLOW_HEIGHT)
    with _tab_math:
        st.caption("The whole 3D stage is these five ideas. Every symbol is a number you can read in report.json.")
        st.markdown("**1 · Camera.** A model vertex S_i is rotated by ω, shifted by t and projected with the camera K:")
        st.latex(r"\pi_i(\Theta)=\mathrm{proj}\big(K\,(R(\omega)\,S_i(\alpha,\beta,\psi,\gamma,\delta)+t)\big)")
        st.markdown("**2 · What is minimised.** The distance between the projected vertices and the landmarks x_i that SAM's masks gave, "
                    "plus an appearance term and priors that keep the face plausible:")
        st.latex(r"E(\Theta)=w_{app}E_{app}+\sum_i \lVert \pi_i(\Theta)-x_i \rVert^2+E_{reg}")
        st.markdown("**3 · How it is minimised.** Gauss-Newton: linearise, solve a damped linear system, take the step only if the residual drops "
                    "(otherwise back off). No learned weights, no network:")
        st.latex(r"\Delta\Theta=-\big(J^{\top}J+\lambda\Lambda\big)^{-1}J^{\top}e,\qquad \lambda=0.01")
        st.markdown("**4 · When to refuse.** Five scores in [0, 1]. The photo is accepted only if the smallest one is above 0.60:")
        st.latex(r"s_{res}=\min\!\big(1,\tfrac{\min(H,W)}{256}\big),\quad s_{pose}=e^{-(\omega_y^2+\omega_x^2)/2\sigma^2},\quad "
                 r"s_{lm}=\tfrac1N\sum_i e^{-\lVert x_i-\hat x_i\rVert^2/2\sigma_{lm}^2},\quad \min(s)>0.60")
        st.markdown("**5 · The final call.** Phase 2 combines landmark fit, motion and pose into one confidence, then thresholds it:")
        st.latex(r"c_t=0.5\,s_{lm}+0.3\,s_{flow}+0.2\,s_{pose}\;\Rightarrow\;"
                 r"\begin{cases}>0.75&\text{PROCEED}\\0.40\text{–}0.75&\text{CONSERVATIVE}\\<0.40&\text{FALLBACK}\end{cases}")
        st.markdown("**And the words.** Muse Spark's sentence is kept only if every number in it also appears in the report:")
        st.latex(r"\text{keep}\iff \mathrm{numbers}(\text{reply})\subseteq \mathrm{numbers}(\text{report.json})")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

col_src, col_tip = st.columns(2)

with col_src:
    st.subheader("Face image")
    src_file = st.file_uploader("Upload face image",
                                type=ACCEPTED_FORMATS, key="src_upload")
    if src_file:
        _key = (src_file.name, getattr(src_file, "size", None))
        if st.session_state.get("_src_key") != _key or st.session_state.get("_src_pil") is None:
            try:                                           # once per upload, not on every rerun
                _pil, _raw, _suf = _prepare_upload(src_file.getvalue(), src_file.name)
                st.session_state.update(_src_key=_key, _src_pil=_pil, src_arr=_pil_to_ndarray(_pil),
                                        src_bytes=_raw, src_suffix=_suf)
            except Exception as _e:
                st.session_state.update(_src_key=None, _src_pil=None, src_arr=None)
                st.error(f"Could not read that image ({_e}). Try a JPG or PNG.")
        if st.session_state.get("_src_pil") is not None:
            st.image(st.session_state["_src_pil"], width="stretch")

with col_tip:
    st.subheader("Tips")
    st.markdown("- One clear, roughly frontal face\n"
                "- Face at least **192 px** wide (Stage-1 check)\n"
                "- Even lighting, no heavy blur\n"
                "- Optional: a 3DMM `.npz` below for real faces")

with st.expander("Optional — upload 3DMM model (.npz)"):
    model_file = st.file_uploader("3DMM model", type=["npz"])


# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

ready = st.session_state.get("src_arr") is not None
from face_params.phase1.stage1_validation import ACCEPT_THRESHOLD as _ACCEPT
strict = st.checkbox("Strict Stage-1 validation (reject low-quality sources)", value=True,
                     help=f"Source face must be ≥192 px wide; all 5 validation scores must exceed {_ACCEPT:.2f}.")
st.subheader("Connect models")
_k1, _k2 = st.columns(2)
with _k1:
    sam_on = st.toggle("🔌 SAM 3.1 — finds the face parts", key="conn_sam")
with _k2:
    muse_on = st.toggle("🔌 Muse Spark — writes the verdict", key="conn_muse")

_sam_err  = _check_sam()  if sam_on  else "SAM 3.1 is switched off."
_muse_err = _check_muse() if muse_on else "Muse Spark is switched off."
for _on, _err, _name, _dk in ((sam_on, _sam_err, "SAM 3.1", "_detail_sam"),
                              (muse_on, _muse_err, "Muse Spark", "_detail_muse")):
    if not _on:
        continue
    if _err:
        st.error(f"{_name} could not connect — {_err}")
        if st.session_state.get(_dk):
            with st.expander("details"):
                st.code(st.session_state[_dk][-1500:], language="text")
    else:
        st.success(f"{_name} connected ✓")

# sound: chime when a model connects, buzzer when it fails (only on a state *change*, not on every rerun)
_conn_snd = []
for _key, _on, _err in (("sam", sam_on, _sam_err), ("muse", muse_on, _muse_err)):
    _state = (("err" if _err else "ok") if _on else None)
    if st.session_state.get(f"_snd_conn_{_key}") != _state:
        st.session_state[f"_snd_conn_{_key}"] = _state
        if _state:
            _conn_snd.append(_state)
if _conn_snd:
    _sfx("error" if "err" in _conn_snd else "connect", delay=180)

run_btn = st.button("▶ Run Extraction", disabled=not ready, type="primary")
if run_btn and (_sam_err or _muse_err):
    st.error("Run not started — connect both models first. "
             + " ".join(m for m in (_sam_err, _muse_err) if m))
    _sfx("error")
    run_btn = False

if not ready:
    st.caption("Upload a face image to enable extraction.")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_btn:
    I_src = st.session_state.src_arr
    st.session_state.reject_report = None

    model_path = None
    if model_file is not None:
        tmp_fd, model_path = tempfile.mkstemp(suffix=".npz")
        try:
            os.write(tmp_fd, model_file.read())
        finally:
            os.close(tmp_fd)

    st.session_state["_boot_acc"] = random.choice([0, 1])      # fresh colourway for every run
    with _overlay_container("fo_boot"):
        loader_slot = st.empty()
    loader      = _LoaderTracker(loader_slot, render=_run_screen_html,
                                 milestones=_LOAD_MILESTONES, height=900)
    loader.update(0, "Warming up the solver…")

    def progress_cb(pct: int, msg: str):
        loader.update(pct, msg)

    log_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buf):
            result = run_extract(I_src, model_path, progress_cb, strict,
                                 landmarks="sam",
                                 src_bytes=st.session_state.src_bytes,
                                 src_suffix=st.session_state.src_suffix)
        st.session_state.extract_result = result
        st.session_state["_toast_pending"] = True
        st.session_state.extract_log = log_buf.getvalue()
        time.sleep(1.6)          # let the K.O.! frame land before it disappears
    except Exception as e:
        st.session_state.extract_log = log_buf.getvalue()
        if _is_rejection(e):                     # expected outcome: explain it, no traceback
            from face_params.audit import build_report
            st.session_state.extract_result = None
            st.session_state.reject_msg = str(e)
            st.session_state["_snd_reject"] = True
            st.session_state.reject_report = build_report(validation=getattr(e, "validation", None))
            try:
                st.session_state.reject_verdict = _single_verdict(st.session_state.reject_report)
                st.session_state.reject_verdict_error = None
            except Exception as _ve:
                st.session_state.reject_verdict, st.session_state.reject_verdict_error = None, str(_ve)
        else:
            _sfx("error")
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc(), language="python")
            loader_slot.empty()
            _render_log_section(None, "ERROR")   # show what the solver printed before it died
            raise
    finally:
        if model_path and os.path.exists(model_path):
            os.unlink(model_path)
        loader_slot.empty()


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

result = st.session_state.get("extract_result")

if result is None and st.session_state.get("reject_report"):
    if st.session_state.pop("_snd_reject", False):
        _sfx("lose")
    _gdiv()
    st.subheader("Source rejected")
    _verdict_panel(st.session_state.reject_report, st.session_state.get("reject_verdict"),
                   st.session_state.get("reject_verdict_error"))
    if st.session_state.get("reject_msg"):
        st.caption(st.session_state.reject_msg)
    st.caption("The face must be ≥192 px wide, sharp, frontal and well fitted. Use a better image, "
               "or untick **Strict Stage-1 validation** to continue anyway.")
    st.divider()
    _render_log_section(None, "REJECTED")

if result is not None:
    _gdiv()
    st.subheader("Result")
    if st.session_state.pop("_toast_pending", False):
        st.toast(f"EXTRACTED!  c_t = {result.quality:.2f} · {result.decision}", icon="🎯")
        _sfx({"PROCEED": "win", "CONSERVATIVE": "ok"}.get(result.decision, "lose"), delay=150)
    _icon = {"PROCEED": "✅", "CONSERVATIVE": "⚠️", "FALLBACK": "❌", "REJECTED": "⛔"}.get(result.decision, "")
    st.markdown(f"**{_icon} {result.decision}** · quality c_t = {result.quality:.3f} · the full verdict and quality gate are at the bottom of this section.")
    from face_params.audit import build_report as _build_report
    _rep = getattr(result, "report", None) or _build_report(
        getattr(result, "session", None), result,
        getattr(result, "sam_measures", None), getattr(result, "sam_missing", []))
    _sm = getattr(result, "sam_measures", None) or {}
    _mouth_open = _sm["mouth_open"] if "mouth_open" in _sm else result.mouth_open
    _p = result.params
    st.image(_ndarray_to_pil(result.I_overlay),
             width="stretch",
             caption="Fitted mesh projected onto the frame "
                     "(cyan=eye · yellow=brow · green=nose · pink=mouth · purple=ear)")



    with st.expander("🧱 Baked UV texture (Stage 3 — source colour in UV space)"):
        st.caption("Source-photo colour baked into the model's UV layout (no illumination removal). "
                   "Empty/black areas are UV regions the source photo did not see.")
        st.image(_ndarray_to_pil(np.clip(result.texture, 0, 1)), width="stretch")

    _payload = _summary_dict(result)                      # same structure as run.py's summary.json
    _payload["params"] = {k: np.asarray(v).tolist() for k, v in _p.to_dict().items() if k not in ("K",)}
    st.divider()

    st.subheader("Head pose & alignment")
    hp = _p.head_pose_deg
    c1, c2, c3 = st.columns(3)
    c1.metric("Yaw", f"{hp['yaw']:+.1f}°"); c2.metric("Pitch", f"{hp['pitch']:+.1f}°")
    c3.metric("Roll", f"{hp['roll']:+.1f}°")
    st.caption("Mean landmark error per region (px) — lower is better")
    _re = {k: v for k, v in result.region_error.items()}
    st.bar_chart({"region": list(_re.keys()), "px": list(_re.values())},
                 x="region", y="px", height=160)

    st.divider()

    _residual_chart(_rep, "ok")

    st.divider()

    # ── Eye parameters ───────────────────────────────────────────────────────
    st.subheader("👁 Eye parameters")
    ep = result.eye_params

    def _fitted_eye_block(ep):
        if all(abs(v) < 1e-9 for v in ep.values()):
            st.warning("Eye parameters are all zero — lm_vidx did not cover eye_idx.")
        st.caption("Aperture: 0=closed · 1=open")
        col_el, col_er = st.columns(2)
        for col, lab, k in ((col_el, "Left aperture", "aperture_left"),
                            (col_er, "Right aperture", "aperture_right")):
            with col:
                st.metric(lab, f"{ep[k]:.3f}", delta="open" if ep[k] > 0.5 else "closed", delta_color="off")
                st.progress(float(np.clip(ep[k], 0.0, 1.0)))
        st.caption("Gaze angles (radians): 0=centre · +H=right · +V=up")
        g1, g2, g3, g4 = st.columns(4)
        for col, lab, k, up, dn in ((g1, "H left", "gaze_h_left", "→", "←"),
                                    (g2, "H right", "gaze_h_right", "→", "←"),
                                    (g3, "V left", "gaze_v_left", "↑", "↓"),
                                    (g4, "V right", "gaze_v_right", "↑", "↓")):
            with col:
                st.metric(lab, f"{ep[k]:.3f}", delta=up if ep[k] > 0 else dn, delta_color="off")

    if "eye_aspect_L" in _sm or "eye_aspect_R" in _sm:
        st.caption("Measured from the SAM eye masks (mask height ÷ width; ~0 = closed, larger = wider open). "
                   "The fitted ψ_eye is poorly determined from a single image, so it is kept separately below.")
        col_el, col_er = st.columns(2)
        for col, lab, k in ((col_el, "Left eye h/w", "eye_aspect_L"), (col_er, "Right eye h/w", "eye_aspect_R")):
            v = _sm.get(k)
            col.metric(lab, "n/a (no mask)" if v is None else f"{float(v):.3f}")
        with st.expander("Fitted ψ_eye (unreliable from a single image)"):
            _fitted_eye_block(ep)
    else:
        _fitted_eye_block(ep)

    st.divider()

    # ── Mouth parameters ─────────────────────────────────────────────────────
    st.subheader("👄 Mouth parameters")
    col_ms, col_mb, col_mw = st.columns(3)
    with col_ms:
        st.metric("Status", "🟡 open" if _mouth_open else "🟢 closed")
    if "mouth_aspect" in _sm:
        with col_mb:
            st.metric("Lips h/w (SAM mask)", f"{float(_sm['mouth_aspect']):.3f}")
        with col_mw:
            st.metric("Fitted β_mouth (unreliable)", f"{result.mouth_score:.3f}")
        st.caption("Open/closed comes from the SAM lip mask. The fitted β_mouth is shown for reference only.")
    else:
        with col_mb:
            st.metric("β_mouth score", f"{result.mouth_score:.3f}", delta="threshold 0.15",
                      delta_color="normal" if result.mouth_score >= 0.15 else "inverse")
        with col_mw:
            st.metric("Openness", f"{float(np.clip(result.mouth_score / 1.5, 0.0, 1.0))*100:.0f}%")

    st.caption("Fitted Δβ[0..3] — jaw separation & lip aperture")
    st.bar_chart({"component": ["β[0] jaw", "β[1] lip", "β[2]", "β[3]"],
                  "value": result.delta_beta[:4].tolist()}, x="component", y="value", height=160)

    st.divider()

    # ── Nose parameters ───────────────────────────────────────────────────────
    st.subheader("👃 Nose parameters  (δ_nose)")
    src_state = getattr(result, "src_state", None)
    if src_state is not None and hasattr(src_state, "delta_nose"):
        dn = src_state.delta_nose
        if dn is None or len(dn) == 0 or np.all(dn == 0):
            st.warning(
                "δ_nose is all zeros — nose_idx not covered by lm_vidx, "
                "or using real 3DMM without nose landmarks."
            )
        else:
            v_nose = len(dn) // 3
            dn_mat = dn.reshape(v_nose, 3)
            mags   = np.linalg.norm(dn_mat, axis=1)

            col_n1, col_n2, col_n3 = st.columns(3)
            with col_n1: st.metric("V_nose", v_nose)
            with col_n2: st.metric("Mean |δ_nose|", f"{mags.mean():.4f}")
            with col_n3: st.metric("Max  |δ_nose|", f"{mags.max():.4f}")

            axis_data = {
                "axis":   ["x (left/right)", "y (up/down)", "z (depth)"],
                "mean δ": [float(dn_mat[:,i].mean()) for i in range(3)],
            }
            st.bar_chart({"axis": axis_data["axis"],
                          "value": axis_data["mean δ"]},
                         x="axis", y="value", height=140)

            vert_data = {
                "vertex": [f"v{i}" for i in range(v_nose)],
                "‖δ‖":   mags.tolist(),
            }
            st.bar_chart(vert_data, x="vertex", y="‖δ‖", height=120)
    else:
        st.info("δ_nose not available")

    st.divider()

    # ── Expression delta ─────────────────────────────────────────────────────
    st.subheader("Expression delta  (Δβ)")
    st.caption("First 16 of 64 expression delta components")
    delta_vals = result.delta_beta[:16].tolist()
    bar_data   = {
        "component": [f"β[{i}]" for i in range(16)],
        "value":     delta_vals,
    }
    st.bar_chart(bar_data, x="component", y="value", height=200)

    st.divider()

    # ── Quality gate ─────────────────────────────────────────────────────────
    st.subheader("Quality gate")
    col_q, col_d, col_m = st.columns(3)
    with col_q:
        st.metric("Quality score c_t", f"{result.quality:.3f}",
                  delta="pass ≥0.75" if result.quality >= 0.75 else "low <0.75",
                  delta_color=_score_color(result.quality))
    with col_d:
        icons = {"PROCEED": "✅", "CONSERVATIVE": "⚠️", "FALLBACK": "❌"}
        st.metric("Decision", f"{icons.get(result.decision,'')} {result.decision}")
    with col_m:
        st.metric("Mouth", "🟡 open" if _mouth_open else "🟢 closed",
                  delta=(f"lips h/w={_sm['mouth_aspect']:.2f} (from SAM mask)" if "mouth_aspect" in _sm
                         else f"β_mouth={result.mouth_score:.3f} threshold=0.15"))

    with st.expander("Quality gate explanation"):
        st.markdown(
"""**c_t = 0.50·s_lm + 0.30·s_flow + 0.20·s_pose**

*(single image: no previous frame, so s_flow = 1.0 and the score comes from landmarks + pose)*

| Decision | Threshold | Action |
|---|---|---|
| PROCEED | c_t > 0.75 | Accept fit |
| CONSERVATIVE | 0.40 ≤ c_t ≤ 0.75 | Accept, lower confidence |
| FALLBACK | c_t < 0.40 | Keep previous frame's motion state |
""")

    st.divider()

    # ── Pipeline log (curated) ───────────────────────────────────────────────
    _render_log_section(result.quality, result.decision)

    st.divider()

    st.subheader("Verdict")
    _verdict_panel(_rep, getattr(result, "verdict", None), getattr(result, "verdict_error", None))

    _d1, _d2, _d3 = st.columns(3)
    _d1.download_button("⬇ Parameters (JSON)", data=json.dumps(_payload, indent=2, default=_json_default),
                        file_name="face_params.json", mime="application/json")
    _d2.download_button("⬇ Overlay PNG", data=_ndarray_to_png_bytes(result.I_overlay),
                        file_name="face_overlay.png", mime="image/png")
    _d3.download_button("⬇ UV texture PNG",
                        data=_ndarray_to_png_bytes(np.clip(result.texture, 0, 1).astype(np.float32)),
                        file_name="texture.png", mime="image/png")


# ---------------------------------------------------------------------------
# Live step tracker (filled last so it reflects this run's final state)
# ---------------------------------------------------------------------------
_step = (3 if st.session_state.get("extract_result") is not None
         else 2 if (st.session_state.get("src_arr") is not None
                    and st.session_state.get("conn_sam") and st.session_state.get("conn_muse"))
         else 1 if st.session_state.get("src_arr") is not None else 0)
_step_slot.markdown(_stepper_html(_step), unsafe_allow_html=True)


# ---- ambient motion: ink bubbles drifting up the side margins (never over the content column) ----
def _sky_html() -> str:
    """Two identical copies of a skyline side by side (so the strip can drift forever); some windows flicker."""
    rnd, one, x = random.Random(21), [], 0.0
    while x < 100:
        w, t = 2 + rnd.random() * 5, 8 + rnd.random() * 26
        one.append((x, w, t))
        x += w
    parts = []
    for off in (0, 100):
        for bx, w, t in one:
            parts.append(f"<rect x='{bx + off:.2f}' y='{40 - t:.2f}' width='{w + .2:.2f}' height='{t:.2f}' fill='#0d0d0d'/>")
            for r in range(int((t - 2) // 3)):
                for c in range(max(1, int(w // 1.5))):
                    if rnd.random() < .3:
                        parts.append(f"<rect class='{'w r' if rnd.random() < .25 else 'w'}' x='{bx + off + .4 + c * 1.5:.2f}' y='{40 - t + 1 + r * 3:.2f}' "
                                     f"width='.8' height='1.3' style='animation-delay:-{rnd.uniform(0, 6):.1f}s;"
                                     f"animation-duration:{rnd.uniform(3, 7):.1f}s'/>")
    return ("<div class='sky'><svg viewBox='0 0 200 40' preserveAspectRatio='none' "
            "xmlns='http://www.w3.org/2000/svg'>" + "".join(parts) + "</svg></div>")


def _bubbles() -> None:
    rnd, spans = random.Random(7), []
    for i in range(14):
        left = rnd.uniform(.5, 6) if i % 2 else rnd.uniform(93, 98.5)
        size = rnd.uniform(14, 46)
        dur = rnd.uniform(16, 30)
        spans.append(f"<span class='{'f' if i % 3 == 0 else ''}' style='left:{left:.1f}%;width:{size:.0f}px;"
                     f"height:{size:.0f}px;animation-duration:{dur:.1f}s;animation-delay:-{rnd.uniform(0, dur):.1f}s'></span>")
    st.markdown("<div class='fxb'>" + "".join(spans) + "</div>" + _sky_html(), unsafe_allow_html=True)


_bubbles()
