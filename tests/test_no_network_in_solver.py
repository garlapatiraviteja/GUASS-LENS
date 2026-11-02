"""Guards the hackathon claim: no neural network inside the 3D solve.
Only perception/ (SAM client) and audit/ (Muse verdict) may import ML / API libraries."""
import ast, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_params")
BANNED = {"torch", "tensorflow", "onnx", "onnxruntime", "keras", "dlib", "mediapipe",
          "sklearn", "openai", "meta_sam_parser", "transformers"}
ALLOWED_DIRS = {"perception", "audit"}


def test_solver_modules_import_no_ml_libraries():
    bad = []
    for d, _, files in os.walk(ROOT):
        if os.path.basename(d) in ALLOWED_DIRS:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(d, f)
            for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0
                         else [])
                bad += [(path, n) for n in names if n.split(".")[0] in BANNED]
    assert not bad, bad
