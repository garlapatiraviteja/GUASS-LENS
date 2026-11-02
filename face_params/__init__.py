"""
face_params — 3DMM face-parameter extraction (Gauss-Newton fitting + pose tracking).
Extracted from the face-swap project; all swap / render / blend code removed.
"""
from .extractor import FaceParamExtractor, FaceParams, load_model_npz
from .shape_model import MorphableModel, SourceState, TargetState
from .geometry import make_K, project_mesh, rodrigues
from .io_utils import load_image, save_image, load_video_frames
from .synthetic_model import make_synthetic_model, make_synthetic_uv
from .mesh_utils import compute_vertex_normals, build_adjacency, silhouette_vertex_idx
from .texture_utils import uv_to_px, sample_bilinear_3c, texture_gradient

__all__ = ["FaceParamExtractor", "FaceParams", "load_model_npz", "MorphableModel",
           "SourceState", "TargetState", "make_K", "project_mesh", "rodrigues",
           "load_image", "save_image", "load_video_frames", "make_synthetic_model", "make_synthetic_uv",
           "compute_vertex_normals", "build_adjacency", "silhouette_vertex_idx",
           "uv_to_px", "sample_bilinear_3c", "texture_gradient"]

# pipeline-level API (original project structure)
from .pipeline import FaceParamPipeline, FrameData, SessionData  # noqa: E402
