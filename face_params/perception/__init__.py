"""Real landmarks from SAM 3.1 masks (the only neural step); the 3D solve stays network-free."""
from .landmarks import SamLandmarks, LandmarkSet, REGION_PROMPTS, init_translation
