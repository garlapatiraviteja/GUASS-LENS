"""
mesh_utils.py — generic mesh helpers kept from the original project
(not swap-specific; usable on any fitted face mesh).

  _compute_vertex_normals : area-weighted vertex normals + 1-ring Laplacian smoothing
  _build_adjacency        : vertex adjacency sets from faces
  _silhouette_vertex_idx  : boundary (silhouette) vertices — edges owned by one face
Public aliases without the underscore are provided.
"""
import numpy as np
from collections import defaultdict


def _build_adjacency(faces, V):
    """Vertex adjacency from mesh faces — used to couple neighboring
    vertices' delta_uv corrections so Cycle A stops solving 625
    independent, uncoupled 2x2 systems."""
    adj = [set() for _ in range(V)]
    for a, b, c in faces:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def _compute_vertex_normals(
        S: np.ndarray, faces: np.ndarray,
        smooth_iters: int = 2, smooth_lambda: float = 0.5,
) -> np.ndarray:
    """Compute area-weighted vertex normals from mesh.

    FIX: area-weighting alone is correct on average but leaves individual
    vertex normals noisy at concave creases (nostril, jaw, chin) where a
    few large, steeply-angled triangles dominate the weighted sum. That
    per-vertex noise is exactly what artifact detection keeps flagging at the
    same regions run after run — sharp normal discontinuities feed
    directly into the shading term, so a noisy normal
    produces a visibly wrong intensity at that single pixel/vertex even
    though neighboring vertices are fine.

    Added an explicit neighbor-smoothing pass on top of area-weighting:
    each vertex's raw (area-weighted) normal is blended with the mean of
    its 1-ring neighbors' raw normals, then renormalized. This is plain
    Laplacian smoothing in normal-space — same idea already used for
    delta_uv in the neighbour-coupled solves (_build_adjacency), applied here
    to normals instead. A couple of iterations is enough to kill the
    single-vertex spikes at creases without washing out genuine curvature
    (eye socket, nose bridge) elsewhere on the mesh.
    """
    V = S.shape[0]
    normals = np.zeros((V, 3), dtype=np.float64)
    for face in faces:
        a, b, c = face
        v0, v1, v2 = S[a], S[b], S[c]
        fn = np.cross(v1-v0, v2-v0)
        normals[a] += fn
        normals[b] += fn
        normals[c] += fn

    def _renormalize(n):
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return n / norms

    normals = _renormalize(normals)

    if smooth_iters > 0:
        adj = _build_adjacency(faces, V)
        # Precompute as a padded neighbor index array so the smoothing
        # pass is vectorized instead of a 625-way Python loop per iter.
        max_deg = max(len(a) for a in adj)
        neighbor_idx = np.full((V, max_deg), -1, dtype=np.int64)
        neighbor_cnt = np.zeros(V, dtype=np.int64)
        for i, nbrs in enumerate(adj):
            nbrs = list(nbrs)
            neighbor_idx[i, :len(nbrs)] = nbrs
            neighbor_cnt[i] = len(nbrs)
        valid = neighbor_idx >= 0
        safe_idx = np.where(valid, neighbor_idx, 0)

        for _ in range(smooth_iters):
            gathered = normals[safe_idx]                      # (V, max_deg, 3)
            gathered = np.where(valid[..., None], gathered, 0.0)
            nbr_sum = gathered.sum(axis=1)                    # (V, 3)
            denom = np.where(neighbor_cnt > 0, neighbor_cnt, 1)[:, None]
            nbr_mean = nbr_sum / denom
            blended = (1.0 - smooth_lambda) * normals + smooth_lambda * nbr_mean
            normals = _renormalize(blended)

    return normals


def _silhouette_vertex_idx(faces: np.ndarray, V: int) -> np.ndarray:
    """Return indices of boundary (silhouette) vertices — edges shared by 1 face."""
    edge_count = defaultdict(int)
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in [(a,b),(b,c),(a,c)]:
            edge_count[(min(u,v), max(u,v))] += 1
    bv = set()
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            bv.add(u); bv.add(v)
    return np.array(sorted(bv), dtype=np.intp)


compute_vertex_normals = _compute_vertex_normals
build_adjacency        = _build_adjacency
silhouette_vertex_idx  = _silhouette_vertex_idx
