"""
shape_model.py — 3DMM and regional sub-models.

Implements exactly:
  Base shape:      S = S̄ + A_id α + A_exp β
  Eye submodel:    S_eye(ψ) = S_eye^{ref} + B_eye · ψ,   ψ ∈ ℝ⁶
  Brow submodel:   S_brow = S_brow^{3DMM} + B_brow · γ_brow,   γ_brow ∈ ℝ⁸
  Nose submodel:   S_nose = S_nose^{3DMM} + δ_nose,   δ_nose ∈ ℝ^{3|V_nose|}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MorphableModel:
    """
    3D Morphable Model container.

    Attributes
    ----------
    S_bar     : (V, 3)    mean shape
    A_id      : (V, 3, 80) identity basis  — reshaped from (3V, 80)
    A_exp     : (V, 3, 64) expression basis — reshaped from (3V, 64)
    B_eye     : (V_eye, 3, 6)  eye basis
    B_brow    : (V_brow, 3, 8) brow basis
    S_eye_ref : (V_eye, 3)    eye reference shape
    S_brow_3dmm : (V_brow, 3) brow 3DMM shape (= S_bar[brow_idx])
    S_nose_3dmm : (V_nose, 3) nose 3DMM shape (= S_bar[nose_idx])
    eye_idx   : (V_eye,)   global vertex indices belonging to eyes
    brow_idx  : (V_brow,)  global vertex indices belonging to brows
    nose_idx  : (V_nose,)  global vertex indices belonging to nose
    faces     : (F, 3)     triangle connectivity
    """
    S_bar:       np.ndarray          # (V, 3)
    A_id:        np.ndarray          # (V, 3, 80)
    A_exp:       np.ndarray          # (V, 3, 64)
    B_eye:       np.ndarray          # (V_eye, 3, 6)
    B_brow:      np.ndarray          # (V_brow, 3, 8)
    S_eye_ref:   np.ndarray          # (V_eye, 3)
    eye_idx:     np.ndarray          # (V_eye,) int
    brow_idx:    np.ndarray          # (V_brow,) int
    nose_idx:    np.ndarray          # (V_nose,) int
    faces:       np.ndarray          # (F, 3) int
     # NEW — defaults so existing MorphableModel(...)
    ear_idx:   np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    mouth_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    B_ear:     np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 4)))
    B_mouth:   np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 4)))
    # Derived — set automatically after __post_init__
    V:    int = field(init=False)
    F:    int = field(init=False)

    def __post_init__(self):
        self.V = self.S_bar.shape[0]
        self.F = self.faces.shape[0]
        # Validate basis shapes
        assert self.A_id.shape  == (self.V, 3, 80), \
            f"A_id shape mismatch: {self.A_id.shape}"
        assert self.A_exp.shape == (self.V, 3, 64), \
            f"A_exp shape mismatch: {self.A_exp.shape}"

    # ------------------------------------------------------------------
    # Core shape assembly
    # ------------------------------------------------------------------

    def base_shape(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        S = S̄ + A_id α + A_exp β
        alpha : (80,)
        beta  : (64,)
        Returns (V, 3)
        """
        alpha = alpha.reshape(80)
        beta  = beta.reshape(64)
        # A_id:  (V, 3, 80) → A_id @ alpha gives (V, 3)
        S = (self.S_bar
             + (self.A_id  @ alpha)   # (V, 3)
             + (self.A_exp @ beta))   # (V, 3)
        return S

    def neutral_shape(self, alpha: np.ndarray) -> np.ndarray:
        """S^{neutral} = S̄ + A_id α  (β = 0)"""
        return self.base_shape(alpha, np.zeros(64))

    # ------------------------------------------------------------------
    # Eye submodel
    # ------------------------------------------------------------------

    def eye_shape(self, psi: np.ndarray) -> np.ndarray:
        """
        S_eye(ψ) = S_eye^{ref} + B_eye · ψ
        psi : (6,)   = [a_L, θ^h_L, θ^v_L, a_R, θ^h_R, θ^v_R]
        Returns (V_eye, 3)
        """
        psi = psi.reshape(6)
        return self.S_eye_ref + (self.B_eye @ psi)   # (V_eye, 3)

    # ------------------------------------------------------------------
    # Brow submodel
    # ------------------------------------------------------------------

    def brow_shape(self, gamma_brow: np.ndarray) -> np.ndarray:
        """
        S_brow = S_brow^{3DMM} + B_brow · γ_brow
        gamma_brow : (8,)
        Returns (V_brow, 3)
        """
        gamma_brow = gamma_brow.reshape(8)
        S_brow_3dmm = self.S_bar[self.brow_idx]   # (V_brow, 3)
        return S_brow_3dmm + (self.B_brow @ gamma_brow)  # (V_brow, 3)
    
    def ear_shape(self, gamma_ear: np.ndarray) -> np.ndarray:
        """S_ear = S_ear^3DMM + B_ear . gamma_ear   gamma_ear: (4,)"""
        gamma_ear = gamma_ear.reshape(4)
        S_ear_3dmm = self.S_bar[self.ear_idx]

        return S_ear_3dmm + (self.B_ear @ gamma_ear)
    
    def mouth_shape(self, psi_mouth: np.ndarray) -> np.ndarray:
        """S_mouth = S_mouth^3DMM + B_mouth . psi_mouth   psi_mouth: (4,)"""
        psi_mouth = psi_mouth.reshape(4)
        S_mouth_3dmm = self.S_bar[self.mouth_idx]
        return S_mouth_3dmm + (self.B_mouth @ psi_mouth)

    # ------------------------------------------------------------------
    # Nose submodel
    # ------------------------------------------------------------------

    def nose_shape(self, delta_nose: np.ndarray) -> np.ndarray:
        """
        S_nose = S_nose^{3DMM} + δ_nose
        delta_nose : (3·|V_nose|,)  — stacked per-vertex displacements
        Returns (V_nose, 3)
        """
        V_nose = len(self.nose_idx)
        delta_nose = delta_nose.reshape(V_nose, 3)
        S_nose_3dmm = self.S_bar[self.nose_idx]   # (V_nose, 3)
        return S_nose_3dmm + delta_nose

    # ------------------------------------------------------------------
    # Full assembled shape with all regional submodels applied
    # ------------------------------------------------------------------

    def full_shape(self, alpha, beta, psi_eye, gamma_brow, delta_nose,
               psi_mouth: np.ndarray = None,
               gamma_ear: np.ndarray = None) -> np.ndarray:
        S = self.base_shape(alpha, beta)
        S[self.eye_idx]  += self.eye_shape(psi_eye)     - self.S_bar[self.eye_idx]
        S[self.brow_idx] += self.brow_shape(gamma_brow) - self.S_bar[self.brow_idx]
        S[self.nose_idx] += self.nose_shape(delta_nose) - self.S_bar[self.nose_idx]
        if psi_mouth is not None:
            S[self.mouth_idx] += self.mouth_shape(psi_mouth) - self.S_bar[self.mouth_idx]
        if gamma_ear is not None:
            S[self.ear_idx] += self.ear_shape(gamma_ear) - self.S_bar[self.ear_idx]
        return S
# ---------------------------------------------------------------------------
# TPS (thin-plate spline) Laplacian for nose regularisation
# ---------------------------------------------------------------------------

def build_tps_laplacian(nose_idx: np.ndarray,
                        faces: np.ndarray) -> np.ndarray:
    """
    Build the sparse Laplacian matrix L_tps ∈ ℝ^{|V_nose| × |V_nose|}
    restricted to the nose submesh.  Used as regulariser E_regularise = ‖L_tps δ_nose‖².

    Uses the cotangent-weight Laplacian approximated here by the uniform
    graph Laplacian (L = D − A) over nose-submesh edges.

    Returns L_tps as a dense (|V_nose|, |V_nose|) matrix.
    The caller embeds it into the full Λ block-diagonal prior.
    """
    V_nose = len(nose_idx)
    # Build a set of local indices
    global_to_local = {g: l for l, g in enumerate(nose_idx)}

    # Extract edges among nose vertices
    adj = [set() for _ in range(V_nose)]
    for tri in faces:
        local = [global_to_local.get(v, -1) for v in tri]
        # Only add edges where both endpoints are nose vertices
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            la, lb = local[a], local[b]
            if la >= 0 and lb >= 0:
                adj[la].add(lb)
                adj[lb].add(la)

    # Uniform Laplacian: L = D - A
    L = np.zeros((V_nose, V_nose), dtype=np.float64)
    for i in range(V_nose):
        nbrs = list(adj[i])
        deg = len(nbrs)
        L[i, i] = deg
        for j in nbrs:
            L[i, j] = -1.0
    return L


# ---------------------------------------------------------------------------
# State vector Θ  (named container — avoids raw index magic everywhere)
# ---------------------------------------------------------------------------

@dataclass
class SourceState:
    """
    Phase-1 fitted source state vector:
    Θ_src = [α, β, ω, t, ψ_eye, γ_brow, δ_nose]
             ℝ⁸⁰ ℝ⁶⁴ ℝ³  ℝ³   ℝ⁶      ℝ⁸      ℝ^{3|V_nose|}
    """
    alpha:      np.ndarray   # (80,)
    beta:       np.ndarray   # (64,)
    omega:      np.ndarray   # (3,)
    t:          np.ndarray   # (3,)
    psi_eye:    np.ndarray   # (6,)
    gamma_brow: np.ndarray   # (8,)
    delta_nose: np.ndarray   # (3·V_nose,)
      # NEW — defaults:
    psi_mouth: np.ndarray = field(default_factory=lambda: np.zeros(4))
    gamma_ear: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def to_vec(self) -> np.ndarray:
        return np.concatenate([
            self.alpha, self.beta, self.omega, self.t,
            self.psi_eye, self.gamma_brow, self.delta_nose,
            self.psi_mouth, self.gamma_ear,          # NEW
        ])

    @staticmethod
    def from_vec(v, V_nose) -> "SourceState":
        i = 0
        alpha      = v[i:i+80];   i += 80
        beta       = v[i:i+64];   i += 64
        omega      = v[i:i+3];    i += 3
        t          = v[i:i+3];    i += 3
        psi_eye    = v[i:i+6];    i += 6
        gamma_brow = v[i:i+8];    i += 8
        delta_nose = v[i:i+3*V_nose]; i += 3*V_nose
        psi_mouth  = v[i:i+4];    i += 4     # NEW
        gamma_ear  = v[i:i+4]                # NEW
        return SourceState(alpha, beta, omega, t, psi_eye, gamma_brow,
                            delta_nose, psi_mouth, gamma_ear)

    def zeros_like(self) -> "SourceState":
        return SourceState(
            np.zeros_like(self.alpha),
            np.zeros_like(self.beta),
            np.zeros_like(self.omega),
            np.zeros_like(self.t),
            np.zeros_like(self.psi_eye),
            np.zeros_like(self.gamma_brow),
            np.zeros_like(self.delta_nose),
        )


# @dataclass
# class TargetState:
#     """
#     Phase-2 per-frame target state (identity frozen after session init):
#     Θ_tgt^t = [β_tgt^t, ω_tgt^t, t_tgt^t]
#     """
#     beta:  np.ndarray   # (64,)
#     omega: np.ndarray   # (3,)
#     t:     np.ndarray   # (3,)


@dataclass
class TargetState:
    """Theta_tgt^t = [beta, omega, t, psi_eye, gamma_brow, psi_mouth, gamma_ear]"""
    beta:       np.ndarray   # (64,)
    omega:      np.ndarray   # (3,)
    t:          np.ndarray   # (3,)
    psi_eye:    np.ndarray   # (6,)
    gamma_brow: np.ndarray   # (8,)
    psi_mouth:  np.ndarray   # NEW (4,)
    gamma_ear:  np.ndarray   # NEW (4,)
    def to_vec(self) -> np.ndarray:
        return np.concatenate([self.beta, self.omega, self.t,
                                self.psi_eye, self.gamma_brow,
                                self.psi_mouth, self.gamma_ear])
    @staticmethod
    def from_vec(v: np.ndarray) -> "TargetState":
        beta, omega, t = v[:64], v[64:67], v[67:70]
        psi_eye, gamma_brow = v[70:76], v[76:84]
        psi_mouth, gamma_ear = v[84:88], v[88:92]
        return TargetState(beta, omega, t, psi_eye, gamma_brow, psi_mouth, gamma_ear)
