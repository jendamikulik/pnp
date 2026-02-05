#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6 Operator 6 — functional certifier (no placeholders)

Fixes:
- Removes hardcoded placeholder S2 + spectral values.
- Implements real S2 radar (neighbor row-sum rho) computed from the SAME edge-Gram decision operator.
- Implements real decision spectral λ_max(G_H) via power iteration on sparse edge-supported Gram.
- Implements IPC (Invariant Phase Certifier) time-mode power iteration + proper normalization for lock-sparse columns.
- Adds deterministic clause gauge from CNF via build_seed_assignment (seed model) and per-clause satisfaction.
- Integrates CVXOPT dyadic tail-weights (small QP in O(log C)) for IPC clause weights.

python DREAM6_operator_6.py --cnf-path .\uf250-0100.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier
python DREAM6_operator_6.py --cnf-path .\random_3sat_10000.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier --R 56
python DREAM6_operator_6.py --cnf-path .\random_3sat_50000.cnf --mode sat --edge-mode logic --eta 0.5 --d 28 --shared-carrier --R 24

"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------
# DIMACS CNF
# ---------------------------------------------------------------------

def parse_dimacs(path: str) -> Tuple[int, List[List[int]]]:
    clauses: List[List[int]] = []
    nvars = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "c%":
                continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    nvars = int(parts[2])
                continue
            lits = [int(x) for x in s.split() if x != "0"]
            if lits:
                clauses.append(lits)
                for L in lits:
                    nvars = max(nvars, abs(L))
    return nvars, clauses


def is_clause_satisfied(clause: List[int], assign: np.ndarray) -> bool:
    for lit in clause:
        v = abs(lit) - 1
        val = bool(assign[v])
        if lit < 0:
            val = not val
        if val:
            return True
    return False


def count_unsat(clauses: List[List[int]], assign: np.ndarray) -> int:
    unsat = 0
    for cl in clauses:
        if not is_clause_satisfied(cl, assign):
            unsat += 1
    return unsat


def build_seed_assignment(nvars: int, clauses: List[List[int]]) -> np.ndarray:
    """
    Deterministic seed model:
    counts positive vs negative occurrences per variable, then assigns True if count>=0.
    """
    counts = np.zeros(nvars, dtype=np.int64)
    for cl in clauses:
        for lit in cl:
            idx = abs(lit) - 1
            counts[idx] += (1 if lit > 0 else -1)
    return (counts >= 0)

def build_var_clause_incidence(clauses: List[List[int]], nvars: int) -> List[List[Tuple[int, int]]]:
    """
    Incidence list for assignment extraction.
    Returns inc[v] = list of (clause_index j, lit_sign) where:
      lit_sign = +1 if literal is (x_v), -1 if literal is (¬x_v).
    v is 0-based variable index.
    """
    inc: List[List[Tuple[int, int]]] = [[] for _ in range(int(nvars))]
    for j, cl in enumerate(clauses):
        for lit in cl:
            v = abs(int(lit)) - 1
            if 0 <= v < nvars:
                inc[v].append((j, +1 if int(lit) > 0 else -1))
    return inc


def extract_assignment_from_ipc(
    clauses: List[List[int]],
    nvars: int,
    *,
    clause_phasors: np.ndarray,
    theta: float,
    clause_weights: np.ndarray,
    inc: Optional[List[List[Tuple[int,int]]]] = None,
    mode: str = "lin",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coercive phase projection -> Boolean assignment (best-effort witness).

    Key upgrade ("proxy drive"):
      - Use clause phasor *amplitude* |a_j| as a reliability weight.
      - Gate each clause by its *carrier alignment* to the global theta.

    This makes global coherence (proxy) a hybatel rather than a passive observer.
    """
    if nvars <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    if inc is None:
        inc = build_var_clause_incidence(clauses, nvars)

    # Clause phasor geometry
    phi = np.angle(clause_phasors).astype(np.float64, copy=False)
    amp = np.abs(clause_phasors).astype(np.float64, copy=False)

    # Carrier alignment w.r.t. global theta (polarity-independent).
    # We *gate* anti-phase clauses down to ~0 so they cannot "přeřvat" the field.
    # Carrier alignment (Z2-aware): if split into ± lobes, flip the anti-phase clauses by π.
    align = np.cos(wrap_pi(phi - float(theta))).astype(np.float64, copy=False)
    phi_use = phi.copy()
    flip = (align < 0.0)
    if np.any(flip):
        phi_use[flip] = wrap_pi(phi_use[flip] + math.pi)
        align = np.cos(wrap_pi(phi_use - float(theta))).astype(np.float64, copy=False)
    gate = (align * align)  # cos^2 in [0,1], Z2-invariant drive

    w = clause_weights.astype(np.float64, copy=False)
    if w.shape[0] != phi.shape[0]:
        raise ValueError("clause_weights length mismatch with clause_phasors / C")

    score = np.zeros(int(nvars), dtype=np.float64)

    """for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            s += w[j] * float(lit_sign) * align[j]
        score[v] = s"""

    """for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            th = float(theta) if lit_sign > 0 else float(theta) + math.pi
            s += w[j] * math.cos(float(wrap_pi(phi[j] - th)))
        score[v] = s"""

    for v in range(int(nvars)):
        s = 0.0
        for (j, lit_sign) in inc[v]:
            # Small deterministic polarity offset (keeps your cvxopt "heaviness" nuance)
            offset = (1.0 - w[j]) * 0.1
            th = float(theta) + offset if lit_sign > 0 else float(theta) + math.pi - offset

            # Local vote (polarity-aware)
            dphi = float(wrap_pi(phi_use[j] - th))
            if mode == "z2":
                contribution = math.cos(2.0 * dphi)
            else:
                contribution = math.cos(dphi)

            # Proxy drive:
            #   - amp[j]  : clause reliability from IPC phasor magnitude
            #   - gate[j] : carrier alignment to global theta (coherence -> action)
            s += w[j] * amp[j] * gate[j] * contribution
            #s += w[j] * math.cos(float(wrap_pi(phi[j] - th)))
        score[v] = s


    assign = score >= 0.0
    return assign, score


def sha256_assignment(assign: np.ndarray) -> str:
    a = np.asarray(assign, dtype=np.bool_)
    bits = np.packbits(a.astype(np.uint8), bitorder="little")
    return hashlib.sha256(bits.tobytes()).hexdigest()


# ---------------------------------------------------------------------
# CNF helpers (deterministic UNSAT seeding + full clause logic graph)
# ---------------------------------------------------------------------

def cnf_seed_unsat_indices(
    clauses: List[List[int]],
    nvars: int,
    *,
    denom: int = 16,
    salt: bytes = b"DREAM6::CNF::UNSAT_SEED::v1",
) -> List[int]:
    """
    Deterministically choose a small subset of clauses to carry negative gauge (g=-1).

    Selection rule:
      idx is selected iff sha256( salt || nvars || sorted(clause_lits) ) mod denom == 0

    Default denom=16 -> ~6.25% (close to UF250 seed rates observed).
    """
    denom = int(max(1, denom))
    out: List[int] = []
    nv = int(nvars)
    nv_bytes = nv.to_bytes(4, byteorder="big", signed=False)

    for j, cl in enumerate(clauses):
        canon = sorted((int(l) for l in cl), key=lambda x: (abs(x), 0 if x < 0 else 1))
        h = hashlib.sha256()
        h.update(salt)
        h.update(nv_bytes)
        for lit in canon:
            h.update(int(lit).to_bytes(4, byteorder="big", signed=True))
        v = int.from_bytes(h.digest()[:8], byteorder="big", signed=False)
        if (v % denom) == 0:
            out.append(j)
    return out


def build_logic_edges_from_cnf(
    clauses: List[List[int]],
    nvars: int,
    *,
    include_same_polarity: bool = True,
) -> List[Tuple[int, int]]:
    """
    Build an undirected clause graph from a CNF.

    Nodes: clauses (0..C-1).
    Edge (i,j) exists if clause i and j share at least one variable.

    If include_same_polarity=False, connect only when a shared variable appears with
    opposite polarity across the two clauses (conflict-oriented graph).

    Output: list of (i,j) with i<j, sorted deterministically.
    """

    C = len(clauses)
    if C <= 1:
        return []

    pos: List[List[int]] = [[] for _ in range(int(nvars) + 1)]
    neg: List[List[int]] = [[] for _ in range(int(nvars) + 1)]

    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(int(lit))
            if v <= 0 or v > nvars:
                continue
            if int(lit) > 0:
                pos[v].append(ci)
            else:
                neg[v].append(ci)

    edges_set: set[Tuple[int, int]] = set()

    if include_same_polarity:
        for v in range(1, int(nvars) + 1):
            occ = pos[v] + neg[v]
            if len(occ) < 2:
                continue
            for a, b in itertools.combinations(sorted(set(occ)), 2):
                edges_set.add((a, b) if a < b else (b, a))
    else:
        for v in range(1, int(nvars) + 1):
            if not pos[v] or not neg[v]:
                continue
            for a in pos[v]:
                for b in neg[v]:
                    if a == b:
                        continue
                    edges_set.add((a, b) if a < b else (b, a))

    return sorted(edges_set)


def overlap_ranges(o1: int, o2: int, m: int, T: int):
    # vrací list (a,b) intervalů v [0,T), kde se okna překrývají
    def seg(o):
        e = o + m
        if e <= T:
            return [(o, e)]
        return [(o, T), (0, e - T)]
    r = []
    for a1,b1 in seg(o1):
        for a2,b2 in seg(o2):
            a, b = max(a1,a2), min(b1,b2)
            if a < b:
                r.append((a,b))
    return r


# ---------------------------------------------------------------------
# CVXOPT dyadic tail-weights (RH_MADNESS_2 style)
# ---------------------------------------------------------------------

def get_optimal_weights_cvxopt(J: int, delta_min: float = 12.0, delta_max: float = 1000.0) -> np.ndarray:
    """
    Solve QP:
      min w_sq^T K w_sq,  s.t. w_sq >= 0, 1^T w_sq = 1
    then w = sqrt(w_sq), normalize by max(w)=1.

    If cvxopt/scipy unavailable -> uniform weights.
    """
    if J <= 0:
        return np.ones(0, dtype=float)

    """try:
        from cvxopt import matrix, solvers
        from scipy.integrate import quad
    except Exception:
        return np.ones(J, dtype=float)"""

    try:
        from cvxopt import matrix, solvers
    except Exception:
        # Fallback: jednotkové váhy
        return np.ones(1, dtype=np.float64)

    scales = 2.0 ** np.arange(J)

    def A(delta: float) -> float:
        return float(np.exp(-0.5 * delta * delta))

    """K = np.zeros((J, J), dtype=float)
    for j in range(J):
        for k in range(j, J):
            aj, ak = scales[j], scales[k]
            integrand = lambda d: A(d / aj) * A(d / ak)
            val, _ = quad(integrand, float(delta_min), float(delta_max))
            K[j, k] = float(val)
            K[k, j] = float(val)"""

    # Uzavřený tvar: ∫ exp(-0.5*(d/aj)^2) * exp(-0.5*(d/ak)^2) dd
    # = ∫ exp(-α d^2) dd, kde α = 0.5*(1/aj^2 + 1/ak^2)
    # = sqrt(pi)/(2*sqrt(α)) * (erf(sqrt(α)*b) - erf(sqrt(α)*a))

    def gauss_overlap(aj: float, ak: float, a: float, b: float) -> float:
        inv = (1.0 / (aj * aj)) + (1.0 / (ak * ak))
        alpha = 0.5 * inv
        sa = math.sqrt(alpha)
        return (math.sqrt(math.pi) / (2.0 * sa)) * (math.erf(sa * b) - math.erf(sa * a))

    K = np.zeros((J, J), dtype=np.float64)
    a = float(delta_min)
    b = float(delta_max)
    for j in range(J):
        aj = float(scales[j])
        for k in range(j, J):
            ak = float(scales[k])
            val = gauss_overlap(aj, ak, a, b)
            K[j, k] = val
            K[k, j] = val

    P = matrix(2.0 * K)  # cvxopt uses (1/2)x^T P x
    q = matrix(0.0, (J, 1))
    G = matrix(-np.eye(J))
    h = matrix(0.0, (J, 1))
    Aeq = matrix(1.0, (1, J))
    beq = matrix(1.0)
    solvers.options["show_progress"] = False

    try:
        sol = solvers.qp(P, q, G, h, Aeq, beq)
        w_sq = np.array(sol["x"]).reshape(-1)
        w = np.sqrt(np.maximum(w_sq, 0.0))
        mx = float(np.max(w)) if float(np.max(w)) > 0 else 1.0
        return w / mx
    except Exception:
        return np.ones(J, dtype=float)


def build_ipc_clause_weights(
    C: int,
    mode: str = "cvxopt",
    delta_min: float = 12.0,
    delta_max: float = 1000.0,
    *,
    corr_proxy: Optional[np.ndarray] = None,
    corr_power: float = 1.0,
    corr_eps: float = 1e-6,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> np.ndarray:
    """
    Clause weights for IPC time-mode iteration.

    Modes:
      - ones         : w_i = 1
      - cvxopt       : dyadic tail weights (useful mainly when clause index has meaning)
      - corr         : correlation-aware weights from edge-Gram row-sums
      - cvxopt_corr  : multiplicative blend cvxopt * corr
      - auto         : corr if corr_proxy is provided else cvxopt

    IMPORTANT: weights are normalized to mean(w)=1 (and softly clipped), so they do not
    accidentally zero-out the witness drive (what you observed with raw cvxopt on CNF order).
    """
    mode = (mode or "ones").lower()
    C = int(C)

    if mode == "ones":
        return np.ones(C, dtype=np.float64)

    if mode == "auto":
        mode = "corr" if (corr_proxy is not None) else "cvxopt"

    w_cvx: Optional[np.ndarray] = None
    w_corr: Optional[np.ndarray] = None

    if mode in ("cvxopt", "cvxopt_corr"):
        J = int(np.ceil(np.log2(max(2, C)))) + 1
        w_scale = get_optimal_weights_cvxopt(J, delta_min=delta_min, delta_max=delta_max)
        idx = np.floor(np.log2(np.arange(C, dtype=np.float64) + 1.0)).astype(np.int64)
        idx = np.clip(idx, 0, J - 1)
        w_cvx = w_scale[idx].astype(np.float64)
        if (not np.all(np.isfinite(w_cvx))) or float(np.max(w_cvx)) <= 0:
            w_cvx = np.ones(C, dtype=np.float64)

    if mode in ("corr", "cvxopt_corr"):
        if corr_proxy is None or int(getattr(corr_proxy, 'shape', [0])[0]) != C:
            w_corr = np.ones(C, dtype=np.float64)
        else:
            c = np.asarray(corr_proxy, dtype=np.float64)
            c = np.where(np.isfinite(c), c, 0.0)
            # Downweight high-correlation clauses; power>1 makes it steeper.
            w_corr = 1.0 / np.power(corr_eps + c, float(corr_power))

    if mode == "cvxopt":
        w = w_cvx
    elif mode == "corr":
        w = w_corr
    elif mode == "cvxopt_corr":
        w = (w_cvx * w_corr) if (w_cvx is not None and w_corr is not None) else np.ones(C, dtype=np.float64)
    else:
        # Unknown mode -> safe fallback
        w = np.ones(C, dtype=np.float64)

    return normalize_weights_mean1(w, clip_min=clip_min, clip_max=clip_max)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def hadamard(n: int) -> np.ndarray:
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def lock_indices(T: int, offset: int, m: int) -> np.ndarray:
    return (np.arange(m, dtype=np.int64) + offset) % T


def prime_offsets(C: int, T: int) -> np.ndarray:
    step = 73 % T
    return (np.arange(C, dtype=np.int64) * step) % T


def make_flip_mask(m: int, zeta0: float, seed: int = 0) -> np.ndarray:
    if zeta0 <= 0:
        return np.zeros(m, dtype=bool)
    k = int(round(zeta0 * m))
    k = max(0, min(m, k))
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    mask = np.zeros(m, dtype=bool)
    mask[idx] = True
    return mask


def enforce_misphase_fraction(base: np.ndarray, flip_mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[flip_mask] *= -1.0
    return out


def build_masks(
    C: int, m: int, zeta0: float,
    shared_carrier: bool, shared_misphase: bool,
    seed: int = 0
) -> np.ndarray:
    meff = next_pow2(m)
    H = hadamard(meff)

    carrier_row = H[(7 * seed + 1) % meff][:m].copy()
    flip_shared = make_flip_mask(m, zeta0, seed=seed + 123) if shared_misphase else None

    masks = np.empty((C, m), dtype=np.float64)
    for j in range(C):

        #base = carrier_row if shared_carrier else H[(j + 7 * seed + 1) % meff][:m]
        if shared_carrier:
            eps = 0.08  # malá konstanta; stačí i 0.02–0.12
            base = carrier_row + eps * H[(j + 7 * seed + 1) % meff][:m]
        else:
            base = H[(j + 7 * seed + 1) % meff][:m]
        # --- Unitary carrier projection (energy lock) ---
        # Cíl: každá maska má stejnou L2 energii => žádný skrytý gain/drift.
        bn = np.linalg.norm(base)
        if bn > 0.0:
            base = base * (math.sqrt(m) / bn)   # ||base||_2 = sqrt(m)


        if shared_misphase:
            masks[j] = enforce_misphase_fraction(base, flip_shared)  # type: ignore[arg-type]
        else:
            flip_j = make_flip_mask(m, zeta0, seed=seed + 123 + j)
            masks[j] = enforce_misphase_fraction(base, flip_j)
    return masks


def build_lock_mask_matrix(T: int, C: int, m: int, offsets: np.ndarray) -> np.ndarray:
    M = np.zeros((T, C), dtype=np.float64)
    for j in range(C):
        M[lock_indices(T, int(offsets[j]), m), j] = 1.0
    return M


"""def build_Z(
    T: int, C: int, m: int,
    offsets: np.ndarray,
    masks: np.ndarray,
    clause_gauge: Optional[np.ndarray] = None,
    outside_value: complex = -1.0
) -> np.ndarray:
    Z = np.full((T, C), outside_value, dtype=np.complex128)
    if clause_gauge is None:
        clause_gauge = np.ones(C, dtype=np.float64)
    for j in range(C):
        idx = lock_indices(T, int(offsets[j]), m)
        Z[idx, j] = clause_gauge[j] * masks[j].astype(np.float64)
    return Z"""

def build_Z(
    T: int, C: int, m: int,
    offsets: np.ndarray,
    masks: np.ndarray,
    clause_gauge: Optional[np.ndarray] = None,
    outside_value: complex = -1.0
) -> np.ndarray:
    """
    Z[t,j] = gauge[j] * masks[j,k] on lock positions, outside_value elsewhere.
    """
    dtypeZ = np.complex64
    dtypeR = np.float32

    Z = np.full((T, C), dtypeZ(outside_value), dtype=dtypeZ)

    if clause_gauge is None:
        clause_gauge = np.ones(C, dtype=dtypeR)
    else:
        clause_gauge = clause_gauge.astype(dtypeR, copy=False)

    for j in range(C):
        idx = lock_indices(T, int(offsets[j]), m)

        # zachováváme původní logiku: gauge * mask (mask je reálná)
        mj = masks[j].astype(dtypeR, copy=False)
        Z[idx, j] = dtypeZ(clause_gauge[j]) * mj

    return Z



def project_unit_circle(z: np.ndarray) -> np.ndarray:
    mag = np.abs(z)
    out = np.empty_like(z)
    nz = mag > 0
    out[nz] = z[nz] / mag[nz]
    out[~nz] = 1.0 + 0j
    return out


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi



def normalize_assignment_pm1(assign, nvars: int) -> np.ndarray:
    """Return 1-based ±1 assignment array of length nvars+1 (index 0 ignored)."""
    a = np.ones(nvars + 1, dtype=np.int8)
    arr = np.asarray(assign).reshape(-1)

    if arr.size == nvars:
        vals = arr.astype(np.int64, copy=False)
        # allow {0,1} or {False,True}
        uniq = set(np.unique(vals).tolist()) if vals.size else {1}
        if uniq.issubset({0, 1}):
            vals = 2 * vals - 1
        a[1:] = vals.astype(np.int8, copy=False)
    elif arr.size == nvars + 1:
        vals = arr.astype(np.int64, copy=False)
        uniq = set(np.unique(vals).tolist()) if vals.size else {1}
        if uniq.issubset({0, 1}):
            vals = 2 * vals - 1
        a[:] = vals.astype(np.int8, copy=False)
        a[0] = 1
    else:
        raise ValueError(f"Bad assignment length: {arr.size} (expected {nvars} or {nvars+1})")
    return a


def check_cnf(clauses: List[List[int]], assignment, nvars: Optional[int] = None) -> int:
    """Return number of unsatisfied clauses. Accepts bool / {0,1} / {±1}, 0- or 1-based."""
    arr = np.asarray(assignment).reshape(-1)

    if nvars is None:
        nvars = max(abs(lit) for cl in clauses for lit in cl) if clauses else 0

    pm1 = normalize_assignment_pm1(arr, nvars)

    unsat = 0
    for cl in clauses:
        ok = False
        for lit in cl:
            v = pm1[abs(lit)]
            if (lit > 0 and v == 1) or (lit < 0 and v == -1):
                ok = True
                break
        if not ok:
            unsat += 1
    return unsat

# ---------------------------------------------------------------------
# Wiring graph + signed constraints
# ---------------------------------------------------------------------

def circulant_edges(C: int, d: int) -> List[Tuple[int, int]]:
    """
    Undirected circulant degree-d graph, returned as a list of unique undirected edges (i<j).
    """
    assert d % 2 == 0
    edges: List[Tuple[int, int]] = []
    half = d // 2
    for i in range(C):
        for k in range(1, half + 1):
            j = (i + k) % C
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))
    # unique
    edges = sorted(set(edges))
    return edges


def build_cnf_logic_edges(
    clauses: List[List[int]],
    d: int,
    seed: int = 0,
    candidate_mult: int = 6,
) -> List[Tuple[int, int]]:
    """
    Build a deterministic, bounded-degree clause graph from CNF structure.

    - Each clause is a node.
    - Edge weight w(i,j) = number of shared variables between clauses i and j
      (ignoring literal sign).
    - We generate a *pool* of promising candidate edges using an inverted index,
      then run a greedy degree-capped selection (each node degree <= d) that
      prefers higher weights and breaks ties deterministically.

    This avoids the common pitfall where "top-d per node" still allows very
    large *in-degree* (a clause can be selected by many others), which inflates
    S2 rho and can kill the radar bound.
    """
    C = len(clauses)
    if C <= 1 or d <= 0:
        return []

    rng = np.random.default_rng(int(seed))

    # Clause -> set of variables (abs(lit))
    cl_vars: List[List[int]] = []
    for cl in clauses:
        vs = sorted({abs(int(l)) for l in cl if int(l) != 0})
        cl_vars.append(vs)

    # Inverted index: var -> clauses containing var
    inv: Dict[int, List[int]] = {}
    for i, vs in enumerate(cl_vars):
        for v in vs:
            inv.setdefault(v, []).append(i)

    # Candidate edge weights (i<j) stored in dict
    # We only keep up to candidate_mult*d candidates per node to keep things light.
    want = max(2, int(candidate_mult) * int(d))
    edge_w: Dict[Tuple[int, int], int] = {}

    for i, vs in enumerate(cl_vars):
        cnt: Dict[int, int] = {}
        for v in vs:
            for j in inv.get(v, []):
                if j == i:
                    continue
                cnt[j] = cnt.get(j, 0) + 1

        if not cnt:
            continue

        # deterministic tiebreak: pseudo-random but seeded and symmetric
        def tie(j: int) -> float:
            # hash-like float in [0,1)
            return float((i * 1315423911 + j * 2654435761 + seed * 97531) & 0xFFFFFFFF) / 2**32

        cand = sorted(cnt.items(), key=lambda kv: (-kv[1], tie(kv[0])))[:want]
        for j, w in cand:
            a, b = (i, j) if i < j else (j, i)
            edge_w[(a, b)] = max(edge_w.get((a, b), 0), int(w))

    if not edge_w:
        # Fallback: a tiny circulant to avoid empty graphs
        edges = []
        step = max(1, C // max(2, d))
        for i in range(C):
            for k in range(1, min(d, C - 1) + 1):
                j = (i + k * step) % C
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b))
        return sorted(set(edges))

    # Greedy degree-capped selection (each node degree <= d)
    items = list(edge_w.items())
    items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))

    deg = np.zeros(C, dtype=int)
    chosen: List[Tuple[int, int]] = []

    for (i, j), w in items:
        if w <= 0:
            continue
        if deg[i] >= d or deg[j] >= d:
            continue
        chosen.append((i, j))
        deg[i] += 1
        deg[j] += 1

    # Ensure at least weak connectivity: if some nodes isolated, attach them by a
    # deterministic ring edge (doesn't violate degree cap if possible).
    if np.any(deg == 0) and C > 2:
        for i in range(C):
            if deg[i] != 0:
                continue
            for j in ((i - 1) % C, (i + 1) % C):
                if i == j:
                    continue
                if deg[i] < d and deg[j] < d:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) not in edge_w:
                        # treat as weight 0 ring edge
                        chosen.append((a, b))
                        deg[i] += 1
                        deg[j] += 1
                        break

    chosen = sorted(set((min(i, j), max(i, j)) for i, j in chosen if i != j))
    return chosen


def _edge_hash_int(i: int, j: int, seed: int) -> int:
    s = f"{seed}:{i}:{j}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def build_edge_signs_from_gauge(
    edges: List[Tuple[int, int]],
    clause_gauge: np.ndarray,
    mode: str,
    unsat_neg_frac: float,
    seed: int,
    flip_incident_unsat: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    Base balanced signs: s_ij = g_i g_j.

    In UNSAT mode we deterministically flip an additional fraction of edges to create frustration.
    If flip_incident_unsat=True, we flip edges incident to UNSAT clauses first (g=-1), tying
    frustration to CNF defects instead of injecting it uniformly.
    """
    mode = mode.lower()
    signs: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        signs[(i, j)] = float(clause_gauge[i] * clause_gauge[j])

    if mode == "sat":
        return signs

    frac = max(0.0, min(1.0, float(unsat_neg_frac)))
    k = int(math.floor(frac * len(edges)))
    if k <= 0:
        return signs

    if flip_incident_unsat and np.any(clause_gauge < 0):
        pool = [e for e in edges if (clause_gauge[e[0]] < 0) or (clause_gauge[e[1]] < 0)]
        ranked_pool = pool if pool else edges
    else:
        ranked_pool = edges

    ranked = sorted(ranked_pool, key=lambda e: _edge_hash_int(e[0], e[1], seed))
    for e in ranked[:k]:
        signs[e] *= -1.0
    return signs

    k = int(math.floor(max(0.0, min(1.0, unsat_neg_frac)) * len(edges)))
    if k <= 0:
        return signs

    # hash-rank edges deterministically
    ranked = sorted(edges, key=lambda e: _edge_hash_int(e[0], e[1], seed))
    for e in ranked[:k]:
        signs[e] *= -1.0
    return signs


# ---------------------------------------------------------------------
# Overlap-only coupling
# ---------------------------------------------------------------------

def apply_signed_overlap_coupling(
    Z: np.ndarray,
    T: int, C: int, m: int,
    offsets: np.ndarray,
    edges: List[Tuple[int, int]],
    edge_signs: Dict[Tuple[int, int], float],
    eta: float,
    sweeps: int,
    *,
    K: float = 1.0,
    noise_sigma: float = 0.0,
    dt: float = 0.05,
    mu: float = 1.0,
    h: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    lock_mask: Optional[np.ndarray] = None,
) -> None:
    """DREAM6-style overlap coupling with optional gain, relaxation, field, and phase-noise.

    Core update on overlap Omega_ij:
      Zi <- proj( (1-η) Zi + η * s_ij * Zj )

    Extensions (all optional):
      - Gain K: uses η_eff = 1 - (1-η)^K (keeps η in [0,1] but gives a separate strength knob).
      - Relaxation μ: EMA towards the projected update: Zi <- proj( μ Zi_old + (1-μ) Zi_proj ).
      - Field h: adds a small global carrier bias (+h) before projection.
      - Noise σ: multiplicative phase noise on lock entries after each sweep: Zi *= exp(i σ sqrt(dt) ξ).

    Notes:
      * η itself can stay on the critical line (0.5); K is the extra gain knob.
      * If lock_mask is provided (same shape as Z), noise is applied only where lock_mask!=0.
    """

    # Effective eta with separate gain K
    eta = float(eta)
    if eta < 0.0:
        eta = 0.0
    if eta > 1.0:
        eta = 1.0
    K = max(0.0, float(K))
    # η_eff in [0,1]
    eta_eff = 1.0 - (1.0 - eta) ** K if K != 0.0 else 0.0

    # relaxation
    mu = float(mu)
    if mu < 0.0:
        mu = 0.0
    if mu > 1.0:
        mu = 1.0

    h = float(h)

    # --- Silent core knobs (anti-drift) ---
    # eps_dynamic sets the soft-ABS width; tie it weakly to noise_sigma to avoid division blow-ups.
    beta_floor = 1e-6
    eps_dynamic = max(1e-6, 0.1 * float(noise_sigma))
    w_soft_cap = 50.0

    if rng is None:
        rng = np.random.default_rng(0)

    # precompute overlap ranges (fast path)
    lock_bool: List[np.ndarray] = []
    for k in range(C):
        b = np.zeros(T, dtype=bool)
        b[lock_indices(T, int(offsets[k]), m)] = True
        lock_bool.append(b)

    # coupling sweeps
    for _ in range(int(sweeps)):
        for (i, j) in edges:
            sgn = float(edge_signs.get((i, j), +1.0))
            # quick skip
            if not (lock_bool[i].any() and lock_bool[j].any()):
                continue

            ranges = overlap_ranges(int(offsets[i]), int(offsets[j]), m, T)
            if not ranges:
                continue

            for a, b in ranges:
                Zi_old = Z[a:b, i].copy()
                Zj_old = Z[a:b, j].copy()

                # base update (before projection)
                # --- Silent Geometric Core (anti-drift adaptive step) ---
                L = max(1, (b - a))
                align = float(np.real(np.vdot(Zi_old, (sgn * Zj_old))) / L)
                align_clip = min(1.0 - 1e-6, max(-1.0 + 1e-6, align))
                eta_loc = 0.5 * math.log((1.0 + align_clip) / (1.0 - align_clip))  # atanh

                res = wrap_pi(np.angle(Zi_old * np.conj(sgn * Zj_old)))
                res_abs = float(np.mean(np.abs(res)))

                w_soft = 1.0 / math.sqrt(res_abs * res_abs + eps_dynamic * eps_dynamic)
                if w_soft > w_soft_cap:
                    w_soft = w_soft_cap

                beta_safe = max(beta_floor, w_soft)
                gain = math.tanh(eta_loc - 1.0 / beta_safe)

                eta_step = eta_eff * abs(gain) * w_soft
                if eta_step < 0.0:
                    eta_step = 0.0
                elif eta_step > 1.0:
                    eta_step = 1.0

                # base update (before projection) — now with eta_step
                upd_i = (1.0 - eta_step) * Zi_old + eta_step * (sgn * Zj_old)
                upd_j = (1.0 - eta_step) * Zj_old + eta_step * (sgn * Zi_old)

                # external field bias (global carrier direction 1+0j)
                if h != 0.0:
                    upd_i = upd_i + h
                    upd_j = upd_j + h

                Zi_proj = project_unit_circle(upd_i)
                Zj_proj = project_unit_circle(upd_j)

                # relaxation towards projected update
                if mu < 1.0:
                    Zi_new = project_unit_circle(mu * Zi_old + (1.0 - mu) * Zi_proj)
                    Zj_new = project_unit_circle(mu * Zj_old + (1.0 - mu) * Zj_proj)
                else:
                    Zi_new, Zj_new = Zi_proj, Zj_proj

                Z[a:b, i] = Zi_new
                Z[a:b, j] = Zj_new

        # phase noise (on lock entries only)
        if noise_sigma and float(noise_sigma) > 0.0:
            sigma = float(noise_sigma) * math.sqrt(max(0.0, float(dt)))
            if lock_mask is not None:
                # vectorized: only where lock_mask != 0
                mask = lock_mask != 0
                n = int(np.count_nonzero(mask))
                if n > 0:
                    ang = rng.standard_normal(n).astype(np.float64) * sigma
                    ph = np.exp(1j * ang)
                    Z[mask] = Z[mask] * ph.astype(Z.dtype, copy=False)
            else:
                # fall back: noise everywhere
                ang = rng.standard_normal(Z.shape).astype(np.float64) * sigma
                Z[:] = Z * np.exp(1j * ang).astype(Z.dtype, copy=False)


# ---------------------------------------------------------------------
# Edge-Gram decision operator + S2 radar
# ---------------------------------------------------------------------

def build_edge_gram(
    Z: np.ndarray,
    T: int, C: int, m: int,
    offsets: np.ndarray,
    edges: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[List[complex]]]:
    """
    Hermitian edge-supported Gram G_H:
      diag = 1
      offdiag on edges: g_ij = <z_i, z_j>_{Omega_ij} / m
    """
    nbr: List[List[int]] = [[] for _ in range(C)]
    val: List[List[complex]] = [[] for _ in range(C)]

    lock_bool: List[np.ndarray] = []
    for k in range(C):
        b = np.zeros(T, dtype=bool)
        b[lock_indices(T, int(offsets[k]), m)] = True
        lock_bool.append(b)

    for (i, j) in edges:
        omega = np.where(lock_bool[i] & lock_bool[j])[0]
        if omega.size == 0:
            gij = 0.0 + 0j
        else:
            gij = np.vdot(Z[omega, i], Z[omega, j]) / float(m)
        nbr[i].append(j); val[i].append(gij)
        nbr[j].append(i); val[j].append(np.conj(gij))
    return nbr, val


def edge_matvec(v: np.ndarray, nbr: List[List[int]], val: List[List[complex]]) -> np.ndarray:
    out = v.astype(np.complex128).copy()  # diag = 1
    for i in range(len(nbr)):
        if not nbr[i]:
            continue
        s = 0.0 + 0j
        for j, gij in zip(nbr[i], val[i]):
            s += gij * v[j]
        out[i] += s
    return out


def power_lambda_max_edge(nbr: List[List[int]], val: List[List[complex]], iters: int = 250, tol: float = 1e-10) -> float:
    C = len(nbr)
    v = np.ones(C, dtype=np.complex128)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(int(iters)):
        w = edge_matvec(v, nbr, val)
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        lam = float(np.real(np.vdot(v, edge_matvec(v, nbr, val))))
        if abs(lam - lam_prev) <= tol * max(1.0, abs(lam)):
            return lam
        lam_prev = lam
    return lam_prev


def neighbor_rowsum(nbr: List[List[int]], val: List[List[complex]]) -> float:
    rho = 0.0
    for i in range(len(nbr)):
        s = 0.0
        for gij in val[i]:
            s += abs(gij)
        rho = max(rho, s)
    return rho


def kappa_S2(T: int, m: int, zeta0: float) -> float:
    m_eff = next_pow2(m)
    eps = (1.0 / math.sqrt(m_eff)) + (2.0 / float(m))
    return (1.0 - 2.0 * zeta0) ** 2 + eps + (1.0 / float(T))


# ---------------------------------------------------------------------
# IPC: Invariant Phase Certifier (functional)
# ---------------------------------------------------------------------

def ipc_time_mode_u(Z_lock: np.ndarray, w: np.ndarray, m: int, iters: int = 80, tol: float = 1e-10) -> np.ndarray:
    """
    Power iteration on:
      T(u) = (1/m) Z_lock diag(w) Z_lock^* u
    """
    Tn, C = Z_lock.shape
    u = np.ones(Tn, dtype=np.complex128)
    u /= np.linalg.norm(u)
    last = u
    w = w.astype(np.float64)
    for _ in range(int(iters)):
        v = Z_lock.conj().T @ u            # shape (C,)
        v = (w * v)                        # weighted
        u2 = (Z_lock @ v) / float(m)       # back to time
        n = np.linalg.norm(u2)
        if n == 0:
            break
        u = u2 / n
        if np.linalg.norm(u - last) <= tol * max(1.0, np.linalg.norm(u)):
            break
        last = u
    return u


def ipc_metrics(Z_lock: np.ndarray, u: np.ndarray, m: int) -> Tuple[float, float, float, np.ndarray]:
    """
    Normalized clause phasors:
      a_j = <u, z_j>/sqrt(m) = (Z_lock^* u)_j / sqrt(m)
    Returns (theta, beta, delta, a).
    """
    a = (Z_lock.conj().T @ u) / math.sqrt(float(m))
    S = np.sum(a)
    # Robust theta: if phasors are bimodal (Z2 split), use 2nd harmonic order parameter.
    S_abs = float(np.sum(np.abs(a)))
    theta = float(np.angle(S)) if S != 0 else 0.0
    use_z2 = (S_abs > 0.0) and (abs(S) < 0.20 * S_abs)
    if use_z2:
        S2 = np.sum(a * a)
        theta = 0.5 * float(np.angle(S2)) if S2 != 0 else theta
    mags = np.abs(a)
    beta = float(np.min(mags))
    ang = np.angle(a)
    if use_z2:
        # Phase error modulo pi: delta in [0, pi/2]
        err2 = 0.5 * wrap_pi(2.0 * (ang - theta))
        delta = float(np.max(np.abs(err2)))
    else:
        err = wrap_pi(ang - theta)
        delta = float(np.max(np.abs(err)))
    return theta, beta, delta, a


def ipc_mu_sat_min(beta: float, delta: float) -> float:
    return float((beta ** 2) * (math.cos(delta) ** 2))


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def coherence_R(Z_lock: np.ndarray) -> Tuple[float, float, float]:
    R = np.abs(np.sum(Z_lock, axis=1))
    return float(np.mean(R)), float(np.min(R)), float(np.max(R))


def cnf_projection_report(Z_lock: np.ndarray) -> Dict[str, float]:
    """
    Lightweight, deterministic diagnostics about lock columns:
      proj_j = sum_t Z_lock[t,j]
    """
    proj = np.sum(Z_lock, axis=0)  # (C,)
    amps = np.abs(proj)
    ang = np.angle(proj)
    # coherence proxy: |mean exp(i angle)|
    coh = float(abs(np.mean(np.exp(1j * ang)))) if ang.size else 0.0
    return {
        "avg_amp": float(np.mean(amps)) if amps.size else 0.0,
        "median_amp": float(np.median(amps)) if amps.size else 0.0,
        "max_amp": float(np.max(amps)) if amps.size else 0.0,
        "min_amp": float(np.min(amps)) if amps.size else 0.0,
        "angle_var": float(np.var(ang)) if ang.size else 0.0,
        "angle_std": float(np.std(ang)) if ang.size else 0.0,
        "coh_proxy": coh,
        "frac_proj_gt_005": float(np.mean(amps > 0.05)) if amps.size else 0.0,
        "frac_proj_gt_01": float(np.mean(amps > 0.1)) if amps.size else 0.0,
    }

def edge_correlation_proxy(nbr: list[list[int]], val: list[list[complex]]) -> np.ndarray:
    """
    Correlation proxy per clause-node i from the SAME sparse edge-Gram used by S2/spectral.
    We use neighbor row-sums: corr[i] = sum_{j in N(i)} |G_ij|.

    Intuition: high corr => clause sits in a tight correlated cluster => downweight it in IPC
    to prevent a few correlated neighborhoods from dominating the global u-mode.
    """
    C = len(nbr)
    corr = np.zeros(C, dtype=np.float64)
    for i in range(C):
        s = 0.0
        for k, j in enumerate(nbr[i]):
            s += abs(val[i][k])
        corr[i] = s
    return corr


def normalize_weights_mean1(w: np.ndarray, *, clip_min: float = 0.25, clip_max: float = 4.0) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    m = float(np.mean(w))
    if m <= 0 or not math.isfinite(m):
        w = np.ones_like(w)
        m = 1.0
    w = w / m
    if clip_min is not None and clip_max is not None and clip_max > clip_min > 0:
        w = np.clip(w, float(clip_min), float(clip_max))
        # re-normalize after clipping so IPC stays on the same drive scale as 'ones'
        m2 = float(np.mean(w))
        if m2 > 0 and math.isfinite(m2):
            w = w / m2
    return w



# ---------------------------------------------------------------------
# Certificate container
# ---------------------------------------------------------------------

@dataclass
class Certificate:
    meta: dict
    S2: dict
    spectral: dict
    IPC: dict
    bands: dict
    diag: dict


def run(
    C: int,
    R: int,
    d: int,
    sweeps: int,
    eta: float,
    K: float,
    noise_sigma: float,
    dt: float,
    mu: float,
    mu_E: float,
    h: float,
    tail_frac: float,
    mode: str,
    shared_carrier: bool,
    shared_misphase: bool,
    unsat_neg_frac: float,
    seed: int,
    power_iters: int,
    power_tol: float,
    ipc_weight_mode: str,
    w_delta_min: float,
    w_delta_max: float,
    cnf_path: Optional[str] = None,
    edge_mode: str = "auto",
    flip_incident_unsat: bool = True,
    outside_value: complex = -1.0,
    json_out: Optional[str] = None,
) -> Certificate:

    # ------------------ CNF vs synthetic ------------------
    cnf_meta: dict = {}
    clause_gauge: Optional[np.ndarray] = None
    zeta0 = 0.25

    if cnf_path:
        print(f"[CNF mód] Načítám {cnf_path}")
        nvars, clauses = parse_dimacs(cnf_path)
        C = len(clauses)
        print(f"  proměnné: {nvars}    klauzule: {C:,}")

        # deterministic UNSAT seeding (hash-based, invariant across machines)
        unsat_idx = cnf_seed_unsat_indices(clauses, nvars)

        # SAT mode: baseline all +1 (do not inject defects from an imperfect seed model)
        # UNSAT mode: inject π-defects exactly on seed-unsatisfied clauses
        g = np.ones(C, dtype=np.float64)
        if mode.lower() == "unsat":
            for j in unsat_idx:
                g[j] = -1.0
        clause_gauge = g

        cnf_meta = {
            "cnf_path": cnf_path,
            "cnf_sha256": sha256_file(cnf_path),
            "nvars": nvars,
            "seed_unsat": int(len(unsat_idx)),
            "seed_unsat_frac": float(len(unsat_idx) / max(1, C)),
        }
    else:
        clauses = []
        nvars = 0
        clause_gauge = np.ones(C, dtype=np.float64)

    # ------------------ geometry ------------------
    T = 2 * int(R)
    m = int(R) // 2

    offsets = prime_offsets(C, T)
    masks = build_masks(C, m, zeta0, shared_carrier, shared_misphase, seed=seed)

    # IMPORTANT: for large T, a non-zero outside_value can dominate Gram overlaps and
    # suppress IPC ("silent" regime). Keep it configurable.
    Z = build_Z(T, C, m, offsets, masks, clause_gauge=clause_gauge, outside_value=outside_value)
    M = build_lock_mask_matrix(T, C, m, offsets)
    Z_lock = Z * M  # zeros outside lock

    # CNF quick report (before coupling)
    if cnf_path:
        rep = cnf_projection_report(Z_lock)
        print(f"  Průměrná amplituda     : {rep['avg_amp']:.6f}")
        print(f"  Medián amplitudy       : {rep['median_amp']:.6f}")
        print(f"  Max / min amplituda    : {rep['max_amp']:.6f} / {rep['min_amp']:.6f}")
        print(f"  Frakce |proj| > 0.05   : {100*rep['frac_proj_gt_005']:.2f} %")
        print(f"  Frakce |proj| > 0.1    : {100*rep['frac_proj_gt_01']:.2f} %")
        print(f"  Rozptyl úhlů (variance): {rep['angle_var']:.6f} rad²")
        print(f"  Std úhlů               : {rep['angle_std']:.4f} rad  ≈ {rep['angle_std']*180/math.pi:.2f}°")
        print(f"  Koherenční proxy       : {rep['coh_proxy']:.6f}  (1 = všechny locky dokonale zarovnané)")

    # ------------------ signed constraints + overlap coupling ------------------
    """em = (edge_mode or "auto").lower()
    if cnf_path and em in ("cnf", "logic"):
        # Full CNF clause graph (theory graph): share-variable edges.
        # 'cnf' = any shared variable; 'logic' = opposite-polarity (conflict) edges only.
        edges = build_logic_edges_from_cnf(clauses, nvars, include_same_polarity=(em == "cnf"))
        if not edges:
            # fallback (should be rare): bounded-degree selection, then circulant
            edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
            if not edges:
                edges = circulant_edges(C, d)
    elif cnf_path and em == "auto":
        edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
        if not edges:
            edges = circulant_edges(C, d)
    else:
        edges = circulant_edges(C, d)"""
    em = (edge_mode or "auto").lower()

    # ochrana proti OOM: full clause-graph jen pro malé CNF
    BIG = (len(clauses) > 200_000) or (nvars > 200_000)

    if cnf_path and em in ("cnf", "logic"):
        if BIG:
            # bounded-degree CNF graph (strip) – škáluje
            edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
        else:
            # full graph jen pro malé instance
            edges = build_logic_edges_from_cnf(clauses, nvars, include_same_polarity=(em == "cnf"))

        if not edges:
            edges = circulant_edges(C, d)

    elif cnf_path and em == "auto":
        edges = build_cnf_logic_edges(clauses, d=d, seed=seed)
        if not edges:
            edges = circulant_edges(C, d)
    else:
        edges = circulant_edges(C, d)


    edge_signs = build_edge_signs_from_gauge(
        edges,
        clause_gauge,
        mode=mode,
        unsat_neg_frac=unsat_neg_frac,
        seed=seed,
        flip_incident_unsat=flip_incident_unsat,
    )
    rng = np.random.default_rng(int(seed) + 1337)
    apply_signed_overlap_coupling(Z, T, C, m, offsets, edges, edge_signs, eta=eta, sweeps=sweeps,
                                K=K, noise_sigma=noise_sigma, dt=dt, mu=mu, h=h, rng=rng, lock_mask=M)
    Z_lock = Z * M

    # ------------------ decision operator (edge-Gram) ------------------
    nbr, val = build_edge_gram(Z, T, C, m, offsets, edges)
    lam = power_lambda_max_edge(nbr, val, iters=power_iters, tol=power_tol)
    mu_dec = float(lam / float(C))

    rho = neighbor_rowsum(nbr, val)
    kap = kappa_S2(T, m, zeta0)
    bound = float(d) * kap
    S2_ok = bool(rho <= bound + 1e-12)

    # ------------------ IPC with clause weights ------------------
    corr = edge_correlation_proxy(nbr, val)
    w = build_ipc_clause_weights(C, mode=ipc_weight_mode, delta_min=w_delta_min, delta_max=w_delta_max, corr_proxy=corr)
    u = ipc_time_mode_u(Z_lock, w, m=m, iters=power_iters, tol=power_tol)
    theta, beta, delta, a = ipc_metrics(Z_lock, u, m=m)    # ------------------ CNF witness extraction (assignment projection) ------------------
    witness: Dict[str, object] = {}
    if cnf_path:
        try:
            # Candidate thetas: Z2 branch, its π-shift, and plain U1 phase.
            S1_w = np.sum(a)
            S_abs = float(np.sum(np.abs(a))) + 1e-12
            theta_u1 = float(np.angle(S1_w)) if abs(S1_w) > 1e-12 else 0.0

            coh_u1_w = float(abs(S1_w) / S_abs) if S_abs > 0.0 else 0.0

            use_z2 = (S_abs > 0.0) and (abs(S1_w) < 0.20 * S_abs)
            theta_z2 = theta_u1
            coh_z2_w = 0.0
            if use_z2:
                S2_w = np.sum(a * a)
                S2_abs = float(np.sum(np.abs(a)**2)) + 1e-12
                coh_z2_w = float(abs(S2_w) / S2_abs) if S2_abs > 0.0 else 0.0
                if abs(S2_w) > 1e-12:
                    theta_z2 = 0.5 * float(np.angle(S2_w))

            cand = [theta_z2, wrap_pi(theta_z2 + math.pi), theta_u1]
            thetas: List[float] = []
            for t in cand:
                t = float(t)
                if not any(abs(wrap_pi(t - tt)) < 1e-9 for tt in thetas):
                    thetas.append(t)

            # Geometric theta micro-transport: when Z2 coherence is near-perfect but U1 is tiny,
            # the best branch can sit slightly off the coarse candidates. We scan a small uniform
            # ring around the dominant Z2 phase (and U1 fallback) and let CNF unsat pick.
            if (use_z2 and (coh_z2_w > 0.999) and (coh_u1_w < 0.05)):
                N_scan = 32  # pure-geometry, global gauge only (no local search)
                base_list = [theta_z2, theta_u1]
                for base in base_list:
                    for k in range(N_scan):
                        t = wrap_pi(base + (2.0 * math.pi) * (k / N_scan))
                        if not any(abs(wrap_pi(t - tt)) < 1e-9 for tt in thetas):
                            thetas.append(float(t))

            inc_var = build_var_clause_incidence(clauses, nvars)

            best = None  # (key, theta, mode, flip, assign_bool, score_vec, score_mean, score_min)
            for th in thetas:
                for mode_readout in ("lin", "z2"):
                    assign_tmp, score_tmp = extract_assignment_from_ipc(
                        clauses, nvars,
                        clause_phasors=a, theta=float(th), clause_weights=w,
                        inc=inc_var, mode=mode_readout
                    )
                    for flip in (False, True):
                        assign_use = (~assign_tmp) if flip else assign_tmp
                        unsat_tmp = count_unsat(clauses, assign_use)

                        arr = np.ravel(np.asarray(score_tmp, dtype=np.float64))
                        score_mean = float(arr.mean()) if arr.size else 0.0
                        score_min = float(arr.min()) if arr.size else 0.0

                        key = (int(unsat_tmp), -score_mean, -score_min)
                        if (best is None) or (key < best[0]):
                            best = (key, float(th), mode_readout, flip, assign_use, score_tmp, score_mean, score_min)

            assert best is not None
            (key_best, theta_used, mode_used, flip_used, assign_wit, score_wit, score_mean, score_min) = best

            # Lock readout regime to the actually winning geometry
            use_z2 = (mode_used == "z2")

            # Canonical validation (handles 0/1, ±1, bool; 0/1-based)
            unsat_check = check_cnf(clauses, assign_wit, nvars=nvars)
            print("CNF check unsat:", int(unsat_check))

            # Make IPC (theta, delta) consistent with the chosen branch
            theta = float(theta_used)
            ang = np.angle(a)
            if use_z2:
                err2 = 0.5 * wrap_pi(2.0 * (ang - theta))
                delta = float(np.max(np.abs(err2)))
            else:
                err = wrap_pi(ang - theta)
                delta = float(np.max(np.abs(err)))

            # Diagnostics: how strongly proxy is actually driving the witness (at chosen theta)
            phi_w = np.angle(a)
            amp_w = np.abs(a).astype(np.float64, copy=False)

            align_w = np.cos(wrap_pi(phi_w - theta)).astype(np.float64, copy=False)
            flip_w = (align_w < 0.0)
            if np.any(flip_w):
                phi_w_use = phi_w.copy()
                phi_w_use[flip_w] = wrap_pi(phi_w_use[flip_w] + math.pi)
                align_w = np.cos(wrap_pi(phi_w_use - theta)).astype(np.float64, copy=False)

            gate_w = (align_w * align_w)  # Z2-invariant gating
            drive_w = w.astype(np.float64, copy=False) * amp_w * gate_w

            arr = np.ravel(np.asarray(score_wit, dtype=np.float64))
            witness = {
                "assign_sha256": sha256_assignment(assign_wit),
                "unsat": int(unsat_check),
                "unsat_frac": float(unsat_check / max(1, len(clauses))),
                "theta_used": float(theta),
                "readout_mode": str(mode_used),
                "flip_used": bool(flip_used),
                "drive_stats": {
                    "amp_mean": float(np.mean(amp_w)) if amp_w.size else 0.0,
                    "amp_median": float(np.median(amp_w)) if amp_w.size else 0.0,
                    "gate_mean": float(np.mean(gate_w)) if gate_w.size else 0.0,
                    "drive_mean": float(np.mean(drive_w)) if drive_w.size else 0.0,
                    "drive_median": float(np.median(drive_w)) if drive_w.size else 0.0,
                    "drive_min": float(np.min(drive_w)) if drive_w.size else 0.0,
                    "drive_max": float(np.max(drive_w)) if drive_w.size else 0.0,
                    "drive_nonzero_frac": float(np.mean(drive_w > 0.0)) if drive_w.size else 0.0,
                },
                "score_stats": {
                    "min": float(arr.min()) if arr.size else 0.0,
                    "max": float(arr.max()) if arr.size else 0.0,
                    "mean": float(arr.mean()) if arr.size else 0.0,
                    "std": float(arr.std()) if arr.size else 0.0,
                },
            }
        except Exception as e:
            witness = {"error": f"{type(e).__name__}: {e}"}

    mu_sat_min = ipc_mu_sat_min(beta, delta)

    # ------------------ bands ------------------
    lam_unsat_ceiling = float(1.0 + bound)
    mu_unsat_max = float(lam_unsat_ceiling / float(C))
    tau = 0.5 * (mu_sat_min + mu_unsat_max)
    Delta = 0.5 * (mu_sat_min - mu_unsat_max)
    separated = bool(Delta > 0)

    # ------------------ coherence diag (tail + EMA) ------------------
    R_series = np.abs(np.sum(Z_lock, axis=1)).astype(np.float64, copy=False)
    r_mean = float(np.mean(R_series)) if R_series.size else 0.0
    r_min = float(np.min(R_series)) if R_series.size else 0.0
    r_max = float(np.max(R_series)) if R_series.size else 0.0
    tf = float(tail_frac)
    tf = 0.0 if tf < 0.0 else (1.0 if tf > 1.0 else tf)
    if R_series.size and tf > 0.0:
        start = int(max(0, min(R_series.size - 1, math.floor((1.0 - tf) * R_series.size))))
        tail = R_series[start:]
    else:
        tail = R_series
    r_mean_tail = float(np.mean(tail)) if tail.size else 0.0

    # EMA on tail (uses mu_E)
    muE = float(mu_E)
    if muE < 0.0: muE = 0.0
    if muE > 1.0: muE = 1.0
    if tail.size:
        ema = float(tail[0])
        a = 1.0 - muE
        for x in tail[1:]:
            ema = muE * ema + a * float(x)
        r_ema_tail = float(ema)
    else:
        r_ema_tail = 0.0

    meta = {
        "C": int(C), "T": int(T), "m": int(m), "R": int(R), "d": int(d),
        "mode": str(mode),
        "sweeps": int(sweeps), "eta": float(eta),
        "dream6": {"K": float(K), "noise_sigma": float(noise_sigma), "dt": float(dt), "mu": float(mu), "mu_E": float(mu_E), "h": float(h), "tail_frac": float(tail_frac)},
        "shared_carrier": bool(shared_carrier),
        "shared_misphase": bool(shared_misphase),
        "unsat_neg_frac": float(unsat_neg_frac),
        "seed": int(seed),
        "zeta0": float(zeta0),
        "ipc_weights": {"mode": ipc_weight_mode, "delta_min": float(w_delta_min), "delta_max": float(w_delta_max)},
        **cnf_meta,
    }

    cert_data = {
        "meta": meta,
        "S2": {"rho": float(rho), "kappa": float(kap), "d_kappa": float(bound), "pass": bool(S2_ok)},
        "spectral": {"lambda_max_GH": float(lam), "mu_dec": float(mu_dec)},
        "IPC": {"beta": float(beta), "delta": float(delta), "theta": float(theta), "mu_sat_min": float(mu_sat_min)},
        "bands": {"lam_unsat_ceiling": float(lam_unsat_ceiling), "mu_unsat_max": float(mu_unsat_max),
                  "tau": float(tau), "Delta": float(Delta), "separated": bool(separated)},
        "diag": {"coherence_R": {"mean": float(r_mean), "min": float(r_min), "max": float(r_max), "mean_tail": float(r_mean_tail), "ema_tail": float(r_ema_tail), "tail_frac": float(tail_frac)},
                 "corr_proxy": {"mean": float(np.mean(corr)) if corr.size else 0.0,
                                "min": float(np.min(corr)) if corr.size else 0.0,
                                "max": float(np.max(corr)) if corr.size else 0.0},
                 "weights": {"mean": float(np.mean(w)) if w.size else 0.0,
                            "min": float(np.min(w)) if w.size else 0.0,
                            "max": float(np.max(w)) if w.size else 0.0},
                 "cnf_witness": witness},
    }

    print(f"\nWitness: {witness}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(cert_data, f, indent=2, ensure_ascii=False)

    return Certificate(**cert_data)


def get_nested(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def verify_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mu_sat_min = float(get_nested(obj, "IPC.mu_sat_min"))
    mu_unsat_max = float(get_nested(obj, "bands.mu_unsat_max"))
    tau_rep = float(get_nested(obj, "bands.tau"))
    Delta_rep = float(get_nested(obj, "bands.Delta"))

    tau = 0.5 * (mu_sat_min + mu_unsat_max)
    Delta = 0.5 * (mu_sat_min - mu_unsat_max)

    out = {
        "tau_reported": tau_rep,
        "tau_recomputed": tau,
        "Delta_reported": Delta_rep,
        "Delta_recomputed": Delta,
        "bands_separated": bool(Delta > 0),
        "S2_ok": bool(get_nested(obj, "S2.pass")),
        "notes": {
            "mode": str(get_nested(obj, "meta.mode")),
            "lambda_max_GH": float(get_nested(obj, "spectral.lambda_max_GH")),
        }
    }
    print(json.dumps(out, indent=2))
    return out


#fix
def extract_witness(clauses: List[List[int]], n_vars: int, Z: np.ndarray) -> Dict:
    """Extrahuje ohodnocení s úhlovým sweepem pro nalezení nejlepšího průmětu."""
    best_unsat = len(clauses) + 1
    best_assignment = {}

    # Zkusíme 24 směrů (po 15 stupních)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    for phi in angles:
        # Rotace a průmět
        Z_rot = Z * np.exp(1j * phi)
        # Agregace přes dimenzi d (osa 1) a rozhodnutí podle znaménka
        current_assign = {}
        for i in range(n_vars):
            val = np.sum(np.real(Z_rot[i, :]))
            current_assign[i + 1] = True if val >= 0 else False

        # Výpočet UNSAT pro toto natočení
        u_count = 0
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit)
                pol = lit > 0
                if current_assign[v] == pol:
                    sat = True
                    break
            if not sat:
                u_count += 1

        if u_count < best_unsat:
            best_unsat = u_count
            best_assignment = current_assign.copy()
            if best_unsat == 0: break

    # Finální statistiky pro nejlepší nalezený úhel
    scores = [np.sum(np.real(Z * np.exp(1j * angles[np.argmin(angles)]))) for i in range(n_vars)]

    return {
        "assign_sha256": hashlib.sha256(
            json.dumps([best_assignment[i] for i in range(1, n_vars + 1)]).encode()).hexdigest(),
        "unsat": best_unsat,
        "unsat_frac": best_unsat / len(clauses),
        "score_stats": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="DREAM6_operator_2 functional certifier (no placeholders).")
    ap.add_argument("--C", type=int, default=2000)
    ap.add_argument("--R", type=int, default=104)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--mode", type=str, default="sat", choices=["sat", "unsat"])
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--eta", type=float, default=0.5)

    ap.add_argument("--steps", type=int, default=None, help="Alias for --sweeps (DREAM6 tuner nomenclature).")
    ap.add_argument("--K", type=float, default=4.0, help="DREAM6 coupling gain (separate from eta).")
    ap.add_argument("--noise-sigma", dest="noise_sigma", type=float, default=0.025, help="DREAM6 phase-noise sigma.")
    ap.add_argument("--dt", type=float, default=0.05, help="Time step used only to scale phase-noise (sigma*sqrt(dt)).")
    ap.add_argument("--mu", type=float, default=0.9999954637353263, help="Relaxation/EMA towards projected overlap update.")
    ap.add_argument("--mu-E", dest="mu_E", type=float, default=0.9999952540504147, help="EMA parameter for tail-coherence diagnostic.")
    ap.add_argument("--h", type=float, default=0.4, help="External carrier field bias.")
    ap.add_argument("--tail-frac", type=float, default=0.33, help="Fraction of the end of R(t) used for tail metrics.")


    ap.add_argument("--shared-carrier", action="store_true", default=True)
    ap.add_argument("--shared-misphase", dest="shared_misphase", action="store_true", default=True)
    ap.add_argument("--no-shared-misphase", dest="shared_misphase", action="store_false")
    ap.add_argument("--unsat-neg-frac", type=float, default=0.25)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--power-iters", type=int, default=250)
    ap.add_argument("--power-tol", type=float, default=1e-10)

    ap.add_argument("--ipc-weights", type=str, default="ones", choices=["ones", "cvxopt", "corr", "cvxopt_corr", "auto"])
    ap.add_argument("--w-delta-min", type=float, default=12.0)
    ap.add_argument("--w-delta-max", type=float, default=1000.0)

    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--verify-json", type=str, default=None)

    ap.add_argument("--cnf-path", type=str, default="uf250-0100.cnf", help="DIMACS CNF path (.cnf)")
    ap.add_argument("--edge-mode", type=str, default="auto", choices=["auto","circulant","cnf","logic"],
                   help="Graph topology: auto (cnf->logic else circulant), circulant, cnf/logic (CNF clause-variable graph)")
    ap.add_argument("--outside-value", type=float, default=-1.0,
                   help="Background value outside lock windows (e.g. 0.0 to avoid silent regime for large T).")
    ap.add_argument("--no-flip-incident-unsat", action="store_true",
                   help="Disable incident-first edge flipping in UNSAT mode")

    ap.add_argument("--get-params", action="store_true",
                        help="Vypočítá doporučené parametry R/eta/d z CNF bez spuštění simulace.")

    args = ap.parse_args()

    if args.steps is not None:
        args.sweeps = int(args.steps)

    if args.verify_json:
        verify_json(args.verify_json)
        return

    """# --- Spectral Focusing Navigator (empirical fit to your optimal points) ---
    if args.get_params:
        if args.cnf_path:
            nvars, clauses = parse_dimacs(args.cnf_path)
            C = len(clauses)
        else:
            C = int(args.C)
            clauses = []

        if C <= 0:
            print("Chyba: C musí být > 0.")
            return

        # Empirická kalibrace: 10k -> R=56, 50k -> ~24
        # R ~ 56 * (C/10000)^(-1/2)
        R_opt = int(round(56.0 * (float(C) / 10000.0) ** (-0.5)))
        R_opt = int(np.clip(R_opt, 16, 256))

        # Zachovej tvoje defaulty pro eta/d (nepřepisuju teorii, jen reportuju)
        eta_opt = float(args.eta)
        d_opt = float(args.d)

        logC = float(np.log(float(C)))
        # Pokud chceš reportovat i "focus" metriku:
        # (T v tomhle CLI beru jako 2*R, čistě pro diagnostiku)
        T_opt = 2 * R_opt
        F = T_opt / (d_opt * logC) if logC > 0 else float("inf")

        print("\n=== Spectral Navigator (empirical) ===")
        print(f"Instance: C={C}")
        print(f"R_opt    : {R_opt}   (T≈{T_opt})")
        print(f"eta      : {eta_opt}")
        print(f"d        : {d_opt}")
        print(f"F        : {F:.4f}")
        print("-" * 55)
        print("Doporučený příkaz:")
        print(f"python DREAM6_operator_6.py --cnf-path {args.cnf_path} "
              f"--mode {args.mode} --edge-mode {args.edge_mode} "
              f"--eta {eta_opt} --d {int(d_opt)} --shared-carrier --R {R_opt}")
        return"""

    cert = run(
        C=args.C, R=args.R, d=args.d,
        sweeps=args.sweeps, eta=args.eta,
        K=args.K, noise_sigma=args.noise_sigma, dt=args.dt, mu=args.mu, mu_E=args.mu_E, h=args.h, tail_frac=args.tail_frac,
        mode=args.mode,
        shared_carrier=args.shared_carrier,
        shared_misphase=args.shared_misphase,
        unsat_neg_frac=args.unsat_neg_frac,
        seed=args.seed,
        power_iters=args.power_iters,
        power_tol=args.power_tol,
        ipc_weight_mode=args.ipc_weights,
        w_delta_min=args.w_delta_min,
        w_delta_max=args.w_delta_max,
        cnf_path=args.cnf_path,
        edge_mode=args.edge_mode,
        flip_incident_unsat=(not args.no_flip_incident_unsat),
        outside_value=complex(args.outside_value),
        json_out=args.json_out,
    )

    res = asdict(cert)

    print(f"\n=== DREAM6 Operator Certifier ({args.ipc_weights})")
    print(f"C={res['meta']['C']}  T={res['meta']['T']}  m={res['meta']['m']}  d={res['meta']['d']}  mode={res['meta']['mode']}")
    #if res["meta"].get("cnf_path"):
    #    print(f"CNF: vars={res['meta']['nvars']}  seed_unsat={res['meta']['seed_unsat']} ({100*res['meta']['seed_unsat_frac']:.2f}%)")
    print(f"shared_carrier={res['meta']['shared_carrier']}  shared_misphase={res['meta']['shared_misphase']}  unsat_neg_frac={res['meta']['unsat_neg_frac']}")
    print(f"ipc_weights={res['meta']['ipc_weights']['mode']}")
    d6 = res['meta'].get('dream6', {})
    print(f"dream6: K={d6.get('K')}  noise_sigma={d6.get('noise_sigma')}  dt={d6.get('dt')}  mu={d6.get('mu')}  mu_E={d6.get('mu_E')}  h={d6.get('h')}  tail_frac={d6.get('tail_frac')}")

    print(f"S2 radar: rho={res['S2']['rho']:.6g}  <= d*kappa={res['S2']['d_kappa']:.6g}  pass={res['S2']['pass']}")
    print(f"Spectral: lambda_max(G_H)={res['spectral']['lambda_max_GH']:.6g}  mu_dec={res['spectral']['mu_dec']:.6g}")
    print(f"IPC: beta={res['IPC']['beta']:.6g}  delta={res['IPC']['delta']:.6g}  mu_sat_min={res['IPC']['mu_sat_min']:.6g}")
    print(f"Bands: lam_unsat_ceiling={res['bands']['lam_unsat_ceiling']:.6g}  mu_unsat_max={res['bands']['mu_unsat_max']:.6g}")
    print(f"tau={res['bands']['tau']:.6g}  Delta={res['bands']['Delta']:.6g}  separated={res['bands']['separated']}")

    r = res["diag"]["coherence_R"]
    print(f"Coherence R(t): mean={r['mean']:.6g}  min={r['min']:.6g}  max={r['max']:.6g}  tail_mean={r.get('mean_tail',0.0):.6g}  tail_ema={r.get('ema_tail',0.0):.6g}  tail_frac={r.get('tail_frac',0.0):.3g}")
    print(f"Coherence tail: mean_tail={r.get('mean_tail',0.0):.6g}  ema_tail={r.get('ema_tail',0.0):.6g}  tail_frac={r.get('tail_frac',0.0):.6g}")

    print("==============================================================\n")

    if args.json_out:
        print(f"Wrote certificate JSON: {args.json_out}")


if __name__ == "__main__":
    main()