
#!/usr/bin/env python3
# DREAM6_SEED v3  — A1/A2/A3 focused (offset geometry + Hadamard masks + bounded cross-terms)
# - De-aliased offsets near T/2 with stride s coprime with T (A1).
# - Walsh–Hadamard masks truncated to m slots, row/column indices coprime (A2).
# - Optional coupling over a d-regular circulant graph with kappa_S2-style attenuation (A3).
# - Power-iteration for principal eigenvector (implicit Gram).
# - Initial assignment + UNSAT report; optional tiny greedy micro-polish.
#
# God-mode defaults (from your message):
#   cR=10, L=3, sigma_up=0.045, rho=0.734296875, zeta0=0.4,
#   neighbor_atten=0.9495, seed=42, couple=1 (enable coupling), d=6.
#
# Usage (exact params you gave):
#   python3 DREAM6_seed_v3.py --cnf random_3sat_10000.cnf --godmode --polish 20000
#
# Or explicitly:
#   python3 DREAM6_seed_v3.py --cnf random_3sat_10000.cnf --mode unsat_hadamard \
#     --rho 0.734296875 --zeta0 0.4 --cR 10 --L 3 --sigma_up 0.045 \
#     --neighbor_atten 0.9495 --d 6 --seed 42 --couple 1 --power_iters 60 --polish 20000
#
# python3 DREAM6_seed_v3.py --cnf uf250-0100.cnf --godmode --sigma_up 0.024 --power_iters 120 --polish 500000 --cR 12
#   --L 4 --neighbor_atten 0.9495 --bias_weight 0.12 --score_norm_alpha 0.6 --rho 0.734296875 --zeta0 0.4

import argparse, math, os, random, sys, time
import numpy as np

# ---------- helpers ----------
def sigma_proxy(C, cR=10.0, L=3, eta_power=3, C_B=1.0):
    C = max(2, int(C))
    R = max(1, int(math.ceil(cR * math.log(C))))
    T = R * L
    eta = C ** (-eta_power)
    sigma_up = C_B * math.sqrt(max(1e-12, math.log(C / eta)) / max(1, T))
    return sigma_up, R, T

def wiring_neighbors_circulant(C, d=6):
    """Return neighbor sets for each clause index in a circulant d-regular graph.

    Robustified: for small C or overlarge d we gracefully clip instead of raising.
    """
    if C <= 2 or d <= 0:
        return [set() for _ in range(C)]
    # even degree
    if d % 2 != 0:
        d -= 1
    # cannot exceed C-1
    d = min(d, C - 1)
    if d <= 0:
        return [set() for _ in range(C)]
    nbrs = []
    for i in range(C):
        s = set()
        for step in range(1, d//2 + 1):
            s.add((i - step) % C)
            s.add((i + step) % C)
        nbrs.append(s)
    return nbrs

def gcd_coprime_stride_near_half(T):
    # choose s near T/2, coprime with T
    s = max(1, T//2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

# ---------- schedules ----------
def schedule_unsat_hadamard(C, R, rho=0.734296875, zeta0=0.40, L=3, sigma_up=0.045,
                             seed=42, couple=True, neighbor_atten=0.9495, d=6, verbose=False):
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(math.floor(rho * T))
    k = int(math.floor(zeta0 * m))

    # A1: de-aliased offsets with stride near T/2
    s = gcd_coprime_stride_near_half(T)
    offsets = [(j * s) % T for j in range(C)]
    lock_idx = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]
    Phi = np.full((T, C), np.pi, dtype=float)

    # A2: Walsh–Hadamard masks truncated to m, with coprime row/col indexing
    Hlen = 1
    while Hlen < m: Hlen <<= 1
    # Build H by recursion (small sizes only stored implicitly via indexing choice)
    # We'll assemble the signs by indexing into a synthetic Sylvester Hadamard using parity of bit dot.
    def hadamard_sign(row, col):
        # Sylvester H: sign = (-1)^{<rowbits, colbits>}
        return 1.0 if (bin(row & col).count("1") % 2 == 0) else -1.0

    # choose row step and column generator coprime
    row_step = (Hlen // 2) + 1
    if math.gcd(row_step, Hlen) != 1:
        row_step |= 1
        while math.gcd(row_step, Hlen) != 1:
            row_step += 2
    g = (Hlen // 3) | 1
    while math.gcd(g, Hlen) != 1:
        g += 2

    cols = np.mod(g * np.arange(m, dtype=int), Hlen)
    for j in range(C):
        row = (j * row_step) % Hlen
        # negatives to pi, positives to ~0 with tiny Gaussian sigma_up
        neg_idx = []
        for t in range(m):
            sgn = hadamard_sign(row, int(cols[t]))
            if sgn < 0: neg_idx.append(t)
        if len(neg_idx) >= k:
            mask_pi = rng.choice(np.array(neg_idx, dtype=int), size=k, replace=False)
        else:
            pool = np.setdiff1d(np.arange(m, dtype=int), np.array(neg_idx, dtype=int), assume_unique=False)
            extra = rng.choice(pool, size=k-len(neg_idx), replace=False) if k > len(neg_idx) else np.empty(0, dtype=int)
            mask_pi = np.concatenate([np.array(neg_idx, dtype=int), extra])
        mask_0 = np.setdiff1d(np.arange(m, dtype=int), mask_pi, assume_unique=False)
        slots = lock_idx[j]
        Phi[slots[mask_pi], j] = np.pi
        if len(mask_0) > 0:
            Phi[slots[mask_0], j] = rng.normal(loc=0.0, scale=sigma_up, size=len(mask_0))

    # A3: coupling (optional): attenuate overlaps along circulant neighbors
    if couple and (abs(neighbor_atten - 1.0) > 1e-12):
        neighbors = wiring_neighbors_circulant(C, d=d)
        lock_sets = [set(li.tolist()) for li in lock_idx]
        # kappa_S2 proxy from your Lemma2 page
        # kappa = (1 - 2*ζ0)^2 + 2^(-⌊log2 m⌋/2) + 2/m + 1/T  (we'll use a mild safe version)
        kappa = (1.0 - 2.0*zeta0)**2 + (2.0**(-int(math.log2(max(2, m))) / 2.0)) + (2.0/max(1,m)) + (1.0/max(1,T))
        kappa = max(0.0, min(1.0, kappa))
        for j in range(C):
            o_j = offsets[j]
            Lj = lock_sets[j]
            for j_adj in neighbors[j]:
                if j_adj == j: continue
                La = lock_sets[j_adj]
                overlap = Lj.intersection(La)
                if not overlap: continue
                overlap_size = len(overlap)
                overlap_fraction = overlap_size / max(1, m)
                # attenuation formula biased by overlap and kappa proxy
                cross_term_weight = min(1.0, (len(neighbors[j]) * kappa) / max(1.0, C * (1 - 0.5*sigma_up)**2)
                                        * (1.0 + 3.0 * overlap_fraction))
                attenuation = max(0.70, neighbor_atten - 0.05 * overlap_size / (m * (1 + 0.25
                                        * math.sqrt(max(1e-9, math.log(C))) * overlap_fraction)) * (1 - cross_term_weight))
                idx = np.fromiter(overlap, dtype=int)
                Phi[idx, j_adj] *= attenuation
                if verbose and j % (max(1, C//20)) == 0:
                    print(f"[A3] j={j}→{j_adj} | overlap={overlap_size} attn={attenuation:.3f} κ≈{kappa:.4f}")
    return Phi

def schedule_sat_aligned(C, R, L=3):
    return np.zeros((R*L, C), dtype=float)

# ---------- spectral weight via power-iteration ----------
def principal_weight_power(Phi, iters=60):
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.normal(size=C) + 1j*rng.normal(size=C)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        y = Z @ x
        x = (Z.conj().T @ y) / T
        x /= (np.linalg.norm(x) + 1e-12)
    return np.abs(x)

# ---------- DIMACS ----------
def parse_dimacs(path):
    clauses, nvars = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "c%":
                continue
            if s[0] == "p":
                parts = s.split()
                if len(parts)>=4 and parts[1].lower()=="cnf":
                    nvars = int(parts[2])
                continue
            lits = [int(x) for x in s.split() if x!="0"]
            if lits:
                clauses.append(lits)
                for L in lits:
                    nvars = max(nvars, abs(L))
    return nvars, clauses

# ---------- scoring & seed ----------
def build_seed_assignment(clauses, nvars, mode='unsat_hadamard', cR=10.0, L=3,
                          rho=0.734296875, zeta0=0.40, sigma_up=0.045,
                          neighbor_atten=0.9495, seed=42, couple=True, d=6,
                          power_iters=60, score_norm_alpha=0.5, bias_weight=0.10):
    C = len(clauses)
    _, R, T = sigma_proxy(C, cR=cR, L=L)
    if mode == 'sat':
        Phi = schedule_sat_aligned(C, R, L)
    else:
        Phi = schedule_unsat_hadamard(C, R, rho, zeta0, L, sigma_up, seed,
                                      couple=couple, neighbor_atten=neighbor_atten, d=d, verbose=False)
    w_clause = principal_weight_power(Phi, iters=power_iters)
    w_clause = (w_clause / (w_clause.mean() + 1e-12)).clip(0.1, 10.0)

    pos = [[] for _ in range(nvars+1)]
    neg = [[] for _ in range(nvars+1)]
    pol = np.zeros(nvars+1, dtype=int)
    deg = np.zeros(nvars+1, dtype=int)
    for ci, cl in enumerate(clauses):
        for LIT in cl:
            v = abs(LIT)
            deg[v] += 1
            if LIT > 0:
                pos[v].append(ci); pol[v] += 1
            else:
                neg[v].append(ci); pol[v] -= 1

    score = np.zeros(nvars+1, dtype=float)
    for v in range(1, nvars+1):
        if pos[v]: score[v] += float(w_clause[pos[v]].sum())
        if neg[v]: score[v] -= float(w_clause[neg[v]].sum())
        if score_norm_alpha > 0.0:
            score[v] /= (deg[v]**score_norm_alpha + 1e-12)
        if bias_weight != 0.0:
            score[v] += bias_weight * float(pol[v]) / max(1, deg[v])

    rng = np.random.default_rng(seed+1337)
    dither = rng.uniform(-1e-7, 1e-7, size=nvars+1)
    assign = [(score[v] + dither[v]) >= 0.0 for v in range(1, nvars+1)]
    return assign, Phi

# ---------- UNSAT ----------
def count_unsat(clauses, assign_bool_list):
    unsat = 0
    for cl in clauses:
        ok = False
        for L in cl:
            v = abs(L); val = assign_bool_list[v-1]
            if L < 0: val = (not val)
            if val: ok = True; break
        if not ok: unsat += 1
    return unsat

def check_sat(clauses, model):
   assignment = {
       i + 1:
           bit == 1
       for i, bit in enumerate(model)
   }
   for clause in clauses:
       clause_satisfied = False
       try:
           clause_satisfied = any(
               (lit > 0 and assignment[abs(lit)]) or
               (lit < 0 and not assignment[abs(lit)])
               for lit in clause
           )
       except KeyError:
           return False
       if not clause_satisfied:
           return False
   return True

# ---------- micro-polish (optional) ----------

from typing import List, Optional

def greedy_polish(
    clauses: List[List[int]],
    assign01: List[int],
    flips: int = 20000,
    seed: int = 49,
    alpha: float = 2.4,     # probSAT make exponent
    beta: float = 0.9,      # probSAT break exponent
    epsilon: float = 1e-3,  # probSAT epsilon
    probsat_quota: int = 2000,  # max kroků pro probSAT fázi v rámci polish
) -> List[int]:
    """
    Micro-polish finisher:
      A) exhaust all zero-break flips (tie by max-make)
      B) min-break + max-make tie-break
      C) short probSAT burst if still UNSAT
    """
    rnd = random.Random(seed)
    nvars = len(assign01)
    C = len(clauses)

    # --- adjacency (1-indexed variables) ---
    pos = [[] for _ in range(nvars + 1)]
    neg = [[] for _ in range(nvars + 1)]
    for ci, cl in enumerate(clauses):
        for L in cl:
            (pos if L > 0 else neg)[abs(L)].append(ci)

    # --- state (1-indexed assign for speed) ---
    assign = [False] + [bool(b) for b in assign01]
    sat_count = [0] * C
    in_unsat = [False] * C
    unsat_list: List[int] = []  # compact container of UNSAT clause indices

    def add_unsat(ci: int):
        if not in_unsat[ci]:
            in_unsat[ci] = True
            unsat_list.append(ci)

    def drop_unsat(ci: int):
        if in_unsat[ci]:
            in_unsat[ci] = False

    # init sat_count / unsat
    for ci, cl in enumerate(clauses):
        cnt = 0
        for L in cl:
            v = abs(L)
            val = assign[v]
            if L < 0:
                val = (not val)
            if val:
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            add_unsat(ci)
    # compaction
    unsat_list = [ci for ci in unsat_list if in_unsat[ci]]

    # --- helpers ---
    def breakcount(v: int) -> int:
        bc = 0
        if assign[v]:  # True->False
            for ci in pos[v]:
                if sat_count[ci] == 1:
                    bc += 1
        else:          # False->True
            for ci in neg[v]:
                if sat_count[ci] == 1:
                    bc += 1
        return bc

    def makecount(v: int) -> int:
        mk = 0
        if assign[v]:
            for ci in neg[v]:
                if in_unsat[ci]:
                    mk += 1
        else:
            for ci in pos[v]:
                if in_unsat[ci]:
                    mk += 1
        return mk

    def flip_var(v: int):
        """Incremental flip with sat_count/unsat maintenance."""
        old = assign[v]
        assign[v] = not old
        if old:
            for ci in pos[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in neg[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)
        else:
            for ci in neg[v]:
                sc = sat_count[ci] - 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            for ci in pos[v]:
                sc = sat_count[ci] + 1
                sat_count[ci] = sc
                if sc > 0:
                    drop_unsat(ci)

    # utility to pick a random UNSAT clause quickly
    def pick_unsat_clause() -> Optional[int]:
        if not unsat_list:
            return None
        # fast cleanup-on-read
        i = rnd.randrange(len(unsat_list))
        for _ in range(3):  # up to 3 tries to hit a live one
            ci = unsat_list[i]
            if in_unsat[ci]:
                return ci
            i = rnd.randrange(len(unsat_list))
        # fallback: compact and retry once
        compact = [ci for ci in unsat_list if in_unsat[ci]]
        unsat_list[:] = compact
        if not compact:
            return None
        return rnd.choice(compact)

    def cur_unsat() -> int:
        # quick count without full compaction
        return sum(1 for ci in unsat_list if in_unsat[ci])

    # --- keep best-so-far ---
    best_assign = assign[:]
    best_uns = cur_unsat()

    steps = 0
    # -------- Phase A: exhaust freebies --------
    made_progress = True
    while made_progress and steps < flips:
        made_progress = False
        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]
        freebies = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            if bc == 0:
                freebies.append((makecount(v), v))
        if freebies:
            freebies.sort(reverse=True)  # prefer max make
            _, v = freebies[0]
            flip_var(v)
            steps += 1
            made_progress = True
            # update best
            u = cur_unsat()
            if u < best_uns:
                best_uns = u
                best_assign = assign[:]
            if u == 0:
                return [1 if b else 0 for b in assign[1:]]

    # -------- Phase B: min-break, max-make --------
    while steps < flips:
        if best_uns == 0:
            return [1 if b else 0 for b in best_assign[1:]]

        ci = pick_unsat_clause()
        if ci is None:
            return [1 if b else 0 for b in assign[1:]]
        clause = clauses[ci]

        # preferentially do freebies if present
        v_choice = None
        freebies = []
        cand = []
        for L in clause:
            v = abs(L)
            bc = breakcount(v)
            mk = makecount(v)
            if bc == 0:
                freebies.append((mk, v))
            cand.append((bc, -mk, v))  # -mk to maximize make on tie
        if freebies:
            freebies.sort(reverse=True)
            v_choice = freebies[0][1]
        else:
            # min break, then max make
            v_choice = min(cand)[2]

        flip_var(v_choice)
        steps += 1

        u = cur_unsat()
        if u < best_uns:
            best_uns = u
            best_assign = assign[:]
        if u == 0:
            return [1 if b else 0 for b in assign[1:]]

        # malý „kick“: když se nic nezlepšilo 1k kroků, zkus probSAT burst
        if steps % 1000 == 0 and u >= best_uns:
            # -------- Phase C: short probSAT burst --------
            for _ in range(min(probsat_quota, flips - steps)):
                ci2 = pick_unsat_clause()
                if ci2 is None:
                    return [1 if b else 0 for b in assign[1:]]
                clause2 = clauses[ci2]
                # scores ~ (make+eps)^alpha / (break+eps)^beta
                scores = []
                tot = 0.0
                last_v = None
                for L in clause2:
                    v = abs(L)
                    mk = makecount(v)
                    bc = breakcount(v)
                    s = ((mk + epsilon) ** alpha) / ((bc + epsilon) ** beta)
                    scores.append((v, s))
                    tot += s
                    last_v = v
                r = rnd.random() * tot
                acc = 0.0
                pick = last_v
                for v, s in scores:
                    acc += s
                    if acc >= r:
                        pick = v
                        break
                flip_var(pick)
                steps += 1
                u2 = cur_unsat()
                if u2 < best_uns:
                    best_uns = u2
                    best_assign = assign[:]
                if u2 == 0 or steps >= flips:
                    return [1 if b else 0 for b in assign[1:]]
            # po burstu pokračujeme B-fází

    # vyčerpán budget – vrať nejlepší nalezené
    return [1 if b else 0 for b in (best_assign[1:] if best_uns < cur_unsat() else assign[1:])]


# --------------------- params -----------------------------------------

import math
import sys

def theory_params(C, want_sigma=None, cR=12, L=4, eta_power=3, zeta0=0.40,
                  rho_lock=0.734296875, neighbor_atten=0.9495,
                  tau_hint=0.40, mu_hint=0.002):
    # 1) R, T
    R = math.ceil(cR * math.log(C))
    T = R * L

    # 2) sigma_up (buď z want_sigma, nebo z C_B=1)
    if want_sigma is not None:
        CB = want_sigma * (T**0.5) / math.sqrt((1+eta_power)*math.log(C))
        sigma_up = want_sigma
    else:
        CB = 1.0
        sigma_up = CB * math.sqrt((1+eta_power)*math.log(C)) / (T**0.5)

    # 3) gamma0 kontrola
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    if gamma0 <= 0:
        # opravíme rho_lock minimálně tak, aby gamma0 > 0 s malou rezervou
        rho_lock = (0.5*sigma_up + 0.01)/zeta0
        gamma0 = rho_lock * zeta0 - 0.5*sigma_up

    # 4) bias a alpha z (tau, mu) – pokud máme hinty
    def clip(x,a,b): return max(a, min(b,x))
    if tau_hint is None: tau_hint = 0.40
    if mu_hint  is None: mu_hint  = 0.002

    bias_weight = clip(0.08 + 0.30*max(0.0, tau_hint - mu_hint), 0.06, 0.22)
    score_norm_alpha = clip(0.5 + 0.5*max(0.0, 0.5 - tau_hint), 0.5, 0.85)

    return {
        "R": R, "T": T,
        "sigma_up": sigma_up, "C_B": CB,
        "rho": rho_lock, "zeta0": zeta0,
        "neighbor_atten": neighbor_atten,
        "bias_weight": bias_weight,
        "score_norm_alpha": score_norm_alpha,
        "gamma0": gamma0
    }

# ------------------- finnisher -----------------------------

def spectral_pretest(nvars, clauses, samples=10000):
    C = len(clauses)
    if C == 0 or nvars == 0:
        return 0.0, 0.5, 0.5
    deg = [0]*(nvars+1)
    pol = [0]*(nvars+1)
    for cl in clauses:
        seen = set()
        for L in cl:
            v = abs(L)
            if v in seen:  # guard duplicate
                continue
            seen.add(v)
            deg[v] += 1
            pol[v] += 1 if L > 0 else -1
    tau0 = sum(abs(pol[v]) for v in range(1, nvars+1)) / max(1, sum(deg))
    x = np.random.rand(nvars) - 0.5
    x /= np.linalg.norm(x) + 1e-12
    sampler = []
    for cl in clauses:
        vs = [abs(L) for L in cl]
        k = len(vs)
        for i in range(k):
            for j in range(i+1, k):
                sampler.append((vs[i]-1, vs[j]-1))
    if len(sampler) > samples:
        random.shuffle(sampler)
        sampler = sampler[:samples]
    def Ax(xvec):
        y = np.zeros_like(xvec)
        for i, j in sampler:
            y[i] += xvec[j]
            y[j] += xvec[i]
        return y
    for _ in range(3):
        y = Ax(x)
        x = y / (np.linalg.norm(y) + 1e-12)
    num = float(np.dot(x, Ax(x)))
    den = float(np.dot(x, x)) + 1e-12
    lam = num/den
    mu = lam / max(1, C)
    base = 0.35
    p0 = base + 0.25*(mu - 0.002) - 0.20*(tau0 - 0.4)
    p0 = max(0.12, min(0.58, p0))
    return mu, tau0, p0
"""
class WalkSATFinisher:
    def __init__(self, nvars, clauses, start_assign, p=0.40, seed=42):
        self.n = nvars
        self.C = len(clauses)
        self.clauses = clauses
        self.p = p
        random.seed(seed)
        self.pos = [[] for _ in range(nvars+1)]
        self.neg = [[] for _ in range(nvars+1)]
        for ci, cl in enumerate(clauses):
            for L in cl:
                (self.pos if L>0 else self.neg)[abs(L)].append(ci)
        self.assign = list(start_assign)
        self.sat_count = [0]*self.C
        self.unsat = set()
        for ci, cl in enumerate(clauses):
            cnt = 0
            for L in cl:
                v = abs(L); val = self.assign[v-1]
                if L < 0: val = (not val)
                if val: cnt += 1
            self.sat_count[ci] = cnt
            if cnt == 0: self.unsat.add(ci)

    def _breakcount(self, v):
        bc = 0
        if self.assign[v-1]:
            for ci in self.pos[v]:
                if self.sat_count[ci] == 1: bc += 1
        else:
            for ci in self.neg[v]:
                if self.sat_count[ci] == 1: bc += 1
        return bc

    def _flip(self, v):
        old = self.assign[v-1]; new = not old
        self.assign[v-1] = new
        if old:
            for ci in self.pos[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.neg[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)
        else:
            for ci in self.neg[v]:
                sc = self.sat_count[ci]-1; self.sat_count[ci]=sc
                if sc == 0: self.unsat.add(ci)
            for ci in self.pos[v]:
                sc = self.sat_count[ci]+1; self.sat_count[ci]=sc
                if sc > 0: self.unsat.discard(ci)

    def solve(self, max_flips=20_000_000, report_every=100000, novelty=0.30,
              restart_base=500_000, restart_mult=1.30, seed=42):
        best_unsat = len(self.unsat); best_state = self.assign[:]
        flips_since_best = 0
        next_restart = restart_base
        t0 = time.time()
        for flips in range(1, max_flips+1):
            if not self.unsat:
                return True, flips, time.time()-t0, self.assign[:]
            ci = random.choice(tuple(self.unsat))
            cl = self.clauses[ci]
            cand = [abs(L) for L in cl]
            if random.random() < self.p:
                v = random.choice(cand)
            else:
                best_bc, pool = 1e9, []
                for v0 in cand:
                    bc = self._breakcount(v0)
                    if bc < best_bc:
                        best_bc, pool = bc, [v0]
                    elif bc == best_bc:
                        pool.append(v0)
                v = random.choice(pool)
            self._flip(v)
            cur_unsat = len(self.unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_state = self.assign[:]
                flips_since_best = 0
            else:
                flips_since_best += 1
            if flips % report_every == 0:
                print(f"[finisher] flips={flips:,} unsat={cur_unsat:,} p≈{self.p:.6f}")
            if flips_since_best >= next_restart:
                # restart around the best state (keep 50%)
                rng = np.random.default_rng(seed)
                mask = rng.random(self.n) < 0.5
                self.assign = [ (best_state[i] if mask[i] else random.choice((False, True)))
                                for i in range(self.n) ]
                self.unsat.clear()
                self.sat_count = [0]*self.C
                for ci2, cl2 in enumerate(self.clauses):
                    cnt = 0
                    for L in cl2:
                        v2 = abs(L); val = self.assign[v2-1]
                        if L < 0: val = (not val)
                        if val: cnt += 1
                    self.sat_count[ci2] = cnt
                    if cnt == 0: self.unsat.add(ci2)
                flips_since_best = 0
                next_restart = int(next_restart * restart_mult)
        self.assign = best_state[:]
        return False, max_flips, time.time()-t0, self.assign[:]
"""
def check_sat(clauses, model):
   assignment = {i + 1: (bit == 1) for i, bit in enumerate(model)}
   for clause in clauses:
       clause_satisfied = False
       try:
           clause_satisfied = any(
               (lit > 0 and assignment[abs(lit)]) or
               (lit < 0 and not assignment[abs(lit)])
               for lit in clause
           )
       except KeyError:
           return False
       if not clause_satisfied:
           return False
   return True

def unsat_indices(clauses, assign01):
    ids = []
    for i, cl in enumerate(clauses):
        sat = False
        for L in cl:
            v = abs(L) - 1;
            val = bool(assign01[v])
            if L < 0: val = not val
            if val: sat = True; break
        if not sat: ids.append(i)
    return ids

import random
from typing import List, Tuple

def _lit_true(lit: int, a: List[bool]) -> bool:
    """Return truth value of literal under assignment a (0-based vars)."""
    v = abs(lit) - 1
    val = a[v]
    return val if lit > 0 else (not val)

def _flip_var(v: int, a: List[bool]) -> None:
    """Flip variable v (0-based)."""
    a[v] = not a[v]

def finisher_epic_incremental(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    # noise control
    p_min: float = 0.02,
    p_max: float = 0.90,
    drift_window: int = 200_000,
    # weights
    w_inc: float = 1.0,
    w_decay: float = 0.9995,
    w_cap: float = 50.0,
    # stall / restarts
    stall_window: int = 500_000,
    restart_shake: int = 64,
    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, dict]:
    """
    Incremental weighted WalkSAT/ProbSAT hybrid.
    Returns (assignment, solved, stats)

    CRITICAL FIX:
    - clause_flip_effect must be computed relative to the variable value BEFORE flip.
      Otherwise sat_count drifts from reality and unsat_list/pos corrupt -> IndexError.
    - Always index unsat_list via len(unsat_list), never a cached 'u'.
    """
    rng = random.Random(seed)

    a = a0[:]  # working
    best = a0[:]
    best_unsat = None
    last_improve_at = 0

    m = len(clauses)

    # var -> incident clauses
    var_occ: List[List[int]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            var_occ[abs(lit) - 1].append(ci)

    # sat_count per clause
    sat_count = [0] * m
    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            if _lit_true(lit, a):
                cnt += 1
        sat_count[ci] = cnt

    # unsat set as list + positions
    unsat_list: List[int] = []
    pos = [-1] * m
    for ci in range(m):
        if sat_count[ci] == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # weights
    w = [1.0] * m

    def unsat_size() -> int:
        return len(unsat_list)

    def add_unsat(ci: int) -> None:
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def remove_unsat(ci: int) -> None:
        i = pos[ci]
        if i == -1:
            return
        last = unsat_list[-1]
        unsat_list[i] = last
        pos[last] = i
        unsat_list.pop()
        pos[ci] = -1

    def clause_flip_effect(ci: int, v: int, old_av: bool) -> int:
        """
        Compute delta (after - before) of sat_count[ci] if we flip v.
        IMPORTANT: old_av is a[v] BEFORE the flip.
        a[] is allowed to already be flipped when calling this function.
        """
        before = 0
        after = 0

        for lit in clauses[ci]:
            var = abs(lit) - 1

            # BEFORE value of this var
            val_b = old_av if var == v else a[var]
            litval_b = val_b if lit > 0 else (not val_b)
            if litval_b:
                before += 1

            # AFTER value of this var (v flipped)
            val_a = (not old_av) if var == v else a[var]
            litval_a = val_a if lit > 0 else (not val_a)
            if litval_a:
                after += 1

        return after - before

    def break_score(v: int) -> float:
        # weighted number of clauses that would go satisfied->unsatisfied (sat_count 1 -> 0)
        b = 0.0
        old_av = a[v]  # not flipping for real, just simulating
        for ci in var_occ[v]:
            if sat_count[ci] == 1:
                # if flip decreases sat_count by 1, and since it's 1, it becomes 0 -> break
                if clause_flip_effect(ci, v, old_av) == -1:
                    b += w[ci]
        return b

    def make_score(v: int) -> float:
        # weighted number of clauses that would go unsat->sat (0 -> >=1)
        mk = 0.0
        old_av = a[v]  # simulate
        for ci in var_occ[v]:
            if sat_count[ci] == 0:
                # if flip increases sat_count, it becomes satisfied -> make
                if clause_flip_effect(ci, v, old_av) > 0:
                    mk += w[ci]
        return mk

    # adaptive p control (based on recent improvement rate)
    p = p_min
    window_best = unsat_size()
    window_at = 0

    flips = 0
    while flips < max_flips:
        if not unsat_list:
            return a, True, {"flips": flips, "best_unsat": 0}

        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flips, "best_unsat": 0}

        # best tracking
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_at = flips

        # drift window update
        if flips - window_at >= drift_window:
            # if no progress in window -> increase noise, else reduce
            if u >= window_best:
                p = min(p_max, p * 1.25 + 0.02)
            else:
                p = max(p_min, p * 0.80)
            window_best = u
            window_at = flips

        # stall restart
        if stall_window > 0 and flips - last_improve_at >= stall_window:
            a = best[:]

            # shake only vars that appear in currently UNSAT clauses
            hard_vars = set()
            for ci0 in unsat_list:
                for lit in clauses[ci0]:
                    hard_vars.add(abs(lit) - 1)
            hard_vars = list(hard_vars)

            for _ in range(min(restart_shake, len(hard_vars))):
                v0 = hard_vars[rng.randrange(len(hard_vars))]
                a[v0] = not a[v0]

            # full rebuild sat_count + unsat_list/pos
            for ci0, c0 in enumerate(clauses):
                cnt0 = 0
                for lit in c0:
                    if _lit_true(lit, a):
                        cnt0 += 1
                sat_count[ci0] = cnt0

            unsat_list.clear()
            for i in range(m):
                pos[i] = -1
            for ci0 in range(m):
                if sat_count[ci0] == 0:
                    pos[ci0] = len(unsat_list)
                    unsat_list.append(ci0)

            last_improve_at = flips
            p = min(p_max, max(p, 0.35))
            continue  # restart done, go next iteration with clean state

        # pick random unsat clause (always from current list length)
        ci = unsat_list[rng.randrange(len(unsat_list))]
        c = clauses[ci]

        # weight bump (focused on persistent conflicts)
        w[ci] = min(w_cap, w[ci] + w_inc)

        # choose candidate vars (from clause lits)
        vars_in = [abs(lit) - 1 for lit in c]

        # with prob p: random flip from clause (noise)
        if rng.random() < p:
            v = vars_in[rng.randrange(len(vars_in))]
        else:
            # greedy on weighted (break - make) with tie-breaking
            best_v = vars_in[0]
            best_key = None
            for v0 in vars_in:
                b = break_score(v0)
                mk = make_score(v0)
                # primary: minimize break, secondary: maximize make
                key = (b, -mk, rng.random())
                if best_key is None or key < best_key:
                    best_key = key
                    best_v = v0
            v = best_v

        # apply flip with incremental updates (CRITICAL: store old_av)
        old_av = a[v]
        _flip_var(v, a)
        flips += 1

        # update affected clauses using correct delta relative to old_av
        for cj in var_occ[v]:
            old = sat_count[cj]
            delta = clause_flip_effect(cj, v, old_av)
            if delta == 0:
                continue
            new = old + delta
            sat_count[cj] = new
            if old == 0 and new > 0:
                remove_unsat(cj)
            elif old > 0 and new == 0:
                add_unsat(cj)

        # occasional global decay (cheap)
        if w_decay < 1.0 and (flips & 0x3FFFF) == 0:
            for i in range(m):
                w[i] = max(1.0, w[i] * w_decay)

        if report_every and flips % report_every == 0:
            print(f"[finisher] flips={flips:,} unsat={len(unsat_list)} p≈{p:.3f} best={best_unsat}")

    # finished budget
    return best, (best_unsat == 0), {"flips": flips, "best_unsat": best_unsat}


# finisher

def core_push(
    clauses, nvars, assign01,
    seed=0,
    loops=3,
    rounds=8,
    step0=0.12,
    decay=0.90,
    per_clause_cap=2,
    beta=1.3,
    bias_damp=0.70,
):
    """
    Deterministic-ish tightening pass:
    - works only on UNSAT core
    - builds a 'pressure' score per var from UNSAT clauses
    - flips a small set of top vars each round (step-controlled)
    """
    rng = random.Random(seed)
    a = [bool(x) for x in assign01]

    def clause_sat(cl):
        for lit in cl:
            v = abs(lit) - 1
            val = a[v]
            if lit < 0:
                val = (not val)
            if val:
                return True
        return False

    for L in range(max(1, loops)):
        step = step0
        best_uns = None

        for r in range(max(1, rounds)):
            # collect UNSAT ids
            unsat_ids = []
            for ci, cl in enumerate(clauses):
                if not clause_sat(cl):
                    unsat_ids.append(ci)

            u = len(unsat_ids)
            if best_uns is None or u < best_uns:
                best_uns = u

            if u == 0:
                return [1 if x else 0 for x in a]

            # build pressure score from UNSAT only
            score = [0.0] * nvars
            # (optional) tiny bias damping: prevent "polarity lock"
            pol = [0.0] * nvars

            for ci in unsat_ids:
                cl = clauses[ci]
                # cap influence per clause: sample up to per_clause_cap literals
                lits = cl[:]
                rng.shuffle(lits)
                lits = lits[:max(1, per_clause_cap)]

                for lit in lits:
                    v = abs(lit) - 1
                    # to satisfy an UNSAT clause, you'd like at least one literal true
                    # so push variable toward making this literal true
                    want_true = (lit > 0)
                    cur = a[v]
                    # if want_true then we'd like a[v]=True else False
                    direction = +1.0 if want_true else -1.0
                    # if already aligned, smaller push; if misaligned, bigger push
                    aligned = (cur == want_true)
                    mag = 0.25 if aligned else 1.0
                    score[v] += direction * mag
                    pol[v] += direction

            # bias damping (keeps you from hard-locking a global polarity)
            if bias_damp < 1.0:
                for v in range(nvars):
                    score[v] = bias_damp * score[v] + (1.0 - bias_damp) * pol[v]

            # choose vars to flip: top-|score| scaled by step
            # number of flips grows with sqrt(core) but is step-limited
            k = max(1, int(step * (u ** 0.5) * beta))
            # take top-k by absolute pressure
            cand = sorted(range(nvars), key=lambda i: abs(score[i]), reverse=True)[:k]

            # flip them (small batch move)
            for v in cand:
                a[v] = not a[v]

            step *= decay

        # loop report (optional)
        print(f"[core push] loop {L+1}/{loops}: best_uns={best_uns}")

    return [1 if x else 0 for x in a]

import random
from typing import List, Tuple

# fix
import random
from typing import List, Tuple, Dict

def finisher_corefocus_v1(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    # noise
    p_min: float = 0.02,
    p_max: float = 0.28,
    p_base: float = 0.18,          # vyšší než 0.15 -> midgame se hýbe
    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9996,
    w_cap: float = 80.0,
    # restart
    stall_window: int = 400_000,
    restart_shake: int = 96,
    # reporting
    report_every: int = 1_000_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Core-focused ProbSAT/WalkSAT hybrid:
    - signed occurrences for correct incremental sat_count updates
    - top-weight focus already for u <= 1024 (not only 256)
    - endgame noise clamp only when u <= 128 / 64
    - surgical restart shake (weighted clause pick + ProbSAT var pick)
    """

    rng = random.Random(seed)
    a = a0[:]
    m = len(clauses)

    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # var_occ[v] = [(clause_id, sign), ...]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    # sat_count + unsat_list
    sat_count = [0] * m
    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if lit_true(s, a[v]):
                cnt += 1
        sat_count[ci] = cnt

    unsat_list: List[int] = []
    pos = [-1] * m
    for ci in range(m):
        if sat_count[ci] == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # weights + var scores
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars
    eps = 1e-3

    def update_scores(v_idx: int):
        br = 0.0
        mk = 0.0
        cur = a[v_idx]
        flipv = not cur
        for (ci, sgn) in var_occ[v_idx]:
            sc = sat_count[ci]
            w = clause_w[ci]
            if sc == 0:
                # clause currently UNSAT; flip "makes" if this literal becomes true
                if lit_true(sgn, flipv):
                    mk += w
            elif sc == 1:
                # clause has exactly one satisfier; flip "breaks" if we remove that satisfier
                if lit_true(sgn, cur) and (not lit_true(sgn, flipv)):
                    br += w
        var_br_w[v_idx] = br
        var_mk_w[v_idx] = mk

    for v in range(nvars):
        update_scores(v)

    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0
    flip = 0

    def pick_clause_corefocus(u: int) -> int:
        # weighted focus starts earlier to avoid “basin wandering”
        if u <= 1024:
            K = min(u, 96)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:12] if len(sample) >= 12 else sample
            # 2-choice inside pool
            c1 = pool[rng.randrange(len(pool))]
            c2 = pool[rng.randrange(len(pool))]
            return c1 if clause_w[c1] >= clause_w[c2] else c2
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            return c1 if clause_w[c1] >= clause_w[c2] else c2

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        stalled = flip - last_improve

        # --- restart shake (surgical) ---
        if stalled >= stall_window:
            for _ in range(restart_shake):
                uu = len(unsat_list)
                if uu == 0:
                    return a, True, {"flips": flip, "best_unsat": 0}

                ci = pick_clause_corefocus(uu)
                clause_w[ci] = min(w_cap, clause_w[ci] + 0.8 * w_inc)

                lits = clauses[ci]
                # ProbSAT var pick inside shake (mostly greedy)
                best_v = None
                best_s = -1.0
                for lit in lits:
                    v = abs(lit) - 1
                    s = ((var_mk_w[v] + eps) ** 2.0) / ((var_br_w[v] + eps) ** 1.1)
                    if s > best_s:
                        best_s = s
                        best_v = v
                v_idx = best_v if best_v is not None else abs(lits[0]) - 1

                old_val = a[v_idx]
                a[v_idx] = not a[v_idx]

                affected_clauses = []
                for (cj, sgn) in var_occ[v_idx]:
                    was = lit_true(sgn, old_val)
                    now = lit_true(sgn, a[v_idx])
                    if was == now:
                        continue
                    old_sc = sat_count[cj]
                    sat_count[cj] = old_sc + (1 if now else -1)
                    new_sc = sat_count[cj]
                    affected_clauses.append(cj)

                    if old_sc == 0 and new_sc > 0:
                        ip = pos[cj]
                        last_c = unsat_list[-1]
                        unsat_list[ip] = last_c
                        pos[last_c] = ip
                        unsat_list.pop()
                        pos[cj] = -1
                    elif old_sc > 0 and new_sc == 0:
                        pos[cj] = len(unsat_list)
                        unsat_list.append(cj)

                # refresh neighborhood
                neigh = {v_idx}
                for cj in affected_clauses:
                    for lt in clauses[cj]:
                        neigh.add(abs(lt) - 1)
                for v in neigh:
                    update_scores(v)

            last_improve = flip
            continue

        # --- noise schedule (midgame vs endgame) ---
        ratio = u / max(1, m)
        p_eff = p_base * (0.25 + 2.6 * ratio)

        # endgame clamp only late
        if u <= 128:
            p_eff = min(p_eff, 0.14)
        if u <= 64:
            p_eff = min(p_eff, 0.10)

        # if stalled but not endgame, allow some extra kick
        if stalled > stall_window // 2 and u > 128:
            ramp = min(1.0, (stalled - stall_window // 2) / max(1, stall_window // 2))
            p_eff = min(p_max, p_eff + 0.10 * ramp)

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # --- choose unsat clause (core focus) ---
        ci = pick_clause_corefocus(u)

        # bump weight of selected clause (stronger when close)
        bump = w_inc * (1.0 + max(0.0, (128.0 - u) / 96.0)) if u < 128 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump)

        lits = clauses[ci]

        # local refresh for vars in this clause (cheap + helps stability)
        for lt in lits:
            update_scores(abs(lt) - 1)

        # --- variable selection (ProbSAT-ish) ---
        if u <= 64:
            p_pow, q_pow = 2.6, 1.25
        elif u <= 256:
            p_pow, q_pow = 2.0, 1.10
        else:
            p_pow, q_pow = 1.6, 1.00

        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_s = -1.0
            for lit in lits:
                v = abs(lit) - 1
                s = ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    best_v = v
            v_idx = best_v if best_v is not None else abs(lits[0]) - 1

        # --- apply flip + incremental updates ---
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        flip += 1

        affected_clauses = []
        for (cj, sgn) in var_occ[v_idx]:
            was = lit_true(sgn, old_val)
            now = lit_true(sgn, a[v_idx])
            if was == now:
                continue
            old_sc = sat_count[cj]
            sat_count[cj] = old_sc + (1 if now else -1)
            new_sc = sat_count[cj]
            affected_clauses.append(cj)

            if old_sc == 0 and new_sc > 0:
                ip = pos[cj]
                last_c = unsat_list[-1]
                unsat_list[ip] = last_c
                pos[last_c] = ip
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

        neigh = {v_idx}
        for cj in affected_clauses:
            for lt in clauses[cj]:
                neigh.add(abs(lt) - 1)
        for v in neigh:
            update_scores(v)

        # global decay occasionally
        if (flip & 0x3FFFF) == 0:
            d = 0.99985 if u <= 256 else w_decay
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * d)
            # partial refresh would be faster, but keep it safe
            for v in range(nvars):
                update_scores(v)

        if report_every and (flip % report_every) == 0:
            cur_u = len(unsat_list)
            print(f"[finisher] flips={flip:,} unsat={cur_u} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#fux

import random
from typing import List, Tuple, Dict

def finisher_epic_vNext_predator_basin_guard(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_min: float = 0.01,
    p_max: float = 0.28,

    # stall / restarts
    stall_window: int = 400_000,
    restart_shake: int = 96,

    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9995,
    w_cap: float = 80.0,

    # guard rails
    snapback_gap: int = 64,          # if u > best + gap -> snapback to best
    basin_mult: float = 3.0,         # meltdown if u > best * mult
    basin_abs: int = 150,            # ignore meltdown guard when best is tiny / early

    # tabu (optional)
    use_tabu: bool = True,
    tabu_u_threshold: int = 96,
    tabu_tenure: int = 45,

    report_every: int = 100_000,      # you can set 5_000_000 for less spam
) -> Tuple[List[bool], bool, Dict]:
    """
    Predator finisher with:
      - correct incremental sat_count updates (signed occurrences)
      - top-weight focus in hard core
      - endgame noise clamp
      - restart shake with neighborhood refresh
      - elite snapback
      - basin guard (meltdown prevention)
      - optional micro-tabu in hard core
    """

    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve_flip = 0
    m = len(clauses)

    # --- helpers ---
    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # Signed occurrences: var_occ[v] = [(clause_idx, sign), ...]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    # sat_count + unsat_list/pos
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    def rebuild_state_from_assignment():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            cnt = 0
            for lit in clauses[ci]:
                v = abs(lit) - 1
                s = 1 if lit > 0 else -1
                if lit_true(s, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    rebuild_state_from_assignment()

    # weights and scores
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars
    eps = 1e-3
    p_base = 0.15

    def update_scores(v_idx: int):
        br = 0.0
        mk = 0.0
        cur_v = a[v_idx]
        flip_v = not cur_v
        for (ci, sign) in var_occ[v_idx]:
            sc = sat_count[ci]
            w = clause_w[ci]
            if sc == 0:
                # clause currently UNSAT; make if flip makes this lit TRUE
                if lit_true(sign, flip_v):
                    mk += w
            elif sc == 1:
                # clause currently has exactly one satisfier; break if this var is that satisfier and flip kills it
                if lit_true(sign, cur_v) and (not lit_true(sign, flip_v)):
                    br += w
        var_br_w[v_idx] = br
        var_mk_w[v_idx] = mk

    for v in range(nvars):
        update_scores(v)

    # tabu
    tabu = [0] * nvars

    def pick_clause(u: int) -> int:
        # Top-weight focus in hard-core
        if u <= 256:
            K = min(u, 64)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:8] if len(sample) >= 8 else sample
            # small greed bias: often take max, sometimes random among elite
            if rng.random() < 0.15:
                return pool[0]
            return pool[rng.randrange(len(pool))]
        # Otherwise 2-choice weighted
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    def apply_flip(v_idx: int):
        """Flip variable and update sat_count + unsat_list incrementally (SIGNED, CORRECT)."""
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        new_val = a[v_idx]

        affected_clauses: List[int] = []
        # update each clause containing this variable
        for (cj, sign) in var_occ[v_idx]:
            was_true = lit_true(sign, old_val)
            now_true = lit_true(sign, new_val)
            if was_true == now_true:
                continue

            old_sc = sat_count[cj]
            sat_count[cj] = old_sc + (1 if now_true else -1)
            new_sc = sat_count[cj]
            affected_clauses.append(cj)

            if old_sc == 0 and new_sc > 0:
                # remove from unsat_list
                idx = pos[cj]
                last_c = unsat_list[-1]
                unsat_list[idx] = last_c
                pos[last_c] = idx
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                # add to unsat_list
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

        # refresh neighborhood scores (variables in affected clauses + self)
        affected_vars = {v_idx}
        for cj in affected_clauses:
            for lit in clauses[cj]:
                affected_vars.add(abs(lit) - 1)
        for vv in affected_vars:
            update_scores(vv)

    flip = 0

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # update best
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip

        # --- BASIN GUARD (meltdown prevention) ---
        # If we explode far beyond best, we lost the basin -> teleport back.
        if best_unsat is not None and best_unsat >= 1:
            if (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs):
                a = best[:]
                rebuild_state_from_assignment()
                for v in range(nvars):
                    update_scores(v)
                last_improve_flip = flip
                if report_every:
                    print(f"!!! [basin_guard] u={u} -> teleport best={best_unsat}")
                continue

        # --- ELITE SNAPBACK (anti-meltdown, softer) ---
        if best_unsat is not None and u > best_unsat + snapback_gap:
            a = best[:]
            rebuild_state_from_assignment()
            for v in range(nvars):
                update_scores(v)
            last_improve_flip = flip
            continue

        stalled = flip - last_improve_flip

        # --- RESTART SHAKE ---
        if stalled > stall_window:
            # controlled shake: hit heavy clauses, keep weights pressure, refresh neighborhoods
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                uu = len(unsat_list)
                ci_r = pick_clause(uu)
                clause_w[ci_r] = min(w_cap, clause_w[ci_r] + 1.5 * w_inc)

                lits = clauses[ci_r]
                # pick var in clause by local ProbSAT score (avoid tabu if active)
                tabu_active = use_tabu and (uu <= tabu_u_threshold)
                best_v = None
                best_s = -1.0
                for lit in lits:
                    v = abs(lit) - 1
                    if tabu_active and tabu[v] > flip:
                        continue
                    s = ((var_mk_w[v] + eps) ** 2.0) / ((var_br_w[v] + eps) ** 1.1)
                    if s > best_s:
                        best_s, best_v = s, v
                v_r = best_v if best_v is not None else (abs(lits[rng.randrange(len(lits))]) - 1)

                apply_flip(v_r)
                if use_tabu and (len(unsat_list) <= tabu_u_threshold):
                    tabu[v_r] = flip + tabu_tenure
                flip += 1

            last_improve_flip = flip
            continue

        # --- NOISE CONTROL (chirurgical profile) ---
        ratio = u / m
        p_eff = p_base * (0.15 + 1.6 * ratio)   # calmer midgame than 2.2

        # endgame clamps
        if u <= 256:
            p_eff = min(p_eff, 0.10)
        if u <= 128:
            p_eff = min(p_eff, 0.07)
        if u <= 64:
            p_eff = min(p_eff, 0.04)

        # mild stalled ramp, late
        if (stalled > int(stall_window * 0.7)) and (u > 32):
            ramp = (stalled - int(stall_window * 0.7)) / max(1, int(stall_window * 0.3))
            if ramp > 1.0:
                ramp = 1.0
            p_eff += 0.10 * ramp

        if p_eff < p_min:
            p_eff = p_min
        if p_eff > p_max:
            p_eff = p_max

        # --- pick clause ---
        ci = pick_clause(u)

        # bump weights (radioactive under 64)
        bump_val = w_inc * (1.0 + max(0.0, (64.0 - u) / 32.0)) if u < 64 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump_val)

        # refresh scores for vars in clause (cheap local sync after weight change)
        for lit in clauses[ci]:
            update_scores(abs(lit) - 1)

        # --- choose variable (ProbSAT-style) ---
        lits = clauses[ci]
        if u <= 64:
            p_pow, q_pow = 2.6, 1.25
        elif u <= 128:
            p_pow, q_pow = 2.0, 1.10
        else:
            p_pow, q_pow = 1.6, 1.00

        tabu_active = use_tabu and (u <= tabu_u_threshold)

        if rng.random() < p_eff:
            # random pick (still respect tabu if hard-core)
            if tabu_active:
                cand = [abs(l)-1 for l in lits if tabu[abs(l)-1] <= flip]
                v_idx = cand[rng.randrange(len(cand))] if cand else (abs(lits[rng.randrange(len(lits))]) - 1)
            else:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_score = -1.0
            for lit in lits:
                v = abs(lit) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                score = ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)
                score *= (1.0 + 0.01 * (rng.random() - 0.5))
                if score > best_score:
                    best_score, best_v = score, v
            v_idx = best_v if best_v is not None else (abs(lits[rng.randrange(len(lits))]) - 1)

        # --- apply flip ---
        apply_flip(v_idx)
        if tabu_active:
            tabu[v_idx] = flip + tabu_tenure
        flip += 1

        # --- global decay occasionally ---
        if (flip & 0x3FFFF) == 0:
            curr_decay = 0.9999 if u <= 128 else w_decay
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * curr_decay)
            # full refresh of scores (expensive but rare)
            for v in range(nvars):
                update_scores(v)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

# new
from typing import List, Tuple, Dict, Tuple as Tup
import random

def finisher_epic_v6_sole_sat_guard(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_min: float = 0.005,
    p_max: float = 0.18,

    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9995,
    w_cap: float = 120.0,

    # stall / restarts
    stall_window: int = 600_000,
    restart_shake: int = 128,

    # basin guard
    guard_mult: float = 3.0,      # teleport if u > best * guard_mult and u > guard_min_u
    guard_min_u: int = 150,

    # clause focus
    focus_u: int = 256,           # enable top-weight focus under this
    focus_K: int = 64,            # sample size
    focus_pool: int = 8,          # elite pool

    # tabu (optional, light)
    tabu_u: int = 96,
    tabu_tenure: int = 45,

    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Finisher with:
      - exact break using sole_sat[clause] (3-SAT cheap watch-for-break)
      - basin guard teleport (meltdown prevention)
      - restart shake with neighborhood refresh
      - top-weight focus in endgame
      - mild tabu in hard core (u <= tabu_u)
    """
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve_flip = 0

    m = len(clauses)
    eps = 1e-3
    p_base = 0.12  # calmer than 0.15

    # literal truth helper
    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # signed occurrences: var_occ[v] = [(clause_id, sign), ...]
    var_occ: List[List[Tup[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    # state
    sat_count = [0] * m
    sole_sat = [-1] * m       # clause -> var idx if exactly 1 satisfier, else -1

    unsat_list: List[int] = []
    pos = [-1] * m

    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars

    tabu = [0] * nvars

    def recompute_clause_state(ci: int):
        """Recompute sat_count and sole_sat for clause ci (3-SAT -> tiny)."""
        cnt = 0
        last_v = -1
        for lit in clauses[ci]:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if lit_true(s, a[v]):
                cnt += 1
                last_v = v
                if cnt >= 2:
                    sat_count[ci] = cnt
                    sole_sat[ci] = -1
                    return
        sat_count[ci] = cnt
        sole_sat[ci] = last_v if cnt == 1 else -1

    def rebuild_all_from_assignment():
        """Full rebuild unsat_list/pos + sat_count + sole_sat. Used for teleport."""
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            recompute_clause_state(ci)
            if sat_count[ci] == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    def update_scores(v_idx: int):
        """Exact weighted make/break using sole_sat."""
        br = 0.0
        mk = 0.0
        cur_v = a[v_idx]
        flip_v = not cur_v

        for (ci, sign) in var_occ[v_idx]:
            sc = sat_count[ci]
            w = clause_w[ci]

            if sc == 0:
                # make if after flip this literal becomes true
                if lit_true(sign, flip_v):
                    mk += w

            elif sc == 1:
                # break only if v_idx is the sole satisfier and flip kills it
                if sole_sat[ci] == v_idx:
                    if lit_true(sign, cur_v) and (not lit_true(sign, flip_v)):
                        br += w

        var_br_w[v_idx] = br
        var_mk_w[v_idx] = mk

    def refresh_neighborhood_from_clauses(clist: List[int]):
        """Update scores for vars appearing in given clauses."""
        aff = set()
        for ci in clist:
            for lit in clauses[ci]:
                aff.add(abs(lit) - 1)
        for v in aff:
            update_scores(v)

    def apply_flip(v_idx: int) -> List[int]:
        """Flip var and maintain sat_count, sole_sat, unsat_list, pos. Return affected clauses."""
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        new_val = a[v_idx]

        affected_cls: List[int] = []

        for (cj, sign) in var_occ[v_idx]:
            was_true = lit_true(sign, old_val)
            now_true = lit_true(sign, new_val)
            if was_true == now_true:
                continue

            old_sc = sat_count[cj]
            new_sc = old_sc + (1 if now_true else -1)
            sat_count[cj] = new_sc
            affected_cls.append(cj)

            # maintain sole_sat
            if new_sc == 1:
                # find the sole satisfier (scan 3 lits -> cheap)
                one = -1
                for lit in clauses[cj]:
                    v2 = abs(lit) - 1
                    s2 = 1 if lit > 0 else -1
                    if lit_true(s2, a[v2]):
                        one = v2
                        break
                sole_sat[cj] = one
            else:
                sole_sat[cj] = -1

            # maintain unsat_list
            if old_sc == 0 and new_sc > 0:
                idx = pos[cj]
                last_c = unsat_list[-1]
                unsat_list[idx] = last_c
                pos[last_c] = idx
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

        # neighborhood refresh (local)
        aff_vars = {v_idx}
        for cj in affected_cls:
            for lit in clauses[cj]:
                aff_vars.add(abs(lit) - 1)
        for v in aff_vars:
            update_scores(v)

        return affected_cls

    # init state + scores
    rebuild_all_from_assignment()
    for v in range(nvars):
        update_scores(v)

    flip = 0
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # best tracking
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip

        stalled = flip - last_improve_flip

        # basin guard (meltdown prevention)
        if best_unsat is not None and u > max(int(best_unsat * guard_mult), best_unsat + 64) and u > guard_min_u:
            a = best[:]
            rebuild_all_from_assignment()
            for v in range(nvars):
                update_scores(v)
            last_improve_flip = flip
            # keep going (no print spam unless you want)
            # print(f"!!! [basin_guard] u={u} -> teleport to best={best_unsat}")
            continue

        # restart shake if stalled
        if stalled > stall_window:
            # do directed shakes from hard unsats by weight
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                uu = len(unsat_list)

                # weighted 2-choice on unsat clauses
                c1 = unsat_list[rng.randrange(uu)]
                c2 = unsat_list[rng.randrange(uu)]
                ci_r = c1 if clause_w[c1] >= clause_w[c2] else c2

                # bump during shake (pressure memory)
                clause_w[ci_r] = min(w_cap, clause_w[ci_r] + 1.5 * w_inc)

                # choose var in clause by ProbSAT score (no noise in shake)
                lits = clauses[ci_r]
                best_v = abs(lits[0]) - 1
                best_s = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    s = ((var_mk_w[v] + eps) ** 2.0) / ((var_br_w[v] + eps) ** 1.1)
                    if s > best_s:
                        best_s = s
                        best_v = v

                apply_flip(best_v)

            last_improve_flip = flip
            continue

        # --- noise schedule (calm) ---
        ratio = u / m
        p_eff = p_base * (0.15 + 1.6 * ratio)

        # endgame clamps
        if u <= 256: p_eff = min(p_eff, 0.10)
        if u <= 128: p_eff = min(p_eff, 0.07)
        if u <= 64:  p_eff = min(p_eff, 0.04)

        # mild stall ramp (late)
        if stalled > int(stall_window * 0.7) and u > 32:
            ramp = min(1.0, (stalled - int(stall_window * 0.7)) / max(1, int(stall_window * 0.3)))
            p_eff += 0.10 * ramp

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # --- choose clause (top-weight focus) ---
        if u <= focus_u:
            K = min(u, focus_K)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:min(focus_pool, len(sample))]
            # mostly pick from pool, small chance pick the very top
            if rng.random() < 0.15:
                ci = pool[0]
            else:
                ci = pool[rng.randrange(len(pool))]
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            ci = c1 if clause_w[c1] >= clause_w[c2] else c2

        # bump selected clause weight (radioactive near goal)
        bump = w_inc * (1.0 + max(0.0, (64.0 - u) / 32.0)) if u < 64 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump)

        # refresh scores for vars in this clause (keeps selection truthful)
        for lt in clauses[ci]:
            update_scores(abs(lt) - 1)

        # --- variable selection (ProbSAT + optional tabu) ---
        lits = clauses[ci]
        if u <= 64:
            p_pow, q_pow = 2.6, 1.25
        elif u <= 128:
            p_pow, q_pow = 2.0, 1.10
        else:
            p_pow, q_pow = 1.6, 1.00

        tabu_active = (u <= tabu_u)

        def probsat_score(v: int) -> float:
            return ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)

        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            v_idx = None
            best_s = -1.0
            for lt in lits:
                v = abs(lt) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                s = probsat_score(v)
                # tiny jitter to avoid 2-cycles
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    v_idx = v
            if v_idx is None:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1

        # apply flip
        apply_flip(v_idx)
        if tabu_active:
            tabu[v_idx] = flip + tabu_tenure

        flip += 1

        # global decay occasionally
        if (flip & 0x3FFFF) == 0:
            curr_decay = 0.9999 if u <= 128 else w_decay
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * curr_decay)
            # do NOT full update_scores(nvars) every time; too expensive on 10k
            # refresh a few random vars to keep things sane
            for _ in range(256):
                update_scores(rng.randrange(nvars))

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

# dalsi

from typing import List, Tuple, Dict, Tuple as Tup
import random

def finisher_otf_sole_sat_guard(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise (keep small!)
    p_base: float = 0.06,
    p_min: float = 0.002,
    p_max: float = 0.10,

    # clause weights (optional; keep tame)
    use_weights: bool = True,
    w_inc: float = 0.4,
    w_cap: float = 8.0,
    w_decay: float = 0.9998,

    # stall / restarts
    stall_window: int = 600_000,
    restart_shake: int = 64,

    # basin guard (THIS matters)
    guard_add: int = 20,        # teleport if u > best + guard_add
    guard_min_u: int = 120,     # don't spam guard when already tiny

    # focus
    focus_u: int = 256,
    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve = 0

    m = len(clauses)

    def lit_true(lit: int, aval: bool) -> bool:
        return aval if lit > 0 else (not aval)

    # signed occurrences: var_occ[v] = [(clause_id, lit), ...] where lit is signed for that var
    var_occ: List[List[Tup[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_occ[v].append((ci, lit))

    sat_count = [0] * m
    sole_sat = [-1] * m
    pos = [-1] * m
    unsat_list: List[int] = []

    clause_w = [1.0] * m

    def recompute_clause(ci: int):
        cnt = 0
        last_v = -1
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                cnt += 1
                last_v = v
                if cnt >= 2:
                    sat_count[ci] = cnt
                    sole_sat[ci] = -1
                    return
        sat_count[ci] = cnt
        sole_sat[ci] = last_v if cnt == 1 else -1

    def rebuild_all():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            recompute_clause(ci)
            if sat_count[ci] == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    def apply_flip(v: int) -> List[int]:
        old = a[v]
        a[v] = not a[v]
        affected: List[int] = []

        for (ci, lit) in var_occ[v]:
            was = lit_true(lit, old)
            now = lit_true(lit, a[v])
            if was == now:
                continue

            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            sat_count[ci] = new_sc
            affected.append(ci)

            # maintain sole_sat
            if new_sc == 1:
                one = -1
                for lt in clauses[ci]:
                    vv = abs(lt) - 1
                    if lit_true(lt, a[vv]):
                        one = vv
                        break
                sole_sat[ci] = one
            else:
                sole_sat[ci] = -1

            # maintain unsat_list
            if old_sc == 0 and new_sc > 0:
                idx = pos[ci]
                lastc = unsat_list[-1]
                unsat_list[idx] = lastc
                pos[lastc] = idx
                unsat_list.pop()
                pos[ci] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

        return affected

    # On-the-fly make/break for candidate v inside current clause
    def score_candidate(v: int) -> Tup[float, float]:
        """
        returns (make, break) using CURRENT sat_count/sole_sat
        """
        mk = 0.0
        br = 0.0
        cur = a[v]
        flipped = not cur

        for (ci, lit) in var_occ[v]:
            sc = sat_count[ci]
            w = clause_w[ci] if use_weights else 1.0

            # does this var's literal in clause become true/false after flip?
            # lit corresponds to this variable already.
            was_true = lit_true(lit, cur)
            now_true = lit_true(lit, flipped)
            if was_true == now_true:
                continue

            if sc == 0:
                # clause currently UNSAT: make if we turn this literal true
                if now_true:
                    mk += w
            elif sc == 1:
                # clause currently has ONE satisfier: break only if that satisfier is v and we kill it
                if sole_sat[ci] == v and was_true and (not now_true):
                    br += w

        return mk, br

    def pick_clause(u: int) -> int:
        # light focus by weight near endgame
        if u <= focus_u:
            # sample 32 and pick max weight (cheap)
            K = min(u, 32)
            best_ci = unsat_list[rng.randrange(u)]
            best_w = clause_w[best_ci] if use_weights else 1.0
            for _ in range(K - 1):
                ci = unsat_list[rng.randrange(u)]
                w = clause_w[ci] if use_weights else 1.0
                if w > best_w:
                    best_w = w
                    best_ci = ci
            return best_ci
        # otherwise random unsat
        return unsat_list[rng.randrange(u)]

    # init
    rebuild_all()

    flip = 0
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        # HARD basin guard (prevents “prasopes” runs)
        if best_unsat is not None and u > best_unsat + guard_add and u > guard_min_u:
            a = best[:]
            rebuild_all()
            last_improve = flip
            continue

        stalled = flip - last_improve

        # restart shake if totally stuck
        if stalled > stall_window:
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                uu = len(unsat_list)
                ci = pick_clause(uu)
                lits = clauses[ci]
                # pick min-break in shake (no noise)
                best_v = abs(lits[0]) - 1
                best_br = 1e30
                for lt in lits:
                    v = abs(lt) - 1
                    mk, br = score_candidate(v)
                    if br < best_br:
                        best_br = br
                        best_v = v
                if use_weights:
                    clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)
                apply_flip(best_v)
            last_improve = flip
            continue

        # noise schedule (calm; smaller when close)
        ratio = u / m
        p_eff = p_base * (0.25 + 1.2 * ratio)
        if u <= 256: p_eff = min(p_eff, 0.03)
        if u <= 128: p_eff = min(p_eff, 0.02)
        if u <= 64:  p_eff = min(p_eff, 0.01)
        p_eff = float(max(p_min, min(p_max, p_eff)))

        # pick clause
        ci = pick_clause(u)
        if use_weights:
            clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        lits = clauses[ci]

        # endgame mode: min-break (with make as tiebreak)
        if u <= 64:
            best_v = abs(lits[0]) - 1
            best_br = 1e30
            best_mk = -1.0
            for lt in lits:
                v = abs(lt) - 1
                mk, br = score_candidate(v)
                if (br < best_br) or (br == best_br and mk > best_mk):
                    best_br, best_mk, best_v = br, mk, v
            apply_flip(best_v)

        else:
            # midgame: ProbSAT-ish on-the-fly, but with true break/make
            if rng.random() < p_eff:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
                apply_flip(v_idx)
            else:
                # score = (mk+eps)^p / (br+eps)^q ; use gentle exponents
                if u <= 128:
                    p_pow, q_pow = 2.0, 1.1
                else:
                    p_pow, q_pow = 1.6, 1.0

                best_v = abs(lits[0]) - 1
                best_s = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    mk, br = score_candidate(v)
                    s = ((mk + 1e-3) ** p_pow) / ((br + 1e-3) ** q_pow)
                    # tiny jitter
                    s *= (1.0 + 0.01 * (rng.random() - 0.5))
                    if s > best_s:
                        best_s = s
                        best_v = v
                apply_flip(best_v)

        flip += 1

        # weight decay rarely
        if use_weights and (flip & 0x3FFFF) == 0:
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * w_decay)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict


def finisher_predator_ultimate(
        clauses: List[List[int]],
        nvars: int,
        a0: List[bool],
        seed: int = 0,
        max_flips: int = 50_000_000,
        p_min: float = 0.005,
        p_max: float = 0.20,
        stall_window: int = 1_000_000,
        restart_shake: int = 64,
        w_inc: float = 1.0,
        w_decay: float = 0.9995,
        w_cap: float = 100.0,
        report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    NEJLEPŠÍ VERZE:
    - Inkrementální update skóre (O(1) na flip)
    - Sole-satisfier tracking (přesný break model)
    - Micro-Tabu (prolamování cyklů v závěru)
    - Basin Guard (ochrana spektrálního seedu)
    """
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve = 0
    m = len(clauses)

    # --- INTERNAL STATE ---
    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars
    tabu = [0] * nvars
    TABU_TENURE = 35  # Pro 10k proměnných ideální

    # Signed occurrences
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_occ[v].append((ci, lit))

    def lit_true(lit: int, val: bool) -> bool:
        return val if lit > 0 else (not val)

    def update_var_scores(v: int):
        mk, br = 0.0, 0.0
        cur_v = a[v]
        for ci, lit in var_occ[v]:
            sc, w = sat_count[ci], clause_w[ci]
            if sc == 0:
                if lit_true(lit, not cur_v): mk += w
            elif sc == 1:
                if sole_sat[ci] == v: br += w
        var_mk_w[v], var_br_w[v] = mk, br

    def rebuild_state():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            cnt, last_v = 0, -1
            for lit in clauses[ci]:
                if lit_true(lit, a[abs(lit) - 1]):
                    cnt += 1;
                    last_v = abs(lit) - 1
            sat_count[ci] = cnt
            sole_sat[ci] = last_v if cnt == 1 else -1
            if cnt == 0:
                pos[ci] = len(unsat_list);
                unsat_list.append(ci)
        for v in range(nvars): update_var_scores(v)

    rebuild_state()
    flip = 0
    eps = 1e-3

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0: return a, True, {"flips": flip, "best_unsat": 0}

        if best_unsat is None or u < best_unsat:
            best_unsat, best, last_improve = u, a[:], flip

        # --- BASIN GUARD ---
        if best_unsat is not None and u > best_unsat * 3 and u > 150:
            a = best[:];
            rebuild_state();
            continue

        stalled = flip - last_improve

        # --- RESTART SHAKE ---
        if stalled > stall_window:
            for _ in range(restart_shake):
                if not unsat_list: break
                ci = unsat_list[rng.randrange(len(unsat_list))]
                v = abs(clauses[ci][rng.randrange(3)]) - 1
                # apply flip
                a[v] = not a[v]
                # v 3-SAT shake stačí rebuild nebo lokální update
            rebuild_state();
            last_improve = flip;
            continue

        # --- NOISE & TABU LOGIC ---
        ratio = u / m
        p_eff = 0.15 * (0.2 + 1.5 * ratio)
        tabu_active = (u <= 128)
        if u <= 256: p_eff = min(p_eff, 0.10)
        if u <= 64:  p_eff = min(p_eff, 0.05)

        # --- PICK CLAUSE & VARIABLE ---
        # 2-choice weighting for clause
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        ci = c1 if clause_w[c1] >= clause_w[c2] else c2

        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)
        lits = clauses[ci]

        if rng.random() < max(p_min, p_eff):
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            p_pow, q_pow = (2.6, 1.3) if u <= 64 else (1.8, 1.1)
            best_v, best_s = -1, -1.0
            for lit in lits:
                v = abs(lit) - 1
                if tabu_active and tabu[v] > flip: continue
                s = ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s: best_s, best_v = s, v
            v_idx = best_v if best_v != -1 else abs(lits[rng.randrange(len(lits))]) - 1

        # --- THE CORE: INCREMENTAL FLIP ---
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        if tabu_active: tabu[v_idx] = flip + TABU_TENURE

        affected_vars = {v_idx}
        for ci_aff, lit_aff in var_occ[v_idx]:
            was_true = lit_true(lit_aff, old_val)
            now_true = not was_true

            old_sc = sat_count[ci_aff]
            new_sc = old_sc + (1 if now_true else -1)
            sat_count[ci_aff] = new_sc

            # Update sole_sat
            if new_sc == 1:
                for l in clauses[ci_aff]:
                    if lit_true(l, a[abs(l) - 1]):
                        sole_sat[ci_aff] = abs(l) - 1;
                        break
            else:
                sole_sat[ci_aff] = -1

            # Update unsat_list
            if old_sc == 0 and new_sc > 0:
                idx = pos[ci_aff];
                last_c = unsat_list[-1]
                unsat_list[idx] = last_c;
                pos[last_c] = idx
                unsat_list.pop();
                pos[ci_aff] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[ci_aff] = len(unsat_list);
                unsat_list.append(ci_aff)

            # All vars in affected clauses need score refresh
            for l in clauses[ci_aff]: affected_vars.add(abs(l) - 1)

        for v in affected_vars: update_var_scores(v)

        flip += 1
        if (flip & 0x7FFFF) == 0:  # Decay
            for i in range(m): clause_w[i] = max(1.0, clause_w[i] * w_decay)
            for v in range(nvars): update_var_scores(v)

        if report_every and flip % report_every == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def finisher_predator_fixed(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise (keep sane)
    p_min: float = 0.002,
    p_max: float = 0.08,

    # stall / restarts
    stall_window: int = 800_000,
    restart_shake: int = 96,

    # clause weights (ONLY for clause selection pressure)
    w_inc: float = 0.8,
    w_decay: float = 0.9997,
    w_cap: float = 40.0,

    # basin guard
    guard_add: int = 40,     # teleport if u > best + guard_add
    guard_min_u: int = 140,  # don't spam when already low

    # tabu
    tabu_tenure: int = 45,   # good for 10k vars
    tabu_u: int = 128,

    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Fixed predator:
    - Correct break logic using sole_sat + literal truth flip check
    - Clause weights used ONLY to pick which UNSAT clause to attack (not inside var scores)
    - Incremental sat_count + sole_sat + unsat_list
    - Micro tabu in hard core
    - Basin guard teleport
    """
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve = 0
    m = len(clauses)

    # signed occurrences: var_occ[v] = [(clause_id, signed_lit_for_v), ...]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_occ[v].append((ci, lit))

    def lit_true(lit: int, val: bool) -> bool:
        return val if lit > 0 else (not val)

    sat_count = [0] * m
    sole_sat = [-1] * m

    unsat_list: List[int] = []
    pos = [-1] * m

    clause_w = [1.0] * m

    var_br = [0.0] * nvars
    var_mk = [0.0] * nvars

    tabu = [0] * nvars
    eps = 1e-3

    def recompute_clause(ci: int):
        cnt = 0
        last_v = -1
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                cnt += 1
                last_v = v
        sat_count[ci] = cnt
        sole_sat[ci] = last_v if cnt == 1 else -1

    def rebuild_state():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            recompute_clause(ci)
            if sat_count[ci] == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

        # rebuild var scores (O(total occurrences), OK on rebuild only)
        for v in range(nvars):
            update_var_scores(v)

    def update_var_scores(v: int):
        """
        UNWEIGHTED exact make/break using sole_sat and literal flip check.
        (weights are NOT used here on purpose – avoids stale-weight drift)
        """
        mk = 0.0
        br = 0.0
        cur = a[v]
        flipped = not cur

        for (ci, lit) in var_occ[v]:
            sc = sat_count[ci]

            was = lit_true(lit, cur)
            now = lit_true(lit, flipped)
            if was == now:
                continue

            if sc == 0:
                if now:
                    mk += 1.0
            elif sc == 1:
                # break only if v is the sole satisfier AND flip kills THIS literal
                if sole_sat[ci] == v and was and (not now):
                    br += 1.0

        var_mk[v] = mk
        var_br[v] = br

    def apply_flip(v: int) -> List[int]:
        old = a[v]
        a[v] = not a[v]
        affected_clauses: List[int] = []

        for (ci, lit) in var_occ[v]:
            was = lit_true(lit, old)
            now = lit_true(lit, a[v])
            if was == now:
                continue

            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            sat_count[ci] = new_sc
            affected_clauses.append(ci)

            # maintain sole_sat
            if new_sc == 1:
                one = -1
                for lt in clauses[ci]:
                    vv = abs(lt) - 1
                    if lit_true(lt, a[vv]):
                        one = vv
                        break
                sole_sat[ci] = one
            else:
                sole_sat[ci] = -1

            # maintain unsat_list
            if old_sc == 0 and new_sc > 0:
                idx = pos[ci]
                lastc = unsat_list[-1]
                unsat_list[idx] = lastc
                pos[lastc] = idx
                unsat_list.pop()
                pos[ci] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

        return affected_clauses

    def pick_unsat_clause(u: int) -> int:
        # 2-choice weighted (safe; clause_w does not affect var scores)
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    # init
    rebuild_state()

    flip = 0
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        # basin guard (additive, not multiplicative)
        if best_unsat is not None and u > best_unsat + guard_add and u > guard_min_u:
            a = best[:]
            rebuild_state()
            last_improve = flip
            continue

        stalled = flip - last_improve

        # restart shake
        if stalled > stall_window:
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                uu = len(unsat_list)
                ci = pick_unsat_clause(uu)
                lits = clauses[ci]

                # pick min-break (tie: max-make) – deterministic shake
                best_v = abs(lits[0]) - 1
                best_br = 1e30
                best_mk = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    br = var_br[v]
                    mk = var_mk[v]
                    if (br < best_br) or (br == best_br and mk > best_mk):
                        best_br, best_mk, best_v = br, mk, v

                clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)
                aff_cls = apply_flip(best_v)

                # refresh scores locally
                aff_vars = {best_v}
                for cj in aff_cls:
                    for lt in clauses[cj]:
                        aff_vars.add(abs(lt) - 1)
                for vv in aff_vars:
                    update_var_scores(vv)

            last_improve = flip
            continue

        # calm noise (never let it explode)
        ratio = u / m
        p_eff = 0.03 + 0.20 * ratio
        if u <= 256: p_eff = min(p_eff, 0.02)
        if u <= 128: p_eff = min(p_eff, 0.012)
        if u <= 64:  p_eff = min(p_eff, 0.006)
        p_eff = float(max(p_min, min(p_max, p_eff)))

        # pick clause
        ci = pick_unsat_clause(u)
        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        lits = clauses[ci]
        tabu_active = (u <= tabu_u)

        # choose variable
        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            # ProbSAT-ish score using PRECOMPUTED mk/br (now correct!)
            if u <= 64:
                p_pow, q_pow = 2.6, 1.3
            elif u <= 128:
                p_pow, q_pow = 2.0, 1.1
            else:
                p_pow, q_pow = 1.6, 1.0

            best_v = None
            best_s = -1.0
            for lt in lits:
                v = abs(lt) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                s = ((var_mk[v] + eps) ** p_pow) / ((var_br[v] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s, best_v = s, v

            v_idx = best_v if best_v is not None else abs(lits[rng.randrange(len(lits))]) - 1

        # apply flip
        if tabu_active:
            tabu[v_idx] = flip + tabu_tenure

        aff_cls = apply_flip(v_idx)

        # local score refresh
        aff_vars = {v_idx}
        for cj in aff_cls:
            for lt in clauses[cj]:
                aff_vars.add(abs(lt) - 1)
        for vv in aff_vars:
            update_var_scores(vv)

        flip += 1

        # decay weights rarely (no score dependency -> safe)
        if (flip & 0x7FFFF) == 0:
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * w_decay)

        if report_every and flip % report_every == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
from typing import List, Tuple, Dict
import random

def finisher_epic_vNext_predator(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    p_min: float = 0.02,
    p_max: float = 0.35,
    stall_window: int = 500_000,
    restart_shake: int = 64,
    w_inc: float = 1.0,
    w_decay: float = 0.9995,
    w_cap: float = 100.0,
    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:

    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve_flip = 0
    m = len(clauses)

    # --- TABU ---
    tabu = [0] * nvars
    TABU_TENURE = 32

    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # var_occ[v] = [(clause_index, sign), ...]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_occ[v].append((ci, 1 if lit > 0 else -1))

    # sat_count
    sat_count = [0] * m
    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            if lit_true(1 if lit > 0 else -1, a[v]):
                cnt += 1
        sat_count[ci] = cnt

    # unsat_list + pos
    unsat_list: List[int] = []
    pos = [-1] * m
    for ci in range(m):
        if sat_count[ci] == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # weights + scores
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars

    eps = 1e-3
    p_base = 0.15
    flip = 0

    def update_scores(v_idx: int):
        br = 0.0
        mk = 0.0
        cur_v = a[v_idx]
        flip_v = not cur_v
        for (ci_idx, sign) in var_occ[v_idx]:
            sc = sat_count[ci_idx]
            w = clause_w[ci_idx]
            if sc == 0:
                if lit_true(sign, flip_v):
                    mk += w
            elif sc == 1:
                if lit_true(sign, cur_v) and (not lit_true(sign, flip_v)):
                    br += w
        var_br_w[v_idx] = br
        var_mk_w[v_idx] = mk

    def rebuild_all_scores():
        for v in range(nvars):
            update_scores(v)

    rebuild_all_scores()

    def rebuild_state_from_assignment(assign: List[bool]):
        # full resync: sat_count, unsat_list, pos, and scores
        nonlocal a, best, best_unsat, last_improve_flip
        a = assign[:]

        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1

        for ci, c in enumerate(clauses):
            cnt = 0
            for lit in c:
                v = abs(lit) - 1
                if lit_true(1 if lit > 0 else -1, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

        rebuild_all_scores()

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # best tracking
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip

        # elite snapback (anti-meltdown)
        if best_unsat is not None and u > best_unsat + 64:
            rebuild_state_from_assignment(best)
            last_improve_flip = flip
            continue

        stalled = flip - last_improve_flip

        # restart shake
        if stalled > stall_window:
            for _ in range(restart_shake):
                uu = len(unsat_list)
                if uu == 0:
                    break

                # weighted 2-choice on UNSAT clauses
                c1 = unsat_list[rng.randrange(uu)]
                c2 = unsat_list[rng.randrange(uu)]
                ci_r = c1 if clause_w[c1] >= clause_w[c2] else c2

                # bump during shake (pressure)
                clause_w[ci_r] = min(w_cap, clause_w[ci_r] + 1.5 * w_inc)

                # pick variable in clause (min-break-ish via current scores)
                lits_r = clauses[ci_r]
                best_v = None
                best_score = -1.0
                for lt in lits_r:
                    v = abs(lt) - 1
                    score = ((var_mk_w[v] + eps) ** 2.0) / ((var_br_w[v] + eps) ** 1.1)
                    if score > best_score:
                        best_score = score
                        best_v = v
                v_r = best_v if best_v is not None else abs(lits_r[0]) - 1

                old_v = a[v_r]
                a[v_r] = not a[v_r]
                new_v = a[v_r]

                affected_clauses = []
                for (cj, sign) in var_occ[v_r]:
                    wt = lit_true(sign, old_v)
                    nt = lit_true(sign, new_v)
                    if wt == nt:
                        continue
                    old_sc = sat_count[cj]
                    sat_count[cj] += (1 if nt else -1)
                    new_sc = sat_count[cj]
                    affected_clauses.append(cj)

                    if old_sc == 0 and new_sc > 0:
                        idx = pos[cj]
                        last_c = unsat_list[-1]
                        unsat_list[idx] = last_c
                        pos[last_c] = idx
                        unsat_list.pop()
                        pos[cj] = -1
                    elif old_sc > 0 and new_sc == 0:
                        pos[cj] = len(unsat_list)
                        unsat_list.append(cj)

                # refresh neighborhood scores
                affected_vars = {v_r}
                for cj in affected_clauses:
                    for lt in clauses[cj]:
                        affected_vars.add(abs(lt) - 1)
                for vv in affected_vars:
                    update_scores(vv)

            last_improve_flip = flip
            continue

        # noise control
        ratio = u / m
        p_eff = p_base * (0.35 + 2.2 * ratio)
        if stalled > stall_window // 2 and u > 32:
            p_eff += 0.20 * min(1.0, (stalled - stall_window // 2) / max(1, stall_window // 2))

        # micro-tabu only in hard core
        tabu_active = (u <= 96)

        # clamp noise in endgame
        if u <= 256:
            p_eff = min(p_eff, 0.12)
        elif u <= 512:
            p_eff = min(p_eff, 0.18)

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # clause pick: top-weight focus (u<=256), else 2-choice weighted
        if u <= 256:
            K = min(u, 64)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:8]
            ci = pool[rng.randrange(len(pool))]
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            ci = c1 if clause_w[c1] >= clause_w[c2] else c2

        # bump & refresh clause vars
        bump_val = w_inc * (1.0 + max(0.0, (64.0 - u) / 32.0)) if u < 64 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump_val)
        for lt in clauses[ci]:
            update_scores(abs(lt) - 1)

        # variable selection with tabu
        lits = clauses[ci]
        p_pow, q_pow = (2.6, 1.25) if u <= 64 else ((2.0, 1.10) if u <= 128 else (1.6, 1.00))

        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_s = -1.0
            for lit in lits:
                v = abs(lit) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                s = ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    best_v = v
            v_idx = best_v if best_v is not None else abs(lits[rng.randrange(len(lits))]) - 1

        # apply flip
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        flip += 1

        # set tabu AFTER flip increment
        if tabu_active:
            tabu[v_idx] = flip + TABU_TENURE

        affected_vars = {v_idx}
        affected_clauses = []

        for (cj, sign) in var_occ[v_idx]:
            wt = lit_true(sign, old_val)
            nt = lit_true(sign, a[v_idx])
            if wt == nt:
                continue
            old_sc = sat_count[cj]
            sat_count[cj] += (1 if nt else -1)
            new_sc = sat_count[cj]
            affected_clauses.append(cj)

            if old_sc == 0 and new_sc > 0:
                idx = pos[cj]
                last_c = unsat_list[-1]
                unsat_list[idx] = last_c
                pos[last_c] = idx
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

            for lt in clauses[cj]:
                affected_vars.add(abs(lt) - 1)

        # refresh neighborhood
        for vv in affected_vars:
            update_scores(vv)

        # periodic decay (CRITICAL: must refresh scores after decay)
        if (flip & 0x3FFFF) == 0:
            curr_d = 0.9999 if u <= 128 else w_decay
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * curr_d)

            # MUST resync scores with new clause weights
            rebuild_all_scores()

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#

import random
from typing import List, Tuple, Dict

def finisher_predator_sole_sat(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise (držet nízko)
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,

    # restart / stall
    stall_window: int = 600_000,
    restart_shake: int = 96,

    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9996,
    w_cap: float = 120.0,

    # guards
    snapback_gap: int = 48,      # když u uteče o tolik nad best -> snapback
    basin_mult: float = 2.2,     # meltdown pokud u > best * basin_mult
    basin_abs: int = 220,        # meltdown guard aktivní až když u fakt ulítne

    # tabu (jen v hard-core)
    use_tabu: bool = True,
    tabu_u_threshold: int = 128,
    tabu_tenure: int = 45,

    report_every: int = 100_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Stabilní finisher pro 3-SAT:
    - signed occurrences
    - sat_count + sole_sat (přesný break)
    - váhy klauzulí (core pressure)
    - inkrementální update skóre jen v affected neighborhood
    - micro-tabu v hard-core
    - snapback + basin-guard (ochrana spektrálního seedu)
    """

    rng = random.Random(seed)
    a = a0[:]
    m = len(clauses)

    # --- signed occurrences: var_occ[v] = [(clause_id, sign), ...]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            sgn = 1 if lit > 0 else -1
            var_occ[v].append((ci, sgn))

    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # --- clause state
    sat_count = [0] * m
    sole_sat = [-1] * m  # pokud sat_count==1, obsahuje var index jediného satisfieru

    # --- UNSAT list as packed array + positions
    unsat_list: List[int] = []
    pos = [-1] * m

    def add_unsat(ci: int) -> None:
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def remove_unsat(ci: int) -> None:
        i = pos[ci]
        if i == -1:
            return
        last = unsat_list[-1]
        unsat_list[i] = last
        pos[last] = i
        unsat_list.pop()
        pos[ci] = -1

    # --- weights + var scores
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars
    eps = 1e-3

    # init clause sat_count + sole_sat + unsat
    def init_clause(ci: int) -> None:
        cnt = 0
        last_v = -1
        for lit in clauses[ci]:
            v = abs(lit) - 1
            sgn = 1 if lit > 0 else -1
            if lit_true(sgn, a[v]):
                cnt += 1
                last_v = v
        sat_count[ci] = cnt
        sole_sat[ci] = last_v if cnt == 1 else -1
        if cnt == 0:
            add_unsat(ci)

    unsat_list.clear()
    for i in range(m):
        pos[i] = -1
    for ci in range(m):
        init_clause(ci)

    # recompute sole_sat for clause (3-SAT => max 3 lits => cheap)
    def recompute_sole(ci: int) -> None:
        if sat_count[ci] != 1:
            sole_sat[ci] = -1
            return
        for lit in clauses[ci]:
            v = abs(lit) - 1
            sgn = 1 if lit > 0 else -1
            if lit_true(sgn, a[v]):
                sole_sat[ci] = v
                return
        # fallback (nemělo by nastat)
        sole_sat[ci] = -1

    # exact mk/br using sole_sat
    def update_scores(v: int) -> None:
        cur = a[v]
        flipv = not cur
        mk = 0.0
        br = 0.0
        for (ci, sgn) in var_occ[v]:
            sc = sat_count[ci]
            w = clause_w[ci]
            if sc == 0:
                # clause UNSAT -> make pokud flip udělá lit TRUE
                if lit_true(sgn, flipv):
                    mk += w
            elif sc == 1:
                # break pouze pokud v je JEDINÝ satisfier
                if sole_sat[ci] == v:
                    # jestli flip zabije jediný TRUE lit
                    if lit_true(sgn, cur) and (not lit_true(sgn, flipv)):
                        br += w
        var_mk_w[v] = mk
        var_br_w[v] = br

    for v in range(nvars):
        update_scores(v)

    tabu = [0] * nvars

    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0

    def rebuild_from_best():
        nonlocal a
        a = best[:]
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1
        for ci in range(m):
            sat_count[ci] = 0
            sole_sat[ci] = -1
        for ci in range(m):
            init_clause(ci)
        for v in range(nvars):
            update_scores(v)

    # clause pick: core-focus (u<=1024) + 2-choice weight
    def pick_clause(u: int) -> int:
        if u <= 1024:
            K = min(u, 96)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:12] if len(sample) >= 12 else sample
            c1 = pool[rng.randrange(len(pool))]
            c2 = pool[rng.randrange(len(pool))]
            return c1 if clause_w[c1] >= clause_w[c2] else c2
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    # incremental flip: update sat_count, sole_sat, unsat_list, and collect affected vars
    def apply_flip(v: int) -> List[int]:
        old_val = a[v]
        a[v] = not a[v]
        new_val = a[v]

        affected_clauses: List[int] = []
        affected_vars = {v}

        for (ci, sgn) in var_occ[v]:
            was_true = lit_true(sgn, old_val)
            now_true = lit_true(sgn, new_val)
            if was_true == now_true:
                continue

            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now_true else -1)
            sat_count[ci] = new_sc
            affected_clauses.append(ci)

            # UNSAT maintenance
            if old_sc == 0 and new_sc > 0:
                remove_unsat(ci)
            elif old_sc > 0 and new_sc == 0:
                add_unsat(ci)

            # sole_sat maintenance
            if new_sc == 1:
                recompute_sole(ci)
            else:
                sole_sat[ci] = -1

            # neighborhood vars
            for lit in clauses[ci]:
                affected_vars.add(abs(lit) - 1)

        # refresh scores in affected neighborhood
        for vv in affected_vars:
            update_scores(vv)

        return affected_clauses

    flip = 0
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        # --- guards: basin + snapback ---
        if best_unsat > 0:
            if (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs):
                rebuild_from_best()
                last_improve = flip
                continue
            """if u > best_unsat + snapback_gap:
                rebuild_from_best()
                last_improve = flip
                continue"""
            if (u > best_unsat + snapback_gap) and (u > basin_abs):
                rebuild_from_best()
                last_improve = flip
                continue

        stalled = flip - last_improve

        # --- restart shake (kontrolovaně, bez full chaosu) ---
        if stalled >= stall_window:
            for _ in range(restart_shake):
                uu = len(unsat_list)
                if uu == 0:
                    return a, True, {"flips": flip, "best_unsat": 0}
                ci = pick_clause(uu)
                clause_w[ci] = min(w_cap, clause_w[ci] + 1.5 * w_inc)

                lits = clauses[ci]
                # pick var by score (bez tabu jen v shake; tabu hlídáme až v mainu)
                bv = abs(lits[0]) - 1
                bs = -1.0
                for lit in lits:
                    v = abs(lit) - 1
                    s = ((var_mk_w[v] + eps) ** 2.0) / ((var_br_w[v] + eps) ** 1.2)
                    if s > bs:
                        bs, bv = s, v

                apply_flip(bv)
                flip += 1
            last_improve = flip
            continue

        # --- noise: klidný profil ---
        ratio = u / max(1, m)
        p_eff = p_base * (0.20 + 1.4 * ratio)

        # endgame clamps (čím blíž, tím méně random)
        if u <= 512: p_eff = min(p_eff, 0.08)
        if u <= 256: p_eff = min(p_eff, 0.06)
        if u <= 128: p_eff = min(p_eff, 0.04)
        if u <= 64:  p_eff = min(p_eff, 0.025)

        # stalled ramp jemně (nepřepálit)
        if stalled > int(stall_window * 0.7) and u > 64:
            ramp = (stalled - int(stall_window * 0.7)) / max(1, int(stall_window * 0.3))
            if ramp > 1.0: ramp = 1.0
            p_eff = min(p_max, p_eff + 0.06 * ramp)

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # --- pick clause (core-focus) ---
        ci = pick_clause(u)

        # bump weight (silnější v hard-core)
        bump = w_inc * (1.0 + max(0.0, (256.0 - u) / 256.0)) if u < 256 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump)

        # lokální refresh po bumpu (jen 3 proměnné)
        for lit in clauses[ci]:
            update_scores(abs(lit) - 1)

        # --- select variable ---
        lits = clauses[ci]
        tabu_active = use_tabu and (u <= tabu_u_threshold)

        # exponents: agresivní až v endgame
        if u <= 64:
            p_pow, q_pow = 2.8, 1.35
        elif u <= 256:
            p_pow, q_pow = 2.1, 1.15
        else:
            p_pow, q_pow = 1.7, 1.05

        if rng.random() < p_eff:
            # random within clause, but respect tabu if active
            if tabu_active:
                cand = [abs(l)-1 for l in lits if tabu[abs(l)-1] <= flip]
                v_idx = cand[rng.randrange(len(cand))] if cand else (abs(lits[rng.randrange(len(lits))]) - 1)
            else:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_s = -1.0
            for lit in lits:
                v = abs(lit) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                s = ((var_mk_w[v] + eps) ** p_pow) / ((var_br_w[v] + eps) ** q_pow)
                # tiny jitter to avoid 2-cycles
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s, best_v = s, v
            v_idx = best_v if best_v is not None else (abs(lits[rng.randrange(len(lits))]) - 1)

        # --- flip ---
        apply_flip(v_idx)
        if tabu_active:
            tabu[v_idx] = flip + tabu_tenure
        flip += 1

        # --- decay (vzácně) ---
        if (flip & 0x3FFFF) == 0:
            d = 0.99985 if u <= 512 else w_decay
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * d)
            # po decay přepočítat skóre všech varů (bezpečné, ale vzácné)
            for v in range(nvars):
                update_scores(v)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict, Tuple as Tup

def finisher_epic_vNext_predator_fixed(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_base: float = 0.12,
    p_min: float = 0.002,
    p_max: float = 0.18,

    # stall / restarts
    stall_window: int = 600_000,
    restart_shake: int = 96,

    # clause weights (ONLY for clause selection pressure)
    w_inc: float = 1.2,
    w_decay: float = 0.9996,
    w_cap: float = 80.0,

    # basin guard (simple + safe)
    guard_mult: float = 3.0,   # teleport if u > best*guard_mult
    guard_min_u: int = 150,    # ignore guard when already low-ish

    # focus
    focus_u: int = 256,
    focus_K: int = 64,
    focus_pool: int = 8,

    # micro tabu (only in hard core)
    use_tabu: bool = True,
    tabu_u: int = 96,
    tabu_tenure: int = 45,

    report_every: int = 100_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Fixed, stable finisher for 3-SAT:
      - exact break via sole_sat
      - make/break UNWEIGHTED (immune to clause weight decay staleness)
      - clause weights used ONLY to pick which UNSAT clause to attack (pressure memory)
      - incremental sat_count + sole_sat + unsat_list maintenance
      - local neighborhood score refresh
      - optional micro-tabu in hard core
      - basin guard teleport
    """

    rng = random.Random(seed)

    # normalize a0 (accept 0/1 as well)
    a = [bool(x) for x in a0]
    best = a[:]
    best_unsat = None
    last_improve_flip = 0
    m = len(clauses)
    eps = 1e-3

    # --- helpers ---
    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # signed occurrences: var_occ[v] = [(clause_id, sign_in_that_clause), ...]
    var_occ: List[List[Tup[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    # --- state ---
    sat_count = [0] * m
    sole_sat = [-1] * m        # clause -> var idx if exactly 1 satisfier, else -1

    unsat_list: List[int] = []
    pos = [-1] * m

    clause_w = [1.0] * m       # only for clause selection

    # UNWEIGHTED var scores (counts of clauses)
    var_br = [0.0] * nvars
    var_mk = [0.0] * nvars

    tabu = [0] * nvars

    def recompute_clause_state(ci: int):
        """3-SAT fast recompute sat_count + sole_sat for clause ci."""
        cnt = 0
        last_v = -1
        for lit in clauses[ci]:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if lit_true(s, a[v]):
                cnt += 1
                last_v = v
                if cnt >= 2:
                    sat_count[ci] = cnt
                    sole_sat[ci] = -1
                    return
        sat_count[ci] = cnt
        sole_sat[ci] = last_v if cnt == 1 else -1

    def rebuild_all_from_assignment():
        """Full rebuild of clause states + unsat_list (used by teleport)."""
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            recompute_clause_state(ci)
            if sat_count[ci] == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    def update_scores(v: int):
        """Exact UNWEIGHTED make/break for variable v using sole_sat."""
        mk = 0.0
        br = 0.0
        cur = a[v]
        flipped = not cur
        for (ci, sign) in var_occ[v]:
            sc = sat_count[ci]

            was = lit_true(sign, cur)
            now = lit_true(sign, flipped)
            if was == now:
                continue

            if sc == 0:
                if now:
                    mk += 1.0
            elif sc == 1:
                # break only if v is the sole satisfier AND flip kills it
                if sole_sat[ci] == v and was and (not now):
                    br += 1.0

        var_mk[v] = mk
        var_br[v] = br

    def refresh_neighborhood(clauses_affected: List[int], v_self: int):
        """Refresh scores for variables in affected clauses + the flipped var."""
        aff_vars = {v_self}
        for cj in clauses_affected:
            for lit in clauses[cj]:
                aff_vars.add(abs(lit) - 1)
        for vv in aff_vars:
            update_scores(vv)

    def apply_flip(v: int) -> List[int]:
        """Incremental flip with sat_count + sole_sat + unsat_list maintenance."""
        old = a[v]
        a[v] = not a[v]
        affected_cls: List[int] = []

        for (ci, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = lit_true(sign, a[v])
            if was == now:
                continue

            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            sat_count[ci] = new_sc
            affected_cls.append(ci)

            # maintain sole_sat for this clause (3 literals -> cheap)
            if new_sc == 1:
                one = -1
                for lit in clauses[ci]:
                    vv = abs(lit) - 1
                    ss = 1 if lit > 0 else -1
                    if lit_true(ss, a[vv]):
                        one = vv
                        break
                sole_sat[ci] = one
            else:
                sole_sat[ci] = -1

            # maintain unsat_list
            if old_sc == 0 and new_sc > 0:
                idx = pos[ci]
                lastc = unsat_list[-1]
                unsat_list[idx] = lastc
                pos[lastc] = idx
                unsat_list.pop()
                pos[ci] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

        return affected_cls

    def pick_clause(u: int) -> int:
        """Top-weight focus under focus_u, else weighted 2-choice."""
        if u <= focus_u:
            K = min(u, focus_K)
            sample = [unsat_list[rng.randrange(u)] for _ in range(K)]
            sample.sort(key=lambda c: clause_w[c], reverse=True)
            pool = sample[:min(focus_pool, len(sample))]
            # mostly pick from pool, sometimes pick the very top
            if rng.random() < 0.20:
                return pool[0]
            return pool[rng.randrange(len(pool))]
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            return c1 if clause_w[c1] >= clause_w[c2] else c2

    # --- init ---
    rebuild_all_from_assignment()
    for v in range(nvars):
        update_scores(v)

    flip = 0
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # best tracking
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip

        stalled = flip - last_improve_flip

        # basin guard teleport (simple, avoids “runaway”)
        if best_unsat is not None and u > int(best_unsat * guard_mult) and u > guard_min_u:
            a = best[:]
            rebuild_all_from_assignment()
            for v in range(nvars):
                update_scores(v)
            last_improve_flip = flip
            continue

        # restart shake when stalled (directed, but safe)
        if stalled > stall_window:
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                uu = len(unsat_list)
                ci_r = pick_clause(uu)
                clause_w[ci_r] = min(w_cap, clause_w[ci_r] + 1.5 * w_inc)

                # choose var in clause by ProbSAT score (no noise in shake)
                lits = clauses[ci_r]
                tabu_active = use_tabu and (uu <= tabu_u)
                best_v = abs(lits[0]) - 1
                best_s = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    if tabu_active and tabu[v] > flip:
                        continue
                    s = ((var_mk[v] + eps) ** 2.0) / ((var_br[v] + eps) ** 1.1)
                    if s > best_s:
                        best_s, best_v = s, v

                aff = apply_flip(best_v)
                if use_tabu and (len(unsat_list) <= tabu_u):
                    tabu[best_v] = flip + tabu_tenure
                refresh_neighborhood(aff, best_v)
                flip += 1

            last_improve_flip = flip
            continue

        # noise schedule (calm, no ramp madness)
        """ratio = u / m
        p_eff = p_base * (0.15 + 1.6 * ratio)
        if u <= 256: p_eff = min(p_eff, 0.10)
        if u <= 128: p_eff = min(p_eff, 0.07)
        if u <= 64:  p_eff = min(p_eff, 0.04)
        p_eff = float(max(p_min, min(p_max, p_eff)))"""

        ratio = u / m

        # víc citlivé při malém ratio (sqrt), plus floor pro mid/endgame
        p_eff = p_base * (0.10 + 2.8 * math.sqrt(max(1e-12, ratio)))

        # floor: když už jsi nízko, občas potřebuješ "odskok"
        if u <= 512: p_eff = max(p_eff, 0.030)
        if u <= 256: p_eff = max(p_eff, 0.025)
        if u <= 128: p_eff = max(p_eff, 0.020)
        if u <= 64:  p_eff = max(p_eff, 0.015)

        # cap (ať se to nezmění v ruletu)
        if u <= 256: p_eff = min(p_eff, 0.12)
        if u <= 128: p_eff = min(p_eff, 0.09)
        if u <= 64:  p_eff = min(p_eff, 0.06)

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # pick clause + bump weight
        ci = pick_clause(u)
        bump = w_inc * (1.0 + max(0.0, (64.0 - u) / 32.0)) if u < 64 else w_inc
        clause_w[ci] = min(w_cap, clause_w[ci] + bump)

        # choose variable (ProbSAT, with tabu in hard core)
        lits = clauses[ci]
        tabu_active = use_tabu and (u <= tabu_u)
        p_pow, q_pow = (2.6, 1.25) if u <= 64 else ((2.0, 1.10) if u <= 128 else (1.6, 1.00))

        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_s = -1.0
            for lt in lits:
                v = abs(lt) - 1
                if tabu_active and tabu[v] > flip:
                    continue
                s = ((var_mk[v] + eps) ** p_pow) / ((var_br[v] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))  # tiny jitter
                if s > best_s:
                    best_s, best_v = s, v
            v_idx = best_v if best_v is not None else abs(lits[rng.randrange(len(lits))]) - 1

        # apply flip
        aff = apply_flip(v_idx)
        if tabu_active:
            tabu[v_idx] = flip + tabu_tenure
        refresh_neighborhood(aff, v_idx)

        flip += 1

        # weight decay rarely (safe because scores are unweighted)
        if (flip & 0x3FFFF) == 0:
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * w_decay)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def finisher_back_to_25(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 5_000_000,

    # noise (drž to klidné)
    p_base: float = 0.12,     # základní random krok
    p_min: float = 0.06,
    p_max: float = 0.22,

    # probsat exponents
    p_pow: float = 2.0,
    q_pow: float = 1.1,

    # clause pressure (lehké)
    use_clause_weights: bool = True,
    w_inc: float = 0.2,
    w_cap: float = 3.0,
) -> Tuple[List[bool], bool, Dict]:
    """
    Minimal finisher that matches the behavior of the runs that reached ~25 UNSAT:
      - incremental sat_count
      - incremental break computation for candidates in selected clause only
      - very mild clause weights (optional)
      - no restarts, no tabu, no global decay, no sole_sat bookkeeping
    """

    rng = random.Random(seed)
    a = [bool(x) for x in a0]
    m = len(clauses)

    # signed occurrences: var_occ[v] = [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # sat_count + unsat_list
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if lit_true(s, a[v]):
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    clause_w = [1.0] * m if use_clause_weights else None

    best_unsat = len(unsat_list)
    best = a[:]

    eps = 1e-3
    flip = 0

    def pick_unsat_clause() -> int:
        u = len(unsat_list)
        if u == 1:
            return unsat_list[0]
        if not use_clause_weights:
            return unsat_list[rng.randrange(u)]
        # 2-choice pressure (mild)
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    def break_count_if_flip(v: int) -> int:
        """How many clauses would become UNSAT if v flips? (local exact)"""
        old = a[v]
        new = not old
        br = 0
        for (ci, sign) in var_occ[v]:
            sc = sat_count[ci]
            if sc == 0:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was and (not now) and sc == 1:
                # this clause would lose its last satisfier -> becomes UNSAT
                br += 1
        return br

    def make_count_if_flip(v: int) -> int:
        """How many currently UNSAT clauses would become SAT if v flips? (local exact)"""
        old = a[v]
        new = not old
        mk = 0
        for (ci, sign) in var_occ[v]:
            if sat_count[ci] != 0:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if (not was) and now:
                mk += 1
        return mk

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]

        # gentle noise schedule (keeps basin)
        ratio = u / m
        p_eff = p_base * (0.35 + 2.0 * ratio)
        p_eff = max(p_min, min(p_max, p_eff))

        ci = pick_unsat_clause()
        if use_clause_weights:
            clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        lits = clauses[ci]

        # choose variable
        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = abs(lits[0]) - 1
            best_s = -1.0
            for lt in lits:
                v = abs(lt) - 1
                br = break_count_if_flip(v)
                mk = make_count_if_flip(v)
                s = ((mk + eps) ** p_pow) / ((br + eps) ** q_pow)
                # tiny jitter
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    best_v = v
            v_idx = best_v

        # apply flip incrementally
        old = a[v_idx]
        a[v_idx] = not a[v_idx]
        flip += 1

        for (cj, sign) in var_occ[v_idx]:
            was = lit_true(sign, old)
            now = not was  # flipping variable toggles that literal truth value for this sign
            if was == now:
                continue

            old_sc = sat_count[cj]
            new_sc = old_sc + (1 if now else -1)
            sat_count[cj] = new_sc

            if old_sc == 0 and new_sc > 0:
                idx = pos[cj]
                lastc = unsat_list[-1]
                unsat_list[idx] = lastc
                pos[lastc] = idx
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def finisher_back_to_25_classic(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 5_000_000,

    # this is the KEY: match the old run behavior
    p: float = 0.386,              # random-walk probability (your log showed ~0.386)
    p_hard: float = 0.45,          # when u is small, allow a bit more randomness

    # very mild clause pressure (optional)
    use_clause_weights: bool = True,
    w_inc: float = 0.4,
    w_cap: float = 5.0,

    # probsat style exponents
    p_pow: float = 1.6,
    q_pow: float = 1.0,
) -> Tuple[List[bool], bool, Dict]:
    """
    Classic WalkSAT/ProbSAT hybrid close to the finisher that reached ~25 UNSAT:
      - fixed noise p ~ 0.386 (crucial)
      - pick UNSAT clause (2-choice weighted if enabled)
      - with prob p: random variable in clause
        else: choose by ProbSAT score make^p / break^q computed EXACTLY locally
      - incremental sat_count and unsat_list maintenance
      - NO restarts, NO tabu, NO global weight decay, NO sole_sat bookkeeping
    """

    rng = random.Random(seed)
    a = [bool(x) for x in a0]
    m = len(clauses)

    # signed occurrences: var_occ[v] = [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # sat_count + unsat list
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            if lit_true(sign, a[v]):
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    clause_w = [1.0] * m if use_clause_weights else None

    best = a[:]
    best_unsat = len(unsat_list)
    eps = 1e-3
    flip = 0

    def pick_unsat_clause() -> int:
        u = len(unsat_list)
        if u == 1:
            return unsat_list[0]
        if not use_clause_weights:
            return unsat_list[rng.randrange(u)]
        # 2-choice pressure toward heavier clauses
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    def break_count_if_flip(v: int) -> int:
        old = a[v]
        new = not old
        br = 0
        for (ci, sign) in var_occ[v]:
            sc = sat_count[ci]
            if sc != 1:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was and (not now):
                br += 1
        return br

    def make_count_if_flip(v: int) -> int:
        old = a[v]
        new = not old
        mk = 0
        for (ci, sign) in var_occ[v]:
            if sat_count[ci] != 0:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if (not was) and now:
                mk += 1
        return mk

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]

        ci = pick_unsat_clause()
        if use_clause_weights:
            clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        lits = clauses[ci]

        # IMPORTANT: match old behavior: keep p high, even late game
        p_eff = p_hard if u <= 128 else p

        # pick variable
        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = abs(lits[0]) - 1
            best_s = -1.0
            for lt in lits:
                v = abs(lt) - 1
                br = break_count_if_flip(v)
                mk = make_count_if_flip(v)
                s = ((mk + eps) ** p_pow) / ((br + eps) ** q_pow)
                # tiny jitter to avoid tight cycles
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    best_v = v
            v_idx = best_v

        # apply flip
        old = a[v_idx]
        a[v_idx] = not a[v_idx]
        flip += 1

        # update affected clauses incrementally
        for (cj, sign) in var_occ[v_idx]:
            was = lit_true(sign, old)
            now = not was  # toggles for this sign

            if was == now:
                continue

            old_sc = sat_count[cj]
            new_sc = old_sc + (1 if now else -1)
            sat_count[cj] = new_sc

            if old_sc == 0 and new_sc > 0:
                idx = pos[cj]
                lastc = unsat_list[-1]
                unsat_list[idx] = lastc
                pos[lastc] = idx
                unsat_list.pop()
                pos[cj] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[cj] = len(unsat_list)
                unsat_list.append(cj)

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} p={p_eff:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}


#
import random
from typing import List, Tuple, Dict

def finisher_classic_to_zero_sniper(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 5_000_000,

    # MIDGAME: keep what works for you
    p_mid: float = 0.386,
    p_mid_hard: float = 0.45,

    # ENDGAME trigger + behavior
    endgame_at: int = 40,          # when best_unsat <= this, activate sniper mode
    p_end: float = 0.06,           # much lower noise in endgame
    tabu_tenure: int = 20,         # short tabu only in endgame
    focus_pool_k: int = 24,        # sample size from unsat_list
    focus_top: int = 6,            # pick from top-weight subset

    # clause pressure (light but useful)
    w_inc: float = 0.6,
    w_cap: float = 25.0,

    # probsat exponents (used outside sniper-minbreak)
    p_pow: float = 1.6,
    q_pow: float = 1.0,

    p: float = None,   # <--- přidat
) -> Tuple[List[bool], bool, Dict]:
    """
    Two-phase finisher:
      Phase A (default): matches your successful "p~0.386-0.45" WalkSAT/ProbSAT feel.
      Phase B (endgame): weighted core focus + min-break + micro tabu + low noise.
    """

    if p is not None:
        # když někdo pošle p=..., beru to jako "noise" pro endgame i hard-mid
        p_mid_hard = p
        p_end = p

    rng = random.Random(seed)
    a = [bool(x) for x in a0]
    m = len(clauses)

    # var_occ[v] = [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # sat_count + unsat list maintenance
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            if lit_true(sign, a[v]):
                cnt += 1
        sat_count[ci] = cnt
        if cnt == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # clause weights: core pressure memory
    clause_w = [1.0] * m

    # endgame micro tabu
    tabu_until = [0] * nvars

    best = a[:]
    best_unsat = len(unsat_list)
    last_best_flip = 0
    flip = 0
    eps = 1e-3

    def remove_unsat(ci: int):
        idx = pos[ci]
        lastc = unsat_list[-1]
        unsat_list[idx] = lastc
        pos[lastc] = idx
        unsat_list.pop()
        pos[ci] = -1

    def add_unsat(ci: int):
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def break_if_flip(v: int) -> int:
        """How many currently-satisfied clauses become UNSAT if we flip v?"""
        old = a[v]
        new = not old
        br = 0
        for (ci, sign) in var_occ[v]:
            if sat_count[ci] != 1:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was and (not now):
                br += 1
        return br

    def make_if_flip(v: int) -> int:
        """How many currently-UNSAT clauses become SAT if we flip v?"""
        old = a[v]
        new = not old
        mk = 0
        for (ci, sign) in var_occ[v]:
            if sat_count[ci] != 0:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if (not was) and now:
                mk += 1
        return mk

    def pick_clause(u: int, sniper: bool) -> int:
        """Core-focused pick in endgame, 2-choice in midgame."""
        if u == 1:
            return unsat_list[0]
        if not sniper or u > 256:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            return c1 if clause_w[c1] >= clause_w[c2] else c2

        # sniper: sample then choose from top-weight subset
        k = min(u, focus_pool_k)
        sample = [unsat_list[rng.randrange(u)] for _ in range(k)]
        sample.sort(key=lambda c: clause_w[c], reverse=True)
        pool = sample[: min(focus_top, len(sample))]
        c1 = pool[rng.randrange(len(pool))]
        c2 = pool[rng.randrange(len(pool))]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_best_flip = flip

        # activate sniper only if we've *ever* reached endgame band
        sniper = (best_unsat <= endgame_at)

        # clause selection
        ci = pick_clause(u, sniper)
        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)
        lits = clauses[ci]

        # noise schedule
        if sniper:
            p_eff = p_end
        else:
            p_eff = p_mid_hard if u <= 128 else p_mid

        # variable selection
        if sniper:
            # SNIPER MODE:
            # 1) prefer break=0 moves
            # 2) avoid tabu if possible
            # 3) tie-break by make, then by clause weight pressure
            best_v = None
            best_key = None  # tuple ordering
            for lt in lits:
                v = abs(lt) - 1
                if tabu_until[v] > flip:
                    continue
                br = break_if_flip(v)
                mk = make_if_flip(v)
                key = (br, -mk)   # minimize break, maximize make
                if best_key is None or key < best_key:
                    best_key = key
                    best_v = v

            if best_v is None:
                # all tabu -> ignore tabu once
                lt = lits[rng.randrange(len(lits))]
                best_v = abs(lt) - 1

            # small randomness to escape tiny traps
            if rng.random() < p_eff:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
            else:
                v_idx = best_v

        else:
            # MIDGAME: ProbSAT-like score
            if rng.random() < p_eff:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
            else:
                best_v = abs(lits[0]) - 1
                best_s = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    br = break_if_flip(v)
                    mk = make_if_flip(v)
                    s = ((mk + eps) ** p_pow) / ((br + eps) ** q_pow)
                    s *= (1.0 + 0.01 * (rng.random() - 0.5))
                    if s > best_s:
                        best_s = s
                        best_v = v
                v_idx = best_v

        # apply flip
        old = a[v_idx]
        a[v_idx] = not a[v_idx]

        # set tabu only in sniper mode
        if sniper:
            tabu_until[v_idx] = flip + tabu_tenure

        # incremental update affected clauses
        for (cj, sign) in var_occ[v_idx]:
            was = lit_true(sign, old)
            now = not was
            if was == now:
                continue

            old_sc = sat_count[cj]
            new_sc = old_sc + (1 if now else -1)
            sat_count[cj] = new_sc

            if old_sc == 0 and new_sc > 0:
                remove_unsat(cj)
            elif old_sc > 0 and new_sc == 0:
                add_unsat(cj)

        flip += 1

        if report_every and (flip % report_every) == 0:
            mode = "SNIPER" if sniper else "MID"
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat} p={p_eff:.3f} mode={mode}")

        # optional: if we stagnate deep in sniper, teleport to best (safe)
        # (this is minimal and doesn't wreck midgame)
        if sniper and (flip - last_best_flip) > 5_000_000:
            a = best[:]
            # rebuild sat_count/unsat quickly (O(m*3))
            unsat_list.clear()
            for ci2 in range(m):
                pos[ci2] = -1
                cnt = 0
                for lit in clauses[ci2]:
                    v2 = abs(lit) - 1
                    sign2 = 1 if lit > 0 else -1
                    if lit_true(sign2, a[v2]):
                        cnt += 1
                sat_count[ci2] = cnt
                if cnt == 0:
                    pos[ci2] = len(unsat_list)
                    unsat_list.append(ci2)
            last_best_flip = flip

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def finisher_back_to_25_then_zero(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 100_000,

    # MIDGAME (your good regime)
    p_mid: float = 0.386,
    p_mid_hard: float = 0.45,

    # ENDGAME trigger
    endgame_at: int = 40,

    # ENDGAME (safe)
    p_end: float = 0.18,        # not too low; keep basin “mixing”
    T0: float = 0.35,           # metropolis temperature (start)
    T_decay: float = 0.99998,   # slow cooling
    tabu_tenure: int = 18,      # micro tabu only in endgame
    leash_add: int = 25,        # if u > best + leash_add => snap back

    # weights (light)
    w_inc: float = 0.7,
    w_cap: float = 30.0,

) -> Tuple[List[bool], bool, Dict]:
    rng = random.Random(seed)
    a = [bool(x) for x in a0]
    m = len(clauses)

    # signed occ: var -> [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # state
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    clause_w = [1.0] * m

    def rebuild_from_assignment():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            cnt = 0
            for lit in clauses[ci]:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if lit_true(sign, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    rebuild_from_assignment()

    def remove_unsat(ci: int):
        idx = pos[ci]
        lastc = unsat_list[-1]
        unsat_list[idx] = lastc
        pos[lastc] = idx
        unsat_list.pop()
        pos[ci] = -1

    def add_unsat(ci: int):
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def clause_delta_if_flip(v: int) -> int:
        """Exact delta in UNSAT count if we flip v, computed incrementally via affected clauses."""
        old = a[v]
        new = not old
        delta = 0
        for (ci, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was == now:
                continue
            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            if old_sc == 0 and new_sc > 0:
                delta -= 1
            elif old_sc > 0 and new_sc == 0:
                delta += 1
        return delta

    def apply_flip(v: int):
        old = a[v]
        a[v] = not a[v]
        for (ci, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = not was
            if was == now:
                continue
            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            sat_count[ci] = new_sc
            if old_sc == 0 and new_sc > 0:
                remove_unsat(ci)
            elif old_sc > 0 and new_sc == 0:
                add_unsat(ci)

    # micro tabu
    tabu_until = [0] * nvars

    best = a[:]
    best_unsat = len(unsat_list)
    flip = 0
    T = T0

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]

        endgame = (best_unsat <= endgame_at)

        # LEASH: never allow the walk to explode in endgame
        if endgame and u > best_unsat + leash_add:
            a = best[:]
            rebuild_from_assignment()
            # cool a tiny bit to avoid immediate repeat
            T *= 0.98
            continue

        # pick unsat clause (2-choice weighted)
        if u == 1:
            ci = unsat_list[0]
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            ci = c1 if clause_w[c1] >= clause_w[c2] else c2

        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)
        lits = clauses[ci]

        # noise
        p_eff = (p_end if endgame else (p_mid_hard if u <= 128 else p_mid))

        # choose variable candidate
        if rng.random() < p_eff:
            v = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            # classic "good enough" heuristic: prefer variables that reduce UNSAT (delta<0), else small delta
            best_v = None
            best_d = 10**9
            for lt in lits:
                vv = abs(lt) - 1
                if endgame and tabu_until[vv] > flip:
                    continue
                d = clause_delta_if_flip(vv)
                if d < best_d:
                    best_d = d
                    best_v = vv
            v = best_v if best_v is not None else abs(lits[rng.randrange(len(lits))]) - 1

        # ENDGAME acceptance: Metropolis gate (prevents 40→240 drift)
        if endgame:
            d = clause_delta_if_flip(v)
            accept = False
            if d <= 0:
                accept = True
            else:
                # accept uphill with probability exp(-d/T)
                # for small T this kills big uphill moves
                accept = (rng.random() < pow(2.718281828, -d / max(1e-9, T)))

            if not accept:
                flip += 1
                T *= T_decay
                continue

            tabu_until[v] = flip + tabu_tenure
            T *= T_decay

        # apply flip
        apply_flip(v)
        flip += 1

        if report_every and (flip % report_every) == 0:
            mode = "END" if endgame else "MID"
            pp = p_eff
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat} p={pp:.3f} mode={mode}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def finisher_mid25_with_safe_sniper(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 100_000,

    # MID params (match your good finisher)
    p_mid: float = 0.386,
    p_mid_hard: float = 0.450,
    hard_at_u: int = 256,

    # weights
    w_inc: float = 1.0,
    w_cap: float = 50.0,

    # safe sniper trigger
    sniper_best_at: int = 40,

    # sniper safety
    p_sniper: float = 0.22,     # NOT 0.06
    leash_add: int = 30,        # if u > best + leash => snap back
    T0: float = 0.25,           # metropolis temp
    T_decay: float = 0.99998,   # slow cooling
    tabu_tenure: int = 24,      # small tabu in sniper only
) -> Tuple[List[bool], bool, Dict]:
    rng = random.Random(seed)
    a = a0[:]
    m = len(clauses)

    # signed occurrences var -> [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # state
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    clause_w = [1.0] * m

    # score arrays
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars

    def rebuild_state():
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            cnt = 0
            for lit in clauses[ci]:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if lit_true(sign, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    def remove_unsat(ci: int):
        idx = pos[ci]
        lastc = unsat_list[-1]
        unsat_list[idx] = lastc
        pos[lastc] = idx
        unsat_list.pop()
        pos[ci] = -1

    def add_unsat(ci: int):
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def update_scores(v: int):
        """Weighted make/break, correct and fast enough (touch only neighbors later)."""
        br = 0.0
        mk = 0.0
        cur = a[v]
        flipv = not cur
        for (ci, sign) in var_occ[v]:
            sc = sat_count[ci]
            w = clause_w[ci]
            if sc == 0:
                # would this literal become true after flip?
                if lit_true(sign, flipv):
                    mk += w
            elif sc == 1:
                # break: if currently this literal is true, flipping breaks the only sat
                # we must check if THIS var is the satisfier (3-SAT: cheap scan)
                if lit_true(sign, cur) and (not lit_true(sign, flipv)):
                    # verify it is the only satisfier
                    only = 0
                    for lt in clauses[ci]:
                        vv = abs(lt) - 1
                        sg = 1 if lt > 0 else -1
                        if lit_true(sg, a[vv]):
                            only += 1
                            if only > 1:
                                break
                    if only == 1:
                        br += w
        var_br_w[v] = br
        var_mk_w[v] = mk

    rebuild_state()
    # initial score build
    for v in range(nvars):
        update_scores(v)

    def clause_delta_unsat_if_flip(v: int) -> int:
        """Exact delta UNSAT if flip v (for sniper accept/reject only)."""
        old = a[v]
        new = not old
        d = 0
        for (ci, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was == now:
                continue
            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            if old_sc == 0 and new_sc > 0:
                d -= 1
            elif old_sc > 0 and new_sc == 0:
                d += 1
        return d

    # tabu only in sniper
    tabu_until = [0] * nvars
    T = T0

    best = a[:]
    best_unsat = len(unsat_list)

    flip = 0
    eps = 1e-3

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]

        sniper = (best_unsat <= sniper_best_at)
        mode = "SNIPER" if sniper else "MID"

        # leash only in sniper (prevents your 40->240 explosion)
        if sniper and u > best_unsat + leash_add:
            a = best[:]
            rebuild_state()
            for v in range(nvars):
                update_scores(v)
            T *= 0.98
            continue

        # pick clause: 2-choice weighted from unsat_list
        if u == 1:
            ci = unsat_list[0]
        else:
            c1 = unsat_list[rng.randrange(u)]
            c2 = unsat_list[rng.randrange(u)]
            ci = c1 if clause_w[c1] >= clause_w[c2] else c2

        # bump
        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        # refresh scores of vars in this clause (keeps weights consistent)
        for lt in clauses[ci]:
            update_scores(abs(lt) - 1)

        # noise p
        if sniper:
            p_eff = p_sniper
        else:
            p_eff = p_mid_hard if u <= hard_at_u else p_mid

        # choose variable (THIS is the MID that worked for you)
        lits = clauses[ci]
        p_pow, q_pow = (2.0, 1.1) if (not sniper) else (2.6, 1.25)

        if rng.random() < p_eff:
            v = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            best_v = None
            best_s = -1.0
            for lt in lits:
                vv = abs(lt) - 1
                if sniper and tabu_until[vv] > flip:
                    continue
                s = ((var_mk_w[vv] + eps) ** p_pow) / ((var_br_w[vv] + eps) ** q_pow)
                s *= (1.0 + 0.01 * (rng.random() - 0.5))
                if s > best_s:
                    best_s = s
                    best_v = vv
            v = best_v if best_v is not None else abs(lits[rng.randrange(len(lits))]) - 1

        # sniper accept/reject ONLY (does not change MID)
        if sniper:
            d = clause_delta_unsat_if_flip(v)
            if d > 0:
                # reject big uphill moves with Metropolis
                if rng.random() >= pow(2.718281828, -d / max(1e-9, T)):
                    flip += 1
                    T *= T_decay
                    if report_every and flip % report_every == 0:
                        print(f"[finisher] flips={flip:,} unsat={u} best={best_unsat} p={p_eff:.3f} mode={mode}")
                    continue
            tabu_until[v] = flip + tabu_tenure
            T *= T_decay

        # apply flip and update incremental state
        old = a[v]
        a[v] = not a[v]

        affected_vars = {v}
        for (cj, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = not was
            if was == now:
                continue
            old_sc = sat_count[cj]
            new_sc = old_sc + (1 if now else -1)
            sat_count[cj] = new_sc

            if old_sc == 0 and new_sc > 0:
                remove_unsat(cj)
            elif old_sc > 0 and new_sc == 0:
                add_unsat(cj)

            for lt in clauses[cj]:
                affected_vars.add(abs(lt) - 1)

        for vv in affected_vars:
            update_scores(vv)

        flip += 1

        if report_every and flip % report_every == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat} p={p_eff:.3f} mode={mode}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}


import random
from typing import List, Tuple

def finisher_epic_incremental(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    # noise control
    p_min: float = 0.02,
    p_max: float = 0.90,
    drift_window: int = 200_000,
    # weights
    w_inc: float = 1.0,
    w_decay: float = 0.9995,
    w_cap: float = 50.0,
    # stall / restarts
    stall_window: int = 500_000,
    restart_shake: int = 64,
    report_every: int = 5_000_000,
    # stabilizers
    explosion_mult: float = 8.0,      # backtrack when u > explosion_mult * best_unsat
    explosion_cool: int = 80_000,     # cooldown flips after explosion backtrack
) -> Tuple[List[bool], bool, dict]:
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve_at = 0
    m = len(clauses)

    # incidence: var -> list of clause indices (unsigned)
    var_occ: List[List[int]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            var_occ[abs(lit) - 1].append(ci)

    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    # weights + lazy global decay multiplier (no O(m) loop each time)
    w = [1.0] * m
    w_scale = 1.0

    def lit_true(lit: int, aval: bool) -> bool:
        return aval if lit > 0 else (not aval)

    def remove_unsat(ci: int) -> None:
        i = pos[ci]
        if i == -1:
            return
        last = unsat_list[-1]
        unsat_list[i] = last
        pos[last] = i
        unsat_list.pop()
        pos[ci] = -1

    def add_unsat(ci: int) -> None:
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def rebuild_state() -> None:
        unsat_list.clear()
        for ci in range(m):
            pos[ci] = -1
            cnt = 0
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if lit_true(lit, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)

    # how much sat_count[ci] changes if we flip variable v (given old value old_av)
    def clause_flip_effect(ci: int, v: int, old_av: bool) -> int:
        # 3-SAT assumed by your generator, but works for any short clause
        delta = 0
        new_av = (not old_av)
        c = clauses[ci]
        for lit in c:
            vv = abs(lit) - 1
            if vv != v:
                continue
            # this literal's truth changes (maybe)
            was = lit_true(lit, old_av)
            now = lit_true(lit, new_av)
            if was == now:
                return 0
            delta += (1 if now else -1)
            return delta
        return 0

    def break_score(v: int) -> int:
        # number of currently satisfied clauses that would become UNSAT by flipping v
        old_av = a[v]
        new_av = (not old_av)
        br = 0
        for ci in var_occ[v]:
            if sat_count[ci] != 1:
                continue
            # if v currently satisfies the only satisfied literal, flipping breaks it
            # check whether v is currently making this clause satisfied
            made_true = False
            for lit in clauses[ci]:
                vv = abs(lit) - 1
                if vv != v:
                    continue
                if lit_true(lit, old_av) and (not lit_true(lit, new_av)):
                    made_true = True
                break
            if made_true:
                br += 1
        return br

    def make_score(v: int) -> int:
        # number of currently UNSAT clauses that would become SAT by flipping v
        old_av = a[v]
        new_av = (not old_av)
        mk = 0
        for ci in var_occ[v]:
            if sat_count[ci] != 0:
                continue
            # if flipping makes any literal in this clause true, it becomes SAT
            for lit in clauses[ci]:
                vv = abs(lit) - 1
                if vv != v:
                    continue
                if lit_true(lit, new_av):
                    mk += 1
                break
        return mk

    rebuild_state()

    flips = 0
    # "temperature" p is adaptive; "cooldown" after explosion backtrack
    p = p_max
    cool_until = 0

    while flips < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flips, "best_unsat": 0}

        # best tracking
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_at = flips

        # explosion backtrack
        if best_unsat is not None and u > explosion_mult * best_unsat and flips > cool_until:
            a = best[:]
            rebuild_state()
            last_improve_at = flips
            cool_until = flips + explosion_cool
            # after teleport: keep p moderately high for a short while
            p = min(p_max, max(p, 0.25))
            continue

        stalled = flips - last_improve_at

        # stall restart shake (flip vars from current UNSAT neighborhood)
        if stalled > stall_window:
            # collect vars from unsat clauses
            hard_vars = []
            seen = set()
            for ci in unsat_list:
                for lit in clauses[ci]:
                    v = abs(lit) - 1
                    if v not in seen:
                        seen.add(v)
                        hard_vars.append(v)

            if hard_vars:
                for _ in range(min(restart_shake, len(hard_vars))):
                    v0 = hard_vars[rng.randrange(len(hard_vars))]
                    a[v0] = not a[v0]

            rebuild_state()
            last_improve_at = flips
            p = min(p_max, max(p, 0.25))
            cool_until = max(cool_until, flips + 20_000)
            continue

        # ----- choose clause (2-choice weighted) -----
        ci1 = unsat_list[rng.randrange(u)]
        ci2 = unsat_list[rng.randrange(u)]
        ci = ci1 if (w[ci1] * w_scale) >= (w[ci2] * w_scale) else ci2

        # bump weight (lazy-scale), cap by effective w_cap
        if w_scale < 1e-12:
            w_scale = 1.0
        w[ci] = min(w_cap / w_scale, w[ci] + (w_inc / w_scale))

        c = clauses[ci]
        vars_in = [abs(lit) - 1 for lit in c]

        # ----- temperature schedule: HARD CLAMP by ratio -----
        ratio = u / m
        if ratio <= 0.001:
            p_eff = min(p, 0.10)
        elif ratio <= 0.01:
            p_eff = min(p, 0.30)
        else:
            p_eff = min(p, 0.60)

        # extra near-sat clamp
        if u <= 64:
            p_eff = min(p_eff, 0.08)

        # keep within [p_min, p_max]
        p_eff = max(p_min, min(p_max, p_eff))

        # choose variable
        if rng.random() < p_eff:
            v = vars_in[rng.randrange(len(vars_in))]
        else:
            # deterministic-ish: minimize break, maximize make
            best_v = vars_in[0]
            best_key = None
            for v0 in vars_in:
                b = break_score(v0)
                mk = make_score(v0)
                key = (b, -mk, rng.random())
                if best_key is None or key < best_key:
                    best_key = key
                    best_v = v0
            v = best_v

        # apply flip + incremental update
        old_av = a[v]
        a[v] = not a[v]
        flips += 1

        for cj in var_occ[v]:
            old = sat_count[cj]
            delta = clause_flip_effect(cj, v, old_av)
            if delta == 0:
                continue
            new = old + delta
            sat_count[cj] = new
            if old == 0 and new > 0:
                remove_unsat(cj)
            elif old > 0 and new == 0:
                add_unsat(cj)

        # lazy decay (global multiplier)
        if w_decay < 1.0 and (flips & 0x3FFFF) == 0:
            w_scale *= w_decay
            # renormalize if too small
            if w_scale < 1e-6:
                for i in range(m):
                    w[i] *= w_scale
                    if w[i] < 1.0:
                        w[i] = 1.0
                w_scale = 1.0

        if report_every and (flips % report_every == 0):
            print(f"[finisher] flips={flips:,} unsat={len(unsat_list)} p_eff={p_eff:.3f} scale={w_scale:.3e} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flips, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict, Set

def finisher_epic_incremental_v4_endgame(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_min: float = 0.02,
    p_max: float = 0.90,

    # weights
    w_inc: float = 1.0,
    w_decay: float = 0.9995,
    w_cap: float = 50.0,

    # stall / restart
    stall_window: int = 600_000,
    restart_shake: int = 128,
    report_every: int = 5_000_000,

    # basin guard (jemný, nepřestřeluj)
    explosion_mult: float = 6.0,      # backtrack když u > explosion_mult*best
    explosion_cool: int = 120_000,    # cooldown po teleportu

    # endgame hook (to je ten rozdíl 25 -> 0, když se zadaří)
    endgame_u: int = 40,
    endgame_prob: float = 0.12,       # jak často zkusit greedy krok
    endgame_minbreak: bool = True,    # vyber proměnnou s min break (jinak náhodně)
    endgame_noisecap_128: float = 0.18,
    endgame_noisecap_64: float = 0.10,
) -> Tuple[List[bool], bool, Dict]:
    """
    V4-based incremental finisher + endgame greedy hook.
    - zachovává chování, které ti padalo nejníž
    - v endgame občas udělá "oprav ten unsat clause" krok s min-break volbou
    """

    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = None
    last_improve_flip = 0
    last_explosion_flip = -10**18
    m = len(clauses)

    # --- literal truth helper ---
    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # --- signed occurrences: var_occ[v] = [(ci, sign), ...] ---
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_occ[v].append((ci, s))

    # --- sat_count & unsat_list ---
    sat_count = [0] * m
    for ci, c in enumerate(clauses):
        cnt = 0
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if lit_true(s, a[v]):
                cnt += 1
        sat_count[ci] = cnt

    unsat_list: List[int] = []
    pos = [-1] * m
    for ci in range(m):
        if sat_count[ci] == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # --- weights & scores ---
    clause_w = [1.0] * m
    var_break_w = [0.0] * nvars
    var_make_w = [0.0] * nvars

    def update_var_scores(v_idx: int):
        """Weighted make/break for flipping v_idx under current assignment a."""
        br = 0.0
        mk = 0.0
        cur_val = a[v_idx]
        flipped_val = (not cur_val)

        for (ci_idx, sign) in var_occ[v_idx]:
            sc = sat_count[ci_idx]
            w = clause_w[ci_idx]

            was_lit_true = lit_true(sign, cur_val)
            after_lit_true = lit_true(sign, flipped_val)

            if sc == 0:
                # clause unsat now; make if flip makes THIS lit true
                if after_lit_true:
                    mk += w
            elif sc == 1:
                # clause has exactly 1 satisfier now; break if that satisfier is THIS lit and flip kills it
                if was_lit_true and (not after_lit_true):
                    br += w

        var_break_w[v_idx] = br
        var_make_w[v_idx] = mk

    for v in range(nvars):
        update_var_scores(v)

    # --- utility: rebuild to best (basin guard) ---
    def rebuild_from_assignment(assign: List[bool]):
        nonlocal a, sat_count, unsat_list, pos
        a = assign[:]
        # rebuild sat_count and unsat_list
        unsat_list.clear()
        for ci, c in enumerate(clauses):
            cnt = 0
            for lit in c:
                v = abs(lit) - 1
                s = 1 if lit > 0 else -1
                if lit_true(s, a[v]):
                    cnt += 1
            sat_count[ci] = cnt
            if cnt == 0:
                pos[ci] = len(unsat_list)
                unsat_list.append(ci)
            else:
                pos[ci] = -1
        # rebuild scores
        for v in range(nvars):
            update_var_scores(v)

    # --- apply flip incrementally ---
    def apply_flip(v_idx: int) -> List[int]:
        """Flip variable, update sat_count + unsat_list, return affected clause indices."""
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        new_val = a[v_idx]

        affected_clauses: List[int] = []
        for (cj_idx, sign) in var_occ[v_idx]:
            was_true = lit_true(sign, old_val)
            now_true = lit_true(sign, new_val)
            if was_true == now_true:
                continue

            old_sc = sat_count[cj_idx]
            sat_count[cj_idx] = old_sc + (1 if now_true else -1)
            new_sc = sat_count[cj_idx]
            affected_clauses.append(cj_idx)

            if old_sc == 0 and new_sc > 0:
                # remove from unsat_list
                idx = pos[cj_idx]
                if idx != -1:
                    last_c = unsat_list[-1]
                    unsat_list[idx] = last_c
                    pos[last_c] = idx
                    unsat_list.pop()
                    pos[cj_idx] = -1
            elif old_sc > 0 and new_sc == 0:
                # add to unsat_list
                pos[cj_idx] = len(unsat_list)
                unsat_list.append(cj_idx)

        return affected_clauses

    # --- pick clause: 2-choice weighted (fast) ---
    def pick_unsat_clause_2choice() -> int:
        u = len(unsat_list)
        c1 = unsat_list[rng.randrange(u)]
        c2 = unsat_list[rng.randrange(u)]
        return c1 if clause_w[c1] >= clause_w[c2] else c2

    eps = 1e-3
    flip = 0
    p_base = 0.15

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # record best
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip

        # basin guard: když se utrhne ze řetězu, vrať se
        if best_unsat is not None:
            if (u > max(150, int(best_unsat * explosion_mult))) and (flip - last_explosion_flip > explosion_cool):
                rebuild_from_assignment(best)
                last_explosion_flip = flip
                last_improve_flip = flip
                continue

        stalled = flip - last_improve_flip

        # restart shake (tvoje osvědčená pojistka)
        if stalled > stall_window:
            for _ in range(restart_shake):
                if not unsat_list:
                    break
                ci_r = pick_unsat_clause_2choice()
                lits = clauses[ci_r]
                v_r = abs(lits[rng.randrange(len(lits))]) - 1

                # bump i v shaku, ale jen mírně
                clause_w[ci_r] = min(w_cap, clause_w[ci_r] + 0.5 * w_inc)

                affected_cls = apply_flip(v_r)

                # refresh scores v lokálním okolí
                affected_vars: Set[int] = {v_r}
                for cj in affected_cls:
                    for lt in clauses[cj]:
                        affected_vars.add(abs(lt) - 1)
                for v_upd in affected_vars:
                    update_var_scores(v_upd)

                flip += 1
                if flip >= max_flips:
                    break

            last_improve_flip = flip
            continue

        # --- noise schedule (ponecháváme “v4 feeling”) ---
        ratio = u / m
        p_eff = p_base * (0.35 + 2.2 * ratio)

        # jemný ramp při stallu
        if stalled > stall_window // 2:
            ramp = min(1.0, (stalled - stall_window // 2) / max(1, stall_window // 2))
            p_eff = min(p_max, p_eff + 0.20 * ramp)

        # endgame clamp (to je důležité, jinak se to opije)
        if u <= 128:
            p_eff = min(p_eff, endgame_noisecap_128)
        if u <= 64:
            p_eff = min(p_eff, endgame_noisecap_64)

        p_eff = float(max(p_min, min(p_max, p_eff)))

        # --- ENDGAME HOOK: občas oprav UNSAT klauzuli přímo (min-break) ---
        if u <= endgame_u and rng.random() < endgame_prob:
            ci = pick_unsat_clause_2choice()
            clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)  # tlak na jádro

            lits = clauses[ci]
            if endgame_minbreak:
                # v unsat klauzuli jsou všechny lit false => flip kterékoliv ji “spraví”
                # zvolíme ale proměnnou s minimálním break, aby to nerozjebalo okolí
                best_v = None
                best_br = 1e300
                for lt in lits:
                    v = abs(lt) - 1
                    br = var_break_w[v]
                    # preferuj menší break; jemný jitter proti cyklu
                    br *= (1.0 + 0.01 * (rng.random() - 0.5))
                    if br < best_br:
                        best_br = br
                        best_v = v
                v_idx = best_v if best_v is not None else abs(lits[0]) - 1
            else:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1

            affected_cls = apply_flip(v_idx)

            affected_vars: Set[int] = {v_idx}
            for cj in affected_cls:
                for lt in clauses[cj]:
                    affected_vars.add(abs(lt) - 1)
            for v_upd in affected_vars:
                update_var_scores(v_upd)

            flip += 1
        else:
            # --- standard v4 step: pick clause + ProbSAT scoring ---
            ci = pick_unsat_clause_2choice()
            lits = clauses[ci]

            clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

            # refresh variables in chosen clause (protože jsme bumpnuli váhu)
            for lt in lits:
                update_var_scores(abs(lt) - 1)

            # ProbSAT-like pick
            p_pow = 2.0 if u <= 128 else 1.6
            q_pow = 1.1 if u <= 128 else 1.0

            if rng.random() < p_eff:
                v_idx = abs(lits[rng.randrange(len(lits))]) - 1
            else:
                best_v = None
                best_score = -1.0
                for lt in lits:
                    v = abs(lt) - 1
                    b = var_break_w[v]
                    mk = var_make_w[v]
                    score = ((mk + eps) ** p_pow) / ((b + eps) ** q_pow)
                    score *= (1.0 + 0.01 * (rng.random() - 0.5))
                    if score > best_score:
                        best_score = score
                        best_v = v
                v_idx = best_v if best_v is not None else abs(lits[0]) - 1

            affected_cls = apply_flip(v_idx)

            affected_vars: Set[int] = {v_idx}
            for cj in affected_cls:
                for lt in clauses[cj]:
                    affected_vars.add(abs(lt) - 1)
            for v_upd in affected_vars:
                update_var_scores(v_upd)

            flip += 1

        # decay (stejně jako v4: vzácně, aby to nebolelo výkon)
        if w_decay < 1.0 and (flip & 0x3FFFF) == 0:
            for i in range(m):
                clause_w[i] = max(1.0, clause_w[i] * w_decay)
            # score rebuild (dražší, ale jen občas)
            for v in range(nvars):
                update_var_scores(v)

        if report_every and (flip % report_every) == 0:
            mode = "END" if len(unsat_list) <= endgame_u else "MID"
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat} p_eff={p_eff:.3f} mode={mode}")

    return best, (best_unsat == 0), {"flips": flip, "best_unsat": best_unsat}

#
import random
from typing import List, Tuple, Dict

def _lit_true(lit: int, a: List[bool]) -> bool:
    v = abs(lit) - 1
    val = a[v]
    return val if lit > 0 else (not val)

def _flip_var(v: int, a: List[bool]) -> None:
    a[v] = not a[v]

def finisher_epic_incremental_v2(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise control
    p_min: float = 0.02,
    p_max: float = 0.90,
    drift_window: int = 200_000,

    # weights (SAPS-ish)
    w_inc: float = 1.0,
    w_decay: float = 0.9995,
    w_cap: float = 50.0,

    # stall / restarts
    stall_window: int = 500_000,
    restart_shake: int = 64,

    report_every: int = 5_000_000,
) -> Tuple[List[bool], bool, Dict]:
    """
    Incremental weighted WalkSAT/ProbSAT hybrid (correct incremental deltas).
    Returns (assignment, solved, stats)
    """
    rng = random.Random(seed)

    a = a0[:]                 # working assignment
    best = a0[:]              # best assignment seen
    best_unsat = 10**18
    last_improve_at = 0

    m = len(clauses)

    # var -> incident clauses
    var_occ: List[List[int]] = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c:
            var_occ[abs(lit) - 1].append(ci)

    # sat_count per clause (truth under a)
    sat_count = [0] * m
    for ci, c in enumerate(clauses):
        sat_count[ci] = sum(1 for lit in c if _lit_true(lit, a))

    # unsat set as list + positions (O(1) remove)
    unsat_list: List[int] = []
    pos = [-1] * m
    for ci in range(m):
        if sat_count[ci] == 0:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    # weights per clause
    w = [1.0] * m

    def unsat_size() -> int:
        return len(unsat_list)

    def add_unsat(ci: int) -> None:
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def remove_unsat(ci: int) -> None:
        i = pos[ci]
        if i == -1:
            return
        last = unsat_list[-1]
        unsat_list[i] = last
        pos[last] = i
        unsat_list.pop()
        pos[ci] = -1

    # --- local delta computation BEFORE flip ---
    # returns: dict clause_index -> delta_satcount
    def compute_deltas_for_flip(v: int) -> Dict[int, int]:
        # we know old a[v], and flipping toggles it
        old_val = a[v]
        new_val = (not old_val)

        deltas: Dict[int, int] = {}
        for ci in var_occ[v]:
            c = clauses[ci]

            before = 0
            after = 0
            # compute satcount under old and under new (only v changes)
            for lit in c:
                var = abs(lit) - 1
                if var == v:
                    val_old = old_val
                    val_new = new_val
                else:
                    val_old = a[var]
                    val_new = a[var]

                lit_old = val_old if lit > 0 else (not val_old)
                lit_new = val_new if lit > 0 else (not val_new)
                if lit_old:
                    before += 1
                if lit_new:
                    after += 1

            if after != before:
                deltas[ci] = (after - before)
        return deltas

    # weighted break and make computed from deltas
    def break_make_scores(v: int) -> Tuple[float, float]:
        deltas = compute_deltas_for_flip(v)
        b = 0.0
        mk = 0.0
        for ci, d in deltas.items():
            sc = sat_count[ci]
            # if clause currently satisfied and would become unsat
            if sc > 0 and sc + d == 0:
                b += w[ci]
            # if clause currently unsat and would become sat
            if sc == 0 and sc + d > 0:
                mk += w[ci]
        return b, mk

    # adaptive p control
    p = p_min
    window_best = unsat_size()
    window_at = 0

    flips = 0
    while flips < max_flips:
        u = unsat_size()
        if u == 0:
            return a, True, {"flips": flips, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_at = flips

        # drift window update
        if flips - window_at >= drift_window:
            if u >= window_best:
                p = min(p_max, p * 1.25 + 0.02)
            else:
                p = max(p_min, p * 0.80)
            window_best = u
            window_at = flips

        # stall restart
        if stall_window > 0 and flips - last_improve_at >= stall_window:
            a = best[:]

            # rebuild sat_count + unsat_list cleanly (no ghosts)
            for ci, c in enumerate(clauses):
                sat_count[ci] = sum(1 for lit in c if _lit_true(lit, a))
            unsat_list.clear()
            for i in range(m):
                pos[i] = -1
            for ci in range(m):
                if sat_count[ci] == 0:
                    pos[ci] = len(unsat_list)
                    unsat_list.append(ci)

            # shake hard vars (appear in unsat clauses)
            hard_vars = set()
            for ci in unsat_list:
                for lit in clauses[ci]:
                    hard_vars.add(abs(lit) - 1)
            hard_vars = list(hard_vars)
            for _ in range(min(restart_shake, len(hard_vars))):
                vsh = hard_vars[rng.randrange(len(hard_vars))]
                a[vsh] = not a[vsh]

            # rebuild again after shake
            for ci, c in enumerate(clauses):
                sat_count[ci] = sum(1 for lit in c if _lit_true(lit, a))
            unsat_list.clear()
            for i in range(m):
                pos[i] = -1
            for ci in range(m):
                if sat_count[ci] == 0:
                    pos[ci] = len(unsat_list)
                    unsat_list.append(ci)

            last_improve_at = flips
            p = min(p_max, max(p, 0.35))
            continue

        # IMPORTANT: recompute u AFTER possible changes (we didn't change since top, so ok)
        u = unsat_size()
        if u == 0:
            return a, True, {"flips": flips, "best_unsat": 0}

        # pick random unsat clause
        ci = unsat_list[rng.randrange(u)]
        c = clauses[ci]

        # weight bump and occasional decay
        w[ci] = min(w_cap, w[ci] + w_inc)
        if w_decay < 1.0 and (flips & 0x3FFFF) == 0:
            for i in range(m):
                w[i] = max(1.0, w[i] * w_decay)

        # choose candidate vars from clause lits
        vars_in = [abs(lit) - 1 for lit in c]

        # choose var
        if rng.random() < p:
            v = vars_in[rng.randrange(len(vars_in))]
            deltas = compute_deltas_for_flip(v)
        else:
            best_v = vars_in[0]
            best_key = None
            best_deltas = None
            for v0 in vars_in:
                deltas0 = compute_deltas_for_flip(v0)
                b = 0.0
                mk = 0.0
                for cj, d in deltas0.items():
                    sc = sat_count[cj]
                    if sc > 0 and sc + d == 0:
                        b += w[cj]
                    if sc == 0 and sc + d > 0:
                        mk += w[cj]
                # primary: minimize break; secondary: maximize make
                key = (b, -mk, rng.random())
                if best_key is None or key < best_key:
                    best_key = key
                    best_v = v0
                    best_deltas = deltas0
            v = best_v
            deltas = best_deltas if best_deltas is not None else compute_deltas_for_flip(v)

        # apply flip
        _flip_var(v, a)
        flips += 1

        # update affected clauses using precomputed deltas
        for cj, d in deltas.items():
            old = sat_count[cj]
            new = old + d
            sat_count[cj] = new
            if old == 0 and new > 0:
                remove_unsat(cj)
            elif old > 0 and new == 0:
                add_unsat(cj)

        if report_every and flips % report_every == 0:
            print(f"[finisher-v2] flips={flips:,} unsat={unsat_size()} p≈{p:.3f} best={best_unsat}")

    return best, (best_unsat == 0), {"flips": flips, "best_unsat": best_unsat}

#
import sys, time, math, random, argparse
from typing import List, Tuple, Dict


# --- CORE UTILS (Striktní bool formát) ---
def count_unsat(clauses: List[List[int]], assignment: List[bool]) -> int:
    c = 0
    for cl in clauses:
        sat = False
        for lit in cl:
            val = assignment[abs(lit) - 1]
            if (lit > 0 and val) or (lit < 0 and not val):
                sat = True;
                break
        if not sat: c += 1
    return c


def check_sat(clauses: List[List[int]], assignment: List[bool]) -> bool:
    return count_unsat(clauses, assignment) == 0


# --- SPECTRÁLNÍ MOTOR (Tvůj A1/A2/A3 základ) ---
# [Zde zůstává tvá implementace generate_spectral_seed, která funguje skvěle]
# (Zkráceno pro přehlednost, v souboru ponech svou verzi z v5)

# --- THE ULTIMATE PREDATOR FINISHER ---
def finisher_ultimate(
        clauses: List[List[int]],
        nvars: int,
        a0: List[bool],
        seed: int = 42,
        max_flips: int = 50_000_000,
        p_min: float = 0.005,
        p_max: float = 0.20,
        stall_window: int = 800_000,
        restart_shake: int = 64,
        w_inc: float = 1.0,
        w_decay: float = 0.9995,
        w_cap: float = 80.0,
        report_every: int = 1_000_000,
) -> Tuple[List[bool], bool]:
    rng = random.Random(seed)
    a = a0[:]
    best = a0[:]
    best_unsat = count_unsat(clauses, a)
    m = len(clauses)

    # Inkrementální struktury
    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list = []
    pos = [-1] * m
    clause_w = [1.0] * m
    var_br_w = [0.0] * nvars
    var_mk_w = [0.0] * nvars
    tabu = [0] * nvars
    tabu_tenure = 40

    var_occ = [[] for _ in range(nvars)]
    for ci, c in enumerate(clauses):
        for lit in c: var_occ[abs(lit) - 1].append((ci, lit))

    def get_mk_br(v: int):
        mk, br = 0.0, 0.0
        cur_v = a[v]
        for ci, lit in var_occ[v]:
            sc, w = sat_count[ci], clause_w[ci]
            if sc == 0:
                if (lit > 0 and not cur_v) or (lit < 0 and cur_v): mk += w
            elif sc == 1:
                if sole_sat[ci] == v: br += w
        return mk, br

    def rebuild():
        unsat_list.clear()
        for ci in range(m):
            cnt, last_v = 0, -1
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if (lit > 0 and a[v]) or (lit < 0 and not a[v]):
                    cnt += 1;
                    last_v = v
            sat_count[ci] = cnt
            sole_sat[ci] = last_v if cnt == 1 else -1
            if cnt == 0:
                pos[ci] = len(unsat_list);
                unsat_list.append(ci)
            else:
                pos[ci] = -1
        for v in range(nvars):
            var_mk_w[v], var_br_w[v] = get_mk_br(v)

    rebuild()
    flip = 0
    last_improve = 0

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0: return a, True

        if u < best_unsat:
            best_unsat, best, last_improve = u, a[:], flip

        # BASIN GUARD: Pokud driftujeme moc vysoko, vrať se k nejlepšímu
        if u > best_unsat + 50 and u > 150:
            a = best[:];
            rebuild();
            continue

        # STALL / SHAKE
        if flip - last_improve > stall_window:
            # Jemný shake náhodným flipem v unsat klauzulích
            for _ in range(restart_shake):
                if not unsat_list: break
                ci = unsat_list[rng.randrange(len(unsat_list))]
                v = abs(clauses[ci][rng.randrange(3)]) - 1
                a[v] = not a[v]
            rebuild();
            last_improve = flip;
            continue

        # NOISE SCHEDULE
        p_eff = p_min + (p_max - p_min) * (u / m * 10)
        if u <= 64: p_eff = 0.01  # Chirurgický režim

        # PICK
        ci = unsat_list[rng.randrange(u)]
        # Váhový bump
        clause_w[ci] = min(w_cap, clause_w[ci] + w_inc)

        lits = clauses[ci]
        if rng.random() < p_eff:
            v_idx = abs(lits[rng.randrange(3)]) - 1
        else:
            # ProbSAT / Predator výběr
            best_v, max_s = -1, -1.0
            for lit in lits:
                v = abs(lit) - 1
                if u <= 128 and tabu[v] > flip: continue
                mk, br = var_mk_w[v], var_br_w[v]
                s = ((mk + 0.001) ** 2.5) / ((br + 0.001) ** 1.5)
                if s > max_s: max_s, best_v = s, v
            v_idx = best_v if best_v != -1 else abs(lits[rng.randrange(3)]) - 1

        # APPLY FLIP (Incremental)
        old_val = a[v_idx]
        a[v_idx] = not a[v_idx]
        tabu[v_idx] = flip + tabu_tenure

        aff_vars = {v_idx}
        for ci_aff, lit_aff in var_occ[v_idx]:
            was_sat = (lit_aff > 0 and old_val) or (lit_aff < 0 and not old_val)
            now_sat = not was_sat

            old_sc = sat_count[ci_aff]
            new_sc = old_sc + (1 if now_sat else -1)
            sat_count[ci_aff] = new_sc

            if new_sc == 1:
                for l in clauses[ci_aff]:
                    vv = abs(l) - 1
                    if (l > 0 and a[vv]) or (l < 0 and not a[vv]):
                        sole_sat[ci_aff] = vv;
                        break
            else:
                sole_sat[ci_aff] = -1

            if old_sc == 0 and new_sc > 0:
                idx = pos[ci_aff];
                last_c = unsat_list[-1]
                unsat_list[idx] = last_c;
                pos[last_c] = idx
                unsat_list.pop();
                pos[ci_aff] = -1
            elif old_sc > 0 and new_sc == 0:
                pos[ci_aff] = len(unsat_list);
                unsat_list.append(ci_aff)

            for l in clauses[ci_aff]: aff_vars.add(abs(l) - 1)

        for v in aff_vars: var_mk_w[v], var_br_w[v] = get_mk_br(v)

        flip += 1
        if (flip & 0x7FFFF) == 0:  # Periodic Decay
            for i in range(m): clause_w[i] = max(1.0, clause_w[i] * w_decay)
            for v in range(nvars): var_mk_w[v], var_br_w[v] = get_mk_br(v)

        if report_every and flip % report_every == 0:
            print(f"[finisher] flips={flip:,} unsat={len(unsat_list)} best={best_unsat}")

    return best, False

#
def finisher_two_phase(
    clauses, nvars, a0, seed=0, max_flips=80_000_000,
    # predator params
    p_min=0.003, p_max=0.18,
    stall_window=600_000, restart_shake=128,
    w_inc=1.2, w_decay=0.9996, w_cap=120.0,
    # phase switch
    switch_u=160,
    # endgame params (match v3 vibe)
    classic_p=0.386,
    classic_report=500_000,
    # endgame hook
    hook_every=50_000,
    hook_budget=2_000,
):
    # 1) run predator until it finds a decent basin (or runs out of flips)
    flips1 = int(max_flips * 0.65)
    model, solved, st = finisher_predator_sole_sat(
        clauses=clauses, nvars=nvars, a0=a0, seed=seed,
        max_flips=flips1,
        p_min=p_min, p_max=p_max,
        stall_window=stall_window, restart_shake=restart_shake,
        w_inc=w_inc, w_decay=w_decay, w_cap=w_cap,
        report_every=100_000,
    )
    if solved:
        return model, True, {"phase": "predator", **st}

    # 2) if best is already small-ish, switch to classic endgame (no restarts)
    u = count_unsat(clauses, model)
    if u > switch_u:
        # still too big: just continue predator a bit more (but keep going)
        model2, solved2, st2 = finisher_predator_sole_sat(
            clauses=clauses, nvars=nvars, a0=model, seed=seed + 1337,
            max_flips=int(max_flips * 0.20),
            p_min=p_min, p_max=p_max,
            stall_window=stall_window, restart_shake=restart_shake,
            w_inc=w_inc, w_decay=w_decay, w_cap=w_cap,
            report_every=100_000,
        )
        if solved2:
            return model2, True, {"phase": "predator2", **st2}
        model = model2

    # 3) classic endgame: higher noise + exact local scoring; add repair hook
    #    (hook = krátké deterministické “repair” bursty když to stojí)
    best = model[:]
    best_u = count_unsat(clauses, best)

    flips_left = max_flips - flips1
    chunk = hook_every
    used = 0

    while used < flips_left and best_u > 0:
        # run classic for a chunk
        m2, ok2, stc = finisher_back_to_25_classic(
            clauses=clauses, nvars=nvars, a0=best, seed=seed + 9001 + used,
            max_flips=min(chunk, flips_left - used),
            report_every=classic_report,
            p=classic_p,
        )
        used += stc.get("flips", chunk)
        u2 = count_unsat(clauses, m2)
        if u2 < best_u:
            best, best_u = m2, u2
            if best_u == 0:
                return best, True, {"phase": "classic", "best_unsat": 0, "flips": flips1 + used}

        # endgame repair hook (krátký greedier repair na aktuální best)
        # idea: zkusit pár kroků vždy jen nad UNSAT klauzulemi (lokální “satisfaction” tlak)
        if best_u <= 96:
            tmp = best[:]
            # vezmi seznam UNSAT clause indexů
            u_ids = unsat_indices(clauses, [1 if b else 0 for b in tmp])
            if u_ids:
                import random
                rng = random.Random(seed + 424242 + used)
                rng.shuffle(u_ids)
                steps = 0
                for ci in u_ids[:min(len(u_ids), hook_budget)]:
                    cl = clauses[ci]
                    # flipni proměnnou z klauzule s nejmenším “damage” (min break aproximace)
                    bestv = None
                    bestscore = None
                    for lit in cl:
                        v = abs(lit) - 1
                        tmp[v] = not tmp[v]
                        uu = count_unsat(clauses, tmp)
                        tmp[v] = not tmp[v]
                        score = uu
                        if bestscore is None or score < bestscore:
                            bestscore = score
                            bestv = v
                    if bestv is not None:
                        tmp[bestv] = not tmp[bestv]
                        steps += 1
                        if steps >= hook_budget:
                            break
                u3 = count_unsat(clauses, tmp)
                if u3 < best_u:
                    best, best_u = tmp, u3
                    if best_u == 0:
                        return best, True, {"phase": "hook", "best_unsat": 0, "flips": flips1 + used}

    return best, False, {"phase": "two_phase_done", "best_unsat": best_u, "flips": flips1 + used}

#
def finisher_predator_sole_sat(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,

    # restart / stall
    stall_window: int = 600_000,
    restart_shake: int = 96,

    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9996,
    w_cap: float = 120.0,

    # guards (tvoje původní defaulty byly moc "na krátkým vodítku")
    snapback_gap: int = 250,     # dřív 48 -> to je zabiják
    basin_mult: float = 2.2,
    basin_abs: int = 350,        # dřív 220 -> moc nízko pro “tunneling”

    # chaos kick
    kick_after: int = 300_000,   # flips bez zlepšení
    kick_len: int = 100_000,     # délka kicku
    kick_p: float = 0.18,        # noise během kicku

    # endgame sniper switch
    sniper_u: int = 64,          # při dosažení -> přepnout na sniper
    sniper_flips: int = 8_000_000,
    sniper_p: float = 0.33,

    # tabu (jen v hard-core; během kicku se vypíná)
    use_tabu: bool = True,
    tabu_u_threshold: int = 128,
    tabu_tenure: int = 45,

    report_every: int = 100_000,
) -> Tuple[List[bool], bool, dict]:
    rng = random.Random(seed)
    m = len(clauses)

    a = a0[:]
    best = a[:]
    best_unsat = 10**9
    last_improve = 0
    flip = 0

    # --- CHAOS KICK state
    kick_left = 0

    # -----------------------------
    # vše níže je tvoje původní infrastruktura:
    # sat_count, sole_sat, unsat_list+pos, weights, rebuild_from_best(),
    # apply_flip_accurate(), mk/br score atd.
    # -----------------------------

    # --- helpers (tady nechávám jak jsi to měl)
    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # --- clause state
    sat_count = [0] * m
    sole_sat = [-1] * m

    # --- UNSAT list as packed array + positions
    unsat_list: List[int] = []
    pos = [-1] * m

    def add_unsat(ci: int) -> None:
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        p = pos[ci]
        if p == -1:
            return
        last = unsat_list[-1]
        unsat_list[p] = last
        pos[last] = p
        unsat_list.pop()
        pos[ci] = -1

    # --- weights
    w = [1.0] * m

    # --- init sat_count/sole_sat/unsat_list
    for ci, cl in enumerate(clauses):
        sc = 0
        sole_v = -1
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(1 if lit > 0 else -1, a[v]):
                sc += 1
                sole_v = v
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)
        elif sc == 1:
            sole_sat[ci] = sole_v

    # --- optional tabu
    tabu_until = [0] * nvars

    # --- accurate flip that maintains sat_count/sole_sat/unsat_list
    def apply_flip_accurate(v: int) -> None:
        av_old = a[v]
        a[v] = not a[v]

        # update every clause containing v (ty to máš už předpočítané; tady volám přes tvůj index)
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            cur = a[v]
            sign = 1 if lit > 0 else -1
            was_true = lit_true(sign, av_old)
            now_true = lit_true(sign, cur)

            if was_true == now_true:
                continue

            if was_true and (not now_true):
                # literal turned off
                if sc == 1:
                    # clause becomes UNSAT
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    # clause remains SAT with exactly one satisfier: need recompute sole_sat
                    sat_count[ci] = 1
                    # find the remaining satisfier variable
                    sole = -1
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(1 if lit2 > 0 else -1, a[vv]):
                            sole = vv
                            break
                    sole_sat[ci] = sole
                else:
                    sat_count[ci] = sc - 1
            else:
                # literal turned on
                if sc == 0:
                    # clause becomes SAT
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    # clause goes from 1->2 satisfiers
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    # --- rebuild from best + shake
    def rebuild_from_best(shake: int = 0) -> None:
        nonlocal a
        a = best[:]
        if shake > 0:
            for _ in range(shake):
                vv = rng.randrange(nvars)
                a[vv] = not a[vv]
        # recompute sat_count / sole_sat / unsat_list fully
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1
        for ci, cl in enumerate(clauses):
            sc = 0
            sole_v = -1
            for lit in cl:
                v = abs(lit) - 1
                if lit_true(1 if lit > 0 else -1, a[v]):
                    sc += 1
                    sole_v = v
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                sole_sat[ci] = sole_v
            else:
                sole_sat[ci] = -1

    # --- score helpers: mk + break from sole_sat
    # mk = how many UNSAT clauses would become SAT by flipping v
    # br = weighted break: how many SAT clauses would become UNSAT by flipping v
    def mk_br(v: int) -> Tuple[float, float]:
        mk = 0.0
        br = 0.0
        av_old = a[v]
        av_new = not av_old
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            sign = 1 if lit > 0 else -1
            was_true = lit_true(sign, av_old)
            now_true = lit_true(sign, av_new)
            if was_true == now_true:
                continue
            if (not was_true) and now_true:
                if sc == 0:
                    mk += w[ci]
            elif was_true and (not now_true):
                if sc == 1 and sole_sat[ci] == v:
                    br += w[ci]
        return mk, br

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    t0 = time.time()
    last_report = 0

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "best_unsat": 0}

        # best tracking
        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

            # endgame handoff: přepni na sniper hned když se dotkneš prahu
            if best_unsat <= sniper_u:
                print(f"[predator] target locked (u={best_unsat}), switching to sniper ...")
                rem = max(0, min(sniper_flips, max_flips - flip))
                """model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses, nvars=nvars, a0=best,
                    seed=seed + 777777,
                    max_flips=rem,
                    p=sniper_p,
                    report_every=100_000
                )"""
                # --- SAFE HANDOFF TO SNIPER (ENDGAME) ---
                model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses,
                    nvars=nvars,
                    a0=best,
                    seed=seed + 1337,
                    max_flips=sniper_flips,

                    # MID: drž vyšší šum, ať to ještě "hledá"
                    p_mid=sniper_p,  # třeba 0.33
                    p_mid_hard=sniper_p,  # třeba 0.33

                    # Endgame spouštěj AŽ fakt pozdě
                    endgame_at=24,  # <<< TADY je fix (ne 64)

                    # Endgame low-noise až pro posledních ~24
                    p_end=0.06,

                    tabu_tenure=45,
                    report_every=report_every,
                )

                if ok2:
                    st2 = dict(st2) if isinstance(st2, dict) else {}
                    st2.update({"phase": "sniper", "flips": flip + st2.get("flips", 0)})
                    return model2, True, st2
                # když sniper nedal 0, vrať se s jeho nejlepším modelem
                a = model2[:]
                # recompute structures
                rebuild_from_best(shake=0)
                # pokračuj v predatoru (ale už jsi nízko, většinou to brzy spadne)
                # nevracím hned, ať to můžeš dotlačit dál

        stalled = flip - last_improve

        # --- CHAOS KICK ---
        if stalled > kick_after and kick_left == 0:
            kick_left = kick_len
            print(f"!!! [kick] stagnation at best={best_unsat} -> chaos kick for {kick_len} flips")

        # --- GUARDS: meltdown + SOFT snapback ---
        # během kicku NIKDY nesnapbackuj (jinak je kick placebo)
        if kick_left == 0 and best_unsat > 0:
            # meltdown guard (ponecháme)
            if (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs):
                rebuild_from_best(shake=restart_shake)
                last_improve = flip
                continue

            # soft snapback: jen když jsme opravdu vysoko (u > basin_abs)
            if (u > best_unsat + snapback_gap) and (u > basin_abs):
                rebuild_from_best(shake=0)
                last_improve = flip
                continue

        # --- weights decay (jemně)
        if flip % 256 == 0:
            # decays all weights a bit
            for i in range(m):
                wi = w[i] * w_decay
                w[i] = wi if wi >= 1.0 else 1.0

        # --- compute p_eff ---
        if kick_left > 0:
            p_eff = min(p_max, max(kick_p, p_min))
            kick_left -= 1
        else:
            ratio = u / m
            p_eff = p_base * (0.10 + 2.8 * math.sqrt(max(1e-12, ratio)))
            # endgame floor (ať ti to nezamrzá na 0.02)
            if u <= 512: p_eff = max(p_eff, 0.030)
            if u <= 256: p_eff = max(p_eff, 0.025)
            if u <= 128: p_eff = max(p_eff, 0.020)
            if u <= 64:  p_eff = max(p_eff, 0.015)
            p_eff = float(max(p_min, min(p_max, p_eff)))

        # --- periodic restart on long stall (optionally)
        if stalled > stall_window and kick_left == 0:
            rebuild_from_best(shake=restart_shake)
            last_improve = flip
            continue

        # --- report
        if report_every and (flip - last_report) >= report_every:
            last_report = flip
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat}")

        # --- pick clause to work on
        ci = unsat_list[rng.randrange(u)]
        cl = clauses[ci]

        # --- choose variable: ProbSAT-ish with mk/br, with noise
        if rng.random() < p_eff:
            # noisy random pick
            lit = cl[rng.randrange(len(cl))]
            v = abs(lit) - 1
        else:
            best_v = None
            best_score = None
            for lit in cl:
                v0 = abs(lit) - 1

                # tabu only when not kicking
                if kick_left == 0 and use_tabu and u <= tabu_u_threshold and tabu_until[v0] > flip:
                    continue

                mk, br = mk_br(v0)
                # score: prefer high make, low break
                # (exponenty můžeš doladit, ale tohle je stabilní)
                score = (mk + 1e-6) ** 2.5 / ((br + 1e-6) ** 1.5)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v0

            if best_v is None:
                lit = cl[rng.randrange(len(cl))]
                v = abs(lit) - 1
            else:
                v = best_v

        # --- apply flip
        apply_flip_accurate(v)

        if kick_left == 0 and use_tabu and u <= tabu_u_threshold:
            tabu_until[v] = flip + tabu_tenure

        # --- clause weights pressure: bump current unsat clause
        wci = w[ci] + w_inc
        w[ci] = wci if wci <= w_cap else w_cap

        flip += 1

    return best, False, {"flips": flip, "best_unsat": best_unsat, "time_s": time.time() - t0}

#
import time, math, random
from typing import List, Tuple, Dict


def build_var_occ(clauses: List[List[int]], nvars: int) -> List[List[Tuple[int, int]]]:
    """Sestaví index: pro každou proměnnou seznam (index_klauzule, literál)."""
    occ = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            occ[v].append((ci, lit))
    return occ


def finisher_predator_sole_sat(
        clauses: List[List[int]],
        nvars: int,
        a0: List[bool],
        var_occ: List[List[Tuple[int, int]]] = None,
        seed: int = 0,
        max_flips: int = 50_000_000,

        # Noise parametry
        p_min: float = 0.003,
        p_max: float = 0.18,
        p_base: float = 0.10,

        # Restart / Stall
        stall_window: int = 600_000,
        restart_shake: int = 96,

        # Váhy (Clause Weighting)
        w_inc: float = 1.2,
        w_decay: float = 0.9996,
        w_cap: float = 120.0,

        # Měkké Guardy (Klíč k uvolnění z basinu)
        snapback_gap: int = 250,
        basin_mult: float = 2.2,
        basin_abs: int = 350,

        # Chaos Kick (Prolomení stagnace)
        kick_after: int = 300_000,
        kick_len: int = 100_000,
        kick_p: float = 0.18,

        # Endgame Sniper Switch
        sniper_u: int = 64,
        sniper_flips: int = 8_000_000,
        sniper_p: float = 0.33,

        # Tabu
        use_tabu: bool = True,
        tabu_u_threshold: int = 128,
        tabu_tenure: int = 45,

        report_every: int = 100_000,
) -> Tuple[List[bool], bool, dict]:
    # Inicializace indexu pokud chybí
    if var_occ is None:
        var_occ = build_var_occ(clauses, nvars)

    rng = random.Random(seed)
    m = len(clauses)
    a = a0[:]
    best = a[:]
    best_unsat = 10 ** 9
    last_improve = 0
    flip = 0
    kick_left = 0
    last_report = 0
    t0 = time.time()

    # --- Clause State Structures ---
    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    w = [1.0] * m
    tabu_until = [0] * nvars

    def add_unsat(ci: int):
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int):
        p = pos[ci]
        if p != -1:
            last = unsat_list[-1]
            unsat_list[p] = last
            pos[last] = p
            unsat_list.pop()
            pos[ci] = -1

    def lit_true(sign: int, aval: bool) -> bool:
        return aval if sign > 0 else (not aval)

    # Inicializace stavu klauzulí
    for ci, cl in enumerate(clauses):
        sc = 0
        sole_v = -1
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(1 if lit > 0 else -1, a[v]):
                sc += 1
                sole_v = v
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)
        elif sc == 1:
            sole_sat[ci] = sole_v

    def apply_flip_accurate(v: int):
        av_old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            sign = 1 if lit > 0 else -1
            was_true = lit_true(sign, av_old)
            now_true = not was_true  # Po flipu je to vždy opačně

            if was_true:  # Literál zhasnul
                if sc == 1:
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    sat_count[ci] = 1
                    # Najdi ten zbývající True literál
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(1 if lit2 > 0 else -1, a[vv]):
                            sole_sat[ci] = vv
                            break
                else:
                    sat_count[ci] = sc - 1
            else:  # Literál se rozsvítil
                if sc == 0:
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    def rebuild_from_best(shake: int = 0):
        nonlocal a
        a = best[:]
        if shake > 0:
            for _ in range(shake):
                vv = rng.randrange(nvars)
                a[vv] = not a[vv]
        unsat_list.clear()
        for i in range(m): pos[i] = -1
        for ci, cl in enumerate(clauses):
            sc = sum(1 for lit in cl if lit_true(1 if lit > 0 else -1, a[abs(lit) - 1]))
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                for lit in cl:
                    v = abs(lit) - 1
                    if lit_true(1 if lit > 0 else -1, a[v]):
                        sole_sat[ci] = v;
                        break
            else:
                sole_sat[ci] = -1

    def mk_br(v: int) -> Tuple[float, float]:
        mk, br = 0.0, 0.0
        cur_v = a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            is_pos = (lit > 0)
            # Pokud se flipem literál stane True (byl False)
            if (is_pos and not cur_v) or (not is_pos and cur_v):
                if sc == 0: mk += w[ci]
            else:  # Literál se stane False (byl True)
                if sc == 1 and sole_sat[ci] == v: br += w[ci]
        return mk, br

    # --- HLAVNÍ LOOP ---
    while flip < max_flips:
        u = len(unsat_list)
        if u == 0: return a, True, {"flips": flip, "best_unsat": 0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip
            if best_unsat <= sniper_u:
                print(f"[predator] target locked (u={best_unsat}), calling sniper...")
                # Zde by se volala tvá sniper funkce, pokud ji máš v main()
                # Pro tento script vracíme model k dočištění sniperem
                return best, False, {"phase": "handoff", "u": best_unsat, "flips": flip}

        stalled = flip - last_improve

        # CHAOS KICK Logic
        if stalled > kick_after and kick_left == 0:
            kick_left = kick_len
            print(f"!!! [kick] stagnation at {best_unsat} -> chaos for {kick_len} flips")

        # SOFT GUARDS
        if kick_left == 0:
            if (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs):
                rebuild_from_best(shake=restart_shake)
                last_improve = flip;
                continue
            if (u > best_unsat + snapback_gap) and (u > basin_abs):
                rebuild_from_best(shake=0)
                last_improve = flip;
                continue

        # Weight Decay
        if (flip & 255) == 0:
            for i in range(m):
                if w[i] > 1.0: w[i] = max(1.0, w[i] * w_decay)

        # Noise Schedule
        if kick_left > 0:
            p_eff = kick_p
            kick_left -= 1
        else:
            ratio = u / m
            p_eff = p_base * (0.1 + 2.8 * math.sqrt(ratio))
            if u <= 128: p_eff = max(p_eff, 0.02)
            p_eff = max(p_min, min(p_max, p_eff))

        if report_every and (flip - last_report) >= report_every:
            last_report = flip
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat}")

        # Pick Clause
        ci = unsat_list[rng.randrange(u)]
        cl = clauses[ci]

        # Pick Variable
        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v, b_score = -1, -1.0
            for lit in cl:
                v0 = abs(lit) - 1
                if kick_left == 0 and use_tabu and u <= tabu_u_threshold and tabu_until[v0] > flip:
                    continue
                mk, br = mk_br(v0)
                score = (mk + 1e-6) ** 2.5 / (br + 1e-6) ** 1.5
                if score > b_score:
                    b_score, best_v = score, v0
            v = best_v if best_v != -1 else abs(cl[rng.randrange(len(cl))]) - 1

        apply_flip_accurate(v)
        if kick_left == 0 and use_tabu and u <= tabu_u_threshold:
            tabu_until[v] = flip + tabu_tenure

        # Clause Weighting Bump
        w[ci] = min(w_cap, w[ci] + w_inc)
        flip += 1

    return best, False, {"flips": flip, "best_unsat": best_unsat, "time_s": time.time() - t0}

#
import math, random, time
from typing import List, Tuple, Dict, Optional

def build_var_occ(clauses: List[List[int]], nvars: int) -> List[List[Tuple[int, int]]]:
    occ = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            occ[abs(lit) - 1].append((ci, lit))
    return occ


def finisher_predator_sole_sat_vFinal(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,

    # stall/restart
    stall_window: int = 600_000,
    restart_shake: int = 96,

    # weights
    w_inc: float = 1.2,
    w_decay: float = 0.9996,
    w_cap: float = 120.0,

    # guards (tuned for low-u start ~100)
    snapback_gap: int = 260,
    basin_mult: float = 2.4,
    basin_abs: int = 520,

    # chaos kick (with cooldown!)
    kick_after: int = 450_000,
    kick_len: int = 30_000,
    kick_p: float = 0.16,
    kick_cool_len: int = 220_000,

    # sniper
    sniper_u: int = 64,
    sniper_flips: int = 10_000_000,
    sniper_p: float = 0.33,

    # tabu (disabled during kick)
    use_tabu: bool = True,
    tabu_u_threshold: int = 128,
    tabu_tenure: int = 45,

    report_every: int = 100_000,
) -> Tuple[List[bool], bool, Dict]:

    print(f"[finisher:init] p_min={p_min} p_max={p_max} p_base={p_base}")

    rng = random.Random(seed)
    m = len(clauses)

    # IMPORTANT: uses the a0 you pass (your spectral/core seed)
    a = a0[:]
    best = a[:]
    best_unsat = 10**9
    last_improve = 0
    flip = 0
    last_report = 0
    t0 = time.time()

    # build occurrences locally (fast enough, and avoids name mismatches)
    var_occ = build_var_occ(clauses, nvars)

    # clause state
    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    # weights + tabu
    w = [1.0] * m
    tabu_until = [0] * nvars

    def lit_true(lit: int, aval: bool) -> bool:
        return aval if lit > 0 else (not aval)

    def add_unsat(ci: int):
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int):
        p0 = pos[ci]
        if p0 != -1:
            last = unsat_list[-1]
            unsat_list[p0] = last
            pos[last] = p0
            unsat_list.pop()
            pos[ci] = -1

    # init state
    for ci, cl in enumerate(clauses):
        sc = 0
        sole_v = -1
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                sc += 1
                sole_v = v
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)
        elif sc == 1:
            sole_sat[ci] = sole_v

    def rebuild_from_best(shake: int = 0):
        nonlocal a
        a = best[:]
        if shake > 0:
            for _ in range(shake):
                vv = rng.randrange(nvars)
                a[vv] = not a[vv]

        unsat_list.clear()
        for i in range(m):
            pos[i] = -1

        for ci, cl in enumerate(clauses):
            sc = 0
            sole_v = -1
            for lit in cl:
                v = abs(lit) - 1
                if lit_true(lit, a[v]):
                    sc += 1
                    sole_v = v
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                sole_sat[ci] = sole_v
            else:
                sole_sat[ci] = -1

    def apply_flip_accurate(v: int):
        av_old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, av_old)
            now_true = not was_true  # because only v flipped

            if was_true and (not now_true):
                if sc == 1:
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    sat_count[ci] = 1
                    sole_sat[ci] = -1
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(lit2, a[vv]):
                            sole_sat[ci] = vv
                            break
                else:
                    sat_count[ci] = sc - 1
            else:
                if sc == 0:
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    def mk_br(v: int) -> Tuple[float, float]:
        mk = 0.0
        br = 0.0
        cur_v = a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            is_pos = (lit > 0)
            becomes_true = (is_pos and (not cur_v)) or ((not is_pos) and cur_v)

            if becomes_true:
                if sc == 0:
                    mk += w[ci]
            else:
                if sc == 1 and sole_sat[ci] == v:
                    br += w[ci]
        return mk, br

    # chaos kick state
    kick_left = 0
    kick_cool = 0

    while flip < max_flips:
        u = len(unsat_list)
        if u == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        if best_unsat <= 160:
            model2, ok2, _ = finisher_classic_to_zero_sniper(
                clauses=clauses, nvars=nvars, a0=best,
                seed=seed + 12345 + best_unsat,
                max_flips=250_000,
                #p=sniper_p,
                report_every=0
            )
            if ok2:
                return model2, True, {"phase": "sniper_burst", "best_unsat": 0, "flips": flip}
            # pokračuj z lepšího, pokud se zlepšil
            # (ať to predator nedojebe)
            a = model2[:]
            best = a[:]
            rebuild_from_best(shake=0)

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

            if best_unsat <= sniper_u:
                print(f"[predator] target locked (u={best_unsat}), calling sniper ...")
                rem = max_flips - flip
                """model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses,
                    nvars=nvars,
                    a0=best,
                    seed=seed + 777777,
                    max_flips=min(sniper_flips, rem),
                    p=sniper_p,
                    report_every=report_every
                )"""
                # --- SAFE HANDOFF TO SNIPER (ENDGAME) ---
                model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses,
                    nvars=nvars,
                    a0=best,
                    seed=seed + 1337,
                    max_flips=sniper_flips,

                    # MID: drž vyšší šum, ať to ještě "hledá"
                    p_mid=sniper_p,  # třeba 0.33
                    p_mid_hard=sniper_p,  # třeba 0.33

                    # Endgame spouštěj AŽ fakt pozdě
                    endgame_at=24,  # <<< TADY je fix (ne 64)

                    # Endgame low-noise až pro posledních ~24
                    p_end=0.06,

                    tabu_tenure=45,
                    report_every=report_every,
                )

                if ok2:
                    st2 = dict(st2) if isinstance(st2, dict) else {}
                    st2.update({"phase": "sniper", "handoff_u": best_unsat})
                    return model2, True, st2

                # continue from sniper's best attempt (no random reset)
                a = model2[:]
                best = a[:]
                best_unsat = 10**9
                rebuild_from_best(shake=0)
                last_improve = flip

        stalled = flip - last_improve

        # cooldown tick
        if kick_cool > 0:
            kick_cool -= 1

        # chaos kick trigger (only if not cooling down)
        if stalled > kick_after and kick_left == 0 and kick_cool == 0:
            kick_left = kick_len
            kick_cool = kick_cool_len
            print(f"!!! [kick] stagnation at best={best_unsat} -> chaos for {kick_len} flips (cooldown {kick_cool_len})")

        # guards (disabled during kick)
        if kick_left == 0 and best_unsat > 0:
            # adapt basin_abs lightly to current best (so it behaves well at best~100)
            #basin_abs_eff = max(basin_abs, best_unsat + 320)
            basin_abs_eff = max(180, best_unsat + 140)

            if (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs_eff):
                rebuild_from_best(shake=restart_shake)
                last_improve = flip
                continue
            if (u > best_unsat + snapback_gap) and (u > basin_abs_eff):
                rebuild_from_best(shake=0)
                last_improve = flip
                continue

        # weights decay
        if (flip & 255) == 0:
            for i in range(m):
                if w[i] > 1.0:
                    w[i] = max(1.0, w[i] * w_decay)

        # noise schedule
        if kick_left > 0:
            p_eff = min(p_max, max(p_min, kick_p))
            kick_left -= 1
            tabu_active = False  # IMPORTANT: disable tabu during kick
        else:
            ratio = u / m
            p_eff = p_base * (0.10 + 2.8 * math.sqrt(max(1e-12, ratio)))
            # endgame floors
            if u <= 512:
                p_eff = max(p_eff, 0.030)
            if u <= 256:
                p_eff = max(p_eff, 0.025)
            if u <= 128:
                p_eff = max(p_eff, 0.020)
            if u <= 64:
                p_eff = max(p_eff, 0.015)
            p_eff = float(max(p_min, min(p_max, p_eff)))
            tabu_active = bool(use_tabu and (u <= tabu_u_threshold))

        # long stall restart (not during kick)
        if stalled > stall_window and kick_left == 0:
            rebuild_from_best(shake=restart_shake)
            last_improve = flip
            continue

        # report
        if report_every and (flip - last_report) >= report_every:
            last_report = flip
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat}")

        # pick an UNSAT clause
        ci = unsat_list[rng.randrange(u)]
        cl = clauses[ci]

        # choose variable
        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = -1
            best_score = -1.0
            for lit in cl:
                v0 = abs(lit) - 1
                if tabu_active and tabu_until[v0] > flip:
                    continue
                mk, br = mk_br(v0)
                score = (mk + 1e-6) ** 2.5 / ((br + 1e-6) ** 1.5)
                if score > best_score:
                    best_score = score
                    best_v = v0
            v = best_v if best_v != -1 else abs(cl[rng.randrange(len(cl))]) - 1

        apply_flip_accurate(v)

        if tabu_active:
            tabu_until[v] = flip + tabu_tenure

        # bump weight
        w[ci] = min(w_cap, w[ci] + w_inc)

        flip += 1

    return best, False, {"phase": "predator", "flips": flip, "best_unsat": best_unsat, "time_s": time.time() - t0}


#
import random, time, math
from typing import List, Tuple, Dict, Optional


def finisher_classic_to_zero_sniper(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,
    report_every: int = 5_000_000,

    # Backward-compat: pokud někdo volá p=..., ber to jako p_mid
    p: Optional[float] = None,

    # MIDGAME
    p_mid: float = 0.33,
    p_mid_hard: float = 0.33,

    # ENDGAME trigger + behavior
    endgame_at: int = 24,          # <= 24 teprve sniper mode
    p_end: float = 0.06,           # nízký šum jen v endgame

    # endgame stabilizace
    snapback_gap: int = 120,       # když current u ujede nad best o tolik, vrať best
    stall_window: int = 400_000,   # když se best dlouho nehýbe, udělej micro-jolt
    micro_jolt_len: int = 15_000,  # krátká perioda vyššího šumu
    micro_jolt_p: float = 0.18,

    # tabu v endgame
    tabu_tenure: int = 25,

    # focus sampling
    focus_pool_k: int = 32,
    focus_top: int = 8,

    # clause pressure (light)
    w_inc: float = 0.5,
    w_cap: float = 18.0,

    # probsat exponents (mid)
    p_pow: float = 1.6,
    q_pow: float = 1.0,
) -> Tuple[List[bool], bool, Dict]:
    """
    Two-phase finisher:
      MID: ProbSAT-ish scoring (mk/br) with moderate noise.
      SNIPER (only when best <= endgame_at): focused clause sampling + min-break + tabu + low noise.
    Snapback prevents the "u jumps to 250 and never returns" failure mode.
    """

    rng = random.Random(seed)
    a = [bool(x) for x in a0]
    m = len(clauses)

    # backward compat: allow p=... caller
    if p is not None:
        p_mid = float(p)
        p_mid_hard = float(p)

    # build var occurrences: var_occ[v] = [(ci, sign)]
    var_occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    def lit_true(sign: int, val: bool) -> bool:
        return val if sign > 0 else (not val)

    # sat_count + unsat_list + pos
    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    def add_unsat(ci: int) -> None:
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        pidx = pos[ci]
        if pidx != -1:
            last = unsat_list[-1]
            unsat_list[pidx] = last
            pos[last] = pidx
            unsat_list.pop()
            pos[ci] = -1

    # init counts
    for ci, cl in enumerate(clauses):
        sc = 0
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            if lit_true(sign, a[v]):
                sc += 1
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)

    # weights for focus pressure
    cw = [1.0] * m

    # best tracking
    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0

    # tabu only used in endgame
    tabu_until = [0] * nvars

    # micro jolt state
    jolt_left = 0

    def count_break_if_flip(v: int) -> int:
        """How many currently SAT clauses would become UNSAT if flip v."""
        old = a[v]
        new = not old
        br = 0
        for ci, sign in var_occ[v]:
            sc = sat_count[ci]
            if sc == 0:
                continue
            # does this literal toggle?
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was == now:
                continue
            # if it turns off and clause had sc==1 -> becomes unsat
            if was and (not now) and sc == 1:
                br += 1
        return br

    def count_make_if_flip(v: int) -> int:
        """How many currently UNSAT clauses would become SAT if flip v."""
        old = a[v]
        new = not old
        mk = 0
        for ci, sign in var_occ[v]:
            sc = sat_count[ci]
            if sc != 0:
                continue
            was = lit_true(sign, old)
            now = lit_true(sign, new)
            if was == now:
                continue
            if (not was) and now:
                mk += 1
        return mk

    def apply_flip(v: int) -> None:
        old = a[v]
        a[v] = not old

        for (ci, sign) in var_occ[v]:
            was = lit_true(sign, old)
            now = not was  # toggles for that sign always

            if was == now:
                continue

            old_sc = sat_count[ci]
            new_sc = old_sc + (1 if now else -1)
            sat_count[ci] = new_sc

            if old_sc == 0 and new_sc > 0:
                rem_unsat(ci)
            elif old_sc > 0 and new_sc == 0:
                add_unsat(ci)

    def pick_unsat_clause_focused() -> int:
        """Focus pool sampling biased by cw weights."""
        u = len(unsat_list)
        if u == 0:
            return 0
        k = min(focus_pool_k, u)
        sample = [unsat_list[rng.randrange(u)] for _ in range(k)]
        # sort by weight desc
        sample.sort(key=lambda ci: cw[ci], reverse=True)
        topk = sample[: min(focus_top, len(sample))]
        return topk[rng.randrange(len(topk))]

    t0 = time.time()

    for flip in range(1, max_flips + 1):
        u = len(unsat_list)
        if u == 0:
            return a, True, {"phase": "sniper", "flips": flip, "time_s": time.time() - t0}

        # track best
        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        # decide mode based on BEST (not current!)
        endgame = (best_unsat <= endgame_at)
        mode = "SNIPER" if endgame else "MID"

        # snapback: když current u brutálně ujede, vrať best
        if u > best_unsat + snapback_gap:
            a = best[:]
            # full rebuild counts (safe, a bit slower but stable)
            unsat_list.clear()
            for i in range(m):
                pos[i] = -1
            for ci, cl in enumerate(clauses):
                sc = 0
                for lit in cl:
                    v = abs(lit) - 1
                    sign = 1 if lit > 0 else -1
                    if lit_true(sign, a[v]):
                        sc += 1
                sat_count[ci] = sc
                if sc == 0:
                    add_unsat(ci)
            continue

        # stall → micro-jolt (krátce zvedni noise, ale jen když nejsi SAT)
        stalled = flip - last_improve
        if stalled > stall_window and jolt_left == 0:
            jolt_left = micro_jolt_len

        # effective noise
        if jolt_left > 0:
            p_eff = micro_jolt_p
            jolt_left -= 1
        else:
            if endgame:
                p_eff = p_end
            else:
                # keep your "works-for-you" behavior
                p_eff = p_mid_hard if u <= 128 else p_mid

        # report
        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={u} best={best_unsat} p={p_eff:.3f} mode={mode}")

        # pick clause
        if endgame:
            ci = pick_unsat_clause_focused()
        else:
            ci = unsat_list[rng.randrange(u)]

        # gentle clause pressure (helps focus set)
        if endgame:
            cw[ci] = min(w_cap, cw[ci] + w_inc)

        lits = clauses[ci]

        # pick variable
        if rng.random() < p_eff:
            v = abs(lits[rng.randrange(len(lits))]) - 1
        else:
            # MID: probsat mk/br ; ENDGAME: min-break + tabu
            eps = 1e-6
            best_v = abs(lits[0]) - 1
            best_s = -1e100

            if endgame:
                # prefer min-break
                best_br = 10**9
                cands = []
                for lit in lits:
                    vv = abs(lit) - 1
                    if tabu_until[vv] > flip:
                        continue
                    br = count_break_if_flip(vv)
                    if br < best_br:
                        best_br = br
                        cands = [vv]
                    elif br == best_br:
                        cands.append(vv)
                if cands:
                    v = cands[rng.randrange(len(cands))]
                else:
                    v = abs(lits[rng.randrange(len(lits))]) - 1
            else:
                for lit in lits:
                    vv = abs(lit) - 1
                    br = count_break_if_flip(vv)
                    mk = count_make_if_flip(vv)
                    s = ((mk + eps) ** p_pow) / ((br + eps) ** q_pow)
                    s *= (1.0 + 0.01 * (rng.random() - 0.5))
                    if s > best_s:
                        best_s = s
                        best_v = vv
                v = best_v

        apply_flip(v)

        if endgame:
            tabu_until[v] = flip + tabu_tenure

    # out of flips
    return best, (best_unsat == 0), {"phase": "sniper", "flips": max_flips, "best_unsat": best_unsat, "time_s": time.time() - t0}

#

import time, math, random
from typing import List, Tuple, Dict, Optional

def build_var_occ(clauses: List[List[int]], nvars: int) -> List[List[Tuple[int, int]]]:
    occ = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            occ[abs(lit) - 1].append((ci, lit))
    return occ


def finisher_predator_sole_sat_vFinal(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 50_000_000,

    # noise (SPRÁVNÉ defaulty)
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,

    # stall / restart
    stall_window: int = 900_000,
    restart_shake: int = 96,

    # endgame handoff
    sniper_u: int = 24,          # <<< finální: sniper až fakt pozdě
    sniper_flips: int = 12_000_000,
    sniper_p_mid: float = 0.33,
    sniper_p_end: float = 0.06,
    sniper_call_after: int = 300_000,  # <<< sniper jen po stagnaci

    # tabu near end
    use_tabu: bool = True,
    tabu_u_threshold: int = 96,
    tabu_tenure: int = 35,

    report_every: int = 100_000,
) -> Tuple[List[bool], bool, Dict]:

    rng = random.Random(seed)
    m = len(clauses)
    a = a0[:]

    var_occ = build_var_occ(clauses, nvars)

    # clause state
    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    tabu_until = [0] * nvars

    def lit_true(lit: int, aval: bool) -> bool:
        return aval if lit > 0 else (not aval)

    def add_unsat(ci: int):
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def rem_unsat(ci: int):
        p0 = pos[ci]
        if p0 == -1:
            return
        last = unsat_list[-1]
        unsat_list[p0] = last
        pos[last] = p0
        unsat_list.pop()
        pos[ci] = -1

    def rebuild_full_from(assign: List[bool]):
        nonlocal a
        a = assign[:]
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1

        for ci, cl in enumerate(clauses):
            sc = 0
            sole_v = -1
            for lit in cl:
                v = abs(lit) - 1
                if lit_true(lit, a[v]):
                    sc += 1
                    sole_v = v
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                sole_sat[ci] = sole_v
            else:
                sole_sat[ci] = -1

    # init
    rebuild_full_from(a)

    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0
    last_report = 0
    t0 = time.time()

    print(f"[finisher:init] p_min={p_min} p_max={p_max} p_base={p_base}")

    def apply_flip_accurate(v: int):
        av_old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, av_old)
            now_true = not was_true  # flip toggles this literal

            if was_true and (not now_true):
                if sc == 1:
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    sat_count[ci] = 1
                    sole_sat[ci] = -1
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(lit2, a[vv]):
                            sole_sat[ci] = vv
                            break
                else:
                    sat_count[ci] = sc - 1
            else:
                if sc == 0:
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    def mk_br(v: int) -> Tuple[float, float]:
        mk = 0.0
        br = 0.0
        cur_v = a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            is_pos = (lit > 0)
            becomes_true = (is_pos and (not cur_v)) or ((not is_pos) and cur_v)

            if becomes_true:
                if sc == 0:
                    mk += 1.0
            else:
                if sc == 1 and sole_sat[ci] == v:
                    br += 1.0
        return mk, br

    def core_based_shake(shake: int):
        """Flip random vars that appear in current UNSAT clauses."""
        if shake <= 0:
            return
        if not unsat_list:
            return
        hard = []
        seen = set()
        # sample subset if huge
        sample_u = unsat_list if len(unsat_list) <= 2048 else [unsat_list[rng.randrange(len(unsat_list))] for _ in range(2048)]
        for ci in sample_u:
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if v not in seen:
                    seen.add(v)
                    hard.append(v)
        if not hard:
            return
        for _ in range(shake):
            v = hard[rng.randrange(len(hard))]
            apply_flip_accurate(v)

    while last_report < max_flips:
        flip = last_report + 1  # we use last_report as flip counter holder (tiny trick to keep code simple)
        last_report = flip

        u = len(unsat_list)
        if u == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve = flip

        stalled = flip - last_improve

        # SNIPER HANDOFF: only if best is low AND we actually stalled
        if best_unsat <= sniper_u and stalled >= sniper_call_after:
            print(f"[predator] handoff to sniper (best={best_unsat}, stalled={stalled})")
            model2, ok2, st2 = finisher_classic_to_zero_sniper(
                clauses=clauses,
                nvars=nvars,
                a0=best,
                seed=seed + 1337,
                max_flips=sniper_flips,
                report_every=report_every,

                p_mid=sniper_p_mid,
                p_mid_hard=sniper_p_mid,
                endgame_at=sniper_u,
                p_end=sniper_p_end,

                tabu_tenure=25,
            )
            if ok2:
                st2 = dict(st2) if isinstance(st2, dict) else {}
                st2.update({"phase": "sniper"})
                return model2, True, st2

            # continue from best we had (don’t let sniper drift)
            rebuild_full_from(best)
            last_improve = flip

        # STALL RESTART: core-based shake around best
        if stall_window > 0 and stalled >= stall_window:
            rebuild_full_from(best)
            core_based_shake(restart_shake)
            last_improve = flip
            continue

        # noise schedule
        ratio = u / m
        p_eff = p_base * (0.25 + 2.2 * math.sqrt(max(1e-12, ratio)))
        if u <= 256:
            p_eff = max(p_eff, 0.04)
        if u <= 128:
            p_eff = max(p_eff, 0.03)
        if u <= 64:
            p_eff = max(p_eff, 0.02)
        p_eff = float(max(p_min, min(p_max, p_eff)))

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat}")

        # robust clause pick
        if not unsat_list:
            rebuild_full_from(best)
            continue
        ci = unsat_list[rng.randrange(len(unsat_list))]
        cl = clauses[ci]

        # choose variable
        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = -1
            best_score = -1.0
            tabu_active = bool(use_tabu and u <= tabu_u_threshold)
            for lit in cl:
                v0 = abs(lit) - 1
                if tabu_active and tabu_until[v0] > flip:
                    continue
                mk, br = mk_br(v0)
                score = (mk + 1e-6) ** 2.3 / ((br + 1e-6) ** 1.4)
                if score > best_score:
                    best_score = score
                    best_v = v0
            v = best_v if best_v != -1 else abs(cl[rng.randrange(len(cl))]) - 1

        apply_flip_accurate(v)

        if use_tabu and u <= tabu_u_threshold:
            tabu_until[v] = flip + tabu_tenure

    return best, False, {"phase": "predator", "flips": max_flips, "best_unsat": best_unsat, "time_s": time.time() - t0}

import random, time
from typing import List, Tuple, Dict, Optional

def finisher_classic_to_zero_sniper(
    clauses: List[List[int]],
    nvars: int,
    a0: List[bool],
    seed: int = 0,
    max_flips: int = 10_000_000,
    report_every: int = 200_000,

    # backward compat (kdyby někde zůstalo p=...)
    p: Optional[float] = None,

    # mid params
    p_mid: float = 0.33,
    p_mid_hard: float = 0.33,

    # endgame
    endgame_at: int = 24,
    p_end: float = 0.06,

    # stabilita (zásadní)
    snapback_gap: int = 120,     # current u > best+gap -> teleport na best
    stall_window: int = 500_000, # dlouhá stagnace -> krátký jolt
    jolt_len: int = 15_000,
    jolt_p: float = 0.18,

    # tabu
    tabu_tenure: int = 25,
) -> Tuple[List[bool], bool, Dict]:

    rng = random.Random(seed)
    a = a0[:]
    m = len(clauses)

    if p is not None:
        p_mid = float(p)
        p_mid_hard = float(p)

    # occurrences (ci, lit)
    var_occ = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_occ[abs(lit) - 1].append((ci, lit))

    def lit_true(lit: int, aval: bool) -> bool:
        return aval if lit > 0 else (not aval)

    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m
    tabu_until = [0] * nvars

    def add_unsat(ci: int):
        if pos[ci] != -1:
            return
        pos[ci] = len(unsat_list)
        unsat_list.append(ci)

    def rem_unsat(ci: int):
        p0 = pos[ci]
        if p0 == -1:
            return
        last = unsat_list[-1]
        unsat_list[p0] = last
        pos[last] = p0
        unsat_list.pop()
        pos[ci] = -1

    def rebuild():
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1
        for ci, cl in enumerate(clauses):
            sc = 0
            for lit in cl:
                v = abs(lit) - 1
                if lit_true(lit, a[v]):
                    sc += 1
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)

    rebuild()

    def break_count(v: int) -> int:
        br = 0
        old = a[v]
        new = not old
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            if sc != 1:
                continue
            was = lit_true(lit, old)
            now = lit_true(lit, new)
            if was and (not now):
                br += 1
        return br

    def apply_flip(v: int):
        old = a[v]
        a[v] = not old
        for ci, lit in var_occ[v]:
            was = lit_true(lit, old)
            now = not was
            if was == now:
                continue
            sc = sat_count[ci]
            nsc = sc + (1 if now else -1)
            sat_count[ci] = nsc
            if sc == 0 and nsc > 0:
                rem_unsat(ci)
            elif sc > 0 and nsc == 0:
                add_unsat(ci)

    best = a[:]
    best_u = len(unsat_list)
    last_improve = 0
    jolt_left = 0
    t0 = time.time()

    for flip in range(1, max_flips + 1):
        u = len(unsat_list)
        if u == 0:
            return a, True, {"flips": flip, "time_s": time.time() - t0}

        if u < best_u:
            best_u = u
            best = a[:]
            last_improve = flip

        endgame = (best_u <= endgame_at)
        mode = "SNIPER" if endgame else "MID"

        # leash
        if u > best_u + snapback_gap:
            a = best[:]
            rebuild()
            last_improve = flip
            continue

        stalled = flip - last_improve
        if stalled > stall_window and jolt_left == 0:
            jolt_left = jolt_len

        if jolt_left > 0:
            p_eff = jolt_p
            jolt_left -= 1
        else:
            if endgame:
                p_eff = p_end
            else:
                p_eff = p_mid_hard if u <= 128 else p_mid

        if report_every and (flip % report_every) == 0:
            print(f"[finisher] flips={flip:,} unsat={u} best={best_u} p={p_eff:.3f} mode={mode}")

        if not unsat_list:
            rebuild()
            continue

        ci = unsat_list[rng.randrange(len(unsat_list))]
        cl = clauses[ci]

        # choose var
        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            # endgame: min-break + tabu ; mid: also min-break (simple and stable)
            best_v = None
            best_br = 10**9
            for lit in cl:
                vv = abs(lit) - 1
                if endgame and tabu_until[vv] > flip:
                    continue
                br = break_count(vv)
                if br < best_br:
                    best_br = br
                    best_v = vv
                    if br == 0:
                        break
            if best_v is None:
                best_v = abs(cl[rng.randrange(len(cl))]) - 1
            v = best_v

        apply_flip(v)
        if endgame:
            tabu_until[v] = flip + tabu_tenure

    return best, (best_u == 0), {"flips": max_flips, "best_unsat": best_u, "time_s": time.time() - t0}


# ---------- IO ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--mode", choices=["unsat_hadamard","sat"], default="unsat_hadamard")
    ap.add_argument("--rho", type=float, default=0.734296875)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--cR", type=float, default=10.0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--sigma_up", type=float, default=0.045)
    ap.add_argument("--neighbor_atten", type=float, default=0.9495)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--couple", type=int, default=1)  # 1=enable, 0=disable
    ap.add_argument("--power_iters", type=int, default=60)
    ap.add_argument("--score_norm_alpha", type=float, default=0.5)
    ap.add_argument("--bias_weight", type=float, default=0.10)
    ap.add_argument("--polish", type=int, default=300_000)
    ap.add_argument("--dump_assign", type=str, default="")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--godmode", action="store_true",
        help="Apply the exact parameter pack you posted: cR=10, L=3, rho=0.734296875, zeta0=0.4, sigma_up=0.045, "
             "neighbor_atten=0.9495, seed=42, couple=1, d=6.")
    # do parse_args()
    ap.add_argument("--finisher_flips", type=int, default=80_000_000)
    ap.add_argument("--p_min", type=float, default=0.003)
    ap.add_argument("--p_max", type=float, default=0.18)
    ap.add_argument("--p_base", type=float, default=0.10)

    ap.add_argument("--drift_window", type=int, default=200_000)
    ap.add_argument("--stall_window", type=int, default=800_000)
    ap.add_argument("--restart_shake", type=int, default=96)
    ap.add_argument("--w_inc", type=float, default=1.0)
    ap.add_argument("--w_decay", type=float, default=0.9996)
    ap.add_argument("--w_cap", type=float, default=40.0)
    # --- core-push (deterministic tightening before finisher) ---
    ap.add_argument("--core_seed_loops", type=int, default=0)
    ap.add_argument("--core_rounds", type=int, default=0)
    ap.add_argument("--core_step0", type=float, default=0.12)
    ap.add_argument("--core_decay", type=float, default=0.90)
    ap.add_argument("--core_per_clause_cap", type=int, default=2)
    ap.add_argument("--core_beta", type=float, default=1.3)
    ap.add_argument("--core_bias_damp", type=float, default=0.70)

    # --- shake (escape) ---
    ap.add_argument("--shake_rounds", type=int, default=0)
    ap.add_argument("--shake_q0", type=float, default=0.012)
    ap.add_argument("--shake_decay", type=float, default=0.92)
    ap.add_argument("--shake_local_flips", type=int, default=250_000)
    ap.add_argument("--shake_hops", type=int, default=2)
    ap.add_argument("--shake_sample", type=int, default=24)


    return ap.parse_args()

def main():

    print("\n*** SOLVER DEMO ***")

    print("\n[theory] generating theory params ...")

    args = parse_args()
    if args.godmode:
        args.cR = 10.0; args.L = 3; args.rho = 0.734296875; args.zeta0 = 0.40
        args.sigma_up = 0.045; args.neighbor_atten = 0.9495; args.seed = 42
        args.couple = 1; args.d = 6; args.mode = "unsat_hadamard"; args.power_iters = max(args.power_iters, 60)

    nvars, clauses = parse_dimacs(args.cnf)
    if args.subset and args.subset > 0:
        clauses = clauses[:args.subset]
    C = len(clauses)

    params = theory_params(C=C, want_sigma=0.02, cR=12, L=4, zeta0=0.4,
                      rho_lock=0.734296875, tau_hint=0.40, mu_hint=0.002)
    print("optimal params: ",params)

    print("\n[spectral] generating spectral seed ...")

    t0 = time.time()
    assign, Phi = build_seed_assignment(
        clauses, nvars, mode=args.mode, cR=args.cR, L=args.L,
        rho=args.rho, zeta0=args.zeta0, sigma_up=args.sigma_up,
        neighbor_atten=args.neighbor_atten, seed=args.seed, couple=bool(args.couple), d=args.d,
        power_iters=args.power_iters, score_norm_alpha=args.score_norm_alpha, bias_weight=args.bias_weight
    )

    # optional: core push tightening before polish/finisher
    if args.core_seed_loops > 0 and args.core_rounds > 0:
        print("\n[spectral] core push ...\n")
        assign = core_push(
            clauses, nvars, assign,
            seed=args.seed,
            loops=args.core_seed_loops,
            rounds=args.core_rounds,
            step0=args.core_step0,
            decay=args.core_decay,
            per_clause_cap=args.core_per_clause_cap,
            beta=args.core_beta,
            bias_damp=args.core_bias_damp,
        )


    t1 = time.time()
    if args.polish > 0:
        assign = greedy_polish(clauses, assign, flips=args.polish, seed=args.seed)
    unsat = count_unsat(clauses, assign)
    t2 = time.time()


    print("\n=== SPECTRAL REPORT ===")
    print(f"File              : {os.path.basename(args.cnf)}")
    print(f"Clauses (C)       : {C}")
    print(f"Vars (n)          : {nvars}")
    print(f"Mode              : {args.mode}")
    print(f"rho, zeta0        : {args.rho}, {args.zeta0}")
    print(f"cR, L             : {args.cR}, {args.L}")
    print(f"sigma_up          : {args.sigma_up}")
    print(f"neighbor_atten,d  : {args.neighbor_atten}, {args.d}")
    print(f"couple            : {args.couple}")
    print(f"score norm alpha  : {args.score_norm_alpha}")
    print(f"bias weight       : {args.bias_weight}")
    print(f"power iters       : {args.power_iters}")
    print(f"polish flips      : {args.polish}")
    print(f"Seed time         : {t1-t0:.3f}s")
    print(f"Check time        : {t2-t1:.3f}s")
    print(f"UNSAT clauses     : {unsat} / {C}  ({100.0 * unsat / max(1, C):.2f}%)")

    if unsat == 0:
        sys.exit(0)


    # optional: shake rounds (each = small perturb + short local finisher)
    if args.shake_rounds > 0:
        print("\n[spectral] reshake …\n")
        q = args.shake_q0
        for rr in range(args.shake_rounds):
            # perturb a small fraction of vars appearing in UNSAT clauses
            u_ids = unsat_indices(clauses, assign)  # you already have this helper in v3; if missing here, copy it
            if not u_ids:
                break

            hard_vars = []
            seen = set()
            for ci in u_ids:
                for lit in clauses[ci]:
                    v = abs(lit) - 1
                    if v not in seen:
                        seen.add(v)
                        hard_vars.append(v)

            rng = random.Random(args.seed + 1000 + rr)
            rng.shuffle(hard_vars)

            # flip ~q fraction, at least 1
            t = max(1, int(q * len(hard_vars)))
            for i in range(min(t, len(hard_vars))):
                v = hard_vars[i]
                assign[v] = 1 - assign[v]

            # short local finisher to recover
            model, solved, st = finisher_epic_incremental(
                clauses=clauses,
                nvars=nvars,
                a0=assign,
                seed=args.seed + 2000 + rr,
                max_flips=args.shake_local_flips,
                p_min=args.p_min,
                p_max=min(0.80, args.p_max + 0.10),
                drift_window=args.drift_window,
                stall_window=max(200_000, args.stall_window // 4),
                restart_shake=max(32, args.restart_shake // 2),
                w_inc=args.w_inc,
                w_decay=args.w_decay,
                w_cap=args.w_cap,
                report_every=0,
            )
            assign = [1 if b else 0 for b in model]

            q *= args.shake_decay
            print(f"[shake r{rr+1}] q={q:.4g} unsat={count_unsat(clauses, assign)}")

    # WalkSAT finisher
    print("\n[spectral] passing to finisher …\n")

    """model, solved, st = finisher_predator_sole_sat(
        clauses=clauses,
        nvars=nvars,
        a0=assign,
        seed=args.seed,
        max_flips=args.finisher_flips,
        p_min=args.p_min,
        p_max=args.p_max,
        stall_window=args.stall_window,
        restart_shake=args.restart_shake,
        w_inc=args.w_inc,
        w_decay=args.w_decay,
        w_cap=args.w_cap,
        report_every=100_000,
    )"""

    model, solved, st = finisher_predator_sole_sat_vFinal(
        clauses=clauses,
        nvars=nvars,
        a0=assign,  # <- tvůj spectral/core seed
        seed=args.seed,
        max_flips=args.finisher_flips,

        p_min=args.p_min,
        p_max=args.p_max,
        p_base=args.p_base,

        stall_window=args.stall_window,
        restart_shake=args.restart_shake,

        report_every=100_000,
    )

    sat = check_sat(clauses, model)
    unsat = count_unsat(clauses, model)

    print("\n=== FINISHER RESULT ===")
    print(f"UNSAT clauses  : {unsat} / {C}  ({100.0 * unsat / max(1, C):.2f}%)")
    print(f"Verified SAT   : {sat}")
    print(f"Solved flag    : {solved}")
    #print(f"Flips          : {flips:,}")
    #print(f"Time           : {tsec:.2f}s\n")


    #if args.dump_assign:
    #    with open(args.dump_assign, "w") as f:
    #        f.write("".join("1" if b else "0" for b in res["assignments"]))
    #    print(f"Wrote assignment to: {args.dump_assign}")

if __name__ == "__main__":
    main()