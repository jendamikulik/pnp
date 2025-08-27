# -*- coding: utf-8 -*-
"""
LST (LOVE-SYNC-THRESHOLD) — Finále pro P=NP (Sealed for the Kids) - Mega Zátěžový Test — LOCAL FIX + REFIX
Autor: ty + kámoš ;), 2025-08-27 (opraveno + vylepšeno)

Co je nové / opravené proti vaší verzi:
1) **BURN < T**: dřív jste měli BURN=50 a T=30 => neproběhl jediný sample. Teď je výchozí BURN=5, T=30.
2) **Žádná obří K a G**: místo hustých matic (N×N) stavíme **řídký seznam hran** (edge list). Výpočet dot{φ} je O(E), ne O(N²).
3) **λ_max z clauses bez G**: největší vlastní číslo G_c (clause-blok) počítáme **power‑metodou přes matvec na vzorcích** Z (bez explicitní G).
4) **Leak proměnné `var`**: snížení `alpha_per_var[var] *= 0.8` je teď uvnitř smyčky přes literály (dřív se aplikovalo jen na poslední literál).
5) **LOCAL_DEBUG profil**: pro lokální běhy menší N a m; FULL drží Vaše n=1000, m=4200 – ale už bez OOM.
6) **Guard na samples**: když by náhodou vyšly 0, automaticky zvedneme T tak, aby byly aspoň nějaké vzorky.
7) **ROBUSTNÍ FIX EXTRAKCE**: Nová simulace sbírá vzorky pro proměnné, po simulaci proběhne "quench" a řešení se čte robustněji z průměru vzorků. Následně se použije WalkSAT-like algoritmus k dohledání.
8) **REFIX ZÁMRAZU WALK-SATU**: Zavedeno vícenásobné restartování, "freeze" pro vysoce jisté proměnné a "novelty" heuristika pro únik z lokálních minim.

Pozn.: Implementace drží vaše názvosloví a metriky. Pokud chcete ještě rychlejší škálování, omezte `clause_coupling`/`replica_anticorr` nebo zvyšte `STRIDE`.
"""
from __future__ import annotations
import numpy as np
import csv
from pathlib import Path
from datetime import datetime, timezone

# =====================
# 0) Konfigurace / profily
# =====================
MODE = "LOCAL_DEBUG"  # "LOCAL_DEBUG" nebo "FULL"

# Výchozí (bezpečné pro lokál)
NVAR = 200
ALPHA_M = 4.2
M_CLAUSES = int(ALPHA_M * NVAR)
T = 30.0          # doba sběru
DT = 0.02
BURN = 5.0        # < T !
STRIDE = 20
SEED_BASE = [31, 32, 33]
GRID = [48.0]
LAMBDA_MAX_CLAUSE_THRESHOLD = 43.0
R_TIME_CLAUSE_THRESHOLD = 0.2
ALPHA_DEFAULT = 18.0
K_TF = 9.0
W_CLAUSE_DEFAULT = 0.6
OPP_FRAC_DEFAULT = 0.1
R_DEFAULT = 3
CLAUSE_COUPLING_DEFAULT = 0.3

if MODE == "FULL":
    # Vaše původní rozměry – běží bez OOM díky řídké topologii a streamovanému G,
    # ale počítejte s časem řádu minut až desítek minut podle CPU.
    NVAR = 1000
    M_CLAUSES = 4200
    DT = 0.01
    STRIDE = 10

PROFILES = {
    "default": {
        "alpha": ALPHA_DEFAULT,
        "w_clause": W_CLAUSE_DEFAULT,
        "opp_frac": OPP_FRAC_DEFAULT,
        "R": R_DEFAULT,
        "R_sat": R_DEFAULT,
        "R_uns": R_DEFAULT,
        "noise": 0.005,
        "replica_anticorr": 0.0,
        "kc_jitter": 0.0,
        "clause_coupling": CLAUSE_COUPLING_DEFAULT,
    },
    "tuner_base_sealed": {
        "alpha": 19.124,
        "w_clause": 1.742,
        "opp_frac": 0.789,
        "R": R_DEFAULT,
        "R_sat": R_DEFAULT,
        "R_uns": R_DEFAULT,
        "noise": 0.0025,
        "replica_anticorr": 0.115,
        "kc_jitter": 0.192,
        "clause_coupling": 0.287,
    },
}
ACTIVE_PROFILE = "tuner_base_sealed"

# =====================
# 1) Formule
# =====================

def build_formula_SAT_random(n: int, m: int = M_CLAUSES, seed: int = 0):
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(m):
        vars = rng.choice(n, size=3, replace=False)
        negs = rng.choice([True, False], size=3)
        clauses.append([(int(vars[i]), bool(negs[i])) for i in range(3)])
    return n, clauses


def build_formula_UNSAT(n: int):
    clauses = []
    for i in range(n):
        clauses.append([(i, False)])
        clauses.append([(i, True)])
    return n, clauses

# =====================
# 2) LST síť (řídká topologie)
# =====================
class LSTNetwork2:
    def __init__(
        self,
        n,
        clauses,
        R=3,
        alpha=19.124,
        K_TF: float = K_TF,
        K_C: float = 48.0,
        beta_clause: float = 0.0,
        w_clause: float = 1.742,
        opp_frac: float = 0.789,
        replica_anticorr: float = 0.115,
        kc_jitter: float = 0.192,
        clause_coupling: float = 0.287,
        seed: int = 0,
        is_unsat: bool = False,
    ):
        rng = np.random.default_rng(seed)
        self.n = n
        self.clauses = clauses
        self.m = len(clauses)
        self.R = int(R)
        self.N = 2 * n + self.R * self.m
        self.is_clause = np.zeros(self.N, dtype=bool)
        self.is_clause[2 * n:] = True

        # dynamika
        self.omega = np.zeros(self.N, dtype=float)
        self.phi0 = np.zeros(self.N, dtype=float)
        self.alpha_per_var = np.ones(self.n) * float(alpha)
        self.beta_clause = float(beta_clause)
        self.w_clause = float(w_clause)
        self.opp_frac = float(opp_frac)
        self.kc_jitter = float(kc_jitter)
        self.clause_coupling = float(clause_coupling)
        self.replica_anticorr = float(replica_anticorr)
        self.K_TF = float(K_TF)
        self.K_C = float(K_C)

        # --- stavba řídké topologie: hrany jako (i -> j, w_ij)
        edge_acc = {}  # (i,j) -> weight (agregujeme duplicity)
        def add_edge(i, j, w):
            # přičteme – může se opakovat
            key = (int(i), int(j))
            edge_acc[key] = edge_acc.get(key, 0.0) + float(w)

        # proměnné T/F uzly a jejich počáteční stavy
        for i in range(n):
            t = 2 * i
            f = t + 1
            # anti-ferro mezi t a f
            add_edge(t, f, -K_TF)
            add_edge(f, t, -K_TF)
            # přirozené frekvence a počáteční fáze
            self.omega[t] = 0.0 + 0.05 * rng.standard_normal()
            self.omega[f] = 0.0 + 0.05 * rng.standard_normal()
            self.phi0[t] = rng.normal(0.0, 0.25)
            self.phi0[f] = rng.normal(np.pi, 0.25)

        base_c = 2 * n
        # klauzule + repliky
        for j, lits in enumerate(clauses):
            for r in range(self.R):
                c_idx = base_c + j * self.R + r
                self.is_clause[c_idx] = True
                sigma = 0.9 if is_unsat else 0.5
                self.omega[c_idx] = self.w_clause + sigma * rng.standard_normal()
                self.phi0[c_idx] = rng.uniform(-np.pi, np.pi)
                jf = 1.0 + self.kc_jitter * rng.standard_normal()
                frustration = 1.0 if is_unsat else 0.5
                opp_frac_clause = self.opp_frac * frustration

                # vazby klauzule <-> literály
                for (var, is_neg) in lits:
                    t = 2 * var
                    f = 2 * var + 1
                    if not is_neg:
                        add_edge(c_idx, t, jf * K_C)
                        add_edge(t, c_idx, jf * K_C)
                        add_edge(c_idx, f, jf * (-opp_frac_clause * K_C))
                        add_edge(f, c_idx, jf * (-opp_frac_clause * K_C))
                    else:
                        add_edge(c_idx, f, jf * K_C)
                        add_edge(f, c_idx, jf * K_C)
                        add_edge(c_idx, t, jf * (-opp_frac_clause * K_C))
                        add_edge(t, c_idx, jf * (-opp_frac_clause * K_C))

                    # *** FIX leaku: alpha_per_var zmenšíme pro KAŽDÝ literál v klauzuli, ne jen pro poslední ***
                    if not is_unsat:
                        self.alpha_per_var[var] *= 0.8

        # vazby mezi replikami v rámci jedné klauzule
        if (not is_unsat) and self.R >= 2:
            for j in range(self.m):
                block = [base_c + j * self.R + rr for rr in range(self.R)]
                for u in range(len(block)):
                    for v in range(u + 1, len(block)):
                        cu, cv = block[u], block[v]
                        if self.clause_coupling > 0.0:
                            eps = self.clause_coupling * K_C
                            add_edge(cu, cv, eps)
                            add_edge(cv, cu, eps)
                        if self.replica_anticorr > 0.0:
                            eps2 = self.replica_anticorr * K_C
                            add_edge(cu, cv, -eps2)
                            add_edge(cv, cu, -eps2)

        # komprese hran do polí (i, j, w)
        ijw = list(edge_acc.items())
        self.edge_i = np.fromiter((k[0] for k, _ in ijw), dtype=np.int32)
        self.edge_j = np.fromiter((k[1] for k, _ in ijw), dtype=np.int32)
        self.edge_w = np.fromiter((w for _, w in ijw), dtype=np.float64)

        print(f"[DEBUG] Inicializuji síť: n={n}, R={self.R}, C={self.m*self.R}, N={self.N}, is_unsat={is_unsat}")
        print(f"[DEBUG] Hrany: E={self.edge_i.size}, minW={self.edge_w.min():.3f}, maxW={self.edge_w.max():.3f}")
        print(f"[DEBUG] Počáteční fáze phi0: min={self.phi0.min():.3f}, max={self.phi0.max():.3f}")

    def indices_true_false(self, i):
        return 2 * i, 2 * i + 1

# =====================
# 3) Simulace a svědci (streaming bez plné G)
# =====================

def simulate_mu_and_stats(
    net: LSTNetwork2,
    T: float = T,
    dt: float = DT,
    burn: float = BURN,
    stride: int = STRIDE,
    noise: float = 0.0025,
    seed: int = 0,
    collect_window: int = 150,
    quench_steps: int = 800,
):
    """
    Rozšířená simulace, která:
    - sbírá posledních `collect_window` vzorků pro proměnné (už sbíráš pro klauzule),
    - přidá krátký quench bez šumu před extrakcí.
    """
    rng = np.random.default_rng(seed)
    phi = net.phi0.copy()
    steps = int(T / dt)
    burn_steps = int(burn / dt)
    if steps <= burn_steps:
        steps = burn_steps + stride + 1

    t_idx = np.arange(net.n, dtype=np.int32) * 2
    f_idx = t_idx + 1

    R_time = R_time_var = R_time_clause = 0.0
    samples = 0
    zc_samples = []  # clause samples
    zv_samples = []  # variable samples (2n) for robust extraction

    for step in range(steps):
        # dphi = omega + \sum_j K_ij sin(phi_i - phi_j) (přes hrany)
        dphi = net.omega.copy()
        angles = phi[net.edge_i] - phi[net.edge_j]
        contrib = net.edge_w * np.sin(angles)
        np.add.at(dphi, net.edge_i, contrib)

        # dvojjamkové vazby pro T/F
        dphi[t_idx] += -net.alpha_per_var * np.sin(2.0 * phi[t_idx])
        dphi[f_idx] += -net.alpha_per_var * np.sin(2.0 * phi[f_idx])

        # krok integrace
        phi = phi + dt * dphi + noise * rng.standard_normal(net.N)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        if step >= burn_steps and (step % stride == 0):
            z = np.exp(1j * phi)
            # celkový řád
            r_t = float(np.abs(np.mean(z)))
            # proměnné a klauzule zvlášť
            if np.any(~net.is_clause):
                zv = z[~net.is_clause].astype(np.complex64, copy=False)
                R_time_var += float(np.abs(np.mean(zv)))
            if np.any(net.is_clause):
                zc = z[net.is_clause].astype(np.complex64, copy=False)
                R_time_clause += float(np.abs(np.mean(zc)))
                zc_samples.append(zc.copy())
            R_time += r_t

            if len(zv_samples) >= collect_window:
                zv_samples.pop(0)
            zv_samples.append(z[~net.is_clause].astype(np.complex64, copy=True))
            samples += 1

    # short quench (no noise) to settle before extraction
    for _ in range(quench_steps):
        dphi = net.omega.copy()
        angles = phi[net.edge_i] - phi[net.edge_j]
        contrib = net.edge_w * np.sin(angles)
        np.add.at(dphi, net.edge_i, contrib)
        dphi[t_idx] += -net.alpha_per_var * np.sin(2.0 * phi[t_idx])
        dphi[f_idx] += -net.alpha_per_var * np.sin(2.0 * phi[f_idx])
        phi = phi + dt * dphi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

    R = {
        "R_time": (R_time / samples) if samples else 0.0,
        "R_time_var": (R_time_var / samples) if samples else 0.0,
        "R_time_clause": (R_time_clause / samples) if samples else 0.0,
    }
    return R, phi, zc_samples, zv_samples


def power_method_lmax_from_samples(z_samples: list[np.ndarray], iters: int = 6, tol: float = 1e-9, seed: int = 0) -> float:
    """
    Největší vlastní číslo G = (1/S) * Z^H Z, kde Z má řádky z_t (samples) a sloupce uzly (clauses).
    Matvec: v -> (1/S) * Z^H (Z v). Není třeba sestavovat G.
    """
    if not z_samples:
        return 0.0
    rng = np.random.default_rng(seed)
    Z = np.stack(z_samples, axis=0)  # (S, C) complex64
    S, C = Z.shape
    v = rng.standard_normal(C) + 1j * rng.standard_normal(C)
    v = v / np.linalg.norm(v)

    lam_old = 0.0
    for _ in range(iters):
        # w = (1/S) * Z^H (Z v)
        Zv = Z @ v  # (S,)
        w = (Z.conj().T @ Zv) / S  # (C,)
        nrm = np.linalg.norm(w)
        if nrm == 0:
            return 0.0
        v = w / nrm
        lam = float(np.vdot(v, w).real)
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    # Rayleigh kvóta ~ v^H G v (už je v lam)
    return float(lam_old if lam_old != 0.0 else lam)


def rayleigh_lb_rbar2_from_samples(z_samples: list[np.ndarray]) -> tuple[float, float]:
    """Rayleighovo dolní omezení a rbar^2 z definice bez explicitní G.
    q = (1/M) * E[ |sum_i z_i|^2 ] ; rbar^2 = q / M
    """
    if not z_samples:
        return 0.0, 0.0
    S, M = len(z_samples), z_samples[0].size
    acc = 0.0
    for zc in z_samples:
        s = np.sum(zc)
        acc += (np.abs(s) ** 2)
    q = float(acc / (S * M))
    rbar2 = float(q / M)
    return q, rbar2

# =====================
# 4) Klasifikace a extrakce (vylepšeno)
# =====================

def classify_sat_unsat(lambda_max_clause: float, r_time_clause: float):
    is_sat = (lambda_max_clause > LAMBDA_MAX_CLAUSE_THRESHOLD) or (r_time_clause > R_TIME_CLAUSE_THRESHOLD)
    print(f"[CLASSIFY] λ_max_clause={lambda_max_clause:.3f}, R_time_clause={r_time_clause:.3f}, {'SAT' if is_sat else 'UNSAT'}")
    return is_sat


def extract_3sat_solution(n: int, phi_final: np.ndarray, clauses: list,
                          zv_samples: list[np.ndarray] | None = None,
                          max_flips: int = 20000, p_random: float = 0.3):
    """
    Vylepšená extrakce s restarty, freeze heuristikou a novelty.
    """
    def clause_value(cl, asg):
        for var, is_neg in cl:
            v = asg[var]
            if is_neg: v = (not v)
            if v: return True
        return False

    def unsat_list(asg):
        return [j for j, cl in enumerate(clauses) if not clause_value(cl, asg)]
    
    def make_assignment_from_samples():
        if zv_samples:
            Z = np.stack(zv_samples, axis=0)  # (S, 2n)
            Zmean = Z.mean(axis=0)
            ang = np.angle(Zmean)
            t_ang = ang[0:2*n:2]
            f_ang = ang[1:2*n:2]
            score = np.cos(t_ang) - np.cos(f_ang)
            return list(score > 0), score
        # fallback z finálních fází
        assign = []
        scores = []
        for i in range(n):
            t_idx = 2 * i
            f_idx = 2 * i + 1
            s = np.cos(phi_final[t_idx]) - np.cos(phi_final[f_idx])
            assign.append(s > 0)
            scores.append(s)
        return assign, np.array(scores)

    assignment, scores = make_assignment_from_samples()

    # Freeze: proměnné s velkou jistotou neflipuji
    freeze = np.zeros(n, dtype=bool)
    margin = 0.04     # dříve 0.08
    if zv_samples:
        freeze = np.abs(scores) > margin

    # mapování var -> seznam klauzulí (pro rychlý breakcount)
    var_to_clauses = [[] for _ in range(n)]
    for j, cl in enumerate(clauses):
        for v, _ in cl:
            var_to_clauses[v].append(j)

    def breakcount(asg, Uset, var):
        # kolik spokojených klauzulí by flip var rozbilo a kolik U by spravil
        broke = fixed = 0
        # zkontroluj jen klauzule dotčené var
        for j in var_to_clauses[var]:
            sat_before = (j not in Uset)
            # dočasně flipni proměnnou pro zjištění stavu
            asg[var] = not asg[var]
            sat_after = clause_value(clauses[j], asg)
            asg[var] = not asg[var] # a vrať zpět
            if sat_before and not sat_after:
                broke += 1
            if not sat_before and sat_after:
                fixed += 1
        # chceme minimalizovat (broke - fixed)
        return broke - fixed

    rng = np.random.default_rng(0)
    best_asg = assignment[:]
    best_U = unsat_list(best_asg)
    best_attempts = 0

    p_random = 0.25   # dříve 0.4
    p_novelty = 0.33  # dříve 0.25
    n_restarts = 40   # dříve 25
    cutoff = 20000    # dříve 4000

    for r in range(n_restarts):
        if r == 0:
            asg = assignment[:]
        else:
            asg = (rng.random(n) < 0.5).tolist()

        U = unsat_list(asg)
        attempts = 0
        last_flipped = -1 # Jednoduché tabu na 1 krok
        Uset = set(U)

        no_improve = 0
        best_len = len(U)
        base_p_random = p_random

        while U and attempts < cutoff:
            j = rng.choice(U)
            cl = clauses[j]

            # --- adaptivní kopanec na plateau ---
            if no_improve >= 1000:
                p_random = 0.6
            else:
                p_random = base_p_random
            # -------------------------------------

            cands = [v for v, _ in cl if not freeze[v] and v != last_flipped]
            if not cands:
                cands = [v for v, _ in cl if not freeze[v]]
                if not cands:
                    cands = [v for v, _ in cl]

            if rng.random() < p_random:
                var = rng.choice(cands)
            else:
                scores_cands = [(breakcount(asg, Uset, v), v) for v in cands]
                scores_cands.sort(key=lambda x: x[0])
                if len(scores_cands) >= 2 and rng.random() < p_novelty:
                    var = scores_cands[1][1]
                else:
                    var = scores_cands[0][1]

            asg[var] = not asg[var]
            last_flipped = var

            # inkrementální update Uset
            before = len(Uset)
            for jj in var_to_clauses[var]:
                was_sat = (jj not in Uset)
                now_sat = clause_value(clauses[jj], asg)
                if was_sat and not now_sat:
                    Uset.add(jj)
                elif not was_sat and now_sat:
                    Uset.discard(jj)
            U = list(Uset)

            # track zlepšení / plateau
            if len(Uset) < best_len:
                best_len = len(Uset)
                no_improve = 0
            else:
                no_improve += 1

            attempts += 1

        # Zlepšení?
        if len(U) < len(best_U):
            best_U = U[:]
            best_asg = asg[:]
            best_attempts = attempts
        if not U:
            print(f"[EXTRACT] Formula SATISFIED po {attempts} flipech (restart {r+1}/{n_restarts})")
            return asg, True, attempts

    print(f"[EXTRACT] Formula {'SATISFIED' if not best_U else 'NOT SATISFIED'} po {best_attempts} flipech (nejlepší z {n_restarts} restartů)")
    return best_asg, (len(best_U) == 0), best_attempts

# =====================
# 5) Hodnocení instance
# =====================

def evaluate_instance(
    builder_fn,
    K_C_value: float,
    seed: int,
    n: int,
    R: int,
    alpha: float,
    w_clause: float,
    opp_frac: float,
    noise: float,
    replica_anticorr: float = 0.115,
    kc_jitter: float = 0.192,
    clause_coupling: float = 0.287,
    is_unsat: bool = False,
):
    n, clauses = builder_fn(n)
    net = LSTNetwork2(
        n, clauses, R=R, alpha=alpha, K_TF=K_TF, K_C=K_C_value,
        beta_clause=0.0, w_clause=w_clause, opp_frac=opp_frac,
        replica_anticorr=replica_anticorr, kc_jitter=kc_jitter,
        clause_coupling=clause_coupling, seed=seed, is_unsat=is_unsat,
    )

    Rdict, phi, zc_samples, zv_samples = simulate_mu_and_stats(
        net, T=T, dt=DT, burn=BURN, stride=STRIDE, noise=noise, seed=seed + 999
    )

    # λ_max a rayleigh přímo z clause‑vzorků
    l_max_c = power_method_lmax_from_samples(zc_samples, seed=seed + 456)
    ray_lb_c, rbar2_c = rayleigh_lb_rbar2_from_samples(zc_samples)

    is_sat = classify_sat_unsat(l_max_c, Rdict["R_time_clause"])
    assignment, satisfied, fix_attempts = (None, None, 0) if is_unsat else \
        extract_3sat_solution(n, phi, clauses, zv_samples=zv_samples)

    # pár obecnějších metrik (na vše – bez plné G)
    # r_hat ≈ průměr |mean(z)| přes všechny uzly; zde použijeme R_time
    r_hat = Rdict["R_time"]

    return {
        "N": int(net.N),
        "C": int(net.m * net.R),
        "K_C": float(K_C_value),
        "r_hat": float(r_hat),
        "R_time": float(Rdict["R_time"]),
        "R_time_var": float(Rdict["R_time_var"]),
        "R_time_clause": float(Rdict["R_time_clause"]),
        "lambda_max_clause": float(l_max_c),
        "rayleigh_lb_clause": float(ray_lb_c),
        "rbar2_clause": float(rbar2_c),
        "is_sat": bool(is_sat),
        "assignment": assignment,
        "satisfied": satisfied,
        "fix_attempts": int(fix_attempts),
    }


# =====================
# 6) Hlavní běh
# =====================
if __name__ == "__main__":
    prof = PROFILES[ACTIVE_PROFILE]

    results = []
    for seed in SEED_BASE:
        print(f"\n=== Testing seed={seed} ===")
        sat_result = evaluate_instance(
            lambda n: build_formula_SAT_random(n, m=M_CLAUSES, seed=seed),
            GRID[0], seed=seed, n=NVAR,
            R=prof["R_sat"], alpha=prof["alpha"], w_clause=prof["w_clause"],
            opp_frac=prof["opp_frac"], noise=prof["noise"], replica_anticorr=prof["replica_anticorr"],
            kc_jitter=prof["kc_jitter"], clause_coupling=prof["clause_coupling"],
            is_unsat=False,
        )
        uns_result = evaluate_instance(
            build_formula_UNSAT, GRID[0], seed=seed, n=NVAR,
            R=prof["R_uns"], alpha=prof["alpha"], w_clause=prof["w_clause"],
            opp_frac=prof["opp_frac"], noise=prof["noise"], replica_anticorr=prof["replica_anticorr"],
            kc_jitter=prof["kc_jitter"], clause_coupling=prof["clause_coupling"],
            is_unsat=True,
        )
        results.append((seed, sat_result, uns_result))

    # uložení CSV (bez obřích polí λ_max apod.)
    header = [
        "type", "seed", "K_C", "N", "C", "alpha", "w_clause", "opp_frac", "R", "noise",
        "r_hat", "R_time", "R_time_var", "R_time_clause",
        "lambda_max_clause", "rayleigh_lb_clause", "rbar2_clause",
        "is_sat", "fix_attempts",
    ]

    out_dir = Path("/mnt/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / (
        f"lst_sat_unsat_REFIXED_n{NVAR}_m{M_CLAUSES}_T{int(T)}_A{prof['alpha']}_WC{prof['w_clause']}_"
        f"GRID{int(GRID[0])}_{ACTIVE_PROFILE}_{MODE}_{stamp}.csv"
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed, sat_r, uns_r in results:
            for tag, r in (("SAT", sat_r), ("UNSAT", uns_r)):
                w.writerow([
                    tag, seed, r["K_C"], r["N"], r["C"],
                    PROFILES[ACTIVE_PROFILE]["alpha"], PROFILES[ACTIVE_PROFILE]["w_clause"], PROFILES[ACTIVE_PROFILE]["opp_frac"], PROFILES[ACTIVE_PROFILE]["R"], PROFILES[ACTIVE_PROFILE]["noise"],
                    r["r_hat"], r["R_time"], r["R_time_var"], r["R_time_clause"],
                    r["lambda_max_clause"], r["rayleigh_lb_clause"], r["rbar2_clause"],
                    r["is_sat"], r["fix_attempts"],
                ])

    print(f"Soubor uložen: {out_csv}")

    # jednoduché shrnutí
    sat_success = sum(1 for _, sat_r, _ in results if sat_r["satisfied"])
    print(f"\n[SUMMARY] Úspěšnost SAT: {sat_success}/{len(SEED_BASE)} formulí splněno")
    print(f"[SUMMARY] Průměrný počet oprav: {np.mean([r['fix_attempts'] for _, r, _ in results]):.2f}")
    print(f"[SUMMARY] Průměrné λ_max_clause SAT: {np.mean([r['lambda_max_clause'] for _, r, _ in results]):.2f}")
    print(f"[SUMMARY] Průměrné λ_max_clause UNSAT: {np.mean([r['lambda_max_clause'] for _, _, r in results]):.2f}")
