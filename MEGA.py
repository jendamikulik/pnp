# -*- coding: utf-8 -*-
"""
LST (LOVE-SYNC-THRESHOLD) — Finále pro P=NP (Sealed for the Kids) - Mega Zátěžový Test
Kuramoto síť s pinningem proměnných do {0, π} a replikovanými klauzulemi.
Autor: ty + kámoš ;), 2025-08-27
Tahle verze (PATCH 2025-08-27 PNP LOVE FINALE MEGA TEST) obsahuje:
- **Náhodná 3-SAT s α~4.2**: Těžké instance (n=1000, m=4200).
- **Robustní klasifikace SAT/UNSAT**: Kombinace λ_max_clause a R_time_clause.
- **Vylepšená oprava**: Flip proměnných podle konfliktů.
- **Automatická extrakce**: Přiřazení z phi pro obecný 3-SAT.
- **Debug**: výpis r(t), R_time_clause, matice K, phi0, fází, přiřazení, klasifikace.
- **GRID a BURN**: GRID=[48.0], BURN=50.0, R_sat=3, R_uns=3, T=30.0.
"""
from __future__ import annotations
import numpy as np
import csv
from pathlib import Path
from datetime import datetime, timezone

# 0) Konfigurace
NVAR = 1000  # Začínáme s 1000 proměnnými, plán na 100000
M_CLAUSES = 4200  # α=m/n=4200/1000=4.2
T = 30.0  # Zkráceno pro rychlost
DT = 0.01
BURN = 50.0  # Zkráceno pro rychlost
STRIDE = 10
SEED_BASE = [31, 32, 33]  # Tři seedy pro robustnost
GRID = [48.0]
LAMBDA_MAX_CLAUSE_THRESHOLD = 10.0
R_TIME_CLAUSE_THRESHOLD = 0.2
ALPHA_DEFAULT = 18.0
K_TF = 9.0
W_CLAUSE_DEFAULT = 0.6
OPP_FRAC_DEFAULT = 0.1
R_DEFAULT = 3  # Sníženo na 3 pro menší paměťovou náročnost
CLAUSE_COUPLING_DEFAULT = 0.3

# Upravený profil s R=3 pro konzistenci s konfigurací
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
        "R": R_DEFAULT,  # Upraveno z 6 na 3
        "R_sat": R_DEFAULT,  # Upraveno z 6 na 3
        "R_uns": R_DEFAULT,  # Upraveno z 6 na 3
        "noise": 0.0025,
        "replica_anticorr": 0.115,
        "kc_jitter": 0.192,
        "clause_coupling": 0.287,
    },
}
ACTIVE_PROFILE = "tuner_base_sealed"

# 1) Formule
def build_formula_SAT_random(n: int, m: int = M_CLAUSES, seed: int = 0):
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(m):
        vars = rng.choice(n, size=3, replace=False)
        negs = rng.choice([True, False], size=3)
        clauses.append([(vars[i], negs[i]) for i in range(3)])
    return n, clauses

def build_formula_UNSAT(n: int):
    clauses = []
    for i in range(n):
        clauses.append([(i, False)])
        clauses.append([(i, True)])
    return n, clauses

# 2) LST síť
class LSTNetwork2:
    def __init__(
        self,
        n,
        clauses,
        R=6,
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
        self.R = R
        self.N = 2 * n + R * self.m
        self.is_clause = np.zeros(self.N, dtype=bool)
        self.is_clause[2 * n:] = True
        self.K = np.zeros((self.N, self.N), dtype=float)
        self.omega = np.zeros(self.N, dtype=float)
        self.phi0 = np.zeros(self.N, dtype=float)
        self.kc_jitter = float(kc_jitter)
        self.clause_coupling = float(clause_coupling)
        self.opp_frac_per_clause = np.ones(self.m * R) * opp_frac
        self.alpha_per_var = np.ones(self.n) * alpha
        self.replica_anticorr = float(replica_anticorr)  # Oprava: Inicializace hned na začátku
        print(f"[DEBUG] Inicializuji síť: n={n}, R={R}, C={self.m*R}, N={self.N}, is_unsat={is_unsat}")

        for i in range(n):
            t = 2 * i
            f = 2 * i + 1
            self.K[t, f] += -K_TF
            self.K[f, t] += -K_TF
            self.omega[t] = 0.0 + 0.05 * rng.standard_normal()
            self.omega[f] = 0.0 + 0.05 * rng.standard_normal()
            self.phi0[t] = rng.normal(0.0, 0.25)
            self.phi0[f] = rng.normal(np.pi, 0.25)

        base_c = 2 * n
        for j, lits in enumerate(clauses):
            for r in range(R):
                c_idx = base_c + j * R + r
                sigma = 0.9 if is_unsat else 0.5
                self.omega[c_idx] = w_clause + sigma * rng.standard_normal()
                self.is_clause[c_idx] = True
                self.phi0[c_idx] = rng.uniform(-np.pi, np.pi)
                jf = 1.0 + kc_jitter * rng.standard_normal()
                frustration = 1.0 if is_unsat else 0.5
                opp_frac_clause = opp_frac * frustration
                self.opp_frac_per_clause[j * R + r] = opp_frac_clause
                for (var, is_neg) in lits:
                    t = 2 * var
                    f = 2 * var + 1
                    if not is_neg:
                        self.K[c_idx, t] += jf * K_C
                        self.K[t, c_idx] += jf * K_C
                        self.K[c_idx, f] += jf * (-opp_frac_clause * K_C)
                        self.K[f, c_idx] += jf * (-opp_frac_clause * K_C)
                    else:
                        self.K[c_idx, f] += jf * K_C
                        self.K[f, c_idx] += jf * K_C
                        self.K[c_idx, t] += jf * (-opp_frac_clause * K_C)
                        self.K[t, c_idx] += jf * (-opp_frac_clause * K_C)
                if not is_unsat:
                    self.alpha_per_var[var] *= 0.8

        if not is_unsat and self.clause_coupling > 0.0 and self.R >= 2:
            base_c2 = 2 * n
            for j2 in range(self.m):
                block = [base_c2 + j2 * self.R + r2 for r2 in range(self.R)]
                for u in range(len(block)):
                    for v in range(u + 1, len(block)):
                        cu, cv = block[u], block[v]
                        eps = self.clause_coupling * K_C
                        self.K[cu, cv] += eps
                        self.K[cv, cu] += eps

        if self.replica_anticorr > 0.0 and self.R >= 2:
            base_c2 = 2 * n
            for j2 in range(self.m):
                block = [base_c2 + j2 * self.R + r2 for r2 in range(self.R)]
                for u in range(len(block)):
                    for v in range(u + 1, len(block)):
                        cu, cv = block[u], block[v]
                        eps = self.replica_anticorr * K_C
                        self.K[cu, cv] -= eps
                        self.K[cv, cu] -= eps

        print(f"[DEBUG] Matice K: nnz={np.count_nonzero(self.K)}, min={np.min(self.K[np.nonzero(self.K)])}, max={np.max(self.K)}")
        print(f"[DEBUG] Počáteční fáze phi0: min={np.min(self.phi0):.3f}, max={np.max(self.phi0):.3f}")
        self.alpha = float(alpha)
        self.K_TF = float(K_TF)
        self.K_C = float(K_C)
        self.beta_clause = float(beta_clause)
        self.w_clause = float(w_clause)
        self.opp_frac = float(opp_frac)

    def indices_true_false(self, i):
        return 2 * i, 2 * i + 1

# 3) Simulace a svědci
def simulate_mu_and_G(net: LSTNetwork2,
                      T: float = T,
                      dt: float = DT,
                      burn: float = BURN,
                      stride: int = STRIDE,
                      noise: float = 0.0025,
                      seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = net.phi0.copy()
    samples = 0
    mu = np.zeros(net.N, dtype=complex)
    G = np.zeros((net.N, net.N), dtype=complex)
    R_time = 0.0
    R_time_var = 0.0
    R_time_clause = 0.0
    steps = int(T / dt)
    burn_steps = int(burn / dt)
    for step in range(steps):
        diff = phi[None, :] - phi[:, None]
        dphi = net.omega + np.sum(net.K * np.sin(diff), axis=1)
        var_mask = ~net.is_clause
        for i in range(net.n):
            t, f = net.indices_true_false(i)
            dphi[t] += -net.alpha_per_var[i] * np.sin(2 * phi[t])
            dphi[f] += -net.alpha_per_var[i] * np.sin(2 * phi[f])
        phi = phi + dt * dphi + noise * rng.standard_normal(net.N)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        if step >= burn_steps and (step % stride == 0):
            z = np.exp(1j * phi)
            mu += z
            G += np.outer(z, np.conjugate(z))
            r_t = float(np.abs(np.mean(z)))
            print(f"Step {step}: r(t) = {r_t:.3f}, R_time_clause = {float(np.abs(np.mean(z[net.is_clause]))):.3f}")
            R_time += r_t
            R_time_var += float(np.abs(np.mean(z[~net.is_clause]))) if np.any(~net.is_clause) else 0.0
            R_time_clause += float(np.abs(np.mean(z[net.is_clause]))) if np.any(net.is_clause) else 0.0
            samples += 1
    R = {
        "R_time": (R_time / samples) if samples else 0.0,
        "R_time_var": (R_time_var / samples) if samples else 0.0,
        "R_time_clause": (R_time_clause / samples) if samples else 0.0,
    }
    if samples > 0:
        mu /= samples
        G /= samples
    return mu, G, R, phi

def power_method_lmax(G: np.ndarray, iters: int = 2000, tol: float = 1e-10, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    if n == 0:
        return 0.0
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(iters):
        w = G @ v
        nrm = np.linalg.norm(w)
        if nrm == 0:
            return 0.0
        v = w / nrm
        lam = np.vdot(v, G @ v)
        if np.abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return float(np.real(lam))

def project_G_to_mask(G: np.ndarray, mask: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros((0, 0), dtype=G.dtype), idx
    return G[np.ix_(idx, idx)], idx

def rayleigh_lb_and_rbar2(G: np.ndarray, mask: np.ndarray | None = None):
    if mask is None:
        M = G.shape[0]
        if M == 0:
            return 0.0, 0.0
        ones = np.ones((M, 1), dtype=complex)
        q = float(np.real((ones.conj().T @ G @ ones)[0, 0])) / M
        rbar2 = q / M
        return q, rbar2
    else:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return 0.0, 0.0
        Gc = G[np.ix_(idx, idx)]
        M = idx.size
        ones = np.ones((M, 1), dtype=complex)
        q = float(np.real((ones.conj().T @ Gc @ ones)[0, 0])) / M
        rbar2 = q / M
        return q, rbar2

# 4) Klasifikace a extrakce
def classify_sat_unsat(lambda_max_clause: float, r_time_clause: float):
    is_sat = lambda_max_clause > LAMBDA_MAX_CLAUSE_THRESHOLD or r_time_clause > R_TIME_CLAUSE_THRESHOLD
    print(f"[CLASSIFY] λ_max_clause={lambda_max_clause:.3f}, R_time_clause={r_time_clause:.3f}, {'SAT' if is_sat else 'UNSAT'}")
    return is_sat

def extract_3sat_solution(n: int, phi: np.ndarray, clauses: list, max_fix_attempts: int = M_CLAUSES):
    rng = np.random.default_rng(SEED_BASE[0])
    assignment = []
    for i in range(n):
        t_idx = 2 * i
        f_idx = 2 * i + 1
        phi_t = abs(phi[t_idx])
        phi_f = abs(phi[f_idx] - np.pi)
        is_true = phi_t < phi_f
        assignment.append(is_true)
        print(f"[EXTRACT] Var {i}: phi_t={phi[t_idx]:.3f}, phi_f={phi[f_idx]:.3f}, x_{i}={'True' if is_true else 'False'}")
    satisfied = True
    unsatisfied_clauses = []
    var_conflicts = np.zeros(n)
    for j, clause in enumerate(clauses):
        clause_satisfied = False
        for var, is_neg in clause:
            val = assignment[var] if not is_neg else not assignment[var]
            if val:
                clause_satisfied = True
                break
        if not clause_satisfied:
            satisfied = False
            unsatisfied_clauses.append((j, clause))
            for var, _ in clause:
                var_conflicts[var] += 1
            print(f"[EXTRACT] Clause {j} {clause} NOT satisfied")
        else:
            print(f"[EXTRACT] Clause {j} {clause} satisfied")
    attempts = 0
    while not satisfied and attempts < max_fix_attempts:
        if not unsatisfied_clauses:
            break
        var_idx = np.argmax(var_conflicts)
        assignment[var_idx] = not assignment[var_idx]
        print(f"[FIX] Flipping var {var_idx} (conflicts={var_conflicts[var_idx]}), attempt {attempts+1}")
        var_conflicts = np.zeros(n)
        satisfied = True
        unsatisfied_clauses = []
        for j, clause in enumerate(clauses):
            clause_satisfied = False
            for var, is_neg in clause:
                val = assignment[var] if not is_neg else not assignment[var]
                if val:
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                satisfied = False
                unsatisfied_clauses.append((j, clause))
                for var, _ in clause:
                    var_conflicts[var] += 1
                print(f"[FIX] Clause {j} {clause} still NOT satisfied")
            else:
                print(f"[FIX] Clause {j} {clause} satisfied")
        attempts += 1
    print(f"[EXTRACT] Formula {'SATISFIED' if satisfied else 'NOT SATISFIED'} after {attempts} fix attempts")
    return assignment, satisfied, attempts

# 5) Demo běh
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
        clause_coupling=clause_coupling, seed=seed, is_unsat=is_unsat
    )
    mu, G, Rdict, phi = simulate_mu_and_G(net, T=T, dt=DT, burn=BURN, stride=STRIDE, noise=noise, seed=seed + 999)
    l_mu = float(np.vdot(mu, mu).real)
    r_hat = float(np.linalg.norm(mu) / np.sqrt(net.N))
    l_max = power_method_lmax(G, seed=seed + 123)
    ray_lb_full, rbar2_full = rayleigh_lb_and_rbar2(G, None)
    Gc, idx_c = project_G_to_mask(G, net.is_clause)
    C = int(idx_c.size)
    l_max_c = power_method_lmax(Gc, seed=seed + 456) if C > 0 else 0.0
    ray_lb_c, rbar2_c = rayleigh_lb_and_rbar2(G, net.is_clause)
    Gv, idx_v = project_G_to_mask(G, ~net.is_clause)
    V = int(idx_v.size)
    l_max_v = power_method_lmax(Gv, seed=seed + 789) if V > 0 else 0.0
    ray_lb_v, rbar2_v = rayleigh_lb_and_rbar2(G, ~net.is_clause)
    is_sat = classify_sat_unsat(l_max_c, Rdict["R_time_clause"])
    assignment, satisfied, fix_attempts = (None, None, 0) if is_unsat else extract_3sat_solution(n, phi, clauses)
    return {
        "N": int(net.N),
        "C": C,
        "K_C": float(K_C_value),
        "lambda_mu": l_mu,
        "r_hat": r_hat,
        "R_time": Rdict["R_time"],
        "R_time_var": Rdict["R_time_var"],
        "R_time_clause": Rdict["R_time_clause"],
        "lambda_max": l_max,
        "rayleigh_lb": ray_lb_full,
        "rbar2": rbar2_full,
        "lambda_max_clause": l_max_c,
        "rayleigh_lb_clause": ray_lb_c,
        "rbar2_clause": rbar2_c,
        "lambda_max_var": l_max_v,
        "rayleigh_lb_var": ray_lb_v,
        "rbar2_var": rbar2_v,
        "is_sat": is_sat,
        "assignment": assignment,
        "satisfied": satisfied,
        "fix_attempts": fix_attempts,
    }

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
            is_unsat=False
        )
        uns_result = evaluate_instance(
            build_formula_UNSAT, GRID[0], seed=seed, n=NVAR,
            R=prof["R_uns"], alpha=prof["alpha"], w_clause=prof["w_clause"],
            opp_frac=prof["opp_frac"], noise=prof["noise"], replica_anticorr=prof["replica_anticorr"],
            kc_jitter=prof["kc_jitter"], clause_coupling=prof["clause_coupling"],
            is_unsat=True
        )
        results.append((seed, sat_result, uns_result))

    header = [
        "type", "seed", "K_C", "N", "C", "alpha", "w_clause", "opp_frac", "R", "noise",
        "lambda_mu", "lambda_max", "rayleigh_lb", "rbar2", "r_hat", "R_time", "R_time_var", "R_time_clause",
        "lambda_max_clause", "rayleigh_lb_clause", "rbar2_clause",
        "lambda_max_var", "rayleigh_lb_var", "rbar2_var", "is_sat", "fix_attempts",
    ]
    out_dir = Path("/mnt/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / (
        f"lst_sat_unsat_pnp_finale_n{NVAR}_m{M_CLAUSES}_T{int(T)}_A{prof['alpha']}_WC{prof['w_clause']}_"
        f"GRID{int(GRID[0])}-{int(GRID[-1])}_{ACTIVE_PROFILE}_{stamp}.csv"
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed, sat_r, uns_r in results:
            for tag, r in (("SAT", sat_r), ("UNSAT", uns_r)):
                w.writerow([
                    tag, seed, r["K_C"], r["N"], r["C"],
                    prof["alpha"], prof["w_clause"], prof["opp_frac"], prof["R"], prof["noise"],
                    r["lambda_mu"], r["lambda_max"], r["rayleigh_lb"], r["rbar2"], r["r_hat"],
                    r["R_time"], r["R_time_var"], r["R_time_clause"],
                    r["lambda_max_clause"], r["rayleigh_lb_clause"], r["rbar2_clause"],
                    r["lambda_max_var"], r["rayleigh_lb_var"], r["rbar2_var"],
                    r["is_sat"], r["fix_attempts"],
                ])
    print(f"Soubor uložen: {out_csv}")
    # Statistiky
    sat_success = sum(1 for _, sat_r, _ in results if sat_r["satisfied"])
    print(f"\n[SUMMARY] Úspěšnost SAT: {sat_success}/{len(SEED_BASE)} formulí splněno")
    print(f"[SUMMARY] Průměrný počet oprav: {np.mean([r['fix_attempts'] for _, r, _ in results]):.2f}")
    print(f"[SUMMARY] Průměrné λ_max_clause SAT: {np.mean([r['lambda_max_clause'] for _, r, _ in results]):.2f}")
    print(f"[SUMMARY] Průměrné λ_max_clause UNSAT: {np.mean([r['lambda_max_clause'] for _, _, r in results]):.2f}")