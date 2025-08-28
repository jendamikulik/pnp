# -*- coding: utf-8 -*-
"""
LST (LOVE-SYNC-THRESHOLD) — Deterministická Simulace (Sealed for the Kids)
Autor: @janmikulik420 + kámoš + Grok 3 (xAI), 2025-08-28
Cíl: Čistě deterministický gradientní systém pro SAT/UNSAT s phase-roundingem.
Nastaveno pro NVAR=100, M_CLAUSES=420.
"""
from __future__ import annotations
import numpy as np
import csv
from pathlib import Path
from datetime import datetime, timezone

# =====================
# 0) Konfigurace / Deterministický profil
# =====================
MODE = "DETERMINISTIC"  # Pouze deterministický režim
NVAR = 100
ALPHA_M = 3.9  # místo 4.2
M_CLAUSES = int(ALPHA_M * NVAR)
T = 80.0

DT = 0.01
BURN = 2.0
STRIDE = 20
SEED_BASE = [31]  # Deterministický seed pro konzistenci
GRID = [1.0]  # K_C fixní podle deterministic_config

# nastav si klidně do konfigurace
GRAD_CLAUSE_UNSAT_THRESHOLD = 0.1   # doladíš podle dat
UNSAT_FRAC_THRESHOLD = 0.25          # >2 % klauzulí je fail → UNSAT


# ---- extra kontrolky / polish ----
SAVE_ASSIGNMENT_BITS = True

EARLY_STOP = True
EARLY_SYNC_DELTA_TOL = 1e-5
EARLY_STABLE_SAMPLES = 20     # přitvrdíme, ať to „těžké SAT“ neukončíme brzy
EARLY_GRAD_TOL = 5e-3         # jemnější limit

deterministic_config = {
    "noise": 0.0,
    "replica_anticorr": 0.0,
    "kc_jitter": 0.0,
    "w_clause": 0.0,
    "w_var": 0.0,
    "w_negvar": 0.0,
    "opp_frac": 0.65,
    "clause_coupling": 1.0,
    "K_TF": 1.15,
    "K_C": 3.0,
    "K_LC": 0.5,      # ↓ slabší literal→clause zpětná vazba (bylo 0.8)
    "alpha": 2.8
}




# Prahové hodnoty
TAU_SAT = 0.7
TAU_UNSAT = 0.3
LAMBDA_SAT = 0.5
LAMBDA_UNSAT = 0.1
LAMBDA_THRESHOLD = 0.3  # Pro gap test

# =====================
# 1) Formule
# =====================
def build_formula_SAT_random(n: int, m: int = M_CLAUSES, seed: int = 0):
    rng = np.random.default_rng(seed)  # Zachováme pro konzistenci, ale nepoužíváme náhodnost dál
    clauses = []
    for _ in range(m):
        vars = rng.choice(n, size=3, replace=False)
        negs = rng.choice([True, False], size=3)
        clauses.append([(int(vars[i]), bool(negs[i])) for i in range(3)])
    return n, clauses

def build_formula_UNSAT(n: int):
    clauses = []
    for i in range(n):
        clauses.append([(i, False)])  # x
        clauses.append([(i, True)])   # ¬x
    return n, clauses

# =====================
# 2) LST síť (řídká topologie, deterministická)
# =====================
def _edge_weight_with_break(i, j, w):
    # malý deterministický offset podle indexů
    return w * (1.0 + EPS_EDGE * ((i + 1) * PHI_A + (j + 1) * PHI_B))


# ---- deterministické symmetry-breakery (globální, ať jsou k dispozici všude) ----
EPS_NODE = 1e-3
EPS_EDGE = 1e-3
PHI_A = 0.61803398875       # ~ zlatý řez
PHI_B = 1.22074408460576    # libovolná iracionální konstanta

class LSTNetwork2:
    def __init__(self, n, clauses, seed: int = 0):
        self.n = n
        self.clauses = clauses
        self.m = len(clauses)
        self.R = 1
        self.N = 2 * n + self.R * self.m

        # --- základní pole nejdřív (ať do nich můžeme hned psát) ---
        self.is_clause = np.zeros(self.N, dtype=bool)
        self.is_clause[2 * n:] = True

        self.edge_acc = {}  # instanční akumulátor hran (pozor na class-var bug)
        self.omega = np.zeros(self.N, dtype=float)
        self.phi0  = np.zeros(self.N, dtype=float)

        # --- parametry z konfigurace ---
        self.alpha_per_var   = np.ones(self.n) * deterministic_config["alpha"]
        self.K_TF            = deterministic_config["K_TF"]
        self.K_C             = deterministic_config["K_C"]
        self.opp_frac        = deterministic_config["opp_frac"]
        self.clause_coupling = deterministic_config["clause_coupling"]

        # --- TF anti-vazba + základní fáze pro literály ---
        for i in range(n):
            t, f = 2 * i, 2 * i + 1
            self.phi0[t] = 0.0
            self.phi0[f] = np.pi
            # anti-edge True <-> False

            self._add_edge(t, f, -self.K_TF, 0)
            self._add_edge(f, t, -self.K_TF, 0)


        # --- drobné deterministické vychýlení fází pro všechny nody (symmetry-break) ---
        for u in range(self.N):
            # malé, ale nula-šumové vychýlení fáze: deterministické dle indexu
            self.phi0[u] += EPS_NODE * ((u + 1) * PHI_A + (u * u + 3) * PHI_B)
        # normalizace do (-pi, pi]
        self.phi0 = (self.phi0 + np.pi) % (2 * np.pi) - np.pi

        # --- klauzule -> literály (s malou deterministickou heterogenitou hran) ---
        # --- klauzule -> literály (s malou deterministickou heterogenitou hran) ---
        # --- klauzule <-> literály (normalizace podle výskytů + různé směry) ---
        base_c = 2 * n

        # spočti výskyty literálů v celé formuli (pro normalizaci)
        deg_pos = np.zeros(n, dtype=int)
        deg_neg = np.zeros(n, dtype=int)
        for lits in clauses:
            for var, is_neg in lits:
                if is_neg:
                    deg_neg[var] += 1
                else:
                    deg_pos[var] += 1

        K_CL = deterministic_config["K_C"]  # clause -> literal
        K_LC = deterministic_config.get("K_LC", 0.6 * K_CL)  # literal -> clause
        opp_frac = deterministic_config["opp_frac"]

        for j, lits in enumerate(clauses):
            c_idx = base_c + j * self.R

            for var, is_neg in lits:
                t, f = 2 * var, 2 * var + 1

                if not is_neg:
                    d_pos = max(1, deg_pos[var])
                    d_opp = max(1, deg_neg[var])
                    w_pos_CL = self._edge_weight_with_break(c_idx, t, +K_CL / d_pos)
                    w_opp_CL = self._edge_weight_with_break(c_idx, f, -K_CL * opp_frac / d_opp)
                    w_pos_LC = self._edge_weight_with_break(t, c_idx, +K_LC / d_pos)
                    w_opp_LC = self._edge_weight_with_break(f, c_idx, -K_LC * opp_frac / d_opp)
                    # clause -> literal (silnější)
                    self._add_edge(t, c_idx, w_pos_CL, 1)  # dphi_t += w*sin(phi_clause - phi_t)
                    self._add_edge(f, c_idx, w_opp_CL, 1)
                    # literal -> clause (slabší)
                    self._add_edge(c_idx, t, w_pos_LC, 1)
                    self._add_edge(c_idx, f, w_opp_LC, 1)
                else:
                    d_pos = max(1, deg_neg[var])
                    d_opp = max(1, deg_pos[var])
                    w_pos_CL = self._edge_weight_with_break(c_idx, f, +K_CL / d_pos)
                    w_opp_CL = self._edge_weight_with_break(c_idx, t, -K_CL * opp_frac / d_opp)
                    w_pos_LC = self._edge_weight_with_break(f, c_idx, +K_LC / d_pos)
                    w_opp_LC = self._edge_weight_with_break(t, c_idx, -K_LC * opp_frac / d_opp)
                    self._add_edge(f, c_idx, w_pos_CL, 1)
                    self._add_edge(t, c_idx, w_opp_CL, 1)
                    self._add_edge(c_idx, f, w_pos_LC, 1)
                    self._add_edge(c_idx, t, w_opp_LC, 1)

        # --- materializace hran ---
        self._finalize_edges()
        print(f"[DEBUG] Inicializace sítě: n={self.n}, C={self.m}, N={self.N}")

    def _edge_weight_with_break(self, i, j, w):
        # malý deterministický offset podle indexů (bez RNG, stabilní)
        return float(w) * (1.0 + EPS_EDGE * ((int(i) + 1) * PHI_A + (int(j) + 1) * PHI_B))

    def _add_edge(self, i, j, w, etype: int):
        # etype: 0 = TF vazba, 1 = clause->literal
        key = (int(i), int(j))
        if key not in self.edge_acc:
            self.edge_acc[key] = [0.0, etype]  # [váha, typ]
        self.edge_acc[key][0] += float(w)

    def _finalize_edges(self):
        ijw = list(self.edge_acc.items())
        self.edge_i = np.fromiter((k[0] for k, _ in ijw), dtype=np.int32)
        self.edge_j = np.fromiter((k[1] for k, _ in ijw), dtype=np.int32)
        self.edge_w = np.fromiter((v[0] for _, v in ijw), dtype=np.float64)
        self.edge_type = np.fromiter((v[1] for _, v in ijw), dtype=np.int8)  # 0=TF, 1=CL


# =====================
# 3) Simulace (čistá Euler integrace)
# =====================
def simulate_mu_and_stats(net: LSTNetwork2, T: float = T, dt: float = DT, burn: float = BURN, stride: int = STRIDE):
    """
    Čistě deterministická simulace (Euler). Vrací (R-metry, efektivní fáze pro rounding, vzorky klauzulí, prázdná hist.).
    - ramp-up škáluje pouze hrany klauzulí (edge_type==1)
    - TF hrany (edge_type==0) navíc „annealujeme“ v čase (silné na začátku, slabší ke konci)
    - TF bistabilita přes -alpha*sin(2*phi)
    - vzorkování po burn-in každých `stride` kroků
    - early stop při stabilním malém grad RMS a nepatrné změně clause synchronie
    - vrací časově průměrované fáze (phi_eff) přes průměr z = e^{i phi}
    """
    phi = net.phi0.copy()

    steps = int(T / dt)
    burn_steps = int(burn / dt)
    if steps <= burn_steps:
        steps = burn_steps + stride + 1

    t_idx = np.arange(net.n, dtype=np.int32) * 2
    f_idx = t_idx + 1

    R_time = 0.0
    R_time_var = 0.0
    R_time_clause = 0.0
    clause_sync_series = []
    grad_rms_clause_acc = 0.0
    grad_rms_var_acc = 0.0
    samples = 0
    zc_samples = []

    node_z_acc = np.zeros(net.N, dtype=np.complex128)

    if EARLY_STOP:
        simulate_mu_and_stats._stable_count = 0

    for step in range(steps):
        dphi = net.omega.copy()

        # časový parametr 0..1
        s = step / max(1, steps - 1)

        # clause ramp-up (jako dřív) + TF-annealing
        beta_cl = 0.55 + 0.45 * (s ** 2)
        beta_cl *= (1.0 + 0.15 * s)
        beta_cl = min(beta_cl, 1.15)

        gamma_tf = 1.35 - 0.85 * s   # 1.35 → 0.50 (lineárně)

        # Kuramoto: dphi_i += w_ij * scale * sin(phi_j - phi_i)
        angles = phi[net.edge_j] - phi[net.edge_i]
        edge_scale = np.ones_like(net.edge_w)
        edge_scale[net.edge_type == 1] *= beta_cl
        edge_scale[net.edge_type == 0] *= gamma_tf
        np.add.at(dphi, net.edge_i, (net.edge_w * edge_scale) * np.sin(angles))

        # bistabilita T/F
        dphi[t_idx] += -net.alpha_per_var * np.sin(2.0 * phi[t_idx])
        dphi[f_idx] += -net.alpha_per_var * np.sin(2.0 * phi[f_idx])

        # Euler + wrap
        phi = phi + dt * dphi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        # vzorkování
        if step >= burn_steps and (step % stride == 0):
            z = np.exp(1j * phi)
            node_z_acc += z

            if np.any(net.is_clause):
                zc = z[net.is_clause]
                val = float(np.abs(np.mean(zc)))
                if val > 1.0: val = 1.0
                R_time_clause += val
                clause_sync_series.append(val)
                zc_samples.append(zc.copy())

            if np.any(~net.is_clause):
                zv = z[~net.is_clause]
                R_time_var += float(np.abs(np.mean(zv)))

            R_time += float(np.abs(np.mean(z)))

            if np.any(net.is_clause):
                g_clause = dphi[net.is_clause]
                grad_rms_clause_acc += float(np.sqrt(np.mean(g_clause * g_clause)))
            if np.any(~net.is_clause):
                g_var = dphi[~net.is_clause]
                grad_rms_var_acc += float(np.sqrt(np.mean(g_var * g_var)))

            # EARLY STOP
            if EARLY_STOP and np.any(net.is_clause):
                g_clause_rms_inst = float(np.sqrt(np.mean(g_clause * g_clause)))
                if len(clause_sync_series) >= 2:
                    delta = abs(clause_sync_series[-1] - clause_sync_series[-2])
                    sync_delta_ok = (delta <= EARLY_SYNC_DELTA_TOL)
                else:
                    sync_delta_ok = False

                if (g_clause_rms_inst <= EARLY_GRAD_TOL) and sync_delta_ok:
                    simulate_mu_and_stats._stable_count += 1
                else:
                    simulate_mu_and_stats._stable_count = 0

                if simulate_mu_and_stats._stable_count >= EARLY_STABLE_SAMPLES:
                    samples += 1
                    break

            samples += 1

    def _clamp01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    R = {
        "R_time": _clamp01(R_time / samples) if samples else 0.0,
        "R_time_var": _clamp01(R_time_var / samples) if samples else 0.0,
        "R_time_clause": _clamp01(R_time_clause / samples) if samples else 0.0,
        "R_clause_var": float(max(0.0, np.var(clause_sync_series))) if clause_sync_series else 0.0,
        "grad_rms_clause": grad_rms_clause_acc / samples if samples else 0.0,
        "grad_rms_var": grad_rms_var_acc / samples if samples else 0.0,
    }

    if samples > 0:
        z_mean = node_z_acc / samples
        phi_eff = np.angle(z_mean + 1e-18)
    else:
        phi_eff = phi

    return R, phi_eff, zc_samples, []




def unsat_fraction(clauses, assignment):
    if assignment is None:
        return 1.0
    m = len(clauses)
    bad = 0
    for cl in clauses:
        sat = False
        for v, is_neg in cl:
            val = assignment[v]
            if is_neg:
                val = not val
            if val:
                sat = True
                break
        if not sat:
            bad += 1
    return bad / m if m > 0 else 0.0

def clause_satisfied(clause, assignment):
    """Vrátí True, pokud klauzule je splněná pod daným přiřazením (0/1 numpy int)."""
    for v, is_neg in clause:
        val = int(assignment[v])
        if is_neg:
            val = 1 - val
        if val == 1:
            return True
    return False


def greedy_deterministic_repair(clauses, assignment, max_passes=2):
    """
    Deterministický lokální repair:
    - v každém průchodu vybere proměnnou, jejíž flip nejvíc sníží počet nesplněných klauzulí
    - při shodě vybere nejnižší index
    - nejvýš 'max_passes' průchodů
    """
    n = len(assignment)
    assign = assignment.copy()

    def count_unsat(assign_vec):
        bad = 0
        for cl in clauses:
            if not clause_satisfied(cl, assign_vec):
                bad += 1
        return bad

    for _ in range(max_passes):
        baseline = count_unsat(assign)
        if baseline == 0:
            break

        best_delta = 0
        best_i = None

        # projdi proměnné v pevném pořadí (determinismus)
        for i in range(n):
            assign[i] ^= 1  # dočasný flip
            after = count_unsat(assign)
            assign[i] ^= 1  # vrátit zpět
            delta = baseline - after
            if (delta > best_delta) or (delta == best_delta and delta > 0 and (best_i is None or i < best_i)):
                best_delta = delta
                best_i = i

        if best_delta > 0 and best_i is not None:
            assign[best_i] ^= 1  # proveď nejlepší flip
        else:
            break

    return assign

def solve_with_rounding_and_repair(net: LSTNetwork2, phi, Rdict=None):
    r_clause = (Rdict or {}).get("R_time_clause", 0.0)

    a0 = extract_3sat_solution(net, phi, r_clause=r_clause)
    if check_assignment(net, a0):
        return a0, 0.0

    a1, uf1 = vote_rounding_with_repair(net, phi, base_assignment=a0, passes=16)
    if uf1 == 0.0:
        return a1, 0.0

    a2 = greedy_deterministic_repair(net.clauses, a1, max_passes=30)
    uf2 = unsat_fraction(net.clauses, a2)

    if uf2 <= uf1:
        return a2, uf2
    else:
        return a1, uf1


# =====================
# 4) Phase-rounding místo flipu
# =====================
def extract_3sat_solution(net: LSTNetwork2, phi, history=None, r_clause: float | None = None):
    n = net.n
    assign = np.full(n, -1, dtype=int)  # -1 = nerozhodnuto

    rc = 0.5 if r_clause is None else float(r_clause)
    base, gain = 0.10, 0.35
    tau = float(np.clip(base + gain * (rc - 0.5), 0.05, 0.30))

    for i in range(n):
        t, f = 2*i, 2*i + 1
        m = np.cos(phi[t]) - np.cos(phi[f])
        if   m >  tau: assign[i] = 1
        elif m < -tau: assign[i] = 0
        # jinak necháme nerozhodnuto

    # fallback pro nerozhodnuté: deterministicky ct>cf
    undec = np.where(assign < 0)[0]
    if undec.size:
        ct = np.cos(phi[2*undec])
        cf = np.cos(phi[2*undec + 1])
        assign[undec] = (ct > cf).astype(int)

    return assign




def _literal_cos(net: LSTNetwork2, phi, var: int, is_neg: bool) -> float:
    """Kosinus příslušného literal-nodu."""
    t, f = 2*var, 2*var + 1
    return float(np.cos(phi[f] if is_neg else phi[t]))

def vote_rounding_with_repair(net: LSTNetwork2, phi, base_assignment=None, passes: int = 16):

    """
    Deterministická oprava po roundingu:
    - Každá nesplněná klauzule dá 'hlas' literálu s největším marginem.
    - Aplikujeme všechny hlasy najednou (deterministické tie-breaky).
    - Opakujeme pár průchodů nebo dokud se nic nezlepší.
    """
    if base_assignment is None:
        base_assignment = extract_3sat_solution(net, phi)

    assign = base_assignment.copy()
    best_uf = unsat_fraction(net.clauses, assign)

    for _ in range(passes):
        # 1) nasbírej hlasy z nesplněných klauzulí
        votes_true = np.zeros(net.n, dtype=int)
        votes_false = np.zeros(net.n, dtype=int)
        improved = False

        for cl in net.clauses:
            # zkontroluj, zda je klauzule splněná
            sat = False
            for v, is_neg in cl:
                val = assign[v]
                if is_neg:
                    val = not val
                if val:
                    sat = True
                    break
            if sat:
                continue

            # vyber literal s nejlepším 'margem'
            # margin = cos(lit) - cos(opposite)
            best = None
            best_margin = -1e9
            for v, is_neg in cl:
                ct = _literal_cos(net, phi, v, is_neg)                # cos pro daný literál
                cf = _literal_cos(net, phi, v, not is_neg)            # cos pro opačný literál
                margin = ct - cf
                if margin > best_margin or (abs(margin - best_margin) < 1e-12 and v < (best[0] if best else net.n)):
                    best = (v, is_neg, margin)
                    best_margin = margin

            vbest, is_neg_best, _ = best
            # hlas znamená: nastav vbest tak, aby literál byl True
            if is_neg_best:
                votes_false[vbest] += 1   # ¬x_v → x_v = False
            else:
                votes_true[vbest] += 1    # x_v → x_v = True

        # 2) aplikuj hlasy deterministicky (většina → změna)
        for i in range(net.n):
            if votes_true[i] > votes_false[i]:
                if assign[i] != 1:
                    assign[i] = 1
                    improved = True
            elif votes_false[i] > votes_true[i]:
                if assign[i] != 0:
                    assign[i] = 0
                    improved = True
            # rovnost = žádná změna (deterministické)

        # 3) zkontroluj zlepšení
        uf = unsat_fraction(net.clauses, assign)
        if uf + 1e-12 < best_uf:
            best_uf = uf
        else:
            # žádné zlepšení — stačí
            break

        # už splněno? končíme
        if best_uf == 0.0:
            break

    return assign, best_uf


def check_assignment(net: LSTNetwork2, assignment):
    """Kontrola, zda přiřazení splňuje všechny klauzule."""
    for clause in net.clauses:
        satisfied = False
        for var, is_neg in clause:
            val = assignment[var]
            if is_neg:
                val = not val
            if val:
                satisfied = True
                break
        if not satisfied:
            return False
    return True

# =====================
# 5) Rozhodovací pravidlo
# =====================
def classify_solution(net: LSTNetwork2, phi, lambda_max, zc_samples, Rdict=None):
    rcl = (Rdict or {}).get("R_time_clause", 0.0)
    grad_c = (Rdict or {}).get("grad_rms_clause", 0.0)

    # zkuste rovnou dvoukrokové dotažení
    assignment, uf = solve_with_rounding_and_repair(net, phi, Rdict=Rdict)

    if uf == 0.0:
        return "SAT", assignment

    # velmi nízká synchronie klauzulí + nízký gradient ⇒ silná indikace UNSAT
    if rcl < 0.05 and grad_c < 0.02:
        return "UNSAT", None

    # tolerantní SAT, když už zbývá málo
    if uf <= 0.05:
        return "SAT", assignment

    # lehce nesplněné a klidná dynamika → UNKNOWN
    if uf <= 0.10 and grad_c < 0.10:
        return "UNKNOWN", assignment

    # jinak rozhodni přísněji
    if uf >= 0.50 or (rcl < 0.05 and grad_c < 0.02):
        return "UNSAT", None
    else:
        return "UNKNOWN", assignment




# =====================
# 6) Hodnocení instance
# =====================

def _to_dimacs(n, clauses):
    return [[-(v+1) if is_neg else (v+1) for (v, is_neg) in cl] for cl in clauses]

def _check_with_pycosat(n, clauses):
    try:
        import pycosat
    except Exception:
        return None
    cnf = _to_dimacs(n, clauses)
    res = pycosat.solve(cnf)
    if isinstance(res, list):
        return "SAT"
    if res == "UNSAT":
        return "UNSAT"
    return None


def evaluate_instance(builder_fn, K_C_value: float, seed: int, n: int):
    n, clauses = builder_fn(n)

    # (volitelné) rychlá diagnostika ground-truth:
    gt = _check_with_pycosat(n, clauses)
    if gt is not None:
        print(f"[GT(pycosat)] {gt}")


    net = LSTNetwork2(n, clauses, seed)
    Rdict, phi, zc_samples, _ = simulate_mu_and_stats(net)
    lambda_max = Rdict["R_time_clause"] if zc_samples else 0.0
    status, assignment = classify_solution(net, phi, lambda_max, zc_samples, Rdict=Rdict)
    uf = unsat_fraction(clauses, assignment)

    assignment_bits = ''
    if SAVE_ASSIGNMENT_BITS and (assignment is not None):
        assignment_bits = ''.join('1' if int(b) else '0' for b in assignment)

    return {
        "N": net.N,
        "C": net.m,
        "K_C": K_C_value,
        "R_time": Rdict["R_time"],
        "R_time_clause": Rdict["R_time_clause"],
        "R_clause_var": Rdict["R_clause_var"],
        "grad_rms_clause": Rdict["grad_rms_clause"],
        "grad_rms_var": Rdict["grad_rms_var"],
        "lambda_max_clause": lambda_max,
        "unsat_frac": uf,
        "status": status,
        "assignment": assignment,
        "assignment_bits": assignment_bits,  # ← NOVÉ
    }


def build_formula_SAT_demo(n: int = 3):
    """
    Vytvoří jednoduchou zaručeně splnitelnou 3-SAT formuli.
    Například: (x1 ∨ x2 ∨ ¬x3) ∧ (¬x1 ∨ x3 ∨ x2)
    Řešení: x1 = True, x2 = True, x3 = True
    """
    clauses = [
        [(0, False), (1, False), (2, True)],   # x1 ∨ x2 ∨ ¬x3
        [(0, True),  (2, False), (1, False)],  # ¬x1 ∨ x3 ∨ x2
    ]
    return n, clauses



def benchmark_demo_suite():
    """
    Spustí malý benchmark: mix SAT a UNSAT instancí různých velikostí.
    Vrátí DataFrame s výsledky.
    """
    testcases = [
        ("SAT_demo_n3", build_formula_SAT_demo, 3),
        ("UNSAT_demo_n3", build_formula_UNSAT, 3),
        ("SAT_rand_n10", lambda n: build_formula_SAT_random(n, m=30, seed=42), 10),
        ("UNSAT_n10", build_formula_UNSAT, 10),
        ("SAT_rand_n20", lambda n: build_formula_SAT_random(n, m=60, seed=123), 20),
        ("UNSAT_n20", build_formula_UNSAT, 20),
    ]

    rows = []
    for tag, builder, n in testcases:
        print(f"\n=== Benchmark {tag} ===")
        result = evaluate_instance(builder, GRID[0], seed=0, n=n)
        result["tag"] = tag
        rows.append(result)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df[["tag", "status", "unsat_frac", "grad_rms_clause", "R_time_clause"]])

    # uložit i do CSV
    out_csv = Path("/mnt/data") / f"lst_benchmark_demo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Benchmark] Výsledky uložené do {out_csv}")
    return df

def build_formula_SAT_planted(n: int, m: int = M_CLAUSES, seed: int = 0, return_planted: bool = False):
    """
    Planted 3-SAT: vygeneruje náhodné přiřazení A a každou klauzuli konstruuje tak,
    aby obsahovala aspoň jeden literál, který je pod A pravdivý.
    """
    rng = np.random.default_rng(seed)
    # planted assignment A in {0,1}^n
    A = rng.integers(0, 2, size=n, dtype=np.int8)

    clauses = []
    for _ in range(m):
        vars3 = rng.choice(n, size=3, replace=False)
        k = int(rng.integers(0, 3))   # index literálu, který bude určitě pravdivý pod A
        lits = []
        for idx, v in enumerate(vars3):
            if idx == k:
                # zvolíme polaritu tak, aby literál byl True pod A:
                # A[v] == 1 -> x_v (is_neg=False), A[v] == 0 -> ¬x_v (is_neg=True)
                is_neg = (A[v] == 0)
            else:
                # ostatní polaritu ponecháme náhodně (může, ale nemusí být True pod A)
                is_neg = bool(rng.integers(0, 2))
            lits.append((int(v), bool(is_neg)))
        clauses.append(lits)

    if return_planted:
        # autokontrola: ověř, že A klauzule fakt plní
        def _sat(cl, assign):
            for v, is_neg in cl:
                val = assign[v]
                if is_neg: val = 1 - val
                if val == 1: return True
            return False
        assert all(_sat(cl, A) for cl in clauses), "Planted generator selhal – klauzule není splněná planted A."

        return n, clauses, A

    return n, clauses



# =====================
# 7) Hlavní běh
# =====================
if __name__ == "__main__":
    results = []
    for seed in SEED_BASE:

        print(f"\n=== Testing build_formula_SAT_random seed={seed} ===")
        sat_result = evaluate_instance(
            lambda n: build_formula_SAT_random(n, m=M_CLAUSES, seed=seed),
            GRID[0], seed, n=NVAR
        )
        uns_result = evaluate_instance(build_formula_UNSAT, GRID[0], seed, n=NVAR)
        results.append((seed, sat_result, uns_result))

    header = [
        "type", "seed", "K_C", "N", "C",
        "R_time", "R_time_clause", "R_clause_var",
        "grad_rms_clause", "grad_rms_var",
        "lambda_max_clause", "unsat_frac", "status",
        "assignment_bits"   # ← NOVÉ
    ]
    out_dir = Path("/mnt/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / (
        f"lst_deterministic_n_random_{NVAR}_m{M_CLAUSES}_T{int(T)}_A{deterministic_config['alpha']}_"
        f"GRID{int(GRID[0])}_{stamp}.csv"
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed, sat_r, uns_r in results:
            for tag, r in (("SAT", sat_r), ("UNSAT", uns_r)):
                w.writerow([
                    tag, seed, r["K_C"], r["N"], r["C"],
                    r["R_time"], r["R_time_clause"], r["R_clause_var"],
                    r["grad_rms_clause"], r["grad_rms_var"],
                    r["lambda_max_clause"], r["unsat_frac"], r["status"],
                    r["assignment_bits"]  # ← NOVÉ
                ])

    print(f"Soubor uložen: {out_csv}")
    sat_success = sum(1 for _, sat_r, _ in results if sat_r["status"] == "SAT")
    print(f"\n[SUMMARY] Úspěšnost SAT: {sat_success}/{len(SEED_BASE)} formulí splněno")


    results = []
    for seed in SEED_BASE:

        print(f"\n=== Testing build_formula_SAT_planted seed={seed} ===")
        _n, _clauses, _A = build_formula_SAT_planted(NVAR, m=M_CLAUSES, seed=SEED_BASE[0], return_planted=True)
        assert check_assignment(type("Dummy", (), {"clauses": _clauses})(), _A)  # sanity check
        # a pak pusť evaluate_instance s builderem:
        sat_result = evaluate_instance(
            lambda n: build_formula_SAT_planted(n, m=M_CLAUSES, seed=SEED_BASE[0]),
            GRID[0], SEED_BASE[0], n=NVAR
        )
        uns_result = evaluate_instance(build_formula_UNSAT, GRID[0], seed, n=NVAR)
        results.append((seed, sat_result, uns_result))

    header = [
        "type", "seed", "K_C", "N", "C",
        "R_time", "R_time_clause", "R_clause_var",
        "grad_rms_clause", "grad_rms_var",
        "lambda_max_clause", "unsat_frac", "status",
        "assignment_bits"   # ← NOVÉ
    ]
    out_dir = Path("/mnt/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / (
        f"lst_deterministic_n_planted_{NVAR}_m{M_CLAUSES}_T{int(T)}_A{deterministic_config['alpha']}_"
        f"GRID{int(GRID[0])}_{stamp}.csv"
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seed, sat_r, uns_r in results:
            for tag, r in (("SAT", sat_r), ("UNSAT", uns_r)):
                w.writerow([
                    tag, seed, r["K_C"], r["N"], r["C"],
                    r["R_time"], r["R_time_clause"], r["R_clause_var"],
                    r["grad_rms_clause"], r["grad_rms_var"],
                    r["lambda_max_clause"], r["unsat_frac"], r["status"],
                    r["assignment_bits"]  # ← NOVÉ
                ])

    print(f"Soubor uložen: {out_csv}")
    sat_success = sum(1 for _, sat_r, _ in results if sat_r["status"] == "SAT")
    print(f"\n[SUMMARY] Úspěšnost SAT: {sat_success}/{len(SEED_BASE)} formulí splněno")






    demo_result = evaluate_instance(build_formula_SAT_demo, GRID[0], seed=0, n=3)
    print("\n=== DEMO TEST ===")
    print(demo_result)

    print("\n=== RUNNING BENCHMARK SUITE ===")
    df = benchmark_demo_suite()
    print(df[["tag", "status", "R_time_clause", "lambda_max_clause"]])


