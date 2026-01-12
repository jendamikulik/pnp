#!/usr/bin/env python3
# DREAM6_clean_vFinal.py
# Spectral seeder is kept unchanged by importing from DREAM6_v5.py (same directory).
from __future__ import annotations

import argparse, math, os, random, time
from typing import List, Tuple, Dict, Optional
from array import array

from DREAM6_v5 import (  # type: ignore
    parse_dimacs,
    theory_params,
    build_seed_assignment,
    greedy_polish,
    core_push,
    count_unsat,
    check_sat,
)

BoolAssign = List[bool]
Clause = List[int]



def _as_bool_assign(a: List[int] | List[bool], nvars: int) -> BoolAssign:
    if len(a) != nvars:
        raise ValueError(f"Assignment length mismatch: got {len(a)} expected {nvars}")
    if a and isinstance(a[0], bool):
        return list(a)  # type: ignore[arg-type]
    return [bool(x) for x in a]  # type: ignore[arg-type]


def _as_int_assign(a: BoolAssign) -> List[int]:
    return [1 if b else 0 for b in a]


def lit_true(lit: int, aval: bool) -> bool:
    return aval if lit > 0 else (not aval)


def build_var_occ(clauses: List[Clause], nvars: int) -> List[List[Tuple[int, int]]]:
    occ: List[List[Tuple[int, int]]] = [[] for _ in range(nvars)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            if 0 <= v < nvars:
                occ[v].append((ci, lit))
    return occ


# -------------------------
# SNIPER
# -------------------------
def finisher_classic_to_zero_sniper(
    clauses: List[Clause],
    nvars: int,
    a0: List[int] | List[bool],
    *,
    var_occ: Optional[List[List[Tuple[int, int]]]] = None,
    seed: int = 0,
    max_flips: int = 8_000_000,
    p_mid: float = 0.33,
    p_mid_hard: float = 0.33,  # kept for compatibility
    endgame_at: int = 24,
    p_end: float = 0.06,
    sniper_end_p: float = 0.03,
    tabu_tenure: int = 45,
    rb_prob: float = 0.001,        # 1/1000
    rb_stall_flips: int = 200_000, # jak dlouho bez zlepšení
    report_every: int = 100_000,
) -> Tuple[BoolAssign, bool, Dict]:
    rng = random.Random(seed)
    m = len(clauses)
    a = _as_bool_assign(a0, nvars)
    if var_occ is None:
        var_occ = build_var_occ(clauses, nvars)

    sat_count = [0] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    def add_unsat(ci: int) -> None:
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        p = pos[ci]
        if p != -1:
            last = unsat_list[-1]
            unsat_list[p] = last
            pos[last] = p
            unsat_list.pop()
            pos[ci] = -1

    for ci, cl in enumerate(clauses):
        sc = 0
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                sc += 1
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)

    tabu_until = [0] * nvars

    def flip_var(v: int) -> None:
        old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, old)
            now_true = not was_true

            if was_true and not now_true:
                if sc == 1:
                    sat_count[ci] = 0
                    add_unsat(ci)
                else:
                    sat_count[ci] = sc - 1
            elif (not was_true) and now_true:
                if sc == 0:
                    sat_count[ci] = 1
                    rem_unsat(ci)
                else:
                    sat_count[ci] = sc + 1

    def breakcount(v: int) -> int:
        bc = 0
        cur = a[v]
        for ci, lit in var_occ[v]:
            if sat_count[ci] == 1 and lit_true(lit, cur):
                bc += 1
        return bc

    def rebuild_from_assign(best_assign: BoolAssign) -> None:
        # reset assignment
        a[:] = best_assign[:]

        # rebuild sat_count + unsat_list + pos
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


    best = a[:]
    best_unsat = len(unsat_list)
    # --- SNIPER EMA + smooth p + rollback ---
    ema_r = 0.0
    u_prev = best_unsat
    flips_since_improve = 0
    rollback_every = 50_000  # soft rollback cadence
    p = float(p_mid)

    last_improve_flip = 0
    reheat_until = 0
    last_report = 0
    t0 = time.time()

    in_endgame = False  # hysteresis state for SNIPER mode


    for flip in range(1, max_flips + 1):
        u = len(unsat_list)

        """if u <= 25 and flip % 500 == 0:
            if focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
                continue"""
        if u <= 22 and flip % 200 == 0:
            if focused_endgame_pulse3(clauses, a, unsat_list, flip_var, rng):
                continue
        elif u <= 25 and flip % 200 == 0:
            if focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
                continue

        if u == 0:
            return a, True, {"phase": "sniper", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip
            # when we improve, cancel any reheat
            reheat_until = 0

        # Endgame should be sticky once we've *ever* reached <= endgame_at.
        # Otherwise the search bounces between MID and SNIPER as u fluctuates around the threshold.
        #endgame = (best_unsat <= endgame_at)
        """enter_at = endgame_at
        exit_at  = endgame_at + 12
        if (not in_endgame) and (u <= enter_at):
            in_endgame = True
        elif in_endgame and (u >= exit_at):
            in_endgame = False
        endgame = in_endgame"""
        # --- endgame is based on BEST (sticky) to avoid MID/SNIPER bouncing ---
        if (not in_endgame) and (best_unsat <= endgame_at):
            in_endgame = True
        endgame = in_endgame
        # --- approach band: start behaving "sniper-ish" before true endgame ---
        approach = (not endgame) and (u <= 3 * endgame_at)


        # EMA of local progress (step-to-step improvement)
        improve = max(0, u_prev - u)
        ema_r = 0.9 * ema_r + 0.1 * improve
        if improve > 0:
            flips_since_improve = 0
        else:
            flips_since_improve += 1
        u_prev = u

        # --- probabilistic rollback when hard-stalled ---
        #if flips_since_improve >= rb_stall_flips:
        # --- probabilistic rollback when hard-stalled (DISABLED in approach/endgame) ---
        if (not endgame) and (not approach) and flips_since_improve >= rb_stall_flips:

            if rb_prob > 0.0 and rng.random() < rb_prob:
                rebuild_from_assign(best)
                flips_since_improve = 0
                # keep noise bounded – no explosion after rollback
                p = max(p, 0.8 * min(p_mid, 0.16))

        # Soft rollback if we're stuck too long: rebuild exact best state (no shake)
        """if flips_since_improve >= rollback_every and best is not None:
            rebuild_from(best, shake=0)
            a[:] = best[:]
            flips_since_improve = 0
            # slight reheat after rollback
            p = max(p, 0.8 * p_mid)"""
        """if flips_since_improve >= rollback_every and best is not None:
            # rollback to best-known assignment (no rebuild_from dependency)
            a[:] = best[:]
            flips_since_improve = 0
            # slight reheat after rollback
            p = max(p, 0.8 * p_mid)"""
        if flips_since_improve >= rollback_every and best is not None:
            rebuild_from_assign(best)
            flips_since_improve = 0
            # slight reheat after rollback (bounded)
            p = max(p, 0.8 * min(p_mid, 0.16))



        # Smooth p toward a target (lower under ~30 UNSAT; contract a bit on progress)
        """if endgame:
            if u <= 30:
                p_target = max(sniper_end_p, min(p_end, 0.06))
            else:
                p_target = max(sniper_end_p, p_end)
            scale = 1.0 - 0.15 * min(1.0, ema_r / 5.0)
            p_target = max(sniper_end_p, p_target * scale)
            if reheat_until != 0 and flip <= reheat_until:
                p_target = max(p_target, p_mid)
            if reheat_until != 0 and flip <= reheat_until:
                # reheat is bounded in endgame
                p_target = max(p_target, min(p_mid, 0.10))

        else:
            p_target = p_mid"""

        if endgame:
            # true endgame: low noise + tabu
            if u <= 30:
                p_target = max(sniper_end_p, min(p_end, 0.06))
            else:
                p_target = max(sniper_end_p, p_end)
            scale = 1.0 - 0.15 * min(1.0, ema_r / 5.0)
            p_target = max(sniper_end_p, p_target * scale)
            if reheat_until != 0 and flip <= reheat_until:
                p_target = max(p_target, p_mid)

        elif approach:
            # approach: already reduce noise (kills the 0.33 random-walk plateau)
            p_target = max(0.08, min(0.16, 0.55 * p_mid))

        else:
            p_target = p_mid


        p = 0.95 * p + 0.05 * p_target

        if report_every and (flip - last_report) >= report_every:
            last_report = flip
            mode = "REHEAT" if (endgame and flip <= reheat_until and reheat_until != 0) else ("SNIPER" if endgame else "MID")
            print(f"[finisher] flips={flip:,} unsat={u} best={best_unsat} p={p:.3f} mode={mode}")

        # Refresh u (unsat_list can change after rebuild/sniper)
        u_pick = len(unsat_list)
        if u_pick == 0:
            return a, True, {"phase": "sniper", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}
        ci = unsat_list[rng.randrange(u_pick)]
        cl = clauses[ci]

        """if rng.random() < p:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = None
            best_bc = 10**9
            for lit in cl:
                vv = abs(lit) - 1
                if endgame and tabu_until[vv] > flip:
                    continue
                bc = breakcount(vv)
                if bc < best_bc:
                    best_bc = bc
                    best_v = vv
                elif bc == best_bc and best_v is not None and rng.random() < 0.35:
                    best_v = vv
            v = best_v if best_v is not None else abs(cl[rng.randrange(len(cl))]) - 1"""
        # --- endgame clamp: never let endgame behave like high-noise walk ---
        p_pick = p
        if endgame:
            # hard clamp in endgame (prevents 0.33 "reheat" from blowing structure)
            if p_pick < 0.02:
                p_pick = 0.02
            elif p_pick > 0.08:
                p_pick = 0.08

        # Evaluate candidates by (breakcount, age/tie)
        cand = []
        for lit in cl:
            vv = abs(lit) - 1
            """if endgame and tabu_until[vv] > flip:
                continue"""
            if (endgame or approach) and tabu_until[vv] > flip:
                continue
            bc = breakcount(vv)
            cand.append((bc, vv))

        if not cand:
            # all tabu or empty -> fallback random in clause
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            cand.sort(key=lambda x: x[0])  # sort by breakcount asc
            # deterministic best is minimal breakcount
            best_bc = cand[0][0]
            best_vars = [vv for (bc, vv) in cand if bc == best_bc]

            # noise: pick from top-K of best breakcount set, not random-any
            if rng.random() < p_pick:
                # topK among best breakcount (cap keeps it local)
                K = 8
                v = best_vars[rng.randrange(min(K, len(best_vars)))]
            else:
                # stable pick (if multiple best, pick one randomly to avoid cycles)
                v = best_vars[rng.randrange(len(best_vars))]


        flip_var(v)
        """if endgame:
            tabu_until[v] = flip + tabu_tenure"""
        if endgame or approach:
            tabu_until[v] = flip + tabu_tenure


    return best, False, {"phase": "sniper", "flips": max_flips, "best_unsat": best_unsat, "time_s": time.time() - t0}


# -------------------------
# PREDATOR
# -------------------------
def finisher_predator_sole_sat_vFinal(
    clauses: List[Clause],
    nvars: int,
    a0: List[int] | List[bool],
    *,
    var_occ: Optional[List[List[Tuple[int, int]]]] = None,
    seed: int = 0,
    max_flips: int = 50_000_000,
    p_min: float = 0.003,
    p_max: float = 0.18,
    p_base: float = 0.10,
    stall_window: int = 800_000,
    restart_shake: int = 96,
    w_inc: float = 1.0,
    w_decay: float = 0.9996,
    w_cap: float = 40.0,
    snapback_gap: int = 250,
    basin_mult: float = 2.2,
    basin_abs: int = 350,
    kick_after: int = 300_000,
    kick_len: int = 100_000,
    kick_p: float = 0.18,

    kick_cooldown: int = 250_000,
    kick_disable_best_mult: int = 2,
    sniper_u: int = 64,
    sniper_flips: int = 8_000_000,
    sniper_p: float = 0.33,
    use_tabu: bool = True,
    tabu_u_threshold: int = 128,
    tabu_tenure: int = 45,
    sniper_end_p: float = 0.03,
    report_every: int = 100_000,
) -> Tuple[BoolAssign, bool, Dict]:
    rng = random.Random(seed)
    m = len(clauses)
    a = _as_bool_assign(a0, nvars)
    if var_occ is None:
        var_occ = build_var_occ(clauses, nvars)

    sat_count = [0] * m
    sole_sat = [-1] * m
    unsat_list: List[int] = []
    pos = [-1] * m

    w = [1.0] * m
    tabu_until = [0] * nvars

    last_sniper_best = 10**9
    next_sniper_call = 0

    def add_unsat(ci: int) -> None:
        if pos[ci] == -1:
            pos[ci] = len(unsat_list)
            unsat_list.append(ci)

    def rem_unsat(ci: int) -> None:
        p = pos[ci]
        if p != -1:
            last = unsat_list[-1]
            unsat_list[p] = last
            pos[last] = p
            unsat_list.pop()
            pos[ci] = -1

    for ci, cl in enumerate(clauses):
        sc = 0
        sv = -1
        for lit in cl:
            v = abs(lit) - 1
            if lit_true(lit, a[v]):
                sc += 1
                sv = v
        sat_count[ci] = sc
        if sc == 0:
            add_unsat(ci)
            sole_sat[ci] = -1
        elif sc == 1:
            sole_sat[ci] = sv
        else:
            sole_sat[ci] = -1

    def apply_flip(v: int) -> None:
        old = a[v]
        a[v] = not a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            was_true = lit_true(lit, old)
            now_true = not was_true
            if was_true and not now_true:
                if sc == 1:
                    sat_count[ci] = 0
                    sole_sat[ci] = -1
                    add_unsat(ci)
                elif sc == 2:
                    sat_count[ci] = 1
                    sv = -1
                    for lit2 in clauses[ci]:
                        vv = abs(lit2) - 1
                        if lit_true(lit2, a[vv]):
                            sv = vv
                            break
                    sole_sat[ci] = sv
                else:
                    sat_count[ci] = sc - 1
            elif (not was_true) and now_true:
                if sc == 0:
                    sat_count[ci] = 1
                    sole_sat[ci] = v
                    rem_unsat(ci)
                elif sc == 1:
                    sat_count[ci] = 2
                    sole_sat[ci] = -1
                else:
                    sat_count[ci] = sc + 1

    def rebuild_from(best_assign: BoolAssign, shake: int) -> None:
        nonlocal a
        a = best_assign[:]
        if shake > 0:
            for _ in range(shake):
                vv = rng.randrange(nvars)
                a[vv] = not a[vv]
        unsat_list.clear()
        for i in range(m):
            pos[i] = -1
        for ci, cl in enumerate(clauses):
            sc = 0
            sv = -1
            for lit in cl:
                vv = abs(lit) - 1
                if lit_true(lit, a[vv]):
                    sc += 1
                    sv = vv
            sat_count[ci] = sc
            if sc == 0:
                add_unsat(ci)
                sole_sat[ci] = -1
            elif sc == 1:
                sole_sat[ci] = sv
            else:
                sole_sat[ci] = -1

    def mk_br(v: int) -> Tuple[float, float]:
        mk = 0.0
        br = 0.0
        cur = a[v]
        for ci, lit in var_occ[v]:
            sc = sat_count[ci]
            becomes_true = not lit_true(lit, cur)
            if becomes_true:
                if sc == 0:
                    mk += w[ci]
            else:
                if sc == 1 and sole_sat[ci] == v:
                    br += w[ci]
        return mk, br

    best = a[:]
    best_unsat = len(unsat_list)
    last_improve = 0
    last_improve_flip = 0
    reheat_until = 0
    next_endgame_reset = 0
    last_report = 0
    kick_left = 0
    kick_cooldown_left = 0
    prev_kick_left = 0
    t0 = time.time()
    next_endgame_jolt = 0
    # --- SNIPER RETRY (micro upgrade): re-run sniper from best if we stall in endgame ---
    next_sniper_retry = 0
    sniper_retry_after = 900_000      # how long we must stall before retrying sniper
    sniper_retry_cooldown = 900_000   # minimum spacing between retries

    # --- Fixpoint controller (bounded adaptivity; avoids thrash in endgame)
    p_state = float(p_base)
    eta_p = 0.05  # EMA rate; <1 ensures contractive adjustment
    stability = 0.0
    no_improve_epochs = 0

    # --- FAST stability tracker (no O(nvars) Hamming scan)
    # We approximate "instability" by the fraction of unique variables flipped since the last report.
    # This is O(1) per flip and O(1) per report.
    epoch = 1
    mark = array('I', [0]) * nvars  # per-var last-seen epoch
    changed_count = 0

    print(f"[finisher:init] p_min={p_min} p_max={p_max} p_base={p_base}")

    # --- HARD CAP: prevent catastrophic over-noise ---
    if p_max > 0.16:
        p_max = 0.16
    if kick_p > 0.16:
        kick_p = 0.16


    flip = 0
    while flip < max_flips:
        u = len(unsat_list)

        if u == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        if u < best_unsat:
            best_unsat = u
            best = a[:]
            last_improve_flip = flip
            # when we improve, cancel any reheat
            reheat_until = 0
            last_improve = flip

            if (
                    best_unsat <= sniper_u
                    and best_unsat <= last_sniper_best - 4  # call only if we improved meaningfully
                    and flip >= next_sniper_call
            ):
                """last_sniper_best = 10 ** 9
                next_sniper_call = 0"""

                last_sniper_best = best_unsat
                next_sniper_call = flip + 600_000  # cooldown

                print(f"[predator] target locked (u={best_unsat}), calling sniper ...")
                model2, ok2, st2 = finisher_classic_to_zero_sniper(
                    clauses=clauses,
                    nvars=nvars,
                    a0=best,
                    var_occ=var_occ,
                    seed=seed + 1337,
                    max_flips=sniper_flips,
                    p_mid=sniper_p,
                    p_mid_hard=sniper_p,
                    endgame_at=min(sniper_u, best_unsat),
                    p_end=0.1,
                    sniper_end_p=sniper_end_p,
                    tabu_tenure=tabu_tenure,
                    rb_prob=0.001,
                    rb_stall_flips=200_000,
                    report_every=report_every,
                )
                if ok2:
                    return model2, True, st2
                # continue from sniper's best, rebuilt
                a = model2[:]
                best = a[:]
                rebuild_from(best, shake=0)
                last_improve = flip

        stalled = flip - last_improve

        # --- micro rollback to best (endgame only, rate-limited via stall) ---
        if best_unsat <= sniper_u and stalled >= 600_000 and kick_left == 0:
            # very low probability to avoid thrashing
            if rng.random() < 0.0005:  # 1/2000
                rebuild_from(best, shake=0)
                last_improve = flip

        # --- SNIPER RETRY (micro upgrade) ---
        # If we're already close (best_unsat small) but haven't improved for a long time,
        # re-run the sniper from the *best* assignment to try to break the last core.
        if (
            best_unsat <= sniper_u
            and stalled >= sniper_retry_after
            and kick_left == 0
            and flip >= next_sniper_retry
        ):
            print(f"[predator] stalled near endgame (best={best_unsat}, stalled={stalled:,}), retrying sniper ...")
            model2, ok2, st2 = finisher_classic_to_zero_sniper(
                clauses=clauses,
                nvars=nvars,
                a0=best,
                var_occ=var_occ,
                seed=seed + 2337 + flip,   # tiny decorrelation per retry
                max_flips=sniper_flips,
                p_mid=sniper_p,
                p_mid_hard=sniper_p,
                endgame_at=24,
                p_end=0.06,
                sniper_end_p=sniper_end_p,
                tabu_tenure=tabu_tenure,
                report_every=report_every,
            )
            if ok2:
                return model2, True, st2

            # continue from sniper's best, rebuilt (no shake)
            a = model2[:]
            best = a[:]
            rebuild_from(best, shake=0)

            last_improve = flip
            next_sniper_retry = flip + sniper_retry_cooldown


        # --- ENDGAME WEIGHT RESET (micro-bypass to avoid weight "cementing") ---
        # Trigger only when very close to SAT and truly stalled; keep it rate-limited.
        if (
            best_unsat <= 32
            and stalled >= 650_000
            and kick_left == 0
            and flip >= next_endgame_reset
        ):
            # Drop clause weights to a neutral baseline and re-center on the best state.
            for i in range(m):
                w[i] = 1.0
            rebuild_from(best, shake=0)
            last_improve = flip  # reset stall timer
            next_endgame_reset = flip + 1_500_000  # cooldown
            # Gently re-inject some exploration so we don't immediately re-cement.
            p_state = min(p_max, max(p_state, p_min * 2.0))

        # --- ENDGAME TARGETED JOLT (break backbone clusters with minimal disruption) ---
        if (
            best_unsat <= 28
            and stalled >= 350_000
            and kick_left == 0
            and flip >= next_endgame_jolt
            and len(unsat_list) > 0
        ):
            # Collect vars from current UNSAT clauses
            cand = []
            for ci in unsat_list:
                for lit in clauses[ci]:
                    cand.append(abs(lit) - 1)

            if cand:
                # flip a few vars from the UNSAT-support set
                k = 16  # 12–24 typical; keep small
                for _ in range(k):
                    v = cand[rng.randrange(len(cand))]
                    apply_flip(v)

                last_improve = flip          # reset stall timer (we intentionally moved)
                next_endgame_jolt = flip + 900_000  # cooldown
                # slight exploration bump (very gentle)
                p_state = min(p_max, max(p_state, p_min * 1.5))

            """if cand:
                # flip a few vars from the UNSAT-support set
                k = 32 if best_unsat <= 20 else 16
                for _ in range(k):
                    v = cand[rng.randrange(len(cand))]
                    apply_flip(v)

                last_improve = flip          # reset stall timer (we intentionally moved)
                next_endgame_jolt = flip + (450_000 if best_unsat <= 20 else 900_000)  # cooldown
                # slight exploration bump (very gentle)
                p_state = min(p_max, max(p_state, p_min * 1.5))"""


        # --- CHAOS KICK (rate-limited, disabled in late endgame) ---
        if kick_cooldown_left > 0:
            kick_cooldown_left -= 1

        # Disable kicks when we're already close to endgame.
        kick_disabled = (best_unsat <= kick_disable_best_mult * sniper_u)

        if (not kick_disabled) and stalled > kick_after and kick_left == 0 and kick_cooldown_left == 0:
            kick_left = kick_len
            kick_cooldown_left = kick_cooldown
            # Reset stall timer so we don't immediately retrigger on kick end.
            last_improve = flip
            print(f"!!! [kick] stagnation at best={best_unsat} -> chaos kick for {kick_len} flips (cooldown {kick_cooldown})")

        """if kick_left == 0 and best_unsat > 0:
            meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs)
            softsnap = (u > best_unsat + snapback_gap) and (u > basin_abs)
            if meltdown:
                rebuild_from(best, shake=restart_shake)
                last_improve = flip
                continue
            if softsnap:
                rebuild_from(best, shake=0)
                last_improve = flip
                continue"""

        if kick_left == 0 and best_unsat > 0:
            if best_unsat <= sniper_u:
                # endgame: basin_abs must NOT block snapback
                meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap))
                softsnap = (u > best_unsat + snapback_gap)
            else:
                meltdown = (u > max(int(best_unsat * basin_mult), best_unsat + snapback_gap)) and (u > basin_abs)
                softsnap = (u > best_unsat + snapback_gap) and (u > basin_abs)

            if meltdown:
                rebuild_from(best, shake=restart_shake)
                last_improve = flip
                continue
            if softsnap:
                rebuild_from(best, shake=0)
                last_improve = flip
                continue

        if stalled > stall_window and kick_left == 0:
            rebuild_from(best, shake=restart_shake)
            last_improve = flip
            continue

        if (flip & 255) == 0:
            for i in range(m):
                wi = w[i] * w_decay
                w[i] = wi if wi >= 1.0 else 1.0

        prev_kick_left = kick_left
        if kick_left > 0:
            p_eff = min(p_max, max(kick_p, p_min))
            if p_eff > 0.16:
                p_eff = 0.16

            kick_left -= 1
            # when a kick ends, immediately re-center on best with a small core-shake
            if prev_kick_left == 1:
                rebuild_from(best, shake=max(8, restart_shake // 2))
                last_improve = flip
                # IMPORTANT: rebuild changes unsat_list length; refresh u to avoid stale index
                u = len(unsat_list)
                if u == 0:
                    return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}

        else:
            # Target noise is a bounded function of (u, stability), with a gentle EMA update.
            # Intuition: when dynamics are already stable (high stability) and u is small, we want a
            # near-fixpoint regime (low noise). When the state is unstable or plateauing, we reheat a bit.
            progress = min(1.0, u / max(1.0, float(sniper_u) * 8.0))
            base = p_min + (p_max - p_min) * math.sqrt(progress)
            if u <= sniper_u:
                base = min(base, float(sniper_end_p))
            stab_factor = 1.0 + 2.0 * (1.0 - stability)  # in [1,3] approximately
            target_p = base * stab_factor
            # Do not ramp noise to max when we're already in endgame.
            if best_unsat > 32 and (flip - last_improve) >= 2000000:
                target_p *= 1.35
            target_p = float(max(p_min, min(p_max, target_p)))
            p_state = (1.0 - eta_p) * p_state + eta_p * target_p
            p_eff = float(max(p_min, min(p_max, p_state)))

            p_eff = float(max(p_min, min(p_max, p_eff)))

        if report_every and (flip - last_report) >= report_every:
            # --- stability estimate from unique vars touched since last report
            instab = changed_count / max(1, nvars)  # 0..1
            stability = 0.95 * stability + 0.05 * (1.0 - instab)

            # reset tracker (no per-var clearing)
            epoch += 1
            if epoch >= 0xFFFFFFFF:
                # extremely rare; keep it safe
                epoch = 1
                for i in range(nvars):
                    mark[i] = 0
            changed_count = 0

            last_report = flip
            print(f"[finisher] flips={flip:,} unsat={u} p_eff={p_eff:.3f} best={best_unsat} stab={stability:.3f}", flush=True)

        u_pick = len(unsat_list)
        if u_pick == 0:
            return a, True, {"phase": "predator", "flips": flip, "best_unsat": 0, "time_s": time.time() - t0}
        ci = unsat_list[rng.randrange(u_pick)]
        cl = clauses[ci]

        if rng.random() < p_eff:
            v = abs(cl[rng.randrange(len(cl))]) - 1
        else:
            best_v = None
            best_score = -1.0
            for lit in cl:
                v0 = abs(lit) - 1
                if kick_left == 0 and use_tabu and u <= tabu_u_threshold and tabu_until[v0] > flip:
                    continue
                mk, br = mk_br(v0)
                eps = 1e-6
                mk1 = mk + eps
                br1 = br + eps
                score = (mk1 * mk1 * mk1 * mk1 * mk1) / (br1 * br1 * br1)  # mk^5 / br^3
                if score > best_score:
                    best_score = score
                    best_v = v0
            v = best_v if best_v is not None else abs(cl[rng.randrange(len(cl))]) - 1

        apply_flip(v)

        # stability tracker: mark var touched in this epoch (O(1), no scans)
        if mark[v] != epoch:
            mark[v] = epoch
            changed_count += 1

        if kick_left == 0 and use_tabu and len(unsat_list) <= tabu_u_threshold:
            tabu_until[v] = flip + tabu_tenure

        w[ci] = min(w_cap, w[ci] + w_inc)
        flip += 1

    return best, False, {"phase": "predator", "flips": flip, "best_unsat": best_unsat, "time_s": time.time() - t0}

"""def focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):

    if not unsat_list:
        return False

    cl = clauses[unsat_list[rng.randrange(len(unsat_list))]]

    best_seq = None
    best_u = len(unsat_list)

    for lit1 in cl:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = 0
        for cid in unsat_list:
            sat = False
            for l in clauses[cid]:
                vv = abs(l) - 1
                if (a[vv] and l > 0) or ((not a[vv]) and l < 0):
                    sat = True
                    break
            if not sat:
                u1 += 1

        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in cl:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)

            u2 = 0
            for cid in unsat_list:
                sat = False
                for l in clauses[cid]:
                    vv = abs(l) - 1
                    if (a[vv] and l > 0) or ((not a[vv]) and l < 0):
                        sat = True
                        break
                if not sat:
                    u2 += 1

            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]

            flip_var(v2)
        flip_var(v1)

    if best_seq and best_u < len(unsat_list):
        for v in best_seq:
            flip_var(v)
        return True

    return False"""

def focused_endgame_pulse(clauses, a, unsat_list, var_occ, breakcount, flip_var, rng):
    """
    1–2 flip pulse for u <= 25: evaluate by REAL solver state (len(unsat_list)).
    """
    if not unsat_list:
        return False

    base_u = len(unsat_list)
    cl = clauses[unsat_list[rng.randrange(base_u)]]
    lits = cl[:]

    best_seq = None
    best_u = base_u

    for lit1 in lits:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = len(unsat_list)
        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in lits:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)
            u2 = len(unsat_list)
            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]
            flip_var(v2)

        flip_var(v1)

    if best_seq and best_u < base_u:
        for v in best_seq:
            flip_var(v)
        return True

    return False



"""def focused_endgame_pulse3(clauses, a, unsat_list, flip_var, rng):

    if not unsat_list:
        return False

    target = list(unsat_list)
    base_u = len(target)

    # pick one UNSAT clause as anchor
    cl = clauses[target[rng.randrange(base_u)]]

    def still_unsat_count():
        u = 0
        for cid in target:
            sat = False
            for l in clauses[cid]:
                vv = abs(l) - 1
                if (a[vv] and l > 0) or ((not a[vv]) and l < 0):
                    sat = True
                    break
            if not sat:
                u += 1
        return u

    best_seq = None
    best_u = base_u
    lits = cl[:]  # local

    for lit1 in lits:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = still_unsat_count()
        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in lits:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)
            u2 = still_unsat_count()
            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]

            for lit3 in lits:
                v3 = abs(lit3) - 1
                if v3 == v1 or v3 == v2:
                    continue
                flip_var(v3)
                u3 = still_unsat_count()
                if u3 < best_u:
                    best_u = u3
                    best_seq = [v1, v2, v3]
                flip_var(v3)

            flip_var(v2)
        flip_var(v1)

    if best_seq and best_u < base_u:
        for v in best_seq:
            flip_var(v)
        return True
    return False"""

def focused_endgame_pulse3(clauses, a, unsat_list, flip_var, rng):
    """
    3-flip pulse for u <= ~22: evaluate by REAL solver state (len(unsat_list)).
    No stale snapshots, no clause rescans.
    """
    if not unsat_list:
        return False

    base_u = len(unsat_list)
    # anchor clause from current UNSAT set
    cl = clauses[unsat_list[rng.randrange(base_u)]]
    lits = cl[:]

    best_seq = None
    best_u = base_u

    for lit1 in lits:
        v1 = abs(lit1) - 1
        flip_var(v1)
        u1 = len(unsat_list)
        if u1 < best_u:
            best_u = u1
            best_seq = [v1]

        for lit2 in lits:
            v2 = abs(lit2) - 1
            if v2 == v1:
                continue
            flip_var(v2)
            u2 = len(unsat_list)
            if u2 < best_u:
                best_u = u2
                best_seq = [v1, v2]

            for lit3 in lits:
                v3 = abs(lit3) - 1
                if v3 == v1 or v3 == v2:
                    continue
                flip_var(v3)
                u3 = len(unsat_list)
                if u3 < best_u:
                    best_u = u3
                    best_seq = [v1, v2, v3]
                flip_var(v3)

            flip_var(v2)
        flip_var(v1)

    if best_seq and best_u < base_u:
        for v in best_seq:
            flip_var(v)
        return True

    return False


# -------------------------
# CLI + spectral passthrough
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DREAM6 clean runner (spectral unchanged, finisher vFinal).")
    ap.add_argument("--cnf", required=True)
    ap.add_argument("--subset", type=int, default=0)

    ap.add_argument("--mode", type=str, default="unsat_hadamard")
    ap.add_argument("--cR", type=float, default=10.0)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.734296875)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--sigma_up", type=float, default=0.045)
    ap.add_argument("--neighbor_atten", type=float, default=0.9495)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--couple", type=int, default=1)
    ap.add_argument("--power_iters", type=int, default=60)
    ap.add_argument("--score_norm_alpha", type=float, default=0.5)
    ap.add_argument("--bias_weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--godmode", action="store_true")

    ap.add_argument("--core_seed_loops", type=int, default=0)
    ap.add_argument("--core_rounds", type=int, default=0)
    ap.add_argument("--core_step0", type=float, default=0.12)
    ap.add_argument("--core_decay", type=float, default=0.90)
    ap.add_argument("--core_per_clause_cap", type=int, default=2)
    ap.add_argument("--core_beta", type=float, default=1.3)
    ap.add_argument("--core_bias_damp", type=float, default=0.70)

    ap.add_argument("--polish", type=int, default=300_000)
    ap.add_argument("--finisher_flips", type=int, default=50_000_000)

    ap.add_argument("--p_min", type=float, default=0.003)
    ap.add_argument("--p_max", type=float, default=0.18)
    ap.add_argument("--p_base", type=float, default=0.10)
    ap.add_argument("--stall_window", type=int, default=800_000)
    ap.add_argument("--restart_shake", type=int, default=96)
    ap.add_argument("--w_inc", type=float, default=1.0)
    ap.add_argument("--w_decay", type=float, default=0.9996)
    ap.add_argument("--w_cap", type=float, default=40.0)

    ap.add_argument("--snapback_gap", type=int, default=250)
    ap.add_argument("--basin_mult", type=float, default=2.2)
    ap.add_argument("--basin_abs", type=int, default=350)

    ap.add_argument("--kick_after", type=int, default=300_000)
    ap.add_argument("--kick_len", type=int, default=100_000)
    ap.add_argument("--kick_p", type=float, default=0.18)

    ap.add_argument("--sniper_u", type=int, default=64)
    ap.add_argument("--sniper_flips", type=int, default=8_000_000)
    ap.add_argument("--sniper_p", type=float, default=0.33)
    ap.add_argument("--sniper_end_p", type=float, default=0.03)

    ap.add_argument("--tabu", action="store_true")
    ap.add_argument("--tabu_u", type=int, default=128)
    ap.add_argument("--tabu_tenure", type=int, default=45)

    ap.add_argument("--report_every", type=int, default=100_000)
    return ap.parse_args()


def main() -> None:
    print("\n*** DREAM6_clean_vFinal2_fixed (spectral unchanged) ***\n")
    args = parse_args()

    if args.godmode:
        args.cR = 10.0
        args.L = 3
        args.rho = 0.734296875
        args.zeta0 = 0.40
        args.sigma_up = 0.045
        args.neighbor_atten = 0.9495
        args.seed = 42
        args.couple = 1
        args.d = 6
        args.mode = "unsat_hadamard"
        args.power_iters = max(args.power_iters, 60)

    nvars, clauses = parse_dimacs(args.cnf)
    if args.subset and args.subset > 0:
        clauses = clauses[: args.subset]
    C = len(clauses)

    print(f"File: {os.path.basename(args.cnf)}")
    print(f"Vars: {nvars}  Clauses: {C}  C/n: {C/max(1,nvars):.4f}")

    params = theory_params(C=C, want_sigma=0.02, cR=12, L=4, zeta0=0.4,
                           rho_lock=0.734296875, tau_hint=0.40, mu_hint=0.002)
    print("optimal params: ", params)

    print("\n[spectral] generating spectral seed ...\n")
    t0 = time.time()
    assign, _Phi = build_seed_assignment(
        clauses,
        nvars,
        mode=args.mode,
        cR=args.cR,
        L=args.L,
        rho=args.rho,
        zeta0=args.zeta0,
        sigma_up=args.sigma_up,
        neighbor_atten=args.neighbor_atten,
        seed=args.seed,
        couple=bool(args.couple),
        d=args.d,
        power_iters=args.power_iters,
        score_norm_alpha=args.score_norm_alpha,
        bias_weight=args.bias_weight,
    )

    if args.core_seed_loops > 0 and args.core_rounds > 0:
        print("\n[spectral] core push ...\n")
        assign = core_push(
            clauses,
            nvars,
            assign,
            seed=args.seed,
            loops=args.core_seed_loops,
            rounds=args.core_rounds,
            step0=args.core_step0,
            decay=args.core_decay,
            per_clause_cap=args.core_per_clause_cap,
            beta=args.core_beta,
            bias_damp=args.core_bias_damp,
        )

    if args.polish and args.polish > 0:
        assign = greedy_polish(clauses, assign, flips=args.polish, seed=args.seed)

    t1 = time.time()
    uns0 = count_unsat(clauses, assign)
    print("\n=== SPECTRAL REPORT ===")
    print(f"UNSAT clauses     : {uns0} / {C}  ({100.0*uns0/max(1,C):.2f}%)")
    print(f"Seed time         : {t1 - t0:.3f}s")

    if uns0 == 0:
        print("\n[spectral] SAT already. ✅")
        return

    print("\n[spectral] passing to finisher …\n")
    var_occ = build_var_occ(clauses, nvars)

    model, solved, st = finisher_predator_sole_sat_vFinal(
        clauses=clauses,
        nvars=nvars,
        a0=assign,
        var_occ=var_occ,
        seed=args.seed,
        max_flips=args.finisher_flips,
        p_min=args.p_min,
        p_max=args.p_max,
        p_base=args.p_base,
        stall_window=args.stall_window,
        restart_shake=args.restart_shake,
        w_inc=args.w_inc,
        w_decay=args.w_decay,
        w_cap=args.w_cap,
        snapback_gap=args.snapback_gap,
        basin_mult=args.basin_mult,
        basin_abs=args.basin_abs,
        kick_after=args.kick_after,
        kick_len=args.kick_len,
        kick_p=args.kick_p,
        sniper_u=args.sniper_u,
        sniper_end_p=args.sniper_end_p,
        sniper_flips=args.sniper_flips,
        sniper_p=args.sniper_p,
        use_tabu=bool(args.tabu),
        tabu_u_threshold=args.tabu_u,
        tabu_tenure=args.tabu_tenure,
        report_every=args.report_every,
    )

    sat = check_sat(clauses, _as_int_assign(model))
    uns = count_unsat(clauses, _as_int_assign(model))

    print("\n=== FINISHER RESULT ===")
    print(f"UNSAT clauses  : {uns} / {C}  ({100.0 * uns / max(1, C):.2f}%)")
    print(f"Verified SAT   : {sat}")
    print(f"Solved flag    : {solved}")
    if isinstance(st, dict):
        print(f"Stats          : {st}")


if __name__ == "__main__":
    main()