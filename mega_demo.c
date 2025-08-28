#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- RNG ----------
static uint32_t rng_state = 1u;
static inline void RandSeed(uint32_t seed){ if(seed==0) seed=1; rng_state=seed; }
static inline uint32_t RandU32(void){
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static inline double Rand01(){ return (double)RandU32() / 4294967296.0; }

// ---------- Problem ----------
typedef struct { int lit[3]; } Clause;
typedef struct { int clause; int sign; } Occ; // sign: +1 for positive literal, -1 for negative

typedef struct {
    int nVars, nClauses;
    Clause *clauses;
    uint8_t *assign;     // current assignment
    int *sat_count;      // per clause: # satisfied literals
    int unsat_count;

    // occurrences
    int *deg;            // degree per var
    int **adj_idx;       // for each var, list of indices into occ[]
    Occ *occ;            // flattened occurrences

    // dynamic unsat set
    int *unsat_list;     // list of clause indices with sat_count==0
    int *unsat_pos;      // per clause: position in unsat_list or -1
} Solver;

static inline uint8_t LitValue(int lit, const uint8_t *assign){
    int v = abs(lit)-1;
    uint8_t val = assign[v];
    return (lit>0) ? val : !val;
}

// Maintain unsat set
static inline void unsat_add(Solver *s, int c){
    if (s->unsat_pos[c] >= 0) return;
    int pos = s->unsat_count;
    s->unsat_list[pos] = c;
    s->unsat_pos[c] = pos;
    s->unsat_count++;
}
static inline void unsat_remove(Solver *s, int c){
    int pos = s->unsat_pos[c];
    if (pos < 0) return;
    int last = s->unsat_count - 1;
    int last_c = s->unsat_list[last];
    s->unsat_list[pos] = last_c;
    s->unsat_pos[last_c] = pos;
    s->unsat_pos[c] = -1;
    s->unsat_count = last;
}

// Build variable occurrences
void BuildOccurrences(Solver *s){
    int n = s->nVars, m = s->nClauses;
    s->deg = (int*)calloc(n, sizeof(int));
    for (int c=0;c<m;++c){
        for (int j=0;j<3;++j){
            int v = abs(s->clauses[c].lit[j]) - 1;
            s->deg[v]++;
        }
    }
    int total = 0;
    for (int v=0; v<n; ++v) total += s->deg[v];
    s->occ = (Occ*)malloc(total * sizeof(Occ));
    s->adj_idx = (int**)malloc(n * sizeof(int*));
    int *cursor = (int*)calloc(n, sizeof(int));
    for (int v=0; v<n; ++v) s->adj_idx[v] = (int*)malloc(s->deg[v] * sizeof(int));

    int idx=0;
    for (int c=0;c<m;++c){
        for (int j=0;j<3;++j){
            int lit = s->clauses[c].lit[j];
            int v = abs(lit)-1;
            s->occ[idx].clause = c;
            s->occ[idx].sign = (lit>0)?+1:-1;
            int pos = cursor[v]++;
            s->adj_idx[v][pos] = idx;
            idx++;
        }
    }
    free(cursor);
}

// Init sat counts and unsat set from current assignment
void InitSatCounts(Solver *s){
    int m = s->nClauses;
    s->sat_count = (int*)malloc(m * sizeof(int));
    s->unsat_list = (int*)malloc(m * sizeof(int));
    s->unsat_pos  = (int*)malloc(m * sizeof(int));
    s->unsat_count = 0;
    for (int c=0;c<m;++c){
        s->unsat_pos[c] = -1;
        int sc = 0;
        for (int j=0;j<3;++j) sc += LitValue(s->clauses[c].lit[j], s->assign);
        s->sat_count[c] = sc;
        if (sc==0) unsat_add(s,c);
    }
}

// Incremental flip with unsat maintenance
static inline void FlipVar(Solver *s, int v){
    uint8_t before = s->assign[v];
    s->assign[v] = !before;
    int deg = s->deg[v];
    for (int i=0;i<deg;++i){
        int occ_idx = s->adj_idx[v][i];
        int c = s->occ[occ_idx].clause;
        int sign = s->occ[occ_idx].sign;
        int was_true = (sign>0) ? before : (!before);
        if (was_true){
            // literal becomes false
            int sc = --s->sat_count[c];
            if (sc==0) unsat_add(s,c);
        } else {
            // literal becomes true
            int sc = ++s->sat_count[c];
            if (sc==1) unsat_remove(s,c);
        }
    }
}

// Compute "breakcount" for variable v: clauses that would become unsatisfied
static inline int BreakCount(Solver *s, int v){
    uint8_t before = s->assign[v];
    int deg = s->deg[v];
    int br = 0;
    for (int i=0;i<deg;++i){
        int occ_idx = s->adj_idx[v][i];
        int c = s->occ[occ_idx].clause;
        int sign = s->occ[occ_idx].sign;
        int was_true = (sign>0) ? before : (!before);
        if (was_true && s->sat_count[c]==1) br++;
    }
    return br;
}

// WalkSAT-style step: pick random unsat clause, choose variable to flip
static inline void WalkSATStep(Solver *s, double wp){
    // pick an unsat clause uniformly
    int idx = (int)(Rand01() * s->unsat_count);
    if (idx >= s->unsat_count) idx = s->unsat_count - 1;
    int c = s->unsat_list[idx];
    int v_idx[3];
    for (int j=0;j<3;++j) v_idx[j] = abs(s->clauses[c].lit[j]) - 1;

    if (Rand01() < wp){
        // random walk: flip random var from the clause
        int pick = v_idx[RandU32()%3];
        FlipVar(s, pick);
        return;
    }
    // greedy: minimize breakcount among the 3
    int best_v = v_idx[0];
    int best_br = BreakCount(s, best_v);
    for (int j=1;j<3;++j){
        int v = v_idx[j];
        int br = BreakCount(s, v);
        if (br < best_br){
            best_br = br;
            best_v = v;
        }
    }
    FlipVar(s, best_v);
}

// Planted instance generator: pick random assignment A, generate clauses satisfied by A
void GenPlanted3SAT(Clause *cls, uint8_t *A, int nVars, int nClauses){
    for (int i=0;i<nVars;++i) A[i] = RandU32() & 1;
    for (int c=0;c<nClauses;++c){
        int v[3];
        // unique vars in clause
        v[0] = RandU32()%nVars;
        do { v[1] = RandU32()%nVars; } while (v[1]==v[0]);
        do { v[2] = RandU32()%nVars; } while (v[2]==v[0] || v[2]==v[1]);
        // choose signs so that clause is satisfied by A
        for (int j=0;j<3;++j){
            int sign = (RandU32() & 1) ? +1 : -1;
            int lit = (A[v[j]] ? +1 : -1) * (v[j]+1); // literal true under A
            // with 1/2 make it the satisfying literal, otherwise random but ensure not all false
            if (RandU32() & 1){
                cls[c].lit[j] = lit;
            } else {
                // random sign, but if we accidentally make all three false, force this one true
                int sgn = (RandU32() & 1) ? +1 : -1;
                cls[c].lit[j] = sgn * (v[j]+1);
            }
        }
        // ensure clause is satisfied by A
        int ok = 0;
        for (int j=0;j<3;++j){
            int lit = cls[c].lit[j];
            int var = abs(lit)-1;
            int val = A[var];
            int true_lit = (lit>0) ? val : !val;
            ok |= true_lit;
        }
        if (!ok){
            // force first literal to be true under A
            int j=0;
            int var = v[j];
            cls[c].lit[j] = (A[var] ? +1 : -1) * (var+1);
        }
    }
}

// Try to solve planted instance with WalkSAT
int SolvePlanted(int nVars, int nClauses, double wp, int64_t maxFlips, int prints, double *out_ms, int *out_flips){
    Solver s;
    s.nVars = nVars;
    s.nClauses = nClauses;
    s.clauses = (Clause*)malloc(nClauses * sizeof(Clause));
    s.assign  = (uint8_t*)malloc(nVars * sizeof(uint8_t));
    uint8_t *A = (uint8_t*)malloc(nVars * sizeof(uint8_t)); // planted sol (unused for solving, only generation)
    GenPlanted3SAT(s.clauses, A, nVars, nClauses);

    // random init assignment
    for (int i=0;i<nVars;++i) s.assign[i] = RandU32() & 1;

    BuildOccurrences(&s);
    InitSatCounts(&s);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int64_t flips=0;
    while (s.unsat_count>0 && flips<maxFlips){
        WalkSATStep(&s, wp);
        flips++;
        if (prints && (flips % (int64_t)1e6 == 0)){
            printf("[%,lld flips] energy=%d\n", (long long)flips, s.unsat_count);
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    if (out_ms) *out_ms = ms;
    if (out_flips) *out_flips = (int)flips;

    int solved = (s.unsat_count==0);
    if (solved){
        // Verify against planted assignment A (should satisfy)
        int ok=1;
        for (int c=0;c<nClauses;++c){
            int sc=0;
            for (int j=0;j<3;++j){
                int lit = s.clauses[c].lit[j];
                int v = abs(lit)-1;
                int val = s.assign[v];
                int truth = (lit>0) ? val : !val;
                sc |= truth;
            }
            if (!sc){ ok=0; break; }
        }
        printf("Solved: energy=0, verified=%s\n", ok ? "OK" : "FAIL");
        printf("Flips: %d, Time: %.3f ms, Speed: %.1f Mflips/s\n",
               (int)flips, ms, (flips/1e6)/(ms/1000.0));
    } else {
        printf("NOT solved: energy=%d, Flips: %d, Time: %.3f ms\n", s.unsat_count, (int)flips, ms);
    }

    // cleanup
    for (int v=0; v<s.nVars; ++v) free(s.adj_idx[v]);
    free(s.adj_idx);
    free(s.occ);
    free(s.deg);
    free(s.unsat_list);
    free(s.unsat_pos);
    free(s.sat_count);
    free(s.assign);
    free(s.clauses);
    free(A);
    return solved;
}

int main(){
    RandSeed(20250828u);

    // Choose a large planted instance
    int n = 10000;
    int m = 42000; // ratio 4.2
    double wp = 0.20;   // noise prob for WalkSAT
    long long maxFlips = 20000000LL; // 20M flips
    int prints = 1;

    double ms; int flips;
    int ok = SolvePlanted(n, m, wp, maxFlips, prints, &ms, &flips);

    if (ok){
        printf("MEGA-DEMO: n=%d, m=%d solved in %d flips (%.2f ms)\n", n, m, flips, ms);
    } else {
        printf("MEGA-DEMO: n=%d, m=%d NOT solved within %,lld flips\n", n, m, maxFlips);
    }
    return 0;
}
