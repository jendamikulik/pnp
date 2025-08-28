#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_CATS 5
#define IDX_LOVE 4

static uint32_t rng_state = 1u;
void RandSeed(uint32_t seed){ if(seed==0) seed=1; rng_state=seed; }
uint32_t RandU32(void){
    // xorshift32
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
double Rand01(){ return (double)RandU32() / 4294967296.0; }
double RandNorm(){
    double u1 = Rand01(); if (u1 <= 0.0) u1 = 1e-12;
    double u2 = Rand01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

typedef struct { int lit[3]; } Clause;

typedef struct {
    int clause;
    int sign; // +1 for positive, -1 for negative
} Occ;

typedef struct {
    int nVars;
    int nClauses;
    Clause *clauses;
    uint8_t *assign;
    double E[NUM_CATS];
    double alpha;
    double zeta;
    // incremental state
    int *sat_count;       // per clause: number of satisfied literals
    int unsat_count;      // number of clauses with sat_count==0
    // variable occurrences
    int *deg;             // degree per var
    int **adj_idx;        // for each var, indices into occ array
    Occ *occ;             // flattened occurrences
} Solver;

// ---------- Clause eval (single) for init / debug ----------
static inline uint8_t LitValue(int lit, const uint8_t *assign){
    int v = abs(lit)-1;
    uint8_t val = assign[v];
    if (lit < 0) val = !val;
    return val;
}

void BuildOccurrences(Solver *s){
    int n = s->nVars, m = s->nClauses;
    s->deg = (int*)calloc(n, sizeof(int));
    // first pass: count degrees
    for (int c=0;c<m;++c){
        for (int j=0;j<3;++j){
            int v = abs(s->clauses[c].lit[j])-1;
            if (v>=0 && v<n) s->deg[v]++;
        }
    }
    // prefix sums to allocate slots
    int total = 0;
    for (int v=0; v<n; ++v) total += s->deg[v];
    s->occ = (Occ*)malloc(total * sizeof(Occ));
    s->adj_idx = (int**)malloc(n * sizeof(int*));
    int *cursor = (int*)calloc(n, sizeof(int));
    for (int v=0; v<n; ++v){
        s->adj_idx[v] = (int*)malloc(s->deg[v] * sizeof(int));
    }
    // second pass: fill occurrences
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

void InitSatCounts(Solver *s){
    int m = s->nClauses;
    s->sat_count = (int*)malloc(m * sizeof(int));
    s->unsat_count = 0;
    for (int c=0;c<m;++c){
        int sc=0;
        for (int j=0;j<3;++j) sc += LitValue(s->clauses[c].lit[j], s->assign);
        s->sat_count[c]=sc;
        if (sc==0) s->unsat_count++;
    }
}

void UpdateEnergyField(Solver *s){
    double S = 0.0;
    for (int k = 0; k < NUM_CATS; ++k) S += s->E[k];
    double mLove = s->E[IDX_LOVE] / S;
    double delta = s->alpha * (1.0 - mLove);
    s->E[IDX_LOVE] += delta;
    for (int k = 0; k < NUM_CATS; ++k) s->E[k] -= delta / NUM_CATS;
    for (int k = 0; k < NUM_CATS; ++k) s->E[k] += s->zeta * RandNorm();
    for (int k = 0; k < NUM_CATS; ++k) if (s->E[k] < 1e-9) s->E[k] = 1e-9;
}

// Flip variable v with incremental unsat update; return new unsat_count
int FlipVar(Solver *s, int v){
    uint8_t before = s->assign[v];
    s->assign[v] = !before;
    int deg = s->deg[v];
    for (int i=0;i<deg;++i){
        int occ_idx = s->adj_idx[v][i];
        int c = s->occ[occ_idx].clause;
        int sign = s->occ[occ_idx].sign; // +1 or -1
        int before_sat = (sign>0) ? before : (!before);
        if (before_sat){
            // literal becomes false
            s->sat_count[c]--;
            if (s->sat_count[c]==0) s->unsat_count++;
        } else {
            // literal becomes true
            s->sat_count[c]++;
            if (s->sat_count[c]==1) s->unsat_count--;
        }
    }
    return s->unsat_count;
}

// Greedy sweep with incremental book-keeping
int GreedyFlipSweep(Solver *s){
    int best = s->unsat_count;
    for (int v=0; v<s->nVars; ++v){
        int before_unsat = s->unsat_count;
        FlipVar(s, v); // tentative flip
        int after_unsat = s->unsat_count;
        if (after_unsat <= best){
            best = after_unsat;
            // keep
        } else {
            // revert
            FlipVar(s, v);
            s->unsat_count = before_unsat; // FlipVar updates it back
        }
    }
    return best;
}

int Solve(Solver *s, int maxSweeps, int tol){
    for (int step=0; step<maxSweeps; ++step){
        int uns = GreedyFlipSweep(s);
        if (uns <= tol){
            printf("Solved after %d sweeps; energy=%d\n", step+1, uns);
            return 1;
        }
        UpdateEnergyField(s);
        if ((step+1)%100==0){
            printf("[sweep %7d] energy=%d\n", step+1, s->unsat_count);
        }
    }
    printf("Not solved; maxSweeps=%d, energy=%d\n", maxSweeps, s->unsat_count);
    return 0;
}

int TrySolveIncreasing(Solver *s, int startSweeps, int maxSweepsCap, int restarts){
    int sweeps = startSweeps;
    for (int r=0; r<restarts; ++r){
        // random init
        for (int i=0;i<s->nVars;++i) s->assign[i] = (RandU32() & 1);
        InitSatCounts(s);
        for (int i=0; i<NUM_CATS; ++i) s->E[i]=1.0;
        // doubling schedule for this restart
        sweeps = startSweeps;
        while (sweeps <= maxSweepsCap){
            if (Solve(s, sweeps, 0)) return 1;
            sweeps *= 2;
        }
        printf("Restart %d done (cap %d). Energy=%d\n", r+1, maxSweepsCap, s->unsat_count);
    }
    return 0;
}

// Random 3-SAT instance generator (no duplicate vars in a clause)
void GenRandom3SAT(Clause *cls, int nVars, int nClauses){
    for (int c=0; c<nClauses; ++c){
        int v[3];
        // pick three distinct vars
        v[0] = RandU32()%nVars;
        do { v[1] = RandU32()%nVars; } while (v[1]==v[0]);
        do { v[2] = RandU32()%nVars; } while (v[2]==v[0] || v[2]==v[1]);
        for (int j=0;j<3;++j){
            int sign = (RandU32() & 1) ? +1 : -1;
            cls[c].lit[j] = sign * (v[j]+1); // 1-based
        }
    }
}

int main(){
    // ---- MEGA parameters (tweakable) ----
    int nVars    = 2000;     // try bigger (e.g., 5000, 10000) if you want
    int nClauses = 8000;     // ~4x nVars ratio (near-ish threshold)
    int restarts = 5;
    int startSweeps = 1000;
    int capSweeps   = 64000;

    RandSeed(123456789u);

    // allocate
    Clause *cls = (Clause*)malloc(nClauses * sizeof(Clause));
    GenRandom3SAT(cls, nVars, nClauses);

    Solver s;
    s.nVars = nVars;
    s.nClauses = nClauses;
    s.clauses = cls;
    s.assign = (uint8_t*)malloc(nVars * sizeof(uint8_t));
    s.alpha = 0.25;
    s.zeta  = 0.05;

    BuildOccurrences(&s);
    // initial random assign + sat counts done inside TrySolveIncreasing

    double t0 = 0.0, t1 = 0.0;
    // (timing left to the caller env; simple print only)
    int solved = TrySolveIncreasing(&s, startSweeps, capSweeps, restarts);

    if (solved){
        printf("MEGA test: SOLVED with energy 0.\n");
        printf("Assignments (first 20 vars):\n");
        for (int i=0;i<20 && i<s.nVars;++i){
            printf("  x%-3d=%d%s", i+1, s.assign[i], ((i%5)==4 || i==19 || i==s.nVars-1) ? "\n" : "  ");
        }
    } else {
        printf("MEGA test: NOT solved within limits. Final energy=%d\n", s.unsat_count);
    }

    // clean
    for (int v=0; v<s.nVars; ++v) free(s.adj_idx[v]);
    free(s.adj_idx);
    free(s.occ);
    free(s.deg);
    free(s.sat_count);
    free(s.assign);
    free(cls);
    return 0;
}
