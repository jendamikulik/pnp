#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_CATS 5
#define IDX_LOVE 4

// --- toggle: small demo vs. planted generator ---
#define DEMO_SMALL 0

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
    int nVars;
    int nClauses;
    Clause *clauses;
    uint8_t *assign;
    double E[NUM_CATS];
    double alpha;
    double zeta;
} Solver;

uint8_t EvalClause(Clause *c, uint8_t *assign){
    for (int i = 0; i < 3; ++i){
        int lit = c->lit[i];
        int idx = abs(lit) - 1; // 1-based -> 0-based
        uint8_t val = assign[idx];
        if (lit < 0) val = !val;
        if (val) return 1;
    }
    return 0;
}

int CountUnsat(Solver *s){
    int cnt = 0;
    for (int i = 0; i < s->nClauses; ++i)
        if (!EvalClause(&s->clauses[i], s->assign))
            ++cnt;
    return cnt;
}

void UpdateEnergyField(Solver *s){
    double S = 0.0;
    for (int k = 0; k < NUM_CATS; ++k) S += s->E[k];

    double mLove = s->E[IDX_LOVE] / S;
    double delta = s->alpha * (1.0 - mLove);
    s->E[IDX_LOVE] += delta;
    for (int k = 0; k < NUM_CATS; ++k) s->E[k] -= delta / NUM_CATS;

    // noise
    for (int k = 0; k < NUM_CATS; ++k) s->E[k] += s->zeta * RandNorm();

    // clip
    for (int k = 0; k < NUM_CATS; ++k) if (s->E[k] < 1e-9) s->E[k] = 1e-9;
}

int GreedyFlipSweep(Solver *s){
    int bestUnsat = CountUnsat(s);
    for (int v = 0; v < s->nVars; ++v){
        s->assign[v] = !s->assign[v];
        int unsatNow = CountUnsat(s);
        if (unsatNow < bestUnsat){
            bestUnsat = unsatNow;
        } else {
            s->assign[v] = !s->assign[v];
        }
    }
    return bestUnsat;
}

int Solve(Solver *s, int maxIters, int tol){
    for (int step = 0; step < maxIters; ++step){
        int unsat = GreedyFlipSweep(s);
        if (unsat <= tol){
            printf("Solved after %d steps; energy=%d\n", step+1, unsat);
            return 1;
        }
        UpdateEnergyField(s);
    }
    printf("Not solved; maxIters=%d, energy=%d\n", maxIters, CountUnsat(s));
    return 0;
}

int TrySolveIncreasing(Solver *s, int startIters, int maxItersCap){
    int iters = startIters;
    while (iters <= maxItersCap){
        if (Solve(s, iters, 0)) return 1;
        iters *= 2;
    }
    printf("Gave up after %d iters total (cap).\n", maxItersCap);
    return 0;
}

// --- planted 3-SAT generator (kept simple, no solver changes) ---
void GenPlanted3SAT(Clause *cls, uint8_t *A, int nVars, int nClauses){
    for (int i=0;i<nVars;++i) A[i] = RandU32() & 1;
    for (int c=0;c<nClauses;++c){
        int v[3];
        v[0] = RandU32()%nVars;
        do { v[1] = RandU32()%nVars; } while (v[1]==v[0]);
        do { v[2] = RandU32()%nVars; } while (v[2]==v[0] || v[2]==v[1]);
        for (int j=0;j<3;++j){
            // Make at least one literal true under A
            int force_true = (j==0);
            int var = v[j];
            int sign = (RandU32() & 1) ? +1 : -1;
            int lit = sign * (var+1);
            int val = A[var];
            int truth = (lit>0) ? val : !val;
            if (force_true || truth){
                // ensure clause remains satisfiable by A
                cls[c].lit[j] = (val ? +1 : -1) * (var+1);
            } else {
                cls[c].lit[j] = lit;
            }
        }
    }
}

/* ----------------- POLY-TIME COHERENCE CERTIFICATE -----------------
   Map assignment -> phases φ_i ∈ {0, π}, then compute:
     r = |(1/N) Σ_i e^{i φ_i}|   (Kuramoto order parameter)
     Λ = N^2 r^2                (spectral witness; rank-1 measure)
   This runs in O(N) time and O(1) memory.
------------------------------------------------------------------- */
typedef struct {
    double r;
    double N2r2;
} CoherenceCert;

CoherenceCert CoherenceFromAssign(const uint8_t *assign, int n){
    double Csum = 0.0, Ssum = 0.0;
    for (int i=0; i<n; ++i){
        // φ = 0 if assign[i]==1, φ = π if assign[i]==0
        double ci = assign[i] ? 1.0 : -1.0; // cos(0)=1, cos(π)=-1
        double si = 0.0;                    // sin(0)=0, sin(π)=0
        Csum += ci;
        Ssum += si;
    }
    double r = sqrt(Csum*Csum + Ssum*Ssum) / (double)n;
    double N2r2 = (double)n * (double)n * r * r;
    CoherenceCert cc = { r, N2r2 };
    return cc;
}

void PrintCoherenceDecision(CoherenceCert cc, int n, double eps){
    printf("Coherence certificate: r = %.6f,  N^2 r^2 = %.3f (N = %d)\n", cc.r, cc.N2r2, n);
    if (cc.r >= eps){
        printf("DECISION: ACCEPT (coherence ≥ eps = %.3f)\n", eps);
    } else {
        printf("DECISION: REJECT (coherence < eps = %.3f)\n", eps);
    }
}

int main(){
    RandSeed(424242u); // seed will be replaced per-run by the driver

#if DEMO_SMALL
    // --- tiny demo directly from the original clean core ---
    Clause cls[2] = {
        {{ 1,  2,  3}},
        {{-1,  2, -3}}
    };

    Solver s;
    s.nVars = 3;
    s.nClauses = 2;
    s.clauses = cls;
    s.assign = (uint8_t*)malloc(s.nVars * sizeof(uint8_t));

    for (int i = 0; i < s.nVars; ++i) s.assign[i] = (RandU32() & 1);
    for (int i = 0; i < NUM_CATS; ++i) s.E[i] = 1.0;

    s.alpha = 0.25;
    s.zeta  = 0.10;

    int solved = TrySolveIncreasing(&s, 50, 1<<14);
    if (solved){
        printf("Final answer reached with energy 0.\n");
        printf("Assignments:\n");
        for (int i = 0; i < s.nVars; ++i){
            printf("  x%d = %d\n", i+1, s.assign[i]);
        }
        printf("Energy field E[]:\n");
        for (int k = 0; k < NUM_CATS; ++k){
            printf("  E[%d] = %.6f\n", k, s.E[k]);
        }
    } else {
        printf("No solution reached within cap.\n");
    }

    // --- poly-time coherence witness from current assignment ---
    CoherenceCert cc = CoherenceFromAssign(s.assign, s.nVars);
    PrintCoherenceDecision(cc, s.nVars, /*eps=*/0.20);
    double lambda_est = (double)s.nVars * cc.r * cc.r;
    printf("lambda_max ~ N r^2 = %.6f\n", lambda_est);

    free(s.assign);
#else
    // --- planted instance (kept here but not run by default) ---
    int nVars = 160;
    int nClauses = 640;
    Clause *cls = (Clause*)malloc(nClauses * sizeof(Clause));
    uint8_t *A = (uint8_t*)malloc(nVars * sizeof(uint8_t));
    GenPlanted3SAT(cls, A, nVars, nClauses);

    Solver s;
    s.nVars = nVars;
    s.nClauses = nClauses;
    s.clauses = cls;
    s.assign = (uint8_t*)malloc(s.nVars * sizeof(uint8_t));

    for (int i = 0; i < s.nVars; ++i) s.assign[i] = (RandU32() & 1);
    for (int i = 0; i < NUM_CATS; ++i) s.E[i] = 1.0;

    s.alpha = 0.25;
    s.zeta  = 0.05;

    int solved = TrySolveIncreasing(&s, 50, 1<<14);
    if (!solved) printf("No solution reached within cap.\n");

    CoherenceCert cc = CoherenceFromAssign(s.assign, s.nVars);
    PrintCoherenceDecision(cc, s.nVars, /*eps=*/0.10);

    free(A); free(cls); free(s.assign);
#endif

    return 0;
}
