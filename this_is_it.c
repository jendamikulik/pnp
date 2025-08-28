#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
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

int main(){
    RandSeed(42);

    // (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3)
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
    s.zeta  = 0.10; // nastav na 0.0 pro čistě deterministický běh

    // zvyšujeme iterace, dokud nedosáhneme energy==0 (unsat==0)
    int solved = TrySolveIncreasing(&s, 50, 1<<20);

    if (solved){
        printf("Final answer reached with energy 0.\n");

        // výpis finálního přiřazení
        printf("Assignments:\n");
        for (int i = 0; i < s.nVars; ++i){
            printf("  x%d = %d\n", i+1, s.assign[i]);
        }

        // výpis aktuálního energetického pole
        printf("Energy field E[]:\n");
        for (int k = 0; k < NUM_CATS; ++k){
            printf("  E[%d] = %.6f\n", k, s.E[k]);
        }
    } else {
        printf("No solution reached within cap.\n");
    }

    free(s.assign);
    return 0;
}
