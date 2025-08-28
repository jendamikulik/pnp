#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_CATS 5
#define IDX_LOVE 4

/* ================= RNG ================= */
static uint32_t rng_state = 1u;
void RandSeed(uint32_t seed){ if(seed==0) seed=1; rng_state=seed; }
uint32_t RandU32(void){
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

/* ================= SAT baseline (clean) ================= */
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
    for (int k = 0; k < NUM_CATS; ++k) s->E[k] += s->zeta * RandNorm();
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

/* planted 3-SAT generator */
void GenPlanted3SAT(Clause *cls, uint8_t *A, int nVars, int nClauses){
    for (int i=0;i<nVars;++i) A[i] = RandU32() & 1u;
    for (int c=0;c<nClauses;++c){
        int v[3];
        v[0] = RandU32()%nVars;
        do { v[1] = RandU32()%nVars; } while (v[1]==v[0]);
        do { v[2] = RandU32()%nVars; } while (v[2]==v[0] || v[2]==v[1]);
        for (int j=0;j<3;++j){
            int force_true = (j==0);
            int var = v[j];
            int sign = (RandU32() & 1u) ? +1 : -1;
            int lit = sign * (var+1);
            int val = A[var];
            int truth = (lit>0) ? val : !val;
            if (force_true || truth){
                cls[c].lit[j] = (val ? +1 : -1) * (var+1);
            } else {
                cls[c].lit[j] = lit;
            }
        }
    }
}

/* ====== Poly-time coherence certificate from assignment ====== */
typedef struct {
    double r;
    double N2r2;
} CoherenceCert;
CoherenceCert CoherenceFromAssign(const uint8_t *assign, int n){
    double Csum = 0.0, Ssum = 0.0;
    for (int i=0; i<n; ++i){
        double ci = assign[i] ? 1.0 : -1.0; // cos(0)=1, cos(pi)=-1
        Csum += ci;
    }
    double r = fabs(Csum) / (double)n; // Ssum=0
    double N2r2 = (double)n * (double)n * r * r;
    CoherenceCert cc = { r, N2r2 };
    return cc;
}
void PrintCoherenceDecision(CoherenceCert cc, int n, double eps){
    printf("Assign-coherence: r = %.6f,  N^2 r^2 = %.1f (N=%d)\n", cc.r, cc.N2r2, n);
    if (cc.r >= eps) printf(" -> ACCEPT (r >= %.3f)\n", eps);
    else             printf(" -> REJECT (r < %.3f)\n", eps);
}

/* ================= LST / Kuramoto surface ================= */
typedef struct {
    int N;
    double K0;      // global coupling
    double g;       // pinning to ref phase (0)
    double dt;
    int steps;
    double zeta;
    double *phi;    // phases
    double *omega;  // natural freqs
} LST;

void LST_Init(LST *L, int N, double K0, double g, double dt, int steps, double zeta){
    L->N=N; L->K0=K0; L->g=g; L->dt=dt; L->steps=steps; L->zeta=zeta;
    L->phi   = (double*)malloc(N*sizeof(double));
    L->omega = (double*)malloc(N*sizeof(double));
    for (int i=0;i<N;++i){
        L->phi[i]   = 2.0*M_PI*Rand01() - M_PI;
        L->omega[i] = 0.1 * ((int32_t)RandU32()/2147483648.0); // narrow band
    }
}
static inline void LST_Step(LST *L){
    int N=L->N;
    double *d = (double*)malloc(N*sizeof(double));
    for (int i=0;i<N;++i){
        double sum=0.0;
        for (int j=0;j<N;++j) if (j!=i) sum += sin(L->phi[j]-L->phi[i]);
        double drift = L->omega[i] + (L->K0/(double)N)*sum + L->g * sin(-L->phi[i]);
        if (L->zeta>0.0) drift += L->zeta * RandNorm();
        d[i] = L->dt * drift;
    }
    for (int i=0;i<N;++i){
        L->phi[i] += d[i];
        if (L->phi[i]>M_PI)  L->phi[i]-=2.0*M_PI;
        if (L->phi[i]<-M_PI) L->phi[i]+=2.0*M_PI;
    }
    free(d);
}
void LST_RunAndRecord(LST *L, double *phi_hist){ // size T*N
    for (int t=0;t<L->steps;++t){
        LST_Step(L);
        for (int i=0;i<L->N;++i) phi_hist[t*L->N + i] = L->phi[i];
    }
}
double Rbar_From_History(const double *phi_hist, int N, int T){
    double acc=0.0;
    for (int t=0;t<T;++t){
        double Cr=0.0, Sr=0.0;
        for (int i=0;i<N;++i){ Cr+=cos(phi_hist[t*N+i]); Sr+=sin(phi_hist[t*N+i]); }
        double r = sqrt(Cr*Cr + Sr*Sr) / (double)N;
        acc += r;
    }
    return acc/(double)T;
}

/* Gram matrix G_ij = <cos(phi_i - phi_j)>_T */
void Gram_From_History(const double *phi_hist, int N, int T, double *G){
    for (int i=0;i<N;++i){
        for (int j=0;j<N;++j){
            double acc=0.0;
            for (int t=0;t<T;++t){
                double di = phi_hist[t*N+i];
                double dj = phi_hist[t*N+j];
                acc += cos(di - dj);
            }
            G[i*N+j] = acc / (double)T;
        }
    }
}
/* Projected clause phases: theta_c(t) = mean of signed variable phases in clause c */
void Clause_Phases_From_History(const double *phi_hist, int N, int T, const Clause *cls, int m, double *theta_hist){
    for (int t=0;t<T;++t){
        for (int c=0;c<m;++c){
            double s=0.0;
            for (int j=0;j<3;++j){
                int lit = cls[c].lit[j];
                int v = abs(lit)-1;
                double sign = (lit>0)? +1.0 : -1.0;
                s += sign * phi_hist[t*N + v];
            }
            theta_hist[t*m + c] = s / 3.0;
        }
    }
}
/* Gram on clauses Gc_cd = <cos(theta_c - theta_d)>_T */
void Gram_From_Clauses(const double *theta_hist, int m, int T, double *Gc){
    for (int c=0;c<m;++c){
        for (int d=0; d<m; ++d){
            double acc=0.0;
            for (int t=0;t<T;++t){
                double tc = theta_hist[t*m + c];
                double td = theta_hist[t*m + d];
                acc += cos(tc - td);
            }
            Gc[c*m + d] = acc / (double)T;
        }
    }
}
/* power method for lambda_max */
double Power_MaxEigen(const double *G, int N, int iters){
    double *v = (double*)malloc(N*sizeof(double));
    double *w = (double*)malloc(N*sizeof(double));
    for (int i=0;i<N;++i) v[i] = 1.0/(double)N;
    double lambda=0.0;
    for (int it=0; it<iters; ++it){
        for (int i=0;i<N;++i){
            double s=0.0;
            for (int j=0;j<N;++j) s += G[i*N+j]*v[j];
            w[i]=s;
        }
        double norm=0.0; for (int i=0;i<N;++i) norm += w[i]*w[i];
        norm = sqrt(norm);
        if (norm<1e-15) break;
        for (int i=0;i<N;++i) v[i] = w[i]/norm;
        /* Rayleigh quotient */
        double num=0.0, den=0.0;
        for (int i=0;i<N;++i){
            double Gi=0.0; for (int j=0;j<N;++j) Gi += G[i*N+j]*v[j];
            num += v[i]*Gi; den += v[i]*v[i];
        }
        lambda = num/den;
    }
    free(v); free(w);
    return lambda;
}

int main(){
    RandSeed(20250828u);

    /* ---------- Build CNF (planted) ---------- */
    int nVars = 80;
    int nClauses = 280; // ratio ~3.5 to keep it reasonable for greedy
    Clause *cls = (Clause*)malloc(nClauses*sizeof(Clause));
    uint8_t *A = (uint8_t*)malloc(nVars*sizeof(uint8_t));
    GenPlanted3SAT(cls, A, nVars, nClauses);

    /* ---------- Solve with clean baseline ---------- */
    Solver s;
    s.nVars = nVars; s.nClauses = nClauses; s.clauses = cls;
    s.assign = (uint8_t*)malloc(nVars*sizeof(uint8_t));
    for (int i=0;i<nVars;++i) s.assign[i] = RandU32() & 1u;
    for (int k=0;k<NUM_CATS;++k) s.E[k] = 1.0;
    s.alpha = 0.25; s.zeta = 0.05;

    clock_t t0 = clock();
    int solved = TrySolveIncreasing(&s, 50, 1<<13); // up to 8192 sweeps
    clock_t t1 = clock();
    double ms = 1000.0*(t1 - t0)/CLOCKS_PER_SEC;
    printf("Solve time: %.2f ms\n", ms);

    if (solved){
        printf("Solver: energy=0 reached.\n");
    } else {
        printf("Solver: residual energy = %d\n", CountUnsat(&s));
    }

    /* Poly-time certificate on final assignment */
    CoherenceCert cc = CoherenceFromAssign(s.assign, s.nVars);
    PrintCoherenceDecision(cc, s.nVars, 0.10);
    printf("lambda_est ~ N r^2 = %.6f\n", (double)s.nVars * cc.r * cc.r);

    /* ---------- LST surface simulation ---------- */
    int N = nVars;            // tie LST nodes to variables
    int T = 1200;             // time steps
    LST L; LST_Init(&L, N, /*K0=*/1.1, /*g=*/0.6, /*dt=*/0.02, /*steps=*/T, /*zeta=*/0.02);
    double *phi_hist = (double*)malloc((size_t)N*(size_t)T*sizeof(double));
    LST_RunAndRecord(&L, phi_hist);

    double rbar = Rbar_From_History(phi_hist, N, T);
    printf("LST: rbar = %.4f\n", rbar);

    /* Gram on variables */
    double *G = (double*)malloc((size_t)N*(size_t)N*sizeof(double));
    Gram_From_History(phi_hist, N, T, G);
    double lambda = Power_MaxEigen(G, N, 60);
    printf("LST: lambda_max(G) = %.2f,  N rbar^2 ~ %.2f\n", lambda, (double)N*rbar*rbar);
    printf("LST decision (eps=0.20): %s\n", (lambda >= N*0.20*0.20) ? "ACCEPT" : "REJECT");

    /* Clause-projected surface */
    double *theta_hist = (double*)malloc((size_t)nClauses*(size_t)T*sizeof(double));
    Clause_Phases_From_History(phi_hist, N, T, cls, nClauses, theta_hist);
    double *Gc = (double*)malloc((size_t)nClauses*(size_t)nClauses*sizeof(double));
    Gram_From_Clauses(theta_hist, nClauses, T, Gc);
    double lambda_c = Power_MaxEigen(Gc, nClauses, 40);
    /* clause-level rbar */
    double acc=0.0;
    for (int t=0;t<T;++t){
        double Cr=0.0, Sr=0.0;
        for (int c=0;c<nClauses;++c){ Cr += cos(theta_hist[t*nClauses+c]); Sr += sin(theta_hist[t*nClauses+c]); }
        double rc = sqrt(Cr*Cr + Sr*Sr) / (double)nClauses;
        acc += rc;
    }
    double rbar_c = acc/(double)T;
    printf("Clause-surface: rbar_c=%.4f,  lambda_max(Gc)=%.2f,  C rbar_c^2 ~ %.2f\n",
           rbar_c, lambda_c, (double)nClauses*rbar_c*rbar_c);

    /* cleanup */
    free(Gc); free(theta_hist);
    free(G);  free(phi_hist);
    free(L.phi); free(L.omega);
    free(s.assign); free(cls); free(A);
    return 0;
}
