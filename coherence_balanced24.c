
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* ================= SAT baseline ================= */
typedef struct { int lit[3]; } Clause;
typedef struct {
    int nVars, nClauses;
    Clause *clauses;
    uint8_t *assign;
    double E[5];
    double alpha, zeta;
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
    for (int k = 0; k < 5; ++k) S += s->E[k];
    double mLove = s->E[4] / S;
    double delta = s->alpha * (1.0 - mLove);
    s->E[4] += delta;
    for (int k = 0; k < 5; ++k) s->E[k] -= delta / 5.0;
    for (int k = 0; k < 5; ++k) s->E[k] += s->zeta * RandNorm();
    for (int k = 0; k < 5; ++k) if (s->E[k] < 1e-9) s->E[k] = 1e-9;
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
            return 1;
        }
        UpdateEnergyField(s);
    }
    return 0;
}
int TrySolveIncreasing(Solver *s, int startIters, int maxItersCap){
    int iters = startIters;
    while (iters <= maxItersCap){
        if (Solve(s, iters, 0)) return 1;
        iters *= 2;
    }
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
typedef struct { double r, N2r2; } CoherenceCert;
CoherenceCert CoherenceFromAssign(const uint8_t *assign, int n){
    double Csum = 0.0;
    for (int i=0; i<n; ++i) Csum += assign[i] ? 1.0 : -1.0;
    double r = fabs(Csum) / (double)n;
    CoherenceCert cc = { r, (double)n * (double)n * r * r };
    return cc;
}

/* ================= LST / Kuramoto surface ================= */
typedef struct {
    int N, steps;
    double K0, g, dt, zeta;
    double *phi, *omega;
} LST;

void LST_Init(LST *L, int N, double K0, double g, double dt, int steps, double zeta){
    L->N=N; L->K0=K0; L->g=g; L->dt=dt; L->steps=steps; L->zeta=zeta;
    L->phi   = (double*)malloc((size_t)N*sizeof(double));
    L->omega = (double*)malloc((size_t)N*sizeof(double));
    for (int i=0;i<N;++i){
        L->phi[i]   = 2.0*M_PI*Rand01() - M_PI;
        L->omega[i] = 0.1 * ((int32_t)RandU32()/2147483648.0);
    }
}
static inline void LST_Step(LST *L){
    int N=L->N;
    double *d = (double*)malloc((size_t)N*sizeof(double));
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
void LST_RunAndRecord(LST *L, double *phi_hist){
    for (int t=0;t<L->steps;++t){
        LST_Step(L);
        for (int i=0;i<L->N;++i) phi_hist[(size_t)t*(size_t)L->N + i] = L->phi[i];
    }
}
double Rbar_From_History(const double *phi_hist, int N, int T){
    double acc=0.0;
    for (int t=0;t<T;++t){
        double Cr=0.0, Sr=0.0;
        for (int i=0;i<N;++i){ Cr+=cos(phi_hist[(size_t)t*(size_t)N+i]); Sr+=sin(phi_hist[(size_t)t*(size_t)N+i]); }
        double r = sqrt(Cr*Cr + Sr*Sr) / (double)N;
        acc += r;
    }
    return acc/(double)T;
}
void Gram_From_History(const double *phi_hist, int N, int T, double *G){
    for (int i=0;i<N;++i){
        for (int j=0;j<N;++j){
            double acc=0.0;
            for (int t=0;t<T;++t){
                double di = phi_hist[(size_t)t*(size_t)N+i];
                double dj = phi_hist[(size_t)t*(size_t)N+j];
                acc += cos(di - dj);
            }
            G[i*(size_t)N+j] = acc / (double)T;
        }
    }
}
void Clause_Phases_From_History(const double *phi_hist, int N, int T, const Clause *cls, int m, double *theta_hist){
    for (int t=0;t<T;++t){
        for (int c=0;c<m;++c){
            double s=0.0;
            for (int j=0;j<3;++j){
                int lit = cls[c].lit[j];
                int v = abs(lit)-1;
                double sign = (lit>0)? +1.0 : -1.0;
                s += sign * phi_hist[(size_t)t*(size_t)N + v];
            }
            theta_hist[(size_t)t*(size_t)m + c] = s / 3.0;
        }
    }
}
void Gram_From_Clauses(const double *theta_hist, int m, int T, double *Gc){
    for (int c=0;c<m;++c){
        for (int d=0; d<m; ++d){
            double acc=0.0;
            for (int t=0;t<T;++t){
                double tc = theta_hist[(size_t)t*(size_t)m + c];
                double td = theta_hist[(size_t)t*(size_t)m + d];
                acc += cos(tc - td);
            }
            Gc[c*(size_t)m + d] = acc / (double)T;
        }
    }
}
double Power_MaxEigen(const double *G, int N, int iters){
    double *v = (double*)malloc((size_t)N*sizeof(double));
    double *w = (double*)malloc((size_t)N*sizeof(double));
    for (int i=0;i<N;++i) v[i] = 1.0/(double)N;
    double lambda=0.0;
    for (int it=0; it<iters; ++it){
        for (int i=0;i<N;++i){
            double s=0.0;
            for (int j=0;j<N;++j) s += G[i*(size_t)N+j]*v[j];
            w[i]=s;
        }
        double norm=0.0; for (int i=0;i<N;++i) norm += w[i]*w[i];
        norm = sqrt(norm);
        if (norm<1e-15) break;
        for (int i=0;i<N;++i) v[i] = w[i]/norm;
        double num=0.0, den=0.0;
        for (int i=0;i<N;++i){
            double Gi=0.0; for (int j=0;j<N;++j) Gi += G[i*(size_t)N+j]*v[j];
            num += v[i]*Gi; den += v[i]*v[i];
        }
        lambda = num/den;
    }
    free(v); free(w);
    return lambda;
}

/* --------------- Helpers --------------- */
static int file_exists_nonempty(const char* path){
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return st.st_size > 0;
}
static int cmp_dbl(const void* a, const void* b){
    double da = *(const double*)a, db = *(const double*)b;
    if (da<db) return -1; if (da>db) return 1; return 0;
}
static double median(double* arr, int n){
    qsort(arr, n, sizeof(double), cmp_dbl);
    if (n%2) return arr[n/2];
    return 0.5*(arr[n/2-1] + arr[n/2]);
}
static double percentile(double* arr, int n, double p){
    qsort(arr, n, sizeof(double), cmp_dbl);
    double idx = p*(n-1);
    int lo = (int)floor(idx), hi = (int)ceil(idx);
    if (lo==hi) return arr[lo];
    double t = idx - lo;
    return (1.0-t)*arr[lo] + t*arr[hi];
}
static double cliffs_delta(const double* a, int na, const double* b, int nb){
    long long more=0, less=0;
    for (int i=0;i<na;++i){
        for (int j=0;j<nb;++j){
            if (a[i]>b[j]) ++more;
            else if (a[i]<b[j]) ++less;
        }
    }
    return (double)(more - less) / (double)(na*nb);
}

/* --------------- Per-seed run --------------- */
typedef struct {
    unsigned seed;
    int solved, energy;
    int N, Mfinal;
    double assign_r;
    double lambda_var, rbar_var;
    double lambda_clause, rbar_clause;
    double ms;
} RunRow;

RunRow RunOne(int nVars, int M_in, int T, double K0, double g, double zeta, double eps, unsigned seed, int unsatk_k){
    RunRow row; memset(&row, 0, sizeof(row));
    row.seed = seed; row.N = nVars;

    RandSeed(seed);

    // Build CNF
    int nClauses = M_in;
    Clause *cls = (Clause*)malloc((size_t)nClauses*sizeof(Clause));
    uint8_t *A  = (uint8_t*)malloc((size_t)nVars*sizeof(uint8_t));
    GenPlanted3SAT(cls, A, nVars, nClauses);
    if (unsatk_k > 0){
        int extra = 2*unsatk_k;
        Clause *cls2 = (Clause*)malloc((size_t)(nClauses+extra)*sizeof(Clause));
        for (int i=0;i<nClauses;++i) cls2[i]=cls[i];
        for (int u=0; u<unsatk_k; ++u){
            int v = u % nVars;
            cls2[nClauses + 2*u + 0].lit[0] =  (v+1);
            cls2[nClauses + 2*u + 0].lit[1] =  (v+1);
            cls2[nClauses + 2*u + 0].lit[2] =  (v+1);
            cls2[nClauses + 2*u + 1].lit[0] = -(v+1);
            cls2[nClauses + 2*u + 1].lit[1] = -(v+1);
            cls2[nClauses + 2*u + 1].lit[2] = -(v+1);
        }
        free(cls);
        cls = cls2;
        nClauses += extra;
    }
    row.Mfinal = nClauses;

    // Solve baseline
    Solver s;
    s.nVars = nVars; s.nClauses = nClauses; s.clauses = cls;
    s.assign = (uint8_t*)malloc((size_t)nVars*sizeof(uint8_t));
    for (int i=0;i<nVars;++i) s.assign[i] = RandU32() & 1u;
    for (int k=0;k<5;++k) s.E[k]=1.0;
    s.alpha=0.25; s.zeta=0.05;

    clock_t t0 = clock();
    int solved = TrySolveIncreasing(&s, 50, 1<<13);
    int energy_now = CountUnsat(&s);
    clock_t t1 = clock();
    row.ms = 1000.0*(t1 - t0)/CLOCKS_PER_SEC;
    row.solved = solved;
    row.energy = energy_now;

    CoherenceCert cc = CoherenceFromAssign(s.assign, s.nVars);
    row.assign_r = cc.r;

    // LST surface tied to variables
    int N = nVars;
    LST L; LST_Init(&L, N, K0, g, 0.02, T, zeta);
    double *phi_hist = (double*)malloc((size_t)N*(size_t)T*sizeof(double));
    LST_RunAndRecord(&L, phi_hist);
    double rbar = Rbar_From_History(phi_hist, N, T);
    double *G = (double*)malloc((size_t)N*(size_t)N*sizeof(double));
    Gram_From_History(phi_hist, N, T, G);
    double lambda = Power_MaxEigen(G, N, 60);
    row.lambda_var = lambda;
    row.rbar_var = rbar;

    // Clause-projected surface
    double *theta_hist = (double*)malloc((size_t)nClauses*(size_t)T*sizeof(double));
    Clause_Phases_From_History(phi_hist, N, T, cls, nClauses, theta_hist);
    double *Gc = (double*)malloc((size_t)nClauses*(size_t)nClauses*sizeof(double));
    Gram_From_Clauses(theta_hist, nClauses, T, Gc);
    double lambda_c = Power_MaxEigen(Gc, nClauses, 40);
    row.lambda_clause = lambda_c;
    row.rbar_clause = Rbar_From_History(theta_hist, nClauses, T);

    // Cleanup
    free(Gc); free(theta_hist);
    free(G); free(phi_hist);
    free(L.phi); free(L.omega);
    free(s.assign); free(cls); free(A);

    return row;
}

/* --------------- CSV write --------------- */
static void csv_write_header(FILE* f){
    fprintf(f, "class,seed,N,M_final,T,K0,g,zeta,eps,unsatk,solved,energy,assign_r,"
               "lambda_var,rbar_var,rho_var,lambda_clause,rbar_clause,rho_clause,solve_ms\n");
}
static void csv_write_row(FILE* f, const char* klass, RunRow r, int T, double K0, double g, double zeta, double eps, int unsatk){
    double rho_v = r.lambda_var / (double)r.N;
    double rho_c = r.lambda_clause / (double)r.Mfinal;
    fprintf(f, "%s,%u,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n",
        klass, r.seed, r.N, r.Mfinal, T, K0, g, zeta, eps, unsatk, r.solved, r.energy, r.assign_r,
        r.lambda_var, r.rbar_var, rho_v, r.lambda_clause, r.rbar_clause, rho_c, r.ms);
    fflush(f);
}

/* --------------- MAIN ---------------
Usage:
  ./coherence_balanced24.bin N M T K0 g zeta eps seeds_per_class seed0 unsatk_k out_csv sweep_csv [emin emax estep]
Defaults (sweet-spot-ish):
  N=96 M=336 T=600 K0=1.0 g=0.3 zeta=0.02 eps=0.20 seeds=24 seed0=424242 unsatk_k=6
  out_csv=/mnt/data/balanced24_runs.csv
  sweep_csv=/mnt/data/balanced24_eps.csv emin=0.01 emax=1.00 estep=0.01
------------------------------------- */
int main(int argc, char** argv){
    int N   = (argc>1)? atoi(argv[1]) : 96;
    int M   = (argc>2)? atoi(argv[2]) : 336;
    int T   = (argc>3)? atoi(argv[3]) : 600;
    double K0   = (argc>4)? atof(argv[4]) : 1.0;
    double g    = (argc>5)? atof(argv[5]) : 0.3;
    double zeta = (argc>6)? atof(argv[6]) : 0.02;
    double eps  = (argc>7)? atof(argv[7]) : 0.20;
    int seeds   = (argc>8)? atoi(argv[8]) : 24;
    unsigned seed0 = (argc>9)? (unsigned)strtoul(argv[9],NULL,10) : 424242u;
    int unsatk_k   = (argc>10)? atoi(argv[10]) : 6;
    const char* out_csv = (argc>11)? argv[11] : "/mnt/data/balanced24_runs.csv";
    const char* sweep_csv = (argc>12)? argv[12] : "/mnt/data/balanced24_eps.csv";
    double emin = (argc>13)? atof(argv[13]) : 0.01;
    double emax = (argc>14)? atof(argv[14]) : 1.00;
    double estep= (argc>15)? atof(argv[15]) : 0.01;

    FILE* fout = fopen(out_csv, "w");
    if (!fout){ fprintf(stderr, "Failed to open out_csv\\n"); return 1; }
    csv_write_header(fout);

    // storage for balanced summary
    double *rho_clause_S = (double*)malloc((size_t)seeds*sizeof(double));
    double *rho_clause_U = (double*)malloc((size_t)seeds*sizeof(double));
    double *lambda_var_S = (double*)malloc((size_t)seeds*sizeof(double));
    double *lambda_var_U = (double*)malloc((size_t)seeds*sizeof(double));
    int    *N_S = (int*)malloc((size_t)seeds*sizeof(int));
    int    *N_U = (int*)malloc((size_t)seeds*sizeof(int));
    int    *C_S = (int*)malloc((size_t)seeds*sizeof(int));
    int    *C_U = (int*)malloc((size_t)seeds*sizeof(int));

    // SAT seeds
    for (int i=0;i<seeds;++i){
        unsigned seed = seed0 + 97u*i;
        RunRow r = RunOne(N,M,T,K0,g,zeta,eps,seed, 0);
        csv_write_row(fout, "SAT", r, T, K0, g, zeta, eps, 0);
        rho_clause_S[i] = r.lambda_clause / (double)r.Mfinal;
        lambda_var_S[i] = r.lambda_var;
        N_S[i] = r.N; C_S[i] = r.Mfinal;
    }
    // UNSAT seeds
    for (int i=0;i<seeds;++i){
        unsigned seed = seed0 + 777u*i + 1000003u;
        RunRow r = RunOne(N,M,T,K0,g,zeta,eps,seed, unsatk_k);
        csv_write_row(fout, "UNSAT", r, T, K0, g, zeta, eps, unsatk_k);
        rho_clause_U[i] = r.lambda_clause / (double)r.Mfinal;
        lambda_var_U[i] = r.lambda_var;
        N_U[i] = r.N; C_U[i] = r.Mfinal;
    }
    fclose(fout);

    // Summary stats
    double med_S = median(rho_clause_S, seeds);
    double med_U = median(rho_clause_U, seeds);
    double p25_S = percentile(rho_clause_S, seeds, 0.25);
    double p75_S = percentile(rho_clause_S, seeds, 0.75);
    double p25_U = percentile(rho_clause_U, seeds, 0.25);
    double p75_U = percentile(rho_clause_U, seeds, 0.75);
    double delta_med = med_S - med_U;
    double cliff = cliffs_delta(rho_clause_S, seeds, rho_clause_U, seeds);

    printf("Balanced %d+%d @ N=%d, M=%d, T=%d, K0=%.2f, g=%.2f\\n", seeds, seeds, N, M, T, K0, g);
    printf("rho_clause: med(SAT)=%.5f [%.5f, %.5f], med(UNSAT)=%.5f [%.5f, %.5f], Δ=%.5f, Cliff's δ=%.3f\\n",
           med_S, p25_S, p75_S, med_U, p25_U, p75_U, delta_med, cliff);

    // Epsilon sweep
    FILE* fsw = fopen(sweep_csv, "w");
    if (!fsw){ fprintf(stderr, "Failed to open sweep_csv\\n"); return 0; }
    fprintf(fsw, "eps,acc_clause_SAT,acc_clause_UNSAT,delta_accept_clause,acc_var_SAT,acc_var_UNSAT,delta_accept_var\\n");
    for (double e=emin; e<=emax+1e-12; e+=estep){
        // acceptance: λ >= dim * e^2
        int acc_c_S=0, acc_c_U=0, acc_v_S=0, acc_v_U=0;
        for (int i=0;i<seeds;++i){
            if ( (rho_clause_S[i]*C_S[i]) >= (C_S[i]*e*e) ) ++acc_c_S;
            if ( (rho_clause_U[i]*C_U[i]) >= (C_U[i]*e*e) ) ++acc_c_U;
            if ( lambda_var_S[i] >= (N_S[i]*e*e) ) ++acc_v_S;
            if ( lambda_var_U[i] >= (N_U[i]*e*e) ) ++acc_v_U;
        }
        double a_c_S = (double)acc_c_S / (double)seeds;
        double a_c_U = (double)acc_c_U / (double)seeds;
        double a_v_S = (double)acc_v_S / (double)seeds;
        double a_v_U = (double)acc_v_U / (double)seeds;
        fprintf(fsw, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\\n",
                e, a_c_S, a_c_U, a_c_S-a_c_U, a_v_S, a_v_U, a_v_S-a_v_U);
    }
    fclose(fsw);

    printf("CSV (runs): %s\\n", out_csv);
    printf("CSV (eps):  %s\\n", sweep_csv);
    return 0;
}
