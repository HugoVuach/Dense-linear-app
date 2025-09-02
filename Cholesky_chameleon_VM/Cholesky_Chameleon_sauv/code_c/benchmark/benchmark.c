#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <stdbool.h>  // pour bool, true, false
#include <ctype.h>

#ifndef TARGET_BIN
#define TARGET_BIN "./v6_test"
#endif

// ===== Couleurs (désactivables) =====
static bool use_color = true;
#define COL_PURPLE (use_color ? "\x1b[35m" : "")
#define COL_RESET  (use_color ? "\x1b[0m"  : "")

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) == -1 && errno != EEXIST) {
            perror("mkdir");
            exit(EXIT_FAILURE);
        }
    }
}

static long ms_since(const struct timeval *a, const struct timeval *b) {
    long sec  = b->tv_sec  - a->tv_sec;
    long usec = b->tv_usec - a->tv_usec;
    return sec*1000 + usec/1000;
}

static void setenv_int(const char *k, int v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%d", v);
    setenv(k, buf, 1);
}

static void parse_metrics(const char *out, double *gflops, double *relerr) {
    const char *p;
    if (gflops) {
        *gflops = -1.0;
        p = strstr(out, "Performance:");
        if (p) {
            double tmp;
            if (sscanf(p, "Performance: %lf Gflop/s", &tmp) == 1) {
                *gflops = tmp;
            }
        }
    }
    if (relerr) {
        *relerr = -1.0;
        p = strstr(out, "||A - LL^T||_inf / ||A||_inf =");
        if (p) {
            double tmp;
            if (sscanf(p, "||A - LL^T||_inf / ||A||_inf = %lf", &tmp) == 1) {
                *relerr = tmp;
            }
        }
    }
}

int main(void) {
    // Active/désactive les couleurs
    use_color = isatty(STDOUT_FILENO);
    const char *env_nocolor = getenv("NO_COLOR");
    if (env_nocolor && *env_nocolor) use_color = false;
    
    // 1000, 5000, 8000, 12000, 16000
    const int Ns[]  = { 16000};
    const size_t nN = sizeof(Ns)/sizeof(Ns[0]);

    // 128, 192, 256,
    const int NBs[]  = { 320, 384, 448, 512};
    const size_t nNB = sizeof(NBs)/sizeof(NBs[0]);

    struct { int ncpu, ngpu; const char *name; } mappings[] = {
        {4, 0, "4_cpu_only"},
        {3, 1, "hybrid"},
    };
    const size_t nMap = sizeof(mappings)/sizeof(mappings[0]);

    // "eager","dm","dmdap","dmda","dmdas","heteroprio","pheft"
    //const char* scheds[] = {"pheft","dm"};
    //const size_t nSched = sizeof(scheds)/sizeof(scheds[0]);
    //const int repeats = 8; // calibration + 7 répétitions
    //long long total_steps = (long long)nN * nNB * nMap * nSched * repeats;


    
    // Listes de schedulers filtrées :
    // CPU-only : éviter ceux orientés GPU ; garder les plus utiles
    static const char* scheds_cpu_only[] = {"dm","dmda"};
    // Hybrid : CPU+GPU only
    static const char* scheds_hybrid[]   = {"dmda","dmdas","heteroprio","pheft"};

    const int repeats = 8; // calibration + 7 répétitions

    long long step_count  = 0;

    ensure_dir("results");
    const char *csv_path = "results/bench.csv";
    FILE *csv = fopen(csv_path, "a");
    if (!csv) { perror("fopen bench.csv"); return EXIT_FAILURE; }

    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) {
        fprintf(csv, "timestamp,scheduler,mapping,ncpu,ngpu,N,NB,run_idx,ms,exit_code,gflops,rel_error\n");
        fflush(csv);
    }

    for (size_t iN = 0; iN < nN; ++iN) {
        int N = Ns[iN];
        for (size_t iNB = 0; iNB < nNB; ++iNB) {
            int NB = NBs[iNB];

            int mb   = NB;
            int nb   = NB;
            int bsiz = NB*NB;
            int lm   = N;
            int ln   = N;
            int ioff = 0, joff = 0;
            int m    = N, n = N;
            int p = 1, q = 1;
            int seed = 42;

            for (size_t im = 0; im < nMap; ++im) {
                int ncpu = mappings[im].ncpu;
                int ngpu = mappings[im].ngpu;

                // Définir NCPU/NCUDA d'après le mapping courant (baseline)
                setenv_int("STARPU_NCPU", ncpu);
                setenv_int("STARPU_NCUDA", ngpu);

                // Sélection de la liste de schedulers selon le mapping
                const char **sched_list = NULL;
                size_t nSched = 0;
                if (strcmp(mappings[im].name, "4_cpu_only") == 0) {
                    sched_list = scheds_cpu_only;
                    nSched = sizeof(scheds_cpu_only)/sizeof(scheds_cpu_only[0]);
                } else { // hybrid
                    sched_list = scheds_hybrid;
                    nSched = sizeof(scheds_hybrid)/sizeof(scheds_hybrid[0]);
                }

                // Pour calculer la progression “totale” proprement pour l’affichage
                long long total_steps = (long long)nN * nNB * nMap * nSched * repeats;


                for (size_t is = 0; is < nSched; ++is) {
                    const char *sched = sched_list[is];

                    // Scheduler défini par la liste filtrée
                    setenv("STARPU_SCHED", sched, 1);

                    // Par défaut, on nettoie les variables de binding spécifiques (par run)
                    unsetenv("STARPU_WORKERS_CPUID");
                    unsetenv("STARPU_WORKERS_GETBIND");
                    unsetenv("STARPU_BIND");
                    unsetenv("CUDA_VISIBLE_DEVICES");
                    // setenv("STARPU_WORKER_STATS", "0", 1); // décommenter si besoin d'éteindre

                    // Si run 4_CPU_only : imposer TON mapping/binding
                    if (strcmp(mappings[im].name, "4_cpu_only") == 0) {

                        // Threading BLAS/OMP toujours à 1 pour ne pas perturber StarPU
                        setenv("OMP_NUM_THREADS", "1", 1);
                        setenv("MKL_NUM_THREADS", "1", 1);
                        setenv("OPENBLAS_NUM_THREADS", "1", 1);
                        setenv("STARPU_NCPU", "4", 1);         // force 4 CPU
                        setenv("STARPU_NCUDA", "0", 1);        // force 0 GPU
                        setenv("STARPU_WORKERS_CPUID", "0,2,4,6", 1);
                        setenv("STARPU_WORKERS_GETBIND", "0", 1);
                        setenv("STARPU_BIND", "1", 1);
                        // setenv("STARPU_WORKER_STATS", "1", 1);
                        setenv("CUDA_VISIBLE_DEVICES", "0", 1);
                    }
                    // Si run HYBRID : imposer TON mapping/binding
                    if (strcmp(mappings[im].name, "hybrid") == 0) {
                        
                        // Threading BLAS/OMP toujours à 1 pour ne pas perturber StarPU
                        setenv("OMP_NUM_THREADS", "1", 1);
                        setenv("MKL_NUM_THREADS", "1", 1);
                        setenv("OPENBLAS_NUM_THREADS", "1", 1);
                        setenv("STARPU_NCPU", "3", 1);         // force 3 CPU
                        setenv("STARPU_NCUDA", "1", 1);        // force 1 GPU
                        setenv("STARPU_WORKERS_CPUID", "2,4,6,0", 1);
                        setenv("STARPU_WORKERS_GETBIND", "0", 1);
                        setenv("STARPU_BIND", "1", 1);
                        // setenv("STARPU_WORKER_STATS", "1", 1);
                        setenv("CUDA_VISIBLE_DEVICES", "0", 1);
                    }

                    for (int r = 0; r < repeats; ++r) {
                        setenv("STARPU_CALIBRATE", (r == 0) ? "1" : "0", 1);

                        step_count++;
                        printf("%s------------------ sched=%s N=%d NB=%d mapping=%s étape %lld sur %lld -----------------------------%s\n",
                                COL_PURPLE, sched, N, NB, mappings[im].name, step_count, total_steps, COL_RESET);
                        fflush(stdout);

                        char ncpu_s[16], ngpu_s[16], N_s[16], NB_s[16], mb_s[16], nb_s[16];
                        char bsiz_s[32], lm_s[16], ln_s[16], ioff_s[16], joff_s[16];
                        char m_s[16], n_s[16], p_s[8], q_s[8], seed_s[16];

                        snprintf(ncpu_s, sizeof(ncpu_s), "%d", ncpu);
                        snprintf(ngpu_s, sizeof(ngpu_s), "%d", ngpu);
                        snprintf(N_s, sizeof(N_s), "%d", N);
                        snprintf(NB_s, sizeof(NB_s), "%d", NB);
                        snprintf(mb_s, sizeof(mb_s), "%d", mb);
                        snprintf(nb_s, sizeof(nb_s), "%d", nb);
                        snprintf(bsiz_s, sizeof(bsiz_s), "%d", bsiz);
                        snprintf(lm_s, sizeof(lm_s), "%d", lm);
                        snprintf(ln_s, sizeof(ln_s), "%d", ln);
                        snprintf(ioff_s, sizeof(ioff_s), "%d", ioff);
                        snprintf(joff_s, sizeof(joff_s), "%d", joff);
                        snprintf(m_s, sizeof(m_s), "%d", m);
                        snprintf(n_s, sizeof(n_s), "%d", n);
                        snprintf(p_s, sizeof(p_s), "%d", p);
                        snprintf(q_s, sizeof(q_s), "%d", q);
                        snprintf(seed_s, sizeof(seed_s), "%d", seed);

                        int pipefd[2];
                        if (pipe(pipefd) == -1) {
                            perror("pipe");
                            fclose(csv);
                            return EXIT_FAILURE;
                        }

                        struct timeval t0, t1;
                        gettimeofday(&t0, NULL);

                        pid_t pid = fork();
                        if (pid < 0) {
                            perror("fork");
                            fclose(csv);
                            return EXIT_FAILURE;
                        } else if (pid == 0) {
                            close(pipefd[0]);
                            dup2(pipefd[1], STDOUT_FILENO);
                            char *args[] = {
                                (char*)TARGET_BIN,
                                ncpu_s, ngpu_s, N_s, NB_s, mb_s, nb_s, bsiz_s,
                                lm_s, ln_s, ioff_s, joff_s, m_s, n_s, p_s, q_s, seed_s,
                                NULL
                            };
                            execvp(TARGET_BIN, args);
                            perror("execvp");
                            _exit(127);
                        } else {
                            close(pipefd[1]);
                            enum { OUT_CAP = 65536 };
                            char *outbuf = (char*)calloc(1, OUT_CAP);
                            ssize_t total = 0, rd;
                            while ((rd = read(pipefd[0], outbuf + total, OUT_CAP - 1 - total)) > 0) {
                                total += rd;
                                if (total >= OUT_CAP - 1) break;
                            }
                            close(pipefd[0]);

                            int status = 0;
                            waitpid(pid, &status, 0);
                            gettimeofday(&t1, NULL);
                            long ms = ms_since(&t0, &t1);
                            int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);

                            double gflops = -1.0, relerr = -1.0;
                            parse_metrics(outbuf, &gflops, &relerr);

                            time_t now = time(NULL);
                            struct tm tm_now;
                            localtime_r(&now, &tm_now);
                            char ts[32];
                            strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_now);

                            fprintf(csv, "%s,%s,%s,%d,%d,%d,%d,%d,%ld,%d,%.6f,%.6e\n",
                                    ts, sched, mappings[im].name, ncpu, ngpu, N, NB, r, ms, exit_code,
                                    gflops, relerr);
                            fflush(csv);

                            printf("   -> ms=%ld  GF=%.2f  err=%.2e  exit=%d\n",
                                   ms, gflops, relerr, exit_code);

                            free(outbuf);
                        }
                    } // repeats
                } // scheds
            } // mappings
        } // NB
    } // N

    fclose(csv);
    return EXIT_SUCCESS;
}
